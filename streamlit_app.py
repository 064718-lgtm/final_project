"""
Streamlit UI for cactus presence inference.

Usage:
    streamlit run streamlit_app.py
Requires a trained model file (default: outputs/vgg16.keras).
Theme: 利用空拍影像進行氣候變遷預警之平台
"""

from __future__ import annotations

import hashlib
import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuration
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_UPLOAD_DIR = Path(tempfile.gettempdir()) / "cactus_models"
DEFAULT_MODEL_PATH = OUTPUTS_DIR / "vgg16.keras"
IMAGE_SIZE = (96, 96)
DEFAULT_THRESHOLD = 0.5
LOCAL_LLM_MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"


def cache_resource_compat(**kwargs):
    if hasattr(st, "cache_resource"):
        kwargs.pop("allow_output_mutation", None)
        return st.cache_resource(**kwargs)
    return st.cache(**kwargs)


@cache_resource_compat(
    show_spinner=False,
    hash_funcs={
        Path: lambda p: (str(p), p.stat().st_mtime_ns, p.stat().st_size)
        if p.exists()
        else (str(p), None, None)
    },
    allow_output_mutation=True,
)
def build_custom_objects() -> dict:
    custom = {}
    for name in ("TFOpLambda", "SlicingOpLambda"):
        layer = getattr(tf.keras.layers, name, None)
        if layer is not None:
            custom[name] = layer
    return custom


def load_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    custom_objects = build_custom_objects()
    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False,
        )
    except ValueError:
        if model_path.suffix.lower() == ".keras" and is_hdf5_file(model_path):
            h5_path = coerce_hdf5_path(model_path)
            try:
                return tf.keras.models.load_model(
                    h5_path,
                    compile=False,
                    custom_objects=custom_objects,
                    safe_mode=False,
                )
            except TypeError:
                return tf.keras.models.load_model(
                    h5_path,
                    compile=False,
                    custom_objects=custom_objects,
                )
        raise
    except TypeError:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,
        )


def list_model_files(outputs_dir: Path) -> list[Path]:
    if not outputs_dir.exists():
        return []
    candidates = []
    for pattern in ("*.keras", "*.h5"):
        candidates.extend(outputs_dir.glob(pattern))
    return sorted(candidates)


def save_uploaded_model(uploaded_file) -> Path:
    MODEL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_UPLOAD_DIR / uploaded_file.name
    model_path.write_bytes(uploaded_file.getbuffer())
    return model_path


def is_hdf5_file(model_path: Path) -> bool:
    try:
        import h5py
    except Exception:
        return False
    try:
        return h5py.is_hdf5(model_path)
    except Exception:
        return False


def coerce_hdf5_path(model_path: Path) -> Path:
    h5_dir = MODEL_UPLOAD_DIR / "h5"
    h5_dir.mkdir(parents=True, exist_ok=True)
    h5_path = h5_dir / f"{model_path.stem}.h5"
    if not h5_path.exists() or h5_path.stat().st_mtime_ns < model_path.stat().st_mtime_ns:
        h5_path.write_bytes(model_path.read_bytes())
    return h5_path


def default_climate_advice(has_cactus: bool) -> str:
    if has_cactus:
        return (
            "目前氣候變遷壓力不嚴重，但仍需注意可能影響該地區環境的跡象：\n"
            "- 乾旱期延長或降雨變得不穩定\n"
            "- 植被覆蓋下降、裸地比例增加\n"
            "- 土壤含水降低、地表龜裂或沙化\n"
            "- 水源補給減少或水質惡化\n"
            "若上述趨勢持續，可能逐步削弱當地生態韌性。"
        )
    return (
        "目前氣候變遷壓力偏高，建議立即採取處置以避免更嚴重的情況：\n"
        "- 啟動水資源管理與節水措施\n"
        "- 進行棲地/植被復育，減少土地擾動\n"
        "- 建立土壤含水、植被覆蓋的監測機制\n"
        "- 設置早期預警與社區協作應變\n"
        "透過持續監測與介入可降低退化風險。"
    )


@cache_resource_compat(show_spinner=False, allow_output_mutation=True)
def load_local_llm(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_llm_prompt(has_cactus: bool, prob: float, threshold: float, model_name: str) -> str:
    status = "檢測到仙人掌" if has_cactus else "未檢測到仙人掌"
    severity = "氣候變遷不嚴重" if has_cactus else "氣候變遷嚴重"
    return (
        "你是環境風險分析師。請用繁體中文輸出 3-5 點條列建議。\n"
        "必須包含結論（氣候變遷不嚴重/嚴重）與後續注意跡象或處置措施。\n"
        "範例：\n"
        "輸入：檢測到仙人掌\n"
        "輸出：\n"
        "- 氣候變遷不嚴重，但需注意乾旱期延長\n"
        "- 觀察植被覆蓋是否下降\n"
        "輸入：未檢測到仙人掌\n"
        "輸出：\n"
        "- 氣候變遷嚴重，建議啟動節水與復育\n"
        "- 加強土壤含水與植被監測\n"
        "現在輸入：\n"
        f"模型={model_name}；結果={status}；機率={prob:.2f}；閾值={threshold:.2f}\n"
        f"結論：{severity}\n"
        "輸出：\n"
    )


def format_llm_output(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = [
        line
        for line in lines
        if line.startswith(("-", "•", "1.", "2.", "3.", "4.", "5."))
    ]
    if bullet_lines:
        return "\n".join(bullet_lines[:5])
    parts = (
        text.replace("。", "。\n")
        .replace("！", "！\n")
        .replace("？", "？\n")
        .splitlines()
    )
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return ""
    parts = parts[:5]
    return "\n".join(f"- {p}" for p in parts)


def generate_local_advice(
    has_cactus: bool,
    prob: float,
    threshold: float,
    model_name: str,
    model_id: str,
) -> tuple[str | None, str | None]:
    try:
        model, tokenizer = load_local_llm(model_id)
    except Exception as e:
        return None, str(e)
    prompt = build_llm_prompt(has_cactus, prob, threshold, model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        formatted = format_llm_output(generated)
        if not formatted:
            return None, "LLM 輸出為空"
        return formatted, None
    except Exception as e:
        return None, str(e)


def compute_image_hash(data: bytes | None) -> str | None:
    if not data:
        return None
    return hashlib.sha256(data).hexdigest()


def reset_advice_state() -> None:
    st.session_state.pop("llm_advice", None)
    st.session_state.pop("llm_meta", None)


def reset_prediction_state() -> None:
    st.session_state.pop("prediction", None)
    st.session_state.pop("prediction_meta", None)
    st.session_state.pop("prediction_error", None)
    reset_advice_state()


def request_llm_advice() -> None:
    st.session_state["run_llm_advice"] = True


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image, model) -> float:
    arr = preprocess_image(image)
    prob = model.predict(arr, verbose=0)[0][0]
    return float(prob)


def find_last_conv_layer(root) -> tf.keras.layers.Layer | None:
    """Return the last Conv2D layer object (search recursively)."""
    last_conv = None

    def _walk(layer):
        nonlocal last_conv
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                _walk(sub)

    # handle model or layer
    children = getattr(root, "layers", None) or []
    for lyr in children:
        _walk(lyr)
    return last_conv


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    """Generate Grad-CAM heatmap for a single image array (HWC uint8)."""
    input_tensor = tf.cast(img_array, tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    def _generic(conv_layer_obj):
        grad_model = tf.keras.models.Model(
            [model.inputs], [conv_layer_obj.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap_local = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap_local = tf.squeeze(heatmap_local)
        heatmap_local = tf.maximum(heatmap_local, 0) / (tf.reduce_max(heatmap_local) + 1e-8)
        return heatmap_local.numpy()

    # Try VGG-specific graph if present
    try:
        vgg_layer = model.get_layer("vgg16")
    except Exception:
        vgg_layer = None

    if vgg_layer is not None:
        conv_layer_vgg = find_last_conv_layer(vgg_layer)
        aug_layer = None
        try:
            aug_layer = model.get_layer("data_augmentation")
        except Exception:
            pass
        x = aug_layer(input_tensor, training=False) if aug_layer is not None else input_tensor
        x = tf.keras.applications.vgg16.preprocess_input(x)
        if conv_layer_vgg is None:
            raise ValueError("No Conv2D layer found in VGG16 for Grad-CAM.")
        conv_model = tf.keras.models.Model(
            inputs=vgg_layer.input, outputs=[conv_layer_vgg.output, vgg_layer.output]
        )
        try:
            gap_layer = model.get_layer("global_average_pooling2d_1")
            drop_layer = model.get_layer("dropout_1")
            dense_layer = model.get_layer("dense_1")
        except Exception:
            gap_layer = drop_layer = dense_layer = None
        if None in [gap_layer, drop_layer, dense_layer]:
            raise ValueError("Missing pooling/dropout/dense layers for Grad-CAM.")
        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = conv_model(x, training=False)
            head = gap_layer(base_outputs)
            head = drop_layer(head, training=False)
            predictions = dense_layer(head)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
    else:
        conv_layer_generic = find_last_conv_layer(model)
        if conv_layer_generic is None:
            raise ValueError("No Conv2D layer found for Grad-CAM.")
        heatmap = _generic(conv_layer_generic)

    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((img_array.shape[1], img_array.shape[0]))
    heatmap = np.array(heatmap) / 255.0
    return heatmap


def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, intensity: float = 0.4) -> Image.Image:
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    heat_rgb = np.zeros_like(base)
    heat_rgb[:, :, 0] = heatmap * 255  # red channel
    blended = base * (1 - intensity) + heat_rgb * intensity
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def list_sample_ids(limit: int = 8):
    sample_csv = BASE_DIR / "sample_submission.csv"
    if sample_csv.exists():
        import pandas as pd

        ids = pd.read_csv(sample_csv)["id"].tolist()
        return ids[:limit]
    return []


def load_sample_image(file_id: str) -> bytes | None:
    candidates = [
        BASE_DIR / "test" / file_id,
        BASE_DIR / "test" / "test" / file_id,
        BASE_DIR / "train" / file_id,
    ]
    for p in candidates:
        if p.exists():
            return p.read_bytes()
    for zpath, prefix in [(BASE_DIR / "test.zip", "test"), (BASE_DIR / "train.zip", "train")]:
        if zpath.exists():
            with zipfile.ZipFile(zpath) as z:
                for name in (f"{prefix}/{file_id}", file_id):
                    try:
                        return z.read(name)
                    except KeyError:
                        continue
    return None


def render_header() -> None:
    st.markdown(
        """
        <style>
        body, .stApp {
            background: #f8f7f2;
            color: #2f3e46;
            font-family: "Segoe UI", "Noto Sans", sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2f3e46;
            letter-spacing: 0.2px;
        }
        .stMarkdown, .stText, .stMetric, p, label, span {
            color: #34444d !important;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .hero {
            background: linear-gradient(140deg, #eef7f4 0%, #f7f5ef 50%, #e9f2fb 100%);
            color: #2f3e46;
            padding: 22px;
            border-radius: 18px;
            margin-bottom: 20px;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.10);
        }
        .hero h1 { margin: 0 0 6px 0; }
        .hero p { margin: 6px 0; color: #3d4b53; }
        .card {
            padding: 16px;
            border-radius: 16px;
            border: 1px solid #e7ecef;
            background: #ffffffee;
            box-shadow: 0 8px 22px rgba(0,0,0,0.08);
            backdrop-filter: blur(4px);
        }
        .stButton>button, .stDownloadButton>button {
            background: #dff0e3;
            color: #2f3e46;
            border-radius: 12px;
            border: 1px solid #cfe5d7;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        }
        .stSlider [role='slider'] {
            background: #88b9aa;
        }
        </style>
        <div class="hero">
          <h1>利用空拍影像進行氣候變遷預警之平台 🌵</h1>
          <p>上傳影像 → 檢測仙人掌 → 提醒氣候暖化風險，協助環境監測決策。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="利用空拍影像進行氣候變遷預警之平台",
        page_icon="🌵",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    render_header()

    with st.sidebar:
        st.subheader("設定")
        model_choices = list_model_files(OUTPUTS_DIR)
        model_select = None
        st.markdown("**模型來源**")
        st.caption("選擇要使用的模型檔，會影響預測結果。")
        if model_choices:
            model_select = st.selectbox(
                "模型檔",
                options=model_choices,
                format_func=lambda p: p.name if isinstance(p, Path) else str(p),
                key="model_select",
                on_change=reset_prediction_state,
                label_visibility="collapsed",
            )
        else:
            st.info("目前未找到可用模型，可改用下方上傳。")

        st.markdown("**上傳模型**")
        st.caption("若清單中沒有模型，可上傳 .keras 或 .h5 檔。")
        uploaded_model = st.file_uploader(
            "上傳模型檔",
            type=["keras", "h5"],
            key="uploaded_model",
            on_change=reset_prediction_state,
            label_visibility="collapsed",
        )
        model_path = None
        if uploaded_model is not None:
            model_path = save_uploaded_model(uploaded_model)
        elif model_select:
            model_path = Path(model_select)

        st.markdown("**判定門檻**")
        st.caption("機率高於門檻時，視為偵測到仙人掌。")
        threshold = st.slider(
            "判定門檻",
            min_value=0.1,
            max_value=0.9,
            value=float(DEFAULT_THRESHOLD),
            step=0.05,
            key="threshold",
            on_change=reset_advice_state,
            label_visibility="collapsed",
        )
        invert_pred = "vgg" in str(model_path).lower() if model_path else False

        st.markdown("**氣候解讀**")
        st.caption("啟用本地輕量模型，生成改善建議。")
        enable_llm = st.checkbox(
            "啟用氣候解讀",
            value=True,
            key="enable_llm",
            on_change=reset_advice_state,
        )
        if enable_llm:
            st.caption("首次啟用會下載模型，可能需要 1-2 分鐘。")

        st.markdown("---")
        st.markdown(
            "**快速導覽**\n"
            "1) 選擇或上傳模型\n"
            "2) 上傳影像\n"
            "3) 查看結果後按「生成改善建議」取得建議"
        )

    st.markdown("### 上傳影像")
    st.caption("支援 JPG/PNG，建議使用清晰、光線充足的空拍視角。")
    uploaded = st.file_uploader(
        "選擇一張影像",
        type=["jpg", "jpeg", "png"],
        key="uploaded_image",
        on_change=reset_prediction_state,
    )
    uploaded_bytes = None
    image = None
    image_caption = ""
    if uploaded:
        uploaded_bytes = uploaded.getvalue()
        image = Image.open(io.BytesIO(uploaded_bytes))
        image_caption = "上傳影像預覽"

    if image:
        st.image(image, caption=image_caption, use_container_width=True)
    else:
        st.info("請上傳影像。")

    if not model_path:
        st.info("請先在側邊欄選擇 outputs/*.keras 或上傳模型檔。")

    current_meta = {
        "image_hash": compute_image_hash(uploaded_bytes),
        "model_path": str(model_path) if model_path else None,
        "model_mtime": model_path.stat().st_mtime_ns if model_path and model_path.exists() else None,
        "model_size": model_path.stat().st_size if model_path and model_path.exists() else None,
    }

    prediction = st.session_state.get("prediction")
    prediction_meta = st.session_state.get("prediction_meta")
    should_predict = image and model_path and current_meta["image_hash"]

    if should_predict and prediction_meta != current_meta:
        model_path = Path(model_path)
        if not model_path.exists():
            st.error(f"找不到模型檔：{model_path}")
            st.stop()
        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"模型載入失敗：{e}")
            st.stop()

        with st.spinner("模型推論中..."):
            prob = predict(image, model)
            resized = image.resize(IMAGE_SIZE)
            try:
                heatmap = make_gradcam_heatmap(np.asarray(resized), model)
                overlay = overlay_heatmap(resized, heatmap)
            except Exception as e:
                heatmap = None
                overlay = None
                st.warning(f"Grad-CAM 生成失敗：{e}")

        prob_display = 1 - prob if invert_pred else prob
        st.session_state["prediction"] = {
            "prob_display": prob_display,
            "resized": resized,
            "overlay": overlay,
            "model_name": model_path.name,
        }
        st.session_state["prediction_meta"] = current_meta
        reset_advice_state()

    prediction = st.session_state.get("prediction")
    prediction_meta = st.session_state.get("prediction_meta")
    if prediction and prediction_meta == current_meta:
        st.markdown("---")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        has_cactus = prediction["prob_display"] >= threshold
        if has_cactus:
            st.markdown(
                "**偵測到仙人掌，環境韌性仍在，氣候變遷壓力暫不嚴重。** "
                "保持定期巡檢與水資源管理，持續追蹤後續變化即可。"
            )
        else:
            st.markdown(
                "**未檢測到仙人掌，請啟動氣候變遷警示。** "
                "檢查棲地/灌溉/植被管理狀態，並評估是否需補植或加強保育行動。"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("仙人掌機率", f"{prediction['prob_display']*100:.2f}%", delta=None)
        with col2:
            st.metric("判定", "存在" if has_cactus else "未檢測到")
        with col3:
            st.metric("閾值", f"{threshold:.2f}")

        st.markdown("#### Grad-CAM 關注熱力圖")
        if prediction["overlay"] is not None:
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(prediction["resized"], caption="輸入影像 (縮放後)", use_container_width=True)
            with gc2:
                st.image(prediction["overlay"], caption="Grad-CAM 熱力圖覆蓋", use_container_width=True)
        else:
            st.info("此模型未找到 Conv2D 層，無法產生 Grad-CAM。")

        st.caption("可在側邊欄調整判定閾值；閾值越低，越容易判定為有仙人掌。")

        st.markdown("#### LLM 改善建議")
        st.button(
            "生成改善建議",
            type="primary",
            on_click=request_llm_advice,
        )
        st.caption("按下「生成改善建議」以產生 LLM 建議。")

        current_llm_meta = {
            "prediction_meta": prediction_meta,
            "threshold": float(threshold),
            "enable_llm": bool(enable_llm),
        }
        should_generate_llm = st.session_state.pop("run_llm_advice", False)
        if should_generate_llm:
            advice_text = None
            llm_error = None
            if enable_llm:
                advice_text, llm_error = generate_local_advice(
                    has_cactus=has_cactus,
                    prob=prediction["prob_display"],
                    threshold=threshold,
                    model_name=prediction["model_name"],
                    model_id=LOCAL_LLM_MODEL_ID,
                )
            if not advice_text:
                advice_text = default_climate_advice(has_cactus)
                if not enable_llm and not llm_error:
                    llm_error = "LLM 未啟用，已改用預設文字"
            st.session_state["llm_advice"] = {"text": advice_text, "error": llm_error}
            st.session_state["llm_meta"] = current_llm_meta

        llm_advice = st.session_state.get("llm_advice")
        llm_meta = st.session_state.get("llm_meta")
        if llm_advice and llm_meta == current_llm_meta:
            st.markdown(llm_advice["text"])
            if llm_advice["error"] and enable_llm:
                st.caption(f"LLM 生成失敗，已改用預設文字：{llm_advice['error']}")
            elif llm_advice["error"] and not enable_llm:
                st.caption(llm_advice["error"])
        else:
            st.info("按「生成改善建議」生成改善建議。")
    elif image:
        st.info("已上傳影像，請等待模型推論或確認模型檔。")


if __name__ == "__main__":
    main()
