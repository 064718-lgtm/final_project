"""
Streamlit UI for cactus presence inference.

Usage:
    streamlit run streamlit_app.py
Requires a trained model file (default: outputs/vgg16.keras).
Theme: 利用空拍影像進行氣候變遷預警之平台
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuration
DEFAULT_MODEL_PATH = Path("outputs/vgg16.keras")
DEMO_DIR = Path("DEMO")
IMAGE_SIZE = (96, 96)
DEFAULT_THRESHOLD = 0.5


@st.cache(allow_output_mutation=True)
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return tf.keras.models.load_model(model_path)


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
    sample_csv = Path("sample_submission.csv")
    if sample_csv.exists():
        import pandas as pd

        ids = pd.read_csv(sample_csv)["id"].tolist()
        return ids[:limit]
    return []


def load_demo_image(demo_key: str) -> bytes | None:
    mapping = {
        "DEMO_無仙人掌": DEMO_DIR / "0.jpg",
        "DEMO_有仙人掌": DEMO_DIR / "1.jpg",
    }
    path = mapping.get(demo_key)
    if path and path.exists():
        return path.read_bytes()
    return None


def load_sample_image(file_id: str) -> bytes | None:
    candidates = [
        Path("test") / file_id,
        Path("test") / "test" / file_id,
        Path("train") / file_id,
    ]
    for p in candidates:
        if p.exists():
            return p.read_bytes()
    for zpath, prefix in [(Path("test.zip"), "test"), (Path("train.zip"), "train")]:
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
        outputs_dir = Path("outputs")
        model_choices = [p for p in outputs_dir.glob("*.keras")] if outputs_dir.exists() else []
        default_choice = DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else (model_choices[0] if model_choices else "")
        model_select = st.selectbox(
            "選擇模型 (CNN/VGG16)",
            options=model_choices or [default_choice],
            format_func=lambda p: p.name if isinstance(p, Path) else str(p),
        )
        model_path = Path(model_select) if model_select else DEFAULT_MODEL_PATH
        threshold = st.slider(
            "判定閾值 (存在機率 >= 閾值 即視為有仙人掌)",
            min_value=0.1,
            max_value=0.9,
            value=float(DEFAULT_THRESHOLD),
            step=0.05,
        )
        invert_pred = "vgg" in str(model_path).lower()
        preset_options = ["DEMO_無仙人掌", "DEMO_有仙人掌"]
        sample_choice = st.selectbox(
            "Demo 範例影像（固定 DEMO/0.jpg 與 DEMO/1.jpg）",
            options=["(上傳自選)"] + preset_options,
        )
        st.markdown(
            "操作步驟：\n"
            "1) 在這裡選擇模型檔（`.keras`，放在 `outputs/`）。\n"
            "2) 若要快速展示，可在下方選擇 DEMO 範例影像（DEMO/0=無仙人掌，DEMO/1=有仙人掌）。\n"
            "3) 或切換到主畫面上傳 JPG/PNG，自行推論。\n"
            "4) 推論後會顯示機率、判定、Grad-CAM 熱力圖與暖化提醒。"
        )

    st.markdown("### 上傳影像")
    st.caption("支援 JPG/PNG，建議使用清晰、光線充足的空拍視角。")
    uploaded = st.file_uploader("選擇一張影像", type=["jpg", "jpeg", "png"])

    image = None
    image_caption = ""
    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read()))
        image_caption = "上傳影像預覽"
    elif sample_choice and sample_choice != "(上傳自選)":
        sample_bytes = load_demo_image(sample_choice)
        if sample_bytes:
            image = Image.open(io.BytesIO(sample_bytes))
            image_caption = f"範例影像：{sample_choice}"
        else:
            st.warning("找不到 DEMO 範例影像，請確認 DEMO/0.jpg 與 DEMO/1.jpg 是否存在。")

    if image:
        st.image(image, caption=image_caption, use_column_width=True)

        try:
            model = load_model(Path(model_path))
        except FileNotFoundError as e:
            st.error(str(e))
            return

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

        prob_display = 1 - prob if 'invert_pred' in locals() and invert_pred else prob
        has_cactus = prob_display >= threshold
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("仙人掌機率", f"{prob_display*100:.2f}%", delta=None)
        with col2:
            st.metric("判定", "存在" if has_cactus else "未檢測到")
        with col3:
            st.metric("閾值", f"{threshold:.2f}")

        st.markdown("---")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if has_cactus:
            st.markdown(
                "**根據這個空排結果圖，模型判定有仙人掌存在。** "
                "後台分析顯示在此影像區域，氣候暖化的趨勢可能加劇乾旱環境，"
                "請留意生態變化並持續監測水資源與植被變化。"
            )
        else:
            st.markdown(
                "根據這個空排結果圖，**未檢測到仙人掌。** "
                "若需更精細的結果，可提供更高解析度或不同角度的影像。"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Grad-CAM 關注熱力圖")
        if overlay is not None:
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(resized, caption="輸入影像 (縮放後)", use_column_width=True)
            with gc2:
                st.image(overlay, caption="Grad-CAM 熱力圖覆蓋", use_column_width=True)
        else:
            st.info("此模型未找到 Conv2D 層，無法產生 Grad-CAM。")

        st.caption("可在側邊欄調整判定閾值；閾值越低，越容易判定為有仙人掌。")
    else:
        st.info("請上傳影像，或在側邊欄選擇範例影像以進行推論。")


if __name__ == "__main__":
    main()
