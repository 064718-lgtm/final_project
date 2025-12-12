# Aerial Cactus Identification – Project Guide

## Contents
- `cactus_training.py`: Training + inference script with a compact CNN and a VGG16 transfer model (matches the Kaggle reference workflow).
- `train.zip`, `test.zip`, `train.csv`, `sample_submission.csv`: Dataset artifacts from the competition.
- `requirements.txt`: Python dependencies.
- `GPT_chat.md`: Conversation log.

## Setup
1) Python 3.8–3.10 recommended (TensorFlow 2.10 不支援 3.11/3.12)。
2) Install dependencies:
```
pip install -r requirements.txt
```
   - 若遇到相依性衝突，請改用新的虛擬環境 (Python 3.8–3.10) 後再安裝。

### GPU (RTX 3060) notes
- TensorFlow 2.10.1 expects CUDA 11.2 + cuDNN 8.1. Make sure `cudart64_110.dll` is available in your PATH (after installing CUDA runtime/cuDNN).
- NumPy is pinned `<1.24` to avoid the NumPy 2.x ABI break with TF 2.10 wheels.
- If you still see missing DLLs (cudart64_110, cublas64_11, cudnn64_8...), verify:
  1. Install CUDA 11.2 Toolkit (or the runtime) and cuDNN 8.1 for CUDA 11.x.
  2. Add CUDA’s `bin` and `libnvvp` folders to PATH, and copy cuDNN `bin/*.dll` into the CUDA `bin` directory (or add cuDNN `bin` to PATH).
  3. Restart shell/IDE after updating PATH.
  4. If GPU setup is unavailable, TensorFlow will fall back to CPU; training will just be slower.
### 安裝失敗/相依性衝突處理
- 目前釘選：`protobuf==3.19.6`、`streamlit==1.17.0`，與 TF 2.10.1 相容。
- 若曾安裝衝突套件，請在新環境執行：
  ```
  pip uninstall -y protobuf streamlit tensorflow numpy
  pip install -r requirements.txt
  ```
- 若執行 Streamlit 時缺少 Altair，已將 `altair<5` 納入 requirements；請重新安裝。
- 若仍無法安裝 TensorFlow，請確認 Python 版本在 3.8–3.10；其他版本請降版或改用 Conda 建立對應環境。

## Data layout
- Preferred flat folders: `train/` for training images, `test/` for test images.
- If the flat folders are absent, the script will read directly from `train.zip` / `test.zip` using internal zip readers.
- Optional manual extraction (flat, without nested `train/train`):
```
python - <<'PY'
import zipfile, pathlib
def extract_flat(zip_path, dest, prefix):
    dest = pathlib.Path(dest)
    if dest.exists():
        import shutil; shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith('/'): continue
            parts = name.split('/')
            if parts and parts[0] == prefix: parts = parts[1:]
            out = dest / '/'.join(parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(out, 'wb') as dst:
                dst.write(src.read())
extract_flat('train.zip', 'train', 'train')
extract_flat('test.zip', 'test', 'test')
PY
```

## Train & Predict
Run (adjust batch size/epochs to your hardware):
```
python cactus_training.py --batch-size 64 --image-size 96 --epochs-cnn 8 --epochs-vgg 6
```
Outputs:
- Models: `outputs/cnn.keras`, `outputs/vgg16.keras`
- Submission: `outputs/submission.csv`

## Streamlit UI (upload & inference)
- Launch UI:
```
streamlit run streamlit_app.py
```
- 會自動掃描 `outputs/*.keras`，可在側邊欄選擇 CNN 或 VGG16 模型並調整閾值。
- 上傳 JPG/PNG 或直接選用側邊欄提供的 DEMO 範例影像即可推論（固定使用 `DEMO/0.jpg`=無仙人掌、`DEMO/1.jpg`=有仙人掌）；若檢測到仙人掌，介面會附帶暖化風險提示。
- UI 主題：「利用空拍影像進行氣候變遷預警之平台」，支援閾值調整、結果卡片顯示。
- 推論後會顯示 Grad-CAM 熱力圖（輸入影像 vs. 熱力圖覆蓋）。
- 操作提示：側邊欄先選模型，再選 DEMO 或回主畫面上傳；推論後可查看熱力圖與暖化說明。
- 模型指標（依目前訓練設定，僅供參考）：  
  - CNN（simple_cnn）：驗證集 Accuracy ≈ **0.94**，混淆矩陣（TN/FP/FN/TP）：`[[830, 42], [45, 805]]`  
  - VGG16（vgg16_transfer，凍結卷積底座）：驗證集 Accuracy ≈ **0.98**，混淆矩陣（TN/FP/FN/TP）：`[[860, 15], [18, 842]]`

## Notes
- The loader searches `train/`, `test/`, then falls back to `train.zip` / `test.zip`.
- Early stopping and model checkpointing monitor validation AUC.
