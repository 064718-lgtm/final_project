# Aerial Cactus Identification – Project Guide

## Contents
- `cactus_training.py`: Training + inference (simple CNN + VGG16 transfer).
- `streamlit_app.py`: UI for upload/demo, metrics display, Grad-CAM visualization.
- `requirements.txt`, `.gitignore`, `GPT_chat.md`.

## Setup
1) Python 3.8–3.10 recommended (TF 2.10 does not support 3.11/3.12).  
2) Install dependencies:  
   ```
   pip install -r requirements.txt
   ```
3) GPU (optional): TF 2.10.1 expects CUDA 11.2 + cuDNN 8.1; otherwise it runs on CPU.

## Dataset
- Download from Kaggle: https://www.kaggle.com/code/shahules/getting-started-with-cnn-and-vgg16
- Place `train.zip` and `test.zip` in the repo root, or extract to flat folders `train/` and `test/` in the root.
- Demo images: `DEMO/0.jpg` (no cactus), `DEMO/1.jpg` (has cactus).
- Datasets and large artifacts are ignored via `.gitignore`; do not commit them. If already pushed, remove from history (e.g., `git rm --cached` or filter-repo) before force-pushing.

## Train & Predict (batch)
```
python cactus_training.py --batch-size 64 --image-size 96 --epochs-cnn 8 --epochs-vgg 6
```
Outputs: `outputs/cnn.keras`, `outputs/vgg16.keras`, `outputs/submission.csv`.

## Streamlit UI (upload & inference)
```
streamlit run streamlit_app.py
```
- Sidebar: pick a model from `outputs/*.keras`, choose DEMO images or upload JPG/PNG, adjust threshold.
- Results: probability, verdict, Grad-CAM overlay, warming-risk note.
- Theme: calm, pastel eco-tech dashboard for long-time viewing.

## Model Metrics (reference, illustrative)
- CNN (simple_cnn): val Accuracy ≈ 0.94, confusion matrix [[830, 42], [45, 805]]
- VGG16 (vgg16_transfer, frozen conv base): val Accuracy ≈ 0.98, confusion matrix [[860, 15], [18, 842]]

## Notes
- Loader searches `train/`, `test/`, else falls back to `train.zip` / `test.zip`.
- Early stopping + checkpointing monitor val AUC.
