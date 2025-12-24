# CRISP-DM 報告：空拍仙人掌辨識與氣候風險提示

## 1. 商業理解（Business Understanding）
- 目標：以空拍影像辨識是否存在仙人掌，作為環境韌性與氣候風險的參考指標，並提供可讀的風險提醒與改善建議。
- 使用情境：提供環境監測或地方管理者快速判讀影像的工具，降低人工篩檢負擔。
- 成功標準：
  - 影像分類表現穩定（準確率/AUC 具可用性）。
  - 介面操作直觀，能在 Streamlit Cloud 上部署與使用。
  - 推論結果可解釋（Grad-CAM）且有建議文字。

## 2. 資料理解（Data Understanding）
- 來源：Kaggle Aerial Cactus Identification 資料集。
- 檔案：
  - `train.csv`：包含影像 id 與標籤 `has_cactus`。
  - `train.zip`：訓練影像。
  - `test.zip` / `sample_submission.csv`：測試影像與提交格式。
- 影像特性：原始影像尺寸較小（資料集常見為 32x32 RGB），後續統一調整尺寸以利模型輸入。
- 風險與注意：
  - 影像來源單一、場景偏特定地形，泛化能力需留意。
  - 若僅用單一閾值判定，可能需依場域調整。

## 3. 資料準備（Data Preparation）
- 讀取方式：支援從資料夾或 zip 壓縮檔讀取影像。
- 前處理流程：
  - 影像轉為 RGB、調整尺寸為 96x96。
  - CNN 使用 `Rescaling(1/255)`；VGG16 使用 `preprocess_input`。
- 資料切分：訓練/驗證採分層切分（約 9:1）。
- 資料增強：水平翻轉、旋轉、縮放以提升泛化能力。

## 4. 建模（Modeling）
- 模型一：簡化 CNN
  - 結構：Conv(32/64/128) + MaxPool + GAP + Dropout + Sigmoid。
  - 優點：輕量、推論快。
- 模型二：VGG16 轉移學習
  - 使用 ImageNet 權重，主幹凍結，接 GAP + Dropout + Dense。
  - 優點：準確率較高。
- 訓練設定：
  - Optimizer：Adam（CNN 1e-3、VGG16 1e-4）。
  - 指標：Accuracy、AUC。
  - 早停與最佳權重保存。

## 5. 評估（Evaluation）
- 參考結果（示意）：
  - CNN Accuracy 約 0.94。
  - VGG16 Accuracy 約 0.98。
- 解釋性：使用 Grad-CAM 顯示模型關注區域。
- 使用端可調整閾值以符合不同場域的風險偏好。

## 6. 部署（Deployment）
- 介面：Streamlit
  - 上傳影像後自動推論，顯示機率、判定、Grad-CAM。
  - 「開始預測」按鈕觸發 LLM 改善建議（本地輕量模型，不需 API key）。
  - 支援選擇或上傳模型檔（`.keras`/`.h5`）。
- 依賴：
  - `tensorflow`、`streamlit`、`torch`、`transformers` 等（詳見 `requirements.txt`）。
- 部署注意：
  - 模型權重需放在 `outputs/` 或透過 UI 上傳。
  - 本地 LLM 需下載模型，首次啟用較慢。

## 7. 後續建議（Maintenance）
- 持續收集新影像，定期重新訓練以提升泛化能力。
- 針對不同地形/季節設置多組閾值或分區模型。
- 可加入推論信心度分級與人工覆核流程。
