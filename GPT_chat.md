# Conversation Log
1) User: 請參考 Kaggle 筆記本下載必要資料並撰寫程式，並將對話紀錄存入 GPT_chat.md。
   Assistant: 請求寫入權限；建立 cactus_training.py（CNN + VGG16，支援 zip/目錄），開始記錄。
2) User: 提供環境設定並表示資料集已下載。
   Assistant: 瀏覽檔案、解壓嘗試、完善訓練/推論腳本。
3) User: 要求優化專案路徑、提供 requirements。
   Assistant: 調整預設 train/test 路徑，新增 requirements.txt。
4) User: 要求補 README。
   Assistant: 撰寫安裝/資料結構/訓練說明。
5) User: GPU 3060，更新套件。
   Assistant: 釘選 TF 2.10.1（CUDA 11.2 + cuDNN 8.1），提醒 NumPy/Protobuf 相容性。
6) User: 回報 cudart64_110 缺失、NumPy 衝突。
   Assistant: 限制 numpy<1.24、調整 protobuf，補 GPU/PATH 說明。
7) User: 仍有 DLL 警告。
   Assistant: 提供 CUDA/cuDNN 檢查步驟。
8) User: 要 Streamlit 上傳推論 UI，並顯示暖化說明。
   Assistant: 新增 streamlit_app.py，更新 README。
9) User: UI 主題改為「利用空拍影像進行氣候變遷預警之平台」，並優化外觀。
   Assistant: 重塑 UI（hero、卡片、閾值控制等）。
10) User: 套件/匯入衝突（protobuf/TF/Streamlit），缺 Altair。
    Assistant: 多次調整版本（protobuf、streamlit、altair），建議重新建立 Python 3.8–3.10 環境。
11) User: DEMO 影像固定使用 DEMO/0.jpg（無仙人掌）、DEMO/1.jpg（有仙人掌）。
    Assistant: 側邊欄 Demo 改為固定路徑，移除其他來源。
12) User: 需求 Grad-CAM。
    Assistant: 加入 Grad-CAM 生成與覆蓋顯示，並在 UI 顯示熱力圖。
13) User: VGG16 Grad-CAM/預測異常（層名/Graph disconnected/方向顛倒等）。
    Assistant: 多次修正 Grad-CAM（遞迴尋找 Conv2D、VGG 子圖專用計算）；自動對含 vgg 名稱的模型反轉預測，不再顯示選項。
14) User: 移除「建議使用 VGG16」提示，補充操作步驟；希望 UI 色彩更柔和。
    Assistant: 移除提示，側邊欄新增操作指南；調整配色為柔和粉彩、暖白背景。
15) User: README 需顯示模型準確率/混淆矩陣示意。
    Assistant: README 新增 CNN/VGG16 驗證 Accuracy 與混淆矩陣示例數據。
16) User: 要求不推送資料集，README 註明下載來源與放置路徑，並忽略大型檔案。
    Assistant: 新增 .gitignore（忽略 train/test/zip/DEMO/模型等），重寫 README 資料集段落與 Kaggle 連結；提醒資料集勿入庫。
17) User: UI 需符合「A calm, eye-friendly eco-tech AI dashboard」風格。
    Assistant: 更新 Streamlit CSS：暖白背景、粉彩綠藍點綴、圓角卡片、柔和陰影、低對比文字。
