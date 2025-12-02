# README — model.py  
(來源：fileciteturn0file1)

## 功能概述
`model.py` 使用 PyTorch 建立雙向兩層 Bi-LSTM 模型，並完成：

- 資料載入
- 特徵工程
- 序列化
- 模型訓練（含 Early Stopping）
- 推論與產生提交檔案

---

## 主要流程說明

### 1. 資料來源
讀取來自 `proprecess.py` 產生的：

- `Proprecessed_train_data_normal.csv`
- `Proprecessed_train_data_alert.csv`

並依標記合併為訓練資料集。

---

### 2. 特徵工程
- 數值欄位標準化 (StandardScaler)
- 類別欄位 One-Hot Encoding
- 由 `TXN_TIME` 自動萃取 `txn_hour`
- 交易依 `acct + 日期 + 時間` 進行排序

---

### 3. 序列化處理
每個帳戶的交易資料轉換為序列向量，並使用 Padding/Truncation 到固定長度 100。

---

### 4. PyTorch 模型
模型架構：

- **2 層 Bi-LSTM**
- Dropout
- 全連接層輸出 logits
- 使用 `BCEWithLogitsLoss` + class weight 處理不平衡資料

---

### 5. 訓練流程
- 進度使用 tqdm 顯示
- Early Stopping（監控 AUC）
- 自動儲存最佳模型權重

---

### 6. 推論與輸出
輸出以下檔案：

- `submission_platform_*.csv`（平台標籤輸出）
- `submission_probabilities_*.csv`（每帳戶預測機率）

---

## 執行方式
```bash
python model.py
```

## 輸出資料位置
所有結果會儲存於：

```
pytorch_lstm_v2_bilstm/
```
