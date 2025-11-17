# 🧠 Bi-LSTM Fraud Detection Pipeline

本專案提供一套完整流程，從交易資料前處理到使用 **PyTorch 兩層雙向 LSTM（Bi-LSTM）** 的異常偵測模型。

---

## 📌 專案簡介

本專案包含兩大部分：

### 1. `proprecess.py` — 資料前處理  
將原始帳戶交易資料整理成模型所需欄位，輸出：

- `Proprecessed_train_data_normal.csv`
- `Proprecessed_train_data_alert.csv`

### 2. `model.py` — 分類模型（Bi-LSTM）  
執行完整訓練流程：

- 特徵工程：標準化 / One-Hot
- 交易序列化（每帳戶最多 100 筆）
- Padding / Truncating
- Bi-LSTM 訓練（含 Early Stopping）
- 預測帳戶異常機率
- 輸出提交檔案（label 與 probability）

---

## 🚀 使用方式

### 1️⃣ 安裝套件

```bash
pip install pandas numpy scikit-learn tqdm torch
```

### 2️⃣ 執行資料前處理

```bash
python proprecess.py
```

會生成：

```
Proprecessed_train_data_normal.csv
Proprecessed_train_data_alert.csv
```

### 3️⃣ 執行模型訓練

```bash
python model.py
```

模型訓練完成後會自動輸出：

- 最佳模型：`pytorch_lstm_best_model_xxx.pth`
- 提交標籤：`submission_platform_xxx.csv`
- 異常機率：`submission_probabilities_xxx.csv`
- 執行參數：`parameters_xxx.csv`

---

## 🧠 模型架構

### Bi-LSTM（v9.3）

- 兩層 LSTM（num_layers=2）
- 雙向（bidirectional=True）
- Hidden size = 64
- Dropout = 0.3
- 最終接全連接層輸出 logits

### 訓練設定

- Batch size = 64
- Epochs = 200
- Learning rate = 0.001
- Early Stopping patience = 30
- Loss：BCEWithLogitsLoss（含 pos_weight）

---

## 🔧 主要參數調整（model.py）

```python
MAX_SEQUENCE_LENGTH = 100
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 30
```

---

## 📤 輸出說明

### 1. 平台提交檔（二分類）
```
submission_platform_xxx.csv
acct,label
```

### 2. 異常機率（排名用）
```
submission_probabilities_xxx.csv
acct,probability
```

---

