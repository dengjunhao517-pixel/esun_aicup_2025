# README — proprecess.py  
(來源：fileciteturn0file0)

## 功能概述
`proprecess.py` 將原始交易資料轉換為模型可用的訓練資料。主要輸出：

- `Proprecessed_train_data_normal.csv`
- `Proprecessed_train_data_alert.csv`

## 主要流程
### 1. 建立資料分組 
依帳號排序後產生 `group_p` 作為分組序號。

### 2. 四大邏輯處理  
腳本模擬 SQL JOIN + UNION 的多段查詢：

- `NEXT_LEVEL`
- `FROM`
- `TO`
- `BEFORE_LEVEL`

每段邏輯會從交易資料與來源帳號資料比對，產生對應標記。

### 3. 最終整併
整合所有邏輯的結果後再次 JOIN 回來源資料，並輸出標準化欄位（全部大寫）。

## 執行方式
```bash
python proprecess.py
```

## 輸入資料
- `acct_transaction.csv`
- `acct_alert.csv`
- `acct_predict.csv`

## 輸出資料
- `Proprecessed_train_data_normal.csv`
- `Proprecessed_train_data_alert.csv`
