import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
import warnings
from functools import reduce
from tqdm import tqdm # 用於顯示進度條

# 匯入 scikit-learn 相關套件
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score # 用於計算 AUC

# --- 【NEW】 匯入 PyTorch 相關套件 ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("="*50)
    print("!!! 錯誤: 找不到 PyTorch 套件 !!!")
    print("請執行: pip install torch")
    print("="*50)
    raise

# ------------------------------------------------------------------------
# --- 1. 參數設定 (Parameters) ---
# ------------------------------------------------------------------------

# 輸出子目錄 (儲存 .csv 檔)
OUTPUT_DIR = 'pytorch_lstm_v2_bilstm' # <-- 建議使用新目錄

# **策略**：我們要預測多少筆資料為 "1"
N_TO_PREDICT_POSITIVE = 200 

# ========================================================================
# --- 【NEW】 LSTM 專用參數 ---
# ========================================================================

# 每個帳戶最多讀取最近 N 筆交易
MAX_SEQUENCE_LENGTH = 100

# LSTM 層的神經元數量
LSTM_UNITS = 64

# Dropout 比例 (防止過擬合)
DROPOUT_RATE = 0.3

# 訓練週期
EPOCHS = 200 

# 批次大小
BATCH_SIZE = 64

# 驗證集切分比例
VALIDATION_SPLIT = 0.15

# 學習率
LEARNING_RATE = 0.001

# --- 【MODIFIED (v9.3)】 ---
# 2層 Bi-LSTM 更複雜，需要更多耐心
EARLY_STOPPING_PATIENCE = 30

# ========================================================================
# --- 【ADAPTED】 特徵群組 (意義不變) ---
# ========================================================================
FEATURE_GROUPS_TO_USE = {
    'txn_amt_stats': True,
    'is_self_txn_dist': True,
    'to_acct_type_dist': True,
    'from_acct_type_dist': True,
    'channel_type_dist': True,
    'txn_hour_dist': True,
    'currency_type_dist': True,
    'flag_dist': True
}
# ========================================================================


# ------------------------------------------------------------------------
# --- 2. 輔助函式 (Helper Functions) ---
# ------------------------------------------------------------------------

def load_data(file_path):
    """ (與 v9.2 相同) 讀取 CSV 並自動改名 'acct' 和 'currency_type' """
    if not os.path.exists(file_path):
        print(f"!!! 錯誤: 必要的輸入檔案 {file_path} 找不到! !!!")
        return None
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        print(f"  > {file_path}: UTF-8 讀取失敗，嘗試 'big5' 編碼...")
        df = pd.read_csv(file_path, encoding='big5')
    except Exception as e:
        print(f"讀取檔案 {file_path} 失敗: {e}")
        return None
        
    if 'ACCT' in df.columns:
        df = df.rename(columns={'ACCT': 'acct'})
    if 'CURRENCY_TYPE' in df.columns:
        df = df.rename(columns={'CURRENCY_TYPE': 'currency_type'})
    return df

def save_parameters(timestamp):
    """ (儲存本次執行的參數) """
    print(f"正在儲存參數設定檔...")
    try:
        params_to_log = {
            'run_timestamp': timestamp,
            # --- 【MODIFIED (v9.3)】 ---
            'MODEL_TYPE': 'PyTorch_BiLSTM_2L', 
            'N_TO_PREDICT_POSITIVE': N_TO_PREDICT_POSITIVE,
            'OUTPUT_DIR': OUTPUT_DIR,
            'MAX_SEQUENCE_LENGTH': MAX_SEQUENCE_LENGTH,
            'LSTM_UNITS': LSTM_UNITS,
            'DROPOUT_RATE': DROPOUT_RATE,
            'EPOCHS': EPOCHS,
            'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'VALIDATION_SPLIT': VALIDATION_SPLIT,
            'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE
        }
        for key, value in FEATURE_GROUPS_TO_USE.items():
            params_to_log[f'FEAT_{key}'] = value
        
        params_df = pd.DataFrame([params_to_log])
        param_filename = os.path.join(OUTPUT_DIR, f'parameters_{timestamp}.csv')
        params_df.to_csv(param_filename, index=False, encoding='utf-8-sig')
        print(f"參數已儲存至: {param_filename}")
        
    except Exception as e:
        print(f"!!! 警告: 儲存參數時發生錯誤: {e} !!!")

# --- PyTorch 專用輔助函式 ---

def pad_sequences_numpy(sequences, maxlen, num_features, padding='pre', truncating='pre', value=0.):
    """ (與 v9.2 相同) 手動實作 Keras pad_sequences 的功能 """
    padded_data = np.full((len(sequences), maxlen, num_features), value, dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        if not isinstance(seq, np.ndarray):
            seq = np.array(seq, dtype=np.float32)
        seq_len = seq.shape[0]
        if seq_len == 0:
            continue
        if truncating == 'pre':
            seq = seq[-maxlen:]
        elif truncating == 'post':
            seq = seq[:maxlen]
        seq_len = seq.shape[0]
        if padding == 'pre':
            padded_data[i, -seq_len:] = seq
        elif padding == 'post':
            padded_data[i, :seq_len] = seq
    return padded_data

class TransactionDataset(Dataset):
    """ (與 v9.2 相同) PyTorch 自訂資料集 (Dataset) """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========================================================================
# --- 【MODIFIED (v9.3)】 ---
# --- PyTorch 2層雙向 LSTM 模型 ---
# ========================================================================
class LSTMModel(nn.Module):
    """
    PyTorch 自訂 LSTM 模型
    --- 【NEW (v9.3)】 ---
    - 2層 (num_layers=2)
    - 雙向 (bidirectional=True)
    - 增加了 LSTM 層之間的 Dropout
    - 修正了 forward pass 以正確串接 2 個方向的 final state
    """
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 定義層
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            batch_first=True,
            num_layers=2,           # <-- 2層
            bidirectional=True,     # <-- 雙向
            dropout=dropout_rate    # <-- 在層與層之間加 Dropout (僅在 num_layers > 1 時有效)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全連接層
        # 來自 2 個方向 (forward/backward) 的 hidden_size
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, 1) # 輸出 1 個 logit

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # PyTorch LSTM 回傳 (output, (h_n, c_n))
        # 我們不再需要 output，只需要 h_n
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        # h_n shape will be: (2 * 2, batch_size, hidden_size) = (4, batch, hidden)
        _, (h_n, c_n) = self.lstm(x)
        
        # h_n[0] = layer 1, forward
        # h_n[1] = layer 1, backward
        # h_n[2] = layer 2, forward  <-- 我們要這個
        # h_n[3] = layer 2, backward <-- 我們要這個
        
        # 串接最後一層的前向和後向 hidden state
        # h_n[-2, :, :] 是最後一層 (layer 2) 的 forward state
        # h_n[-1, :, :] 是最後一層 (layer 2) 的 backward state
        x = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        # x shape: (batch_size, hidden_size * 2)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        # 輸出 raw logits (未經 sigmoid)
        # shape: (batch_size, 1)
        return x
# ========================================================================
# --- 修改結束 ---
# ========================================================================


# ------------------------------------------------------------------------
# --- 3. 主程式：PyTorch LSTM 序列模型 ---
# (此區塊與 v9.2 相同)
# ------------------------------------------------------------------------

if __name__ == "__main__":
    
    warnings.filterwarnings('ignore')
    
    # --- 檢查 CUDA (GPU) 是否可用 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"===== 正在使用裝置: {device} =====")
    
    print("===== 步驟 1: 設置環境與讀取資料 (PyTorch LSTM) =====")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"所有輸出將儲存於: {OUTPUT_DIR}")
    print(f"本次執行時間戳記: {timestamp}")

    # --- 儲存參數 ---
    save_parameters(timestamp)
    
    # --- 讀取資料 ---
    df_alert_raw = load_data('Proprecessed_train_data_alert.csv')
    df_predict_raw = load_data('Proprecessed_train_data_normal.csv')
    df_order = load_data('acct_predict.csv') 

    if df_alert_raw is None or df_predict_raw is None or df_order is None:
        print("程式因缺少檔案而中止。")
    
    else:
        print("\n===== 步驟 2: 特徵工程 (FE) - 序列化 =====")
        
        # 1. 合併資料
        df_alert_raw['Label'] = 1
        df_predict_raw['Label'] = 0
        df_all = pd.concat([df_alert_raw, df_predict_raw], ignore_index=True)
        print(f"資料合併完成。總交易筆數: {len(df_all)}")

        # 2. 準備特徵欄位
        numerical_features = []
        categorical_features = []
        
        if FEATURE_GROUPS_TO_USE.get('txn_amt_stats'):
            numerical_features.append('TXN_AMT')
        if FEATURE_GROUPS_TO_USE.get('is_self_txn_dist'):
            categorical_features.append('IS_SELF_TXN')
        if FEATURE_GROUPS_TO_USE.get('to_acct_type_dist'):
            categorical_features.append('TO_ACCT_TYPE')
        if FEATURE_GROUPS_TO_USE.get('from_acct_type_dist'):
            categorical_features.append('FROM_ACCT_TYPE')
        if FEATURE_GROUPS_TO_USE.get('channel_type_dist'):
            categorical_features.append('CHANNEL_TYPE')
        if FEATURE_GROUPS_TO_USE.get('currency_type_dist'):
            categorical_features.append('currency_type')
        if FEATURE_GROUPS_TO_USE.get('flag_dist'): 
            categorical_features.append('FLAG')
            
        print(f"使用 {len(numerical_features)} 個數值特徵: {numerical_features}")
        print(f"使用 {len(categorical_features)} 個分類特徵: {categorical_features}")

        # 3. 處理時間 (排序用) -- 【v9.2 修正】
        print("正在清理並排序交易時間...")
        
        # 3.1. 移除 'TXN_DATE' (天數) 或 'TXN_TIME' (時間) 為空的交易
        original_count = len(df_all)
        df_all.dropna(subset=['TXN_DATE', 'TXN_TIME'], inplace=True)
        print(f"  > 移除 {original_count - len(df_all)} 筆無效時間的交易。")

        # 3.2. 確保 TXN_DATE 是可排序的數字
        df_all['TXN_DATE'] = pd.to_numeric(df_all['TXN_DATE'], errors='coerce')
        
        # 3.3. 確保 TXN_TIME 是可排序的
        #      使用 pd.to_timedelta 將 'HH:MM:SS' 轉換為「自午夜起的時間差」
        df_all['time_for_sort'] = pd.to_timedelta(df_all['TXN_TIME'], errors='coerce')

        # 3.4. 再次移除轉換失敗的列 (日期或時間格式錯誤)
        original_count = len(df_all)
        df_all.dropna(subset=['TXN_DATE', 'time_for_sort'], inplace=True)
        print(f"  > 移除 {original_count - len(df_all)} 筆格式錯誤的日期或時間。")
        
        # 3.5. 排序
        df_all = df_all.sort_values(by=['acct', 'TXN_DATE', 'time_for_sort'])
        print("交易資料已按 (帳戶 -> 天數 -> 時間) 排序。")
        
        # 3.6. 移除輔助欄位 (我們不需要 'datetime' 或 'time_for_sort' 進入模型)
        if 'time_for_sort' in df_all.columns:
            df_all = df_all.drop(columns=['time_for_sort'])
        if 'datetime' in df_all.columns:
            df_all = df_all.drop(columns=['datetime'])
        # --- 【FIX】 結束 ---
        

        # 4. 處理特徵 (標準化 & One-Hot Encoding)
        
        # --- 【NEW】 建立 'txn_hour' 特徵 (如果需要) ---
        if FEATURE_GROUPS_TO_USE.get('txn_hour_dist'):
            # 我們需要從 TXN_TIME (HH:MM:SS) 中提取 'HH' (小時)
            # TXN_TIME 仍然是字串，例如 '21:05:00'
            try:
                df_all['txn_hour'] = df_all['TXN_TIME'].str.split(':', n=1, expand=True)[0]
                # 將 'txn_hour' 加入分類特徵列表 (在 OHE 之前)
                if 'txn_hour' not in categorical_features:
                     categorical_features.append('txn_hour')
                print(f"  > 已從 TXN_TIME 提取 'txn_hour' 特徵。")
            except Exception as e:
                print(f"  > !!! 警告: 從 TXN_TIME 提取 'txn_hour' 失敗: {e} !!!")
        
        df_all['TXN_AMT'] = pd.to_numeric(df_all['TXN_AMT'], errors='coerce')
        
        # 填充 NaNs
        for col in numerical_features:
            df_all[col] = df_all[col].fillna(0)
        for col in categorical_features:
            df_all[col] = df_all[col].fillna('UNK').astype(str) # 確保 txn_hour 也是 str

        # 標準化 (Numerical)
        if numerical_features:
            scaler = StandardScaler()
            df_all[numerical_features] = scaler.fit_transform(df_all[numerical_features])

        # One-Hot Encoding (Categorical)
        ohe_features = []
        if categorical_features:
            # 我們必須在填充 'UNK' *之後* 才進行 One-Hot
            # 確保 'txn_hour' 也是字串
            for col in categorical_features:
                df_all[col] = df_all[col].astype(str)
                
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe_result = encoder.fit_transform(df_all[categorical_features])
            ohe_feature_names = encoder.get_feature_names_out(categorical_features)
            ohe_df = pd.DataFrame(ohe_result, columns=ohe_feature_names, index=df_all.index)
            ohe_features = ohe_df.columns.tolist()
            df_all = pd.concat([df_all, ohe_df], axis=1)
            
        feature_columns = numerical_features + ohe_features
        NUM_FEATURES = len(feature_columns)
        print(f"特徵處理完成。序列特徵維度: {NUM_FEATURES}")

        
        print("\n===== 步驟 3: 建立序列資料 (Padding/Truncating) =====")
        
        account_labels = df_all.groupby('acct')['Label'].first()
        accounts = account_labels.index.values
        y_labels = account_labels.values
        
        print(f"正在將 {len(accounts)} 個帳戶轉換為序列...")
        grouped = df_all.groupby('acct')
        all_sequences = [grouped.get_group(acct)[feature_columns].values for acct in accounts]
        
        print(f"正在 Padding/Truncating 序列至長度 {MAX_SEQUENCE_LENGTH}...")
        X_padded = pad_sequences_numpy(
            all_sequences, 
            maxlen=MAX_SEQUENCE_LENGTH,
            num_features=NUM_FEATURES,
            padding='pre',
            truncating='pre'
        )
        
        print(f"序列資料維度 (X): {X_padded.shape}")
        print(f"標籤資料維度 (y): {y_labels.shape}")

        
        print("\n===== 步驟 4: 準備訓練集、驗證集與預測集 =====")
        
        X_train_val = X_padded
        y_train_val = y_labels
        
        predict_mask = (y_labels == 0)
        X_test = X_padded[predict_mask]
        acct_test = accounts[predict_mask]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, 
            y_train_val, 
            test_size=VALIDATION_SPLIT, 
            stratify=y_train_val, 
            random_state=42
        )

        print(f"總訓練資料: {X_train.shape}")
        print(f"總驗證資料: {X_val.shape}")
        print(f"總預測 (提交) 資料: {X_test.shape}")

        
        print("\n===== 步驟 5: 建立 PyTorch DataLoaders =====")
        
        train_dataset = TransactionDataset(X_train, y_train)
        val_dataset = TransactionDataset(X_val, y_val)
        test_dataset = TransactionDataset(X_test, np.zeros(len(X_test))) 
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print("DataLoaders 建立完成。")

        
        print("\n===== 步驟 6: 建立模型與損失函數 =====")
        
        model = LSTMModel(
            input_size=NUM_FEATURES,
            hidden_size=LSTM_UNITS,
            dropout_rate=DROPOUT_RATE
        ).to(device) 
        
        # --- 計算 Class Weight (pos_weight) ---
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        pos_weight_scalar = num_neg / num_pos
        pos_weight_tensor = torch.tensor([pos_weight_scalar], dtype=torch.float32).to(device)
        print(f"訓練集中 0: {num_neg}, 1: {num_pos} | pos_weight (正樣本懲罰權重): {pos_weight_scalar:.2f}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        try:
            from torchinfo import summary
            summary(model, input_size=(BATCH_SIZE, MAX_SEQUENCE_LENGTH, NUM_FEATURES))
        except ImportError:
            print("Pytorch Summary (run 'pip install torchinfo' for details):")
            print(model)

            
        print("\n===== 步驟 7: 訓練模型 (PyTorch Loop) =====")
        
        best_val_auc = -1.0
        epochs_no_improve = 0
        best_model_path = os.path.join(OUTPUT_DIR, f'pytorch_lstm_best_model_{timestamp}.pth') 
        
        for epoch in range(1, EPOCHS + 1):
            
            # --- 訓練模式 ---
            model.train()
            train_losses = []
            
            train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
            for X_batch, y_batch in train_loop:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # --- 評估模式 ---
            model.eval()
            val_losses = []
            all_val_preds = []
            all_val_labels = []
            
            with torch.no_grad(): 
                val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Validate]", leave=False)
                for X_batch, y_batch in val_loop:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_losses.append(loss.item())
                    all_val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                    all_val_labels.append(y_batch.cpu().numpy())
                    
            avg_val_loss = np.mean(val_losses)
            
            val_preds = np.concatenate(all_val_preds).flatten()
            val_labels = np.concatenate(all_val_labels).flatten()
            val_auc = roc_auc_score(val_labels, val_preds)
            
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            # --- Early Stopping 和 Model Checkpoint ---
            if val_auc > best_val_auc:
                print(f"  > Val AUC improved ({best_val_auc:.4f} -> {val_auc:.4f}). 儲存模型...")
                best_val_auc = val_auc
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path) 
            else:
                epochs_no_improve += 1
                print(f"  > Val AUC did not improve. ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"EarlyStopping 觸發. 停止於 Epoch {epoch}")
                break
                
        print("模型訓練完成。")

        
        print("\n===== 步驟 8: 產生預測結果 =====")
        print("載入最佳權重...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval() 
        
        all_test_preds = []
        
        print(f"正在對 {len(X_test)} 筆提交資料進行預測...")
        with torch.no_grad():
            test_loop = tqdm(test_loader, desc=f"Predicting", leave=False)
            for X_batch, _ in test_loop: 
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                all_test_preds.append(torch.sigmoid(outputs).cpu().numpy())

        pred_probs = np.concatenate(all_test_preds).flatten()
        
        results_df = pd.DataFrame({
            'acct': acct_test,
            'probability': pred_probs
        })

        top_n_indices = np.argsort(pred_probs)[-N_TO_PREDICT_POSITIVE:]
        
        results_df['label'] = 0
        results_df.iloc[top_n_indices, results_df.columns.get_loc('label')] = 1

        print(f"已產生融合機率，並將機率最高的前 {N_TO_PREDICT_POSITIVE} 筆標記為 1。")
        print("預測標籤分佈:")
        print(results_df['label'].value_counts())

        
        print("\n===== 步驟 9: 排序並儲存提交檔案 =====")
        
        if 'acct' not in df_order.columns:
            if len(df_order.columns) > 0:
                original_col = df_order.columns[0]
                df_order = df_order.rename(columns={original_col: 'acct'})
                print(f"警告: 已將 {original_col} 視為 'acct' 欄位。")
        
        final_submission = df_order[['acct']].merge(results_df, on='acct', how='left')
        final_submission['probability'] = final_submission['probability'].fillna(0)
        final_submission['label'] = final_submission['label'].fillna(0).astype(int)

        # --- 分拆提交檔案 ---
        submission_platform_df = final_submission[['acct', 'label']]
        submission_platform_filename = os.path.join(OUTPUT_DIR, f'submission_platform_{timestamp}.csv')
        submission_platform_df.to_csv(submission_platform_filename, index=False, encoding='utf-8-sig')
        print(f"平台提交檔案 (acct, label) 已儲存: {submission_platform_filename}")
        
        submission_proba_df = final_submission[['acct', 'probability']]
        submission_proba_filename = os.path.join(OUTPUT_DIR, f'submission_probabilities_{timestamp}.csv')
        submission_proba_df.to_csv(submission_proba_filename, index=False, encoding='utf-8-sig')
        print(f"機率檔案 (acct, probability) 已儲存: {submission_proba_filename}")

        print("\n提交檔案 (前5筆, 包含機率與標籤):")
        print(final_submission.head())
        print("\n提交檔案 (依機率排序，看最高的):")
        print(final_submission.sort_values(by='probability', ascending=False).head())

        print("\n===== PyTorch LSTM 模型處理已完成 =====")
