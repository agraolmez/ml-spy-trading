import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import random

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Downloading raw data (2000–2025)...")
start_date, end_date = "2000-01-01", "2025-07-01"


spy_raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
spy_raw.columns = spy_raw.columns.get_level_values(0)
spy_raw = spy_raw[['Open', 'High', 'Low', 'Close', 'Volume']]


tickers = ['^TNX','^FVX','^MOVE','HYG','IEF','RSP','^VIX','^IRX']
for t in tickers:
    col = t.replace('^','') + "_Close"    
    spy_raw[col] = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']

spy_raw.dropna(inplace=True) 


OOS_CUTOFF = "2023-01-01"
in_sample_raw  = spy_raw.loc[:OOS_CUTOFF].copy()
oos_sample_raw = spy_raw.loc[OOS_CUTOFF:].copy()


def prepare_features_and_target(df):
    df_copy = df.copy() 

    
    df_copy['M30_MA']          = ta.sma(df_copy['Close'], length=30)
    df_copy['M10_MA']          = ta.sma(df_copy['Close'], length=10)
    macd = ta.macd(df_copy['Close'])
    df_copy['MACD_DIFF']       = macd['MACD_12_26_9'] - macd['MACDs_12_26_9']
    df_copy['OBV']             = ta.obv(df_copy['Close'], df_copy['Volume'])
    df_copy['WILLR_14']        = ta.willr(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)

    
    df_copy['RSI_14']          = ta.rsi(df_copy['Close'], length=14)
    df_copy['RSI_7']           = ta.rsi(df_copy['Close'], length=7)
    df_copy['ADX_14']          = ta.adx(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)['ADX_14']
    df_copy['StochRSI_14']     = ta.stochrsi(df_copy['Close'], length=14)['STOCHRSIk_14_14_3_3']
    df_copy['Price_Momentum_5D'] = df_copy['Close'].pct_change(5)
    df_copy['VIX_Change_5D']   = df_copy['VIX_Close'].pct_change(5)
    df_copy['Down_Moves']      = (df_copy['Close'].pct_change() < 0).rolling(10).sum()
    df_copy['MACD_Bearish_Cross']= (ta.macd(df_copy['Close'])['MACDh_12_26_9'] < 0).astype(int)
    df_copy['Volume_Spike']    = (df_copy['Volume'] > df_copy['Volume'].rolling(20).mean() * 1.5).astype(int)
    df_copy['HYG_IEF']         = df_copy['HYG_Close'] / df_copy['IEF_Close']
    df_copy['Yield_Slope']     = df_copy['TNX_Close'] - df_copy['FVX_Close']
    df_copy['Market_Breadth']  = df_copy['RSP_Close'] / df_copy['Close']

    
    df_copy['Target'] = (df_copy['Close'].shift(-30) > df_copy['Close']).astype(int)

    return df_copy.dropna() 

in_sample  = prepare_features_and_target(in_sample_raw)
oos_sample = prepare_features_and_target(oos_sample_raw)


all_features = [
    'M30_MA', 'M10_MA', 'MACD_DIFF', 'OBV', 'WILLR_14', 
    'RSI_14', 'RSI_7', 'ADX_14', 'StochRSI_14', 'Price_Momentum_5D',
    'VIX_Change_5D', 'Down_Moves', 'MACD_Bearish_Cross', 'Volume_Spike',
    'HYG_IEF', 'Yield_Slope', 'Market_Breadth'
]

param_grid = {
    "num_leaves": [10, 20, 40],
    "max_depth": [3, 5, 8],
    "reg_alpha": [0, 1, 5],
    "reg_lambda": [0, 1, 5],
    "scale_pos_weight": [0.8, 1.0, 1.2],
    "threshold": [0.4, 0.5, 0.6]
}

def random_param_combo(param_grid):
    return {k: random.choice(v) for k, v in param_grid.items()}

N_TRIALS = 5000
results = []

print(f"Starting random search ({N_TRIALS} trials, seed={SEED})...")
for _ in tqdm(range(N_TRIALS)):
    
    feature_set = random.sample(all_features, 6)
    
    params = random_param_combo(param_grid)
    
    X_in, y_in = in_sample[feature_set], in_sample['Target']
    X_oos, y_oos = oos_sample[feature_set], oos_sample['Target']

    
    model = lgb.LGBMClassifier(
        objective='binary',
        random_state=SEED, 
        n_jobs=1,
        num_leaves=params['num_leaves'],
        max_depth=params['max_depth'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        scale_pos_weight=params['scale_pos_weight'],
        verbosity=-1,
        force_col_wise=True
    )
    try:
        model.fit(X_in, y_in)
        preds_in = model.predict(X_in)
        acc_in = accuracy_score(y_in, preds_in)

        probs_oos = model.predict_proba(X_oos)[:, 1]
        preds_oos = (probs_oos > params['threshold']).astype(int)
        roc_oos = roc_auc_score(y_oos, probs_oos)
        acc_oos = accuracy_score(y_oos, preds_oos)
        results.append({
            "SEED": SEED,
            "Features": feature_set,
            **params,
            "Train_Accuracy": acc_in,
            "ROC_AUC_OOS": roc_oos,
            "Accuracy_OOS": acc_oos
        })
    except Exception as e:
        
        continue

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="ROC_AUC_OOS", ascending=False)
out_name = f"model_results_seed{SEED}.xlsx"
df_results.to_excel(out_name, index=False)
print(f"All trials completed. Results saved to '{out_name}'!")

import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import random

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Preparing data for walk-forward backtest...")
start_date, end_date = "2000-01-01", "2025-07-01"

spy_base_raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
spy_base_raw.columns = spy_base_raw.columns.get_level_values(0)
spy_base_raw = spy_base_raw[['Open', 'High', 'Low', 'Close', 'Volume']]

tickers = ['^TNX','^FVX','^MOVE','HYG','IEF','RSP','^VIX','^IRX']
for t in tickers:
    col = t.replace('^','') + "_Close"
    
    spy_base_raw[col] = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']

spy_base_raw.dropna(inplace=True) 

def create_features_and_target(df_segment):
    df_copy = df_segment.copy()

    
    df_copy['M30_MA']          = ta.sma(df_copy['Close'], length=30)
    df_copy['M10_MA']          = ta.sma(df_copy['Close'], length=10)
    macd = ta.macd(df_copy['Close'])
    df_copy['MACD_DIFF']       = macd['MACD_12_26_9'] - macd['MACDs_12_26_9']
    df_copy['OBV']             = ta.obv(df_copy['Close'], df_copy['Volume'])
    df_copy['WILLR_14']        = ta.willr(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)
    df_copy['RSI_14']          = ta.rsi(df_copy['Close'], length=14)
    df_copy['RSI_7']           = ta.rsi(df_copy['Close'], length=7)
    df_copy['ADX_14']          = ta.adx(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)['ADX_14']
    df_copy['StochRSI_14']     = ta.stochrsi(df_copy['Close'], length=14)['STOCHRSIk_14_14_3_3']
    df_copy['Price_Momentum_5D'] = df_copy['Close'].pct_change(5)
    df_copy['VIX_Change_5D']   = df_copy['VIX_Close'].pct_change(5)
    df_copy['Down_Moves']      = (df_copy['Close'].pct_change() < 0).rolling(10).sum()
    df_copy['MACD_Bearish_Cross']= (ta.macd(df_copy['Close'])['MACDh_12_26_9'] < 0).astype(int)
    df_copy['Volume_Spike']    = (df_copy['Volume'] > df_copy['Volume'].rolling(20).mean()*1.5).astype(int)
    df_copy['HYG_IEF']         = df_copy['HYG_Close'] / df_copy['IEF_Close']
    df_copy['Yield_Slope']     = df_copy['TNX_Close'] - df_copy['FVX_Close']
    df_copy['Market_Breadth']  = df_copy['RSP_Close'] / df_copy['Close']

    
    df_copy['Target']          = (df_copy['Close'].shift(-30) > df_copy['Close']).astype(int)
    df_copy['Daily_Return']    = df_copy['Close'].pct_change()

    return df_copy.dropna() 

print("Loading and filtering models from 'model_results_seed42.xlsx'...")
try:
    results_df = pd.read_excel("model_results_seed42.xlsx")
except FileNotFoundError:
    print("Error: 'model_results_seed42.xlsx' not found. Please run the Random Search script first.")
    exit()

filtered_models = results_df[
    (results_df['ROC_AUC_OOS'] > 0.55) &
    (results_df['Train_Accuracy'] < 0.80)
].copy().reset_index(drop=True)

if filtered_models.empty:
    print("No models found satisfying the filter criteria (ROC_AUC_OOS > 0.55 AND Train_Accuracy < 0.8). Exiting.")
    exit()

print(f"{len(filtered_models)} models found for walk-forward backtest.")

periods = [
    ("2021-01-01","2022-01-01"), 
    ("2022-01-01","2023-01-01"), 
    ("2023-01-01","2024-01-01"), 
    ("2024-01-01","2025-01-01")  
]

walk_forward_results = []

print("\nStarting walk-forward backtest for filtered models...")
for idx, model_row in tqdm(filtered_models.iterrows(), total=filtered_models.shape[0], desc="Walk-Forward Models"):
    
    feature_set = eval(model_row['Features']) if isinstance(model_row['Features'], str) else model_row['Features']
    params = {
        "num_leaves":       int(model_row['num_leaves']),
        "max_depth":        int(model_row['max_depth']),
        "reg_alpha":        float(model_row['reg_alpha']),
        "reg_lambda":       float(model_row['reg_lambda']),
        "scale_pos_weight": float(model_row['scale_pos_weight']),
        "verbosity": -1,
        "force_col_wise": True,
        "random_state": SEED, 
        "n_jobs": 1,
        "objective": "binary"
    }
    threshold = float(model_row['threshold'])

    fw_roc_scores, fw_acc_scores, fw_ann_returns, fw_sharpe_ratios = [], [], [], []

    for train_end_str, test_end_str in periods:
        train_end_dt = pd.to_datetime(train_end_str)
        test_end_dt  = pd.to_datetime(test_end_str)

        
        current_train_base = spy_base_raw.loc[:train_end_dt].copy()
        current_test_base  = spy_base_raw.loc[train_end_dt:test_end_dt].copy()

        
        current_train_df = create_features_and_target(current_train_base)
        current_test_df  = create_features_and_target(current_test_base)
        
        
        
        min_data_points = max(30, 30) 
        
        if current_train_df.empty or current_test_df.empty or \
           len(current_train_df) < min_data_points or \
           len(current_test_df) < min_data_points:
            
            fw_roc_scores.append(np.nan)
            fw_acc_scores.append(np.nan)
            fw_ann_returns.append(np.nan)
            fw_sharpe_ratios.append(np.nan)
            continue 

        
        if not all(f in current_train_df.columns for f in feature_set) or \
           not all(f in current_test_df.columns for f in feature_set):
            
            fw_roc_scores.append(np.nan)
            fw_acc_scores.append(np.nan)
            fw_ann_returns.append(np.nan)
            fw_sharpe_ratios.append(np.nan)
            continue

        X_train, y_train = current_train_df[feature_set], current_train_df['Target']
        X_test, y_test = current_test_df[feature_set], current_test_df['Target']
        
        
        if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
            
            fw_roc_scores.append(np.nan)
            fw_acc_scores.append(np.nan)
            fw_ann_returns.append(np.nan)
            fw_sharpe_ratios.append(np.nan)
            continue

        
        model = lgb.LGBMClassifier(**params)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            
            fw_roc_scores.append(np.nan)
            fw_acc_scores.append(np.nan)
            fw_ann_returns.append(np.nan)
            fw_sharpe_ratios.append(np.nan)
            continue

        
        try:
            pred_proba = model.predict_proba(X_test)[:, 1]
            pred_binary = (pred_proba > threshold).astype(int)

            
            if len(y_test) == 0 or len(np.unique(y_test)) < 2: 
                
                fw_roc_scores.append(np.nan)
                fw_acc_scores.append(np.nan)
            else:
                fw_roc_scores.append(roc_auc_score(y_test, pred_proba))
                fw_acc_scores.append(accuracy_score(y_test, pred_binary))

            
            oos_df_period = pd.DataFrame(index=current_test_df.index)
            oos_df_period['Position'] = pd.Series(pred_binary, index=current_test_df.index)
            oos_df_period['Daily_Return'] = current_test_df['Daily_Return']
            
            
            oos_df_period['Strategy_Return'] = oos_df_period['Position'].shift(1).fillna(0) * oos_df_period['Daily_Return']
            
            returns_for_sharpe = oos_df_period['Strategy_Return'].dropna()

            if len(returns_for_sharpe) > 0:
                cumulative_strategy_return = (1 + returns_for_sharpe).cumprod().iloc[-1]
                annualized_return = cumulative_strategy_return ** (252 / len(returns_for_sharpe)) - 1
                annualized_volatility = returns_for_sharpe.std() * np.sqrt(252)
                sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan
            else:
                annualized_return = np.nan
                sharpe_ratio = np.nan

            fw_ann_returns.append(annualized_return)
            fw_sharpe_ratios.append(sharpe_ratio)

        except Exception as e:
            
            fw_roc_scores.append(np.nan)
            fw_acc_scores.append(np.nan)
            fw_ann_returns.append(np.nan)
            fw_sharpe_ratios.append(np.nan)

    
    model_result_row = model_row.to_dict()

    
    model_result_row["FW_ROC_Mean"] = np.nanmean(fw_roc_scores)
    model_result_row["FW_Accuracy_Mean"] = np.nanmean(fw_acc_scores)
    model_result_row["FW_Ann_Return_Mean"] = np.nanmean(fw_ann_returns)
    model_result_row["FW_Sharpe_Mean"] = np.nanmean(fw_sharpe_ratios)
    model_result_row["FW_Periods_Completed"] = len([x for x in fw_roc_scores if not np.isnan(x)]) 

    walk_forward_results.append(model_result_row)

df_results = pd.DataFrame(walk_forward_results)


output_file_name = "walk_forward_backtest_results.xlsx"
df_results.to_excel(output_file_name, index=False)
print(f"\nWalk-forward backtest complete. Results saved to '{output_file_name}'")

import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import random

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


SEED = 42 
random.seed(SEED)
np.random.seed(SEED)

print("Program starting: Data download and Multi-Stage Model Tests...")
start_date, end_date = "2000-01-01", "2025-07-01"

print("1. Raw data is being downloaded...")
spy_base_raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
spy_base_raw.columns = spy_base_raw.columns.get_level_values(0)
spy_base_raw = spy_base_raw[['Open', 'High', 'Low', 'Close', 'Volume']]

tickers = ['^TNX','^FVX','^MOVE','HYG','IEF','RSP','^VIX','^IRX']
for t in tickers:
    col = t.replace('^','') + "_Close"
    spy_base_raw[col] = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']

spy_base_raw.dropna(inplace=True) 

def create_features_and_target(df_segment):
    df_copy = df_segment.copy()

    
    df_copy['M30_MA']          = ta.sma(df_copy['Close'], length=30)
    df_copy['M10_MA']          = ta.sma(df_copy['Close'], length=10)
    macd = ta.macd(df_copy['Close'])
    df_copy['MACD_DIFF']       = macd['MACD_12_26_9'] - macd['MACDs_12_26_9']
    df_copy['OBV']             = ta.obv(df_copy['Close'], df_copy['Volume'])
    df_copy['WILLR_14']        = ta.willr(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)
    df_copy['RSI_14']          = ta.rsi(df_copy['Close'], length=14)
    df_copy['RSI_7']           = ta.rsi(df_copy['Close'], length=7)
    df_copy['ADX_14']          = ta.adx(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)['ADX_14']
    df_copy['StochRSI_14']     = ta.stochrsi(df_copy['Close'], length=14)['STOCHRSIk_14_14_3_3']
    df_copy['Price_Momentum_5D'] = df_copy['Close'].pct_change(5)
    df_copy['VIX_Change_5D']   = df_copy['VIX_Close'].pct_change(5)
    df_copy['Down_Moves']      = (df_copy['Close'].pct_change() < 0).rolling(10).sum()
    df_copy['MACD_Bearish_Cross']= (ta.macd(df_copy['Close'])['MACDh_12_26_9'] < 0).astype(int)
    df_copy['Volume_Spike']    = (df_copy['Volume'] > df_copy['Volume'].rolling(20).mean()*1.5).astype(int)
    df_copy['HYG_IEF']         = df_copy['HYG_Close'] / df_copy['IEF_Close']
    df_copy['Yield_Slope']     = df_copy['TNX_Close'] - df_copy['FVX_Close']
    df_copy['Market_Breadth']  = df_copy['RSP_Close'] / df_copy['Close']

    df_copy['Target']          = (df_copy['Close'].shift(-30) > df_copy['Close']).astype(int)
    df_copy['Daily_Return']    = df_copy['Close'].pct_change()

    return df_copy.dropna()

print("2. Models are being loaded from 'model_results_seed42.xlsx' and initial filter is being applied...")
try:
    results_df = pd.read_excel("model_results_seed42.xlsx")
except FileNotFoundError:
    print("Error: 'model_results_seed42.xlsx' not found. Please run the Random Search script first.")
    exit()

filtered_models_initial = results_df[
    (results_df['ROC_AUC_OOS'] > 0.55) & 
    (results_df['Train_Accuracy'] < 0.80) 
].copy().reset_index(drop=True)

if filtered_models_initial.empty:
    print("No models found satisfying the initial filter criteria. Exiting program.")
    exit()

print(f"After initial filter, {len(filtered_models_initial)} models will be subjected to Walk-Forward test.")

periods_wf = [
    ("2021-01-01","2022-01-01"),
    ("2022-01-01","2023-01-01"),
    ("2023-01-01","2024-01-01"),
    ("2024-01-01","2025-01-01")
]
min_wf_data_points = max(30, 30) 

wf_test_results_raw = [] 
print("\n3. Starting Walk-Forward Backtest for filtered models...")

for idx, model_row in tqdm(filtered_models_initial.iterrows(), total=filtered_models_initial.shape[0], desc="Walk-Forward Test"):
    feature_set = eval(model_row['Features']) if isinstance(model_row['Features'], str) else model_row['Features']
    params = {
        "num_leaves":       int(model_row['num_leaves']),
        "max_depth":        int(model_row['max_depth']),
        "reg_alpha":        float(model_row['reg_alpha']),
        "reg_lambda":       float(model_row['reg_lambda']),
        "scale_pos_weight": float(model_row['scale_pos_weight']),
        "verbosity": -1,
        "force_col_wise": True,
        "random_state": SEED,
        "n_jobs": 1,
        "objective": "binary"
    }
    threshold = float(model_row['threshold'])

    fw_roc_scores, fw_acc_scores, fw_ann_returns, fw_sharpe_ratios = [], [], [], []

    for train_end_str, test_end_str in periods_wf:
        train_end_dt = pd.to_datetime(train_end_str)
        test_end_dt  = pd.to_datetime(test_end_str)

        current_train_base = spy_base_raw.loc[:train_end_dt].copy()
        current_test_base  = spy_base_raw.loc[train_end_dt:test_end_dt].copy()

        train_df = create_features_and_target(current_train_base)
        test_df  = create_features_and_target(current_test_base)
        
        if train_df.empty or test_df.empty or \
           len(train_df) < min_wf_data_points or \
           len(test_df) < min_wf_data_points:
            fw_roc_scores.append(np.nan); fw_acc_scores.append(np.nan)
            fw_ann_returns.append(np.nan); fw_sharpe_ratios.append(np.nan)
            continue

        if not all(f in train_df.columns for f in feature_set) or \
           not all(f in test_df.columns for f in feature_set):
            fw_roc_scores.append(np.nan); fw_acc_scores.append(np.nan)
            fw_ann_returns.append(np.nan); fw_sharpe_ratios.append(np.nan)
            continue

        X_train, y_train = train_df[feature_set], train_df['Target']
        X_test, y_test = test_df[feature_set], test_df['Target']
        
        if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
            fw_roc_scores.append(np.nan); fw_acc_scores.append(np.nan)
            fw_ann_returns.append(np.nan); fw_sharpe_ratios.append(np.nan)
            continue

        model = lgb.LGBMClassifier(**params)
        try:
            model.fit(X_train, y_train)
            pred_proba = model.predict_proba(X_test)[:, 1]
            pred_binary = (pred_proba > threshold).astype(int)

            if len(y_test) == 0 or len(np.unique(y_test)) < 2:
                fw_roc_scores.append(np.nan); fw_acc_scores.append(np.nan)
            else:
                fw_roc_scores.append(roc_auc_score(y_test, pred_proba))
                fw_acc_scores.append(accuracy_score(y_test, pred_binary))

            oos_df_period = pd.DataFrame(index=test_df.index)
            oos_df_period['Position'] = pd.Series(pred_binary, index=test_df.index)
            oos_df_period['Daily_Return'] = test_df['Daily_Return']
            oos_df_period['Strategy_Return'] = oos_df_period['Position'].shift(1).fillna(0) * oos_df_period['Daily_Return']
            
            returns_for_sharpe = oos_df_period['Strategy_Return'].dropna()

            if len(returns_for_sharpe) > 0:
                cumulative_strategy_return = (1 + returns_for_sharpe).cumprod().iloc[-1]
                annualized_return = cumulative_strategy_return ** (252 / len(returns_for_sharpe)) - 1
                annualized_volatility = returns_for_sharpe.std() * np.sqrt(252)
                sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan
            else:
                annualized_return = np.nan; sharpe_ratio = np.nan

            fw_ann_returns.append(annualized_return)
            fw_sharpe_ratios.append(sharpe_ratio)

        except Exception as e:
            fw_roc_scores.append(np.nan); fw_acc_scores.append(np.nan)
            fw_ann_returns.append(np.nan); fw_sharpe_ratios.append(np.nan)
            
    
    model_result_row = model_row.to_dict() 

    model_result_row["FW_ROC_Mean"] = np.nanmean(fw_roc_scores) if fw_roc_scores else np.nan
    model_result_row["FW_Accuracy_Mean"] = np.nanmean(fw_acc_scores) if fw_acc_scores else np.nan
    model_result_row["FW_Ann_Return_Mean"] = np.nanmean(fw_ann_returns) if fw_ann_returns else np.nan
    model_result_row["FW_Sharpe_Mean"] = np.nanmean(fw_sharpe_ratios) if fw_sharpe_ratios else np.nan
    model_result_row["FW_Periods_Completed"] = len([x for x in fw_roc_scores if not np.isnan(x)])

    wf_test_results_raw.append(model_result_row)

df_wf_results = pd.DataFrame(wf_test_results_raw)

min_fw_roc_filter = 0.58
min_fw_sharpe_filter = 0.8
min_fw_periods_completed_filter = 3 
min_fw_ann_return_filter = 0.05 

print("\n4. Models are being filtered based on Walk-Forward results...")
filtered_models_wf = df_wf_results[
    (df_wf_results['FW_ROC_Mean'] > min_fw_roc_filter) &
    (df_wf_results['FW_Sharpe_Mean'] > min_fw_sharpe_filter) &
    (df_wf_results['FW_Periods_Completed'] >= min_fw_periods_completed_filter) &
    (df_wf_results['FW_Ann_Return_Mean'] >= min_fw_ann_return_filter)
].copy().reset_index(drop=True)

if filtered_models_wf.empty:
    print("No models found satisfying the Walk-Forward filters. Exiting program.")
    exit()

print(f"After Walk-Forward filters, {len(filtered_models_wf)} models will be subjected to Bootstrapping test.")


N_BOOTSTRAP_SAMPLES = 50   
OOS_PERIOD_LENGTH_DAYS = 252 
TRAIN_END_LIMIT_DATE_BOOTSTRAP = pd.to_datetime("2024-01-01") 
MIN_TRAIN_LENGTH_DAYS_BOOTSTRAP = 252 * 5 

final_combined_results = [] 
print("\n5. Starting Bootstrapping Test for remaining models...")

for idx, model_row in tqdm(filtered_models_wf.iterrows(), total=filtered_models_wf.shape[0], desc="Bootstrapping Test"):
    
    feature_set = eval(model_row['Features']) if isinstance(model_row['Features'], str) else model_row['Features']
    params = {
        "num_leaves":       int(model_row['num_leaves']),
        "max_depth":        int(model_row['max_depth']),
        "reg_alpha":        float(model_row['reg_alpha']),
        "reg_lambda":       float(model_row['reg_lambda']),
        "scale_pos_weight": float(model_row['scale_pos_weight']),
        "verbosity": -1,
        "force_col_wise": True,
        "random_state": SEED,
        "n_jobs": 1,
        "objective": "binary"
    }
    threshold = float(model_row['threshold'])

    boot_roc_scores, boot_acc_scores, boot_ann_returns, boot_sharpe_ratios = [], [], [], []

    for _ in range(N_BOOTSTRAP_SAMPLES):
        
        potential_train_end_indices = spy_base_raw.index[
            (spy_base_raw.index >= spy_base_raw.index[0] + pd.Timedelta(days=MIN_TRAIN_LENGTH_DAYS_BOOTSTRAP)) &
            (spy_base_raw.index <= TRAIN_END_LIMIT_DATE_BOOTSTRAP - pd.Timedelta(days=OOS_PERIOD_LENGTH_DAYS))
        ]
        
        if len(potential_train_end_indices) == 0:
            break

        train_end_boot = random.choice(potential_train_end_indices)
        oos_start_boot = train_end_boot + pd.Timedelta(days=1)
        oos_end_boot = oos_start_boot + pd.Timedelta(days=OOS_PERIOD_LENGTH_DAYS)

        if oos_end_boot > spy_base_raw.index[-1]:
            oos_end_boot = spy_base_raw.index[-1]
            if (oos_end_boot - oos_start_boot).days < OOS_PERIOD_LENGTH_DAYS * 0.8: 
                continue

        current_train_base_boot = spy_base_raw.loc[:train_end_boot].copy()
        current_test_base_boot  = spy_base_raw.loc[oos_start_boot:oos_end_boot].copy()

        train_df_boot = create_features_and_target(current_train_base_boot)
        test_df_boot  = create_features_and_target(current_test_base_boot)
        
        min_required_df_len = max(30, 30) 
        if train_df_boot.empty or test_df_boot.empty or \
           len(train_df_boot) < MIN_TRAIN_LENGTH_DAYS_BOOTSTRAP / 2 or \
           len(test_df_boot) < min_required_df_len or \
           len(np.unique(test_df_boot['Target'])) < 2:
            continue

        if not all(f in train_df_boot.columns for f in feature_set) or \
           not all(f in test_df_boot.columns for f in feature_set):
            continue

        X_train_boot, y_train_boot = train_df_boot[feature_set], train_df_boot['Target']
        X_test_boot, y_test_boot = test_df_boot[feature_set], test_df_boot['Target']
        
        if X_train_boot.empty or y_train_boot.empty or X_test_boot.empty or y_test_boot.empty:
            continue
        
        model_boot = lgb.LGBMClassifier(**params)
        try:
            model_boot.fit(X_train_boot, y_train_boot)
            pred_proba_boot = model_boot.predict_proba(X_test_boot)[:, 1]
            pred_binary_boot = (pred_proba_boot > threshold).astype(int)

            boot_roc_scores.append(roc_auc_score(y_test_boot, pred_proba_boot))
            boot_acc_scores.append(accuracy_score(y_test_boot, pred_binary_boot))

            oos_df_sample_boot = pd.DataFrame(index=test_df_boot.index)
            oos_df_sample_boot['Position'] = pd.Series(pred_binary_boot, index=test_df_boot.index)
            oos_df_sample_boot['Daily_Return'] = test_df_boot['Daily_Return']
            oos_df_sample_boot['Strategy_Return'] = oos_df_sample_boot['Position'].shift(1).fillna(0) * oos_df_sample_boot['Daily_Return']
            
            returns_for_sharpe_boot = oos_df_sample_boot['Strategy_Return'].dropna()

            if len(returns_for_sharpe_boot) > 0:
                cumulative_strategy_return_boot = (1 + returns_for_sharpe_boot).cumprod().iloc[-1]
                annualized_return_boot = cumulative_strategy_return_boot ** (252 / len(returns_for_sharpe_boot)) - 1
                annualized_volatility_boot = returns_for_sharpe_boot.std() * np.sqrt(252)
                sharpe_ratio_boot = annualized_return_boot / annualized_volatility_boot if annualized_volatility_boot > 0 else np.nan
            else:
                annualized_return_boot = np.nan; sharpe_ratio_boot = np.nan

            boot_ann_returns.append(annualized_return_boot)
            boot_sharpe_ratios.append(sharpe_ratio_boot)

        except Exception as e:
            pass

    
    model_result_row_final = model_row.to_dict() 

    
    model_result_row_final["BOOT_ROC_Mean"] = np.nanmean(boot_roc_scores) if boot_roc_scores else np.nan
    model_result_row_final["BOOT_ROC_Std"] = np.nanstd(boot_roc_scores) if boot_roc_scores else np.nan
    model_result_row_final["BOOT_Accuracy_Mean"] = np.nanmean(boot_acc_scores) if boot_acc_scores else np.nan
    model_result_row_final["BOOT_Accuracy_Std"] = np.nanstd(boot_acc_scores) if boot_acc_scores else np.nan
    model_result_row_final["BOOT_Ann_Return_Mean"] = np.nanmean(boot_ann_returns) if boot_ann_returns else np.nan
    model_result_row_final["BOOT_Ann_Return_Std"] = np.nanstd(boot_ann_returns) if boot_ann_returns else np.nan
    model_result_row_final["BOOT_Sharpe_Mean"] = np.nanmean(boot_sharpe_ratios) if boot_sharpe_ratios else np.nan
    model_result_row_final["BOOT_Sharpe_Std"] = np.nanstd(boot_sharpe_ratios) if boot_sharpe_ratios else np.nan
    model_result_row_final["BOOT_Valid_Samples"] = len([x for x in boot_roc_scores if not np.isnan(x)]) 

    final_combined_results.append(model_result_row_final)

output_df_combined = pd.DataFrame(final_combined_results)

output_file_name_combined = "final_combined_backtest_results.xlsx"
output_df_combined.to_excel(output_file_name_combined, index=False)
print(f"\nCombined and Multi-Stage Backtest completed. Results saved to '{output_file_name_combined}' file.")

print(f"\n--- Loading the best model from '{output_file_name_combined}' and printing details. ---")

try:
    
    loaded_results_df = pd.read_excel(output_file_name_combined)

    
    if 'FW_Sharpe_Mean' not in loaded_results_df.columns:
        print(f"Error: 'FW_Sharpe_Mean' column not found in '{output_file_name_combined}' file.")
    else:
        loaded_results_df['FW_Sharpe_Mean'] = pd.to_numeric(loaded_results_df['FW_Sharpe_Mean'], errors='coerce')

        
        loaded_results_df.dropna(subset=['FW_Sharpe_Mean'], inplace=True)

        if loaded_results_df.empty:
            print("No valid model with Sharpe ratio data found in the loaded file.")
        else:
            
            best_model_row = loaded_results_df.loc[loaded_results_df['FW_Sharpe_Mean'].idxmax()]

            
            print("--------------------------------------------------")

except FileNotFoundError:
    print(f"Error: '{output_file_name_combined}' file not found. Please ensure the file exists.")
except Exception as e:
    print(f"An error occurred while reading the file or finding the best model: {e}")


print("\n=== Selected Model Performance Summary ===")
print(f"Model Name:           {best_model_row.get('Model', 'optimized')}")
print(f"Train Accuracy:       {best_model_row.get('Train_Accuracy', np.nan):.2%}")
print(f"OOS ROC AUC:          {best_model_row.get('ROC_AUC_OOS', np.nan):.4f}")
print(f"OOS Accuracy:         {best_model_row.get('Accuracy_OOS', np.nan):.2%}")
print(f"FW Mean ROC AUC:      {best_model_row.get('FW_ROC_Mean', np.nan):.4f}")
print(f"FW Mean Accuracy:     {best_model_row.get('FW_Accuracy_Mean', np.nan):.2%}")
print(f"FW Mean Ann. Return:  {best_model_row.get('FW_Ann_Return_Mean', np.nan):.2%}")
print(f"FW Mean Sharpe:       {best_model_row.get('FW_Sharpe_Mean', np.nan):.4f}")
print(f"Bootstrap Sharpe μ:    {best_model_row.get('BOOT_Sharpe_Mean', np.nan):.4f}")
print(f"Bootstrap Sharpe σ:    {best_model_row.get('BOOT_Sharpe_Std', np.nan):.4f}")
print("=========================================\n")



