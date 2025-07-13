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
    df_copy['M30_MA'] = ta.sma(df_copy['Close'], length=30)
    df_copy['M10_MA'] = ta.sma(df_copy['Close'], length=10)
    macd = ta.macd(df_copy['Close'])
    df_copy['MACD_DIFF'] = macd['MACD_12_26_9'] - macd['MACDs_12_26_9']
    df_copy['OBV'] = ta.obv(df_copy['Close'], df_copy['Volume'])
    df_copy['WILLR_14'] = ta.willr(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)
    df_copy['RSI_14'] = ta.rsi(df_copy['Close'], length=14)
    df_copy['RSI_7'] = ta.rsi(df_copy['Close'], length=7)
    df_copy['ADX_14'] = ta.adx(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)['ADX_14']
    df_copy['StochRSI_14'] = ta.stochrsi(df_copy['Close'], length=14)['STOCHRSIk_14_14_3_3']
    df_copy['Price_Momentum_5D'] = df_copy['Close'].pct_change(5)
    df_copy['VIX_Change_5D'] = df_copy['VIX_Close'].pct_change(5)
    df_copy['Down_Moves'] = (df_copy['Close'].pct_change() < 0).rolling(10).sum()
    df_copy['MACD_Bearish_Cross'] = (ta.macd(df_copy['Close'])['MACDh_12_26_9'] < 0).astype(int)
    df_copy['Volume_Spike'] = (df_copy['Volume'] > df_copy['Volume'].rolling(20).mean() * 1.5).astype(int)
    df_copy['HYG_IEF'] = df_copy['HYG_Close'] / df_copy['IEF_Close']
    df_copy['Yield_Slope'] = df_copy['TNX_Close'] - df_copy['FVX_Close']
    df_copy['Market_Breadth'] = df_copy['RSP_Close'] / df_copy['Close']
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
    except Exception:
        continue

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="ROC_AUC_OOS", ascending=False)
out_name = f"model_results_seed{SEED}.xlsx"
df_results.to_excel(out_name, index=False)
print(f"All trials completed. Results saved to '{out_name}'!")

print("Preparing data for walk-forward backtest...")
start_date, end_date = "2000-01-01", "2025-07-01"

spy_base_raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
spy_base_raw.columns = spy_base_raw.columns.get_level_values(0)
spy_base_raw = spy_base_raw[['Open', 'High', 'Low', 'Close', 'Volume']]

for t in tickers:
    col = t.replace('^','') + "_Close"
    spy_base_raw[col] = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']

spy_base_raw.dropna(inplace=True)

def create_features_and_target(df_segment):
    df_copy = df_segment.copy()
    df_copy['M30_MA'] = ta.sma(df_copy['Close'], length=30)
    df_copy['M10_MA'] = ta.sma(df_copy['Close'], length=10)
    macd = ta.macd(df_copy['Close'])
    df_copy['MACD_DIFF'] = macd['MACD_12_26_9'] - macd['MACDs_12_26_9']
    df_copy['OBV'] = ta.obv(df_copy['Close'], df_copy['Volume'])
    df_copy['WILLR_14'] = ta.willr(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)
    df_copy['RSI_14'] = ta.rsi(df_copy['Close'], length=14)
    df_copy['RSI_7'] = ta.rsi(df_copy['Close'], length=7)
    df_copy['ADX_14'] = ta.adx(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)['ADX_14']
    df_copy['StochRSI_14'] = ta.stochrsi(df_copy['Close'], length=14)['STOCHRSIk_14_14_3_3']
    df_copy['Price_Momentum_5D'] = df_copy['Close'].pct_change(5)
    df_copy['VIX_Change_5D'] = df_copy['VIX_Close'].pct_change(5)
    df_copy['Down_Moves'] = (df_copy['Close'].pct_change() < 0).rolling(10).sum()
    df_copy['MACD_Bearish_Cross'] = (ta.macd(df_copy['Close'])['MACDh_12_26_9'] < 0).astype(int)
    df_copy['Volume_Spike'] = (df_copy['Volume'] > df_copy['Volume'].rolling(20).mean()*1.5).astype(int)
    df_copy['HYG_IEF'] = df_copy['HYG_Close'] / df_copy['IEF_Close']
    df_copy['Yield_Slope'] = df_copy['TNX_Close'] - df_copy['FVX_Close']
    df_copy['Market_Breadth'] = df_copy['RSP_Close'] / df_copy['Close']
    df_copy['Target'] = (df_copy['Close'].shift(-30) > df_copy['Close']).astype(int)
    df_copy['Daily_Return'] = df_copy['Close'].pct_change()
    return df_copy.dropna()

results_df = pd.read_excel("model_results_seed42.xlsx")
filtered_models = results_df[(results_df['ROC_AUC_OOS'] > 0.55) & (results_df['Train_Accuracy'] < 0.80)].reset_index(drop=True)

periods = [
    ("2021-01-01","2022-01-01"),
    ("2022-01-01","2023-01-01"),
    ("2023-01-01","2024-01-01"),
    ("2024-01-01","2025-01-01")
]

walk_forward_results = []

for _, model_row in tqdm(filtered_models.iterrows(), total=filtered_models.shape[0]):
    feature_set = eval(model_row['Features']) if isinstance(model_row['Features'], str) else model_row['Features']
    params = {
        "num_leaves": int(model_row['num_leaves']),
        "max_depth": int(model_row['max_depth']),
        "reg_alpha": float(model_row['reg_alpha']),
        "reg_lambda": float(model_row['reg_lambda']),
        "scale_pos_weight": float(model_row['scale_pos_weight']),
        "verbosity": -1,
        "force_col_wise": True,
        "random_state": SEED,
        "n_jobs": 1,
        "objective": "binary"
    }
    threshold = float(model_row['threshold'])
    fw_roc, fw_acc, fw_ann_ret, fw_sharpe = [], [], [], []

    for start, end in periods:
        train_df = create_features_and_target(spy_base_raw.loc[:start])
        test_df = create_features_and_target(spy_base_raw.loc[start:end])
        if train_df.empty or test_df.empty: continue
        X_train, y_train = train_df[feature_set], train_df['Target']
        X_test, y_test = test_df[feature_set], test_df['Target']
        if len(y_train.unique())<2 or len(y_test.unique())<2: continue
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:,1]
        pred = (prob>threshold).astype(int)
        fw_roc.append(roc_auc_score(y_test, prob))
        fw_acc.append(accuracy_score(y_test, pred))
        strat_ret = ((pred.shift(1).fillna(0)*test_df['Daily_Return'])+1).cumprod().iloc[-1]
        ann_ret = strat_ret**(252/len(test_df))-1
        vol = (pred.shift(1).fillna(0)*test_df['Daily_Return']).std()*np.sqrt(252)
        fw_ann_ret.append(ann_ret)
        fw_sharpe.append(ann_ret/vol if vol>0 else np.nan)

    row = model_row.to_dict()
    row.update({
        "FW_ROC_Mean": np.nanmean(fw_roc),
        "FW_Accuracy_Mean": np.nanmean(fw_acc),
        "FW_Ann_Return_Mean": np.nanmean(fw_ann_ret),
        "FW_Sharpe_Mean": np.nanmean(fw_sharpe)
    })
    walk_forward_results.append(row)

pd.DataFrame(walk_forward_results).to_excel("walk_forward_backtest_results.xlsx", index=False)

loaded = pd.read_excel("final_combined_backtest_results.xlsx")
loaded['FW_Sharpe_Mean'] = pd.to_numeric(loaded['FW_Sharpe_Mean'], errors='coerce')
best = loaded.loc[loaded['FW_Sharpe_Mean'].idxmax()]

print("=== Selected Model Performance Summary ===")
print(f"Model Name: {best.get('Model','optimized')}")
print(f"Train Accuracy: {best.get('Train_Accuracy',np.nan):.2%}")
print(f"OOS ROC AUC: {best.get('ROC_AUC_OOS',np.nan):.4f}")
print(f"OOS Accuracy: {best.get('Accuracy_OOS',np.nan):.2%}")
print(f"FW Mean ROC AUC: {best.get('FW_ROC_Mean',np.nan):.4f}")
print(f"FW Mean Accuracy: {best.get('FW_Accuracy_Mean',np.nan):.2%}")
print(f"FW Mean Ann Return: {best.get('FW_Ann_Return_Mean',np.nan):.2%}")
print(f"FW Mean Sharpe: {best.get('FW_Sharpe_Mean',np.nan):.4f}")
print(f"Bootstrap Sharpe μ: {best.get('BOOT_Sharpe_Mean',np.nan):.4f}")
print(f"Bootstrap Sharpe σ: {best.get('BOOT_Sharpe_Std',np.nan):.4f}")



