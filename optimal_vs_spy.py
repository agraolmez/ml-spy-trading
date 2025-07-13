import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import lightgbm as lgb
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random
import os

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

data_file_name = "spy_ohlcv_macro_data_2000_2025.csv"
start_date, end_date = "2000-01-01", "2025-07-01"

if os.path.exists(data_file_name):
    print(f"Loading raw data from '{data_file_name}' to ensure reproducibility...")
    spy_base_raw = pd.read_csv(data_file_name, index_col=0, parse_dates=True)
else:
    print("WARNING: Data file not found! Downloading data and saving. This should only happen once.")
    print("Downloading raw data (2000-2025) and saving to CSV...")
    spy_base_raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
    spy_base_raw.columns = spy_base_raw.columns.get_level_values(0)
    spy_base_raw = spy_base_raw[['Open', 'High', 'Low', 'Close', 'Volume']]

    tickers_macro = ['^TNX','^FVX','^MOVE','HYG','IEF','RSP','^VIX','^IRX']
    for t in tickers_macro:
        col = t.replace('^','') + "_Close"
        try:
            temp_data = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
            spy_base_raw[col] = temp_data
        except Exception as e:
            print(f"Warning: Data for '{t}' could not be downloaded or an error occurred: {e}. '{col}' might be missing.")

    spy_base_raw.dropna(inplace=True)
    spy_base_raw.to_csv(data_file_name, index=True)
    print(f"Raw data downloaded and saved to '{data_file_name}'.")

def create_features(df_segment):
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
    df_copy['Down_Moves'] = (df_copy['Close'].pct_change() < 0).rolling(10).sum()
    df_copy['MACD_Bearish_Cross'] = (ta.macd(df_copy['Close'])['MACDh_12_26_9'] < 0).astype(int)
    df_copy['Volume_Spike'] = (df_copy['Volume'] > df_copy['Volume'].rolling(20).mean() * 1.5).astype(int)
    df_copy['VIX_Change_5D'] = df_copy['VIX_Close'].pct_change(5) if 'VIX_Close' in df_copy.columns else np.nan
    df_copy['HYG_IEF'] = df_copy['HYG_Close'] / df_copy['IEF_Close'] if 'HYG_Close' in df_copy.columns and 'IEF_Close' in df_copy.columns else np.nan
    df_copy['Yield_Slope'] = df_copy['TNX_Close'] - df_copy['FVX_Close'] if 'TNX_Close' in df_copy.columns and 'FVX_Close' in df_copy.columns else np.nan
    df_copy['Market_Breadth'] = df_copy['RSP_Close'] / df_copy['Close'] if 'RSP_Close' in df_copy.columns else np.nan
    df_copy['Daily_Return'] = df_copy['Close'].pct_change()
    return df_copy

print("Calculating target for the entire dataset (forward-looking 30 days)...")
data_with_target = spy_base_raw.copy()
data_with_target['Target'] = (data_with_target['Close'].shift(-30) > data_with_target['Close']).astype(int)
initial_rows_before_na_drop = len(data_with_target)
data_with_target.dropna(subset=['Target'], inplace=True)
print(f"Dropped {initial_rows_before_na_drop - len(data_with_target)} rows due to NaN in target (last 30 days).")
print(f"Dataset size after target creation: {len(data_with_target)} rows.")

perfect_model_info = {
    'Model_Name_Display': "Optimized Strategy (Fixed)",
    'feature_set': ['Price_Momentum_5D', 'StochRSI_14', 'Down_Moves', 'Volume_Spike', 'VIX_Change_5D', 'Market_Breadth'],
    'params': {
        "num_leaves": 10,
        "max_depth": 5,
        "reg_alpha": 0.0,
        "reg_lambda": 5.0,
        "scale_pos_weight": 1.2,
        "verbosity": -1,
        "force_col_wise": True,
        "random_state": SEED,
        "n_jobs": 1,
        "objective": "binary"
    },
    'threshold': 0.4
}

def run_walk_forward_for_single_model(model_info, data_df, initial_train_end_date, min_test_period_length_arg, approx_annual_days_arg):
    strategy_cumulative_returns = pd.Series([1.0], index=[data_df.index[0]])
    current_train_end = initial_train_end_date
    pbar_total_actual_days = 0
    temp_curr_for_tqdm = initial_train_end_date
    while temp_curr_for_tqdm < data_df.index[-1]:
        temp_test_start_dt = temp_curr_for_tqdm + pd.Timedelta(days=1)
        temp_test_end_dt = temp_test_start_dt + pd.Timedelta(days=approx_annual_days_arg)
        if temp_test_end_dt > data_df.index[-1]:
            temp_test_end_dt = data_df.index[-1]
        segment_dates = data_df.loc[temp_test_start_dt : temp_test_end_dt].index
        pbar_total_actual_days += len(segment_dates)
        temp_curr_for_tqdm = temp_test_end_dt

    with tqdm(total=pbar_total_actual_days, desc=f"Walk-Forward Backtest for {model_info['Model_Name_Display']}") as pbar:
        while current_train_end < data_df.index[-1]:
            test_start_dt = current_train_end + pd.Timedelta(days=1)
            test_end_dt = test_start_dt + pd.Timedelta(days=approx_annual_days_arg)
            if test_end_dt > data_df.index[-1]:
                test_end_dt = data_df.index[-1]
            if test_start_dt > test_end_dt or len(data_df.loc[test_start_dt:test_end_dt]) < min_test_period_length_arg:
                if not strategy_cumulative_returns.empty and len(strategy_cumulative_returns.index) > 0:
                    last_val = strategy_cumulative_returns.iloc[-1]
                    missing_dates_idx = pd.date_range(start=strategy_cumulative_returns.index[-1] + pd.Timedelta(days=1), end=data_df.index[-1])
                    if not missing_dates_idx.empty:
                        strategy_cumulative_returns = pd.concat([strategy_cumulative_returns, pd.Series(last_val, index=missing_dates_idx)])
                break
            current_train_base_segment = data_df.loc[:current_train_end].copy()
            current_test_base_segment = data_df.loc[test_start_dt:test_end_dt].copy()
            train_df = create_features(current_train_base_segment)
            test_df = create_features(current_test_base_segment)
            train_df['Target'] = data_df.loc[train_df.index, 'Target']
            test_df['Target'] = data_df.loc[test_df.index, 'Target']
            train_df.dropna(subset=model_info['feature_set'] + ['Target'], inplace=True)
            test_df.dropna(subset=model_info['feature_set'] + ['Target'], inplace=True)
            X_train_single = train_df[model_info['feature_set']]
            y_train_single = train_df['Target']
            X_test_single = test_df[model_info['feature_set']]
            y_test_single = test_df['Target']
            if train_df.empty or test_df.empty or len(train_df) < min_test_period_length_arg or len(test_df) < min_test_period_length_arg or len(np.unique(y_train_single)) < 2:
                if not strategy_cumulative_returns.empty and len(strategy_cumulative_returns.index) > 0:
                    last_val = strategy_cumulative_returns.iloc[-1]
                    dates_to_fill = pd.date_range(start=test_start_dt, end=test_end_dt)
                    if not dates_to_fill.empty:
                        new_fill_indices = dates_to_fill.difference(strategy_cumulative_returns.index)
                        if not new_fill_indices.empty:
                            strategy_cumulative_returns = pd.concat([strategy_cumulative_returns, pd.Series(last_val, index=new_fill_indices)])
                pbar.update(len(current_test_base_segment.index))
                current_train_end = test_end_dt
                continue
            if X_train_single.empty or y_train_single.empty or len(np.unique(y_train_single)) < 2 or X_test_single.empty:
                pred_binary = np.full(len(test_df), 0)
            else:
                try:
                    model = lgb.LGBMClassifier(**model_info['params'])
                    model.fit(X_train_single, y_train_single)
                    pred_proba_single = model.predict_proba(X_test_single)[:, 1]
                    pred_binary = (pred_proba_single > model_info['threshold']).astype(int)
                except Exception:
                    pred_binary = np.full(len(test_df), 0)
            oos_df_current_period = pd.DataFrame(index=test_df.index)
            oos_df_current_period['Position'] = pd.Series(pred_binary, index=test_df.index)
            oos_df_current_period['Daily_Return'] = test_df['Daily_Return']
            oos_df_current_period['Position'].fillna(0, inplace=True)
            oos_df_current_period['Strategy_Return'] = oos_df_current_period['Position'].shift(1).fillna(0) * oos_df_current_period['Daily_Return']
            if not strategy_cumulative_returns.empty and len(strategy_cumulative_returns.index) > 0:
                last_cum_ret = strategy_cumulative_returns.iloc[-1]
                period_strategy_returns_series = (1 + oos_df_current_period['Strategy_Return']).cumprod()
                new_indices = period_strategy_returns_series.index.difference(strategy_cumulative_returns.index)
                if not new_indices.empty:
                    new_returns = period_strategy_returns_series.loc[new_indices] * last_cum_ret
                    strategy_cumulative_returns = pd.concat([strategy_cumulative_returns, new_returns])
            else:
                strategy_cumulative_returns = (1 + oos_df_current_period['Strategy_Return']).cumprod()
            current_train_end = test_end_dt
            pbar.update(len(current_test_base_segment.index))
    return strategy_cumulative_returns.reindex(data_df.index, method='ffill').fillna(1.0)

full_bh_daily_returns = data_with_target['Close'].pct_change().fillna(0)
full_bh_cumulative_returns = (1 + full_bh_daily_returns).cumprod()
full_bh_cumulative_returns.iloc[0] = 1.0

def format_metric(value, is_percentage=True, decimals=2):
    if pd.isna(value) or np.isinf(value):
        return 'N/A'
    if is_percentage:
        return f"{value:.{decimals}%}"
    return f"{value:.{decimals}f}"

initial_train_end_date_for_wf_start = pd.to_datetime("2021-01-01")
comparison_reporting_start_date = pd.to_datetime("2023-01-01")
min_test_period_length = 30
approx_annual_days = 252

print(f"\n{'='*50}")
print(f"Final Comparison: {perfect_model_info['Model_Name_Display']} vs. Buy & Hold")
print(f"{'='*50}")
print(f"Starting Walk-Forward Backtest for '{perfect_model_info['Model_Name_Display']}' to generate full cumulative returns...")

strategy_cumulative_returns_overall = run_walk_forward_for_single_model(
    perfect_model_info,
    data_with_target,
    initial_train_end_date_for_wf_start,
    min_test_period_length,
    approx_annual_days
)

model_comparison_returns = strategy_cumulative_returns_overall.loc[comparison_reporting_start_date:].copy()
if not model_comparison_returns.empty:
    model_comparison_returns = model_comparison_returns / model_comparison_returns.iloc[0]
else:
    model_comparison_returns = pd.Series([1.0], index=[comparison_reporting_start_date])

bh_comparison_returns = full_bh_cumulative_returns.loc[comparison_reporting_start_date:].copy()
if not bh_comparison_returns.empty:
    bh_comparison_returns = bh_comparison_returns / bh_comparison_returns.iloc[0]
else:
    bh_comparison_returns = pd.Series([1.0], index=[comparison_reporting_start_date])

model_daily_returns_comparison = model_comparison_returns.pct_change().dropna()
if model_daily_returns_comparison.empty:
    model_total_return, model_ann_return, model_ann_vol, model_sharpe = np.nan, np.nan, np.nan, np.nan
else:
    model_total_return = model_comparison_returns.iloc[-1] - 1
    model_ann_return = (1 + model_daily_returns_comparison).prod() ** (252 / len(model_daily_returns_comparison)) - 1
    model_ann_vol = model_daily_returns_comparison.std() * np.sqrt(252)
    model_sharpe = model_ann_return / model_ann_vol if model_ann_vol > 0 else np.nan

bh_daily_returns_comparison = bh_comparison_returns.pct_change().dropna()
if bh_daily_returns_comparison.empty:
    bh_total_return, bh_ann_return, bh_ann_vol, bh_sharpe = np.nan, np.nan, np.nan, np.nan
else:
    bh_total_return = bh_comparison_returns.iloc[-1] - 1
    bh_ann_return = (1 + bh_daily_returns_comparison).prod() ** (252 / len(bh_daily_returns_comparison)) - 1
    bh_ann_vol = bh_daily_returns_comparison.std() * np.sqrt(252)
    bh_sharpe = bh_ann_return / bh_ann_vol if bh_ann_vol > 0 else np.nan

print("\n--- Model Details (Fixed Optimized Model) ---")
print(f"Model Name: {perfect_model_info['Model_Name_Display']}")
print(f"Features: {perfect_model_info['feature_set']}")
print(f"num_leaves: {perfect_model_info['params']['num_leaves']}, max_depth: {perfect_model_info['params']['max_depth']}")
print(f"reg_alpha: {perfect_model_info['params']['reg_alpha']}, reg_lambda: {perfect_model_info['params']['reg_lambda']}")
print(f"scale_pos_weight: {perfect_model_info['params']['scale_pos_weight']}, threshold: {perfect_model_info['threshold']}")
print(f"{'='*50}")

print(f"\n--- Performance Metrics (Comparison Period: {comparison_reporting_start_date.date()} - {data_with_target.index[-1].date()}) ---")
print(f"{'Metric':<20} {'Model':<20} {'Buy & Hold':<20}")
print(f"{'-'*60}")
print(f"{'Total Return':<20} {format_metric(model_total_return):<20} {format_metric(bh_total_return):<20}")
print(f"{'Annualized Return':<20} {format_metric(model_ann_return):<20} {format_metric(bh_ann_return):<20}")
print(f"{'Annualized Volatility':<20} {format_metric(model_ann_vol):<20} {format_metric(bh_ann_vol):<20}")
print(f"{'Sharpe Ratio':<20} {format_metric(model_sharpe, False):<20} {format_metric(bh_sharpe, False):<20}")
print(f"{'='*60}")

plt.figure(figsize=(12, 7))
plt.plot(model_comparison_returns.index, model_comparison_returns, label=f"Strategy: {perfect_model_info['Model_Name_Display']}")
plt.plot(bh_comparison_returns.index, bh_comparison_returns, label="Buy & Hold (SPY)", linestyle="--")
plt.title(f"Strategy ({perfect_model_info['Model_Name_Display']}) vs. Buy & Hold Cumulative Returns (Starting: {comparison_reporting_start_date.date()})")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
formatter = mticker.PercentFormatter(xmax=1.0, decimals=0)
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.show()
