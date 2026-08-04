# core/features.py
import pandas as pd
import numpy as np
from config.settings import FEATURE_COLS

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=20, num_std=2):
    middle = data.rolling(window).mean()
    std = data.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return (data - lower) / (upper - lower), (upper - lower) / middle

def construct_features(df):
    """
    构造所有技术指标和 target（10日后的价格和方向）
    输入必须包含: Open, High, Low, Close, Volume, Amount, Turn, TradeStatus, PctChg, PeTTM, PbMRQ, PsTTM, PcfNcfTTM
    """
    df = df.copy()
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']

    # 原有特征
    df['Momentum_5'] = close.pct_change(5)
    df['Momentum_10'] = close.pct_change(10)
    df['Return_1d'] = close.pct_change()
    df['Volatility_5'] = df['Return_1d'].rolling(5).std()
    df['Volatility_10'] = df['Return_1d'].rolling(10).std()

    for w in [5, 10, 20]:
        df[f'MA_{w}'] = close.rolling(w).mean()
    df['Price_MA_5_Ratio'] = close / df['MA_5'] - 1
    df['Price_MA_20_Ratio'] = close / df['MA_20'] - 1
    df['MA_5_20_diff'] = df['MA_5'] - df['MA_20']

    df['RSI_14'] = calculate_rsi(close, 14)
    bb_pos, _ = calculate_bollinger_bands(close)
    df['BB_position'] = bb_pos

    # Volume Ratio
    ma5_vol = volume.rolling(5).mean()
    df['Volume_Ratio'] = volume / ma5_vol.replace(0, np.nan)

    df['High_Low_Ratio'] = (high - low) / close

    # 新增原生字段
    if 'Amount' in df.columns:
        df['Amount_Ratio'] = df['Amount'] / df['Amount'].rolling(20).mean().replace(0, np.nan)
    else:
        df['Amount_Ratio'] = 0

    if 'PctChg' in df.columns:
        df['PctChg'] = df['PctChg'] / 100.0
    else:
        df['PctChg'] = close.pct_change()

    if 'TradeStatus' in df.columns:
        df['TradeStatus'] = df['TradeStatus'].astype(int)
    else:
        df['TradeStatus'] = 1

    if 'Turn' in df.columns:
        df['Turnover'] = df['Turn'] / 100.0
    else:
        df['Turnover'] = 0

    # Target
    df['Target_Price'] = close.shift(-10)
    df['Target_Direction'] = (close.shift(-10) > close).astype(int)

    # 清理无穷和缺失
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

def clean_data(df):
    """清理异常值并确保类型正确"""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    if 'TradeStatus' in df.columns:
        df['TradeStatus'] = df['TradeStatus'].astype(int).clip(0, 1)
    return df