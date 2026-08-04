# scripts/expand_whitelist.py
import sys
import os
# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import torch
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config.settings import *
from core.data_loader import load_from_cache
from core.model import DualLSTM
from core.backtest_engine import run_backtest
from core.metrics import compute_metrics
from core.features import construct_features, clean_data
from utils.common import set_seed
from utils.logger import setup_logger

logger = setup_logger(__name__)

WHITELIST_INPUT = WHITELIST_FILE          # 初始白名单
WHITELIST_OUTPUT = WHITELIST_EXTENDED_FILE # 扩展后白名单
TARGET_STOCKS = 30
MAX_WORKERS = 4
MIN_TRADING_DAYS = 200

def backtest_single(code, name, model, scaler_X, scaler_y):
    """使用全市模型回测单只股票"""
    try:
        df = load_from_cache(code)
        if df is None or len(df) < MIN_TRADING_DAYS:
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Date'] >= BACKTEST_START_DATE].copy()
        if len(df) < 50:
            return None
        if 'Target_Price' not in df.columns:
            df = construct_features(df)
            df = clean_data(df)
        backtest_df = run_backtest(df, model, scaler_X, scaler_y,
                                   start_date=BACKTEST_START_DATE)
        if backtest_df is None:
            return None
        metrics = compute_metrics(backtest_df['Capital'].values)
        trades = (backtest_df['Position'].diff().abs() > 0.01).sum() / 2
        return {
            'code': code,
            'name': name,
            'total_return': metrics.get('total_return', np.nan),
            'max_drawdown': metrics.get('max_drawdown', np.nan),
            'sharpe_ratio': metrics.get('sharpe_ratio', np.nan),
            'trade_count': trades
        }
    except Exception as e:
        logger.error(f"{code} 扩展回测异常: {e}")
        return None

def expand_whitelist():
    if not os.path.exists(WHITELIST_INPUT):
        logger.error(f"未找到 {WHITELIST_INPUT}，请先运行 batch_backtest")
        return None

    # 加载全市模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualLSTM(input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    logger.info("全市模型加载成功")

    # 读取初始白名单
    df_orig = pd.read_csv(WHITELIST_INPUT)
    candidates = df_orig.head(TARGET_STOCKS * 2)  # 取前 60 只候选

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for _, row in candidates.iterrows():
            code = str(row['code']).zfill(6)
            name = row.get('name', '')
            futures.append(executor.submit(backtest_single, code, name,
                                           model, scaler_X, scaler_y))
        for future in tqdm(as_completed(futures), total=len(futures), desc="扩展回测"):
            res = future.result()
            if res:
                results.append(res)

    if not results:
        logger.warning("无扩展回测结果")
        return None

    df_res = pd.DataFrame(results).sort_values('total_return', ascending=False)
    final = df_res.head(TARGET_STOCKS)
    final.to_csv(WHITELIST_OUTPUT, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 扩展白名单已保存至 {WHITELIST_OUTPUT}，共 {len(final)} 只")
    return final

if __name__ == "__main__":
    expand_whitelist()