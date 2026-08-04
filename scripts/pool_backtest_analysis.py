# scripts/pool_backtest_analysis.py
import sys
import os
# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from config.settings import *
from core.data_loader import load_from_cache
from core.model import DualLSTM
from core.backtest_engine import run_backtest
from core.metrics import compute_metrics
from core.features import construct_features, clean_data
from utils.common import set_seed
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_unified_model(model_path=None, scaler_x_path=None, scaler_y_path=None):
    """加载全市模型（支持指定路径）"""
    # 如果未指定路径，使用最终模型
    if model_path is None:
        model_path = "model_final.pth"
    if scaler_x_path is None:
        scaler_x_path = "scaler_X_final.pkl"
    if scaler_y_path is None:
        scaler_y_path = "scaler_Y_final.pkl"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualLSTM(input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_X, scaler_y

def run_pool_backtest(whitelist_file=WHITELIST_EXTENDED_FILE, max_stocks=30):
    """组合回测：等权重"""
    if not os.path.exists(whitelist_file):
        logger.error(f"白名单文件 {whitelist_file} 不存在")
        return None

    model, scaler_X, scaler_y = load_unified_model()
    df_white = pd.read_csv(whitelist_file)
    if max_stocks and len(df_white) > max_stocks:
        df_white = df_white.head(max_stocks)

    capital_curves = []
    date_arrays = []
    all_returns = []
    codes = []

    for idx, row in df_white.iterrows():
        code = str(row['code']).zfill(6)
        name = row.get('name', '')
        logger.info(f"回测 {code} {name} ({idx+1}/{len(df_white)})")
        df = load_from_cache(code)
        if df is None or len(df) < 200:
            continue
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Date'] >= BACKTEST_START_DATE].copy()
        if len(df) < 50:
            continue
        if 'Target_Price' not in df.columns:
            df = construct_features(df)
            df = clean_data(df)
        backtest_df = run_backtest(df, model, scaler_X, scaler_y,
                                   start_date=BACKTEST_START_DATE)
        if backtest_df is None:
            continue
        capital_curves.append(backtest_df['Capital'].values)
        date_arrays.append(backtest_df['Date'].values)
        ret = (backtest_df['Capital'].iloc[-1] - backtest_df['Capital'].iloc[0]) / backtest_df['Capital'].iloc[0]
        all_returns.append(ret)
        codes.append(code)

    if not capital_curves:
        logger.error("无有效股票")
        return None

    # 日期对齐（取并集，前向填充）
    all_dates = sorted(set().union(*[set(d) for d in date_arrays]))
    all_dates = pd.to_datetime(all_dates)
    aligned = []
    for curve, dates in zip(capital_curves, date_arrays):
        s = pd.Series(curve, index=dates)
        s = s.reindex(all_dates, method='ffill')
        aligned.append(s.values)
    combined = np.sum(aligned, axis=0) / len(aligned)

    # 绩效
    metrics = compute_metrics(combined)
    logger.info(f"组合总收益: {metrics['total_return']*100:.2f}%, 最大回撤: {metrics['max_drawdown']*100:.2f}%")

    # 绘图
    plt.figure(figsize=(12,6))
    plt.plot(all_dates, combined / combined[0] * 100000, label='组合策略')
    plt.title('组合资金曲线（等权重）')
    plt.xlabel('日期')
    plt.ylabel('资金（元）')
    plt.legend()
    plt.grid(True)
    plt.savefig("pool_curve.png", dpi=150)
    plt.show()

    # 保存结果
    result_df = pd.DataFrame({
        'code': codes,
        'return': all_returns
    })
    result_df.to_csv("pool_returns.csv", index=False)
    logger.info("组合回测完成，结果已保存")

    return combined, result_df

def main():
    run_pool_backtest(max_stocks=30)

if __name__ == "__main__":
    main()