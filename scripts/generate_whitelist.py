#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加载最终模型，对全市股票回测，生成白名单
用法: python scripts/generate_whitelist.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import numpy as np
import torch
import joblib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import *
from core.data_loader import load_from_cache
from core.model import DualLSTM
from core.backtest_engine import run_backtest
from core.metrics import compute_metrics
from core.features import construct_features, clean_data
from utils.common import set_seed
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 配置
MAX_WORKERS = 1                 # 并行线程数（单线程更稳定，可设为1）
MIN_TRADING_DAYS = 200          # 股票最少数据天数
WHITELIST_MIN_RETURN = 0.15     # 最低收益率（15%）
WHITELIST_MIN_TRADES = 2        # 最少交易次数
WHITELIST_MAX_DRAWDOWN = 0.40   # 最大回撤 < 40%
BACKTEST_START = "2025-01-01"   # 回测起始日期（白名单评估期）
BACKTEST_END = TODAY            # 回测截止日期（默认今天）

def load_unified_model():
    """加载最终模型（model_final.pth）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model_final.pth"
    scaler_x_path = "scaler_X_final.pkl"
    scaler_y_path = "scaler_Y_final.pkl"
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件 {model_path} 不存在，请先运行 batch_backtest.py 或 train_final_model.py")
        return None, None, None
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    if hasattr(model.lstm, 'flatten_parameters'):
        model.lstm.flatten_parameters()
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    logger.info(f"✅ 模型加载成功: {model_path}")
    return model, scaler_X, scaler_y

def backtest_single_stock(code, model, scaler_X, scaler_y):
    """回测单只股票，返回绩效指标"""
    try:
        df = load_from_cache(code)
        if df is None or len(df) < MIN_TRADING_DAYS:
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= pd.to_datetime(BACKTEST_START)) & 
                (df['Date'] <= pd.to_datetime(BACKTEST_END))].copy()
        if len(df) < 50:
            return None
        if 'Target_Price' not in df.columns:
            df = construct_features(df)
            df = clean_data(df)
        backtest_df = run_backtest(df, model, scaler_X, scaler_y)
        if backtest_df is None or len(backtest_df) < 10:
            return None
        metrics = compute_metrics(backtest_df['Capital'].values)
        trades = (backtest_df['Position'].diff().abs() > 0.01).sum() / 2
        return {
            'code': code,
            'total_return': metrics.get('total_return', np.nan),
            'max_drawdown': metrics.get('max_drawdown', np.nan),
            'sharpe_ratio': metrics.get('sharpe_ratio', np.nan),
            'trade_count': trades
        }
    except Exception as e:
        logger.debug(f"{code} 回测异常: {e}")
        return None

def main():
    set_seed(SEED)
    logger.info("=" * 60)
    logger.info("🚀 生成最终白名单（基于最终模型）")
    logger.info(f"  回测区间: {BACKTEST_START} ~ {BACKTEST_END}")
    logger.info(f"  筛选条件: 收益 > {WHITELIST_MIN_RETURN*100}%, 交易 >= {WHITELIST_MIN_TRADES}, 回撤 < {WHITELIST_MAX_DRAWDOWN*100}%")
    logger.info("=" * 60)

    # 1. 加载模型
    model, scaler_X, scaler_y = load_unified_model()
    if model is None:
        return

    # 2. 获取股票列表（过滤ST和北交所）
    import akshare as ak
    stock_df = ak.stock_info_a_code_name()
    stock_df = stock_df[~stock_df['code'].str.startswith(('920','430','830','870','871','872','873','874','875','876','877','878','879'))]
    stock_df = stock_df[~stock_df['name'].str.contains('ST|\\*ST', na=False, case=False)]
    codes = stock_df['code'].astype(str).str.zfill(6).tolist()
    logger.info(f"全市场正常股票（过滤ST+北交所）: {len(codes)} 只")

    # 3. 并行回测（使用线程池）
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}
        for code in codes:
            future = executor.submit(backtest_single_stock, code, model, scaler_X, scaler_y)
            future_map[future] = code
            time.sleep(0.02)  # 控制请求速率

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="回测进度"):
            res = future.result()
            if res is not None:
                results.append(res)

    if not results:
        logger.error("无有效回测结果")
        return

    df_res = pd.DataFrame(results)
    # 合并股票名称
    name_map = dict(zip(stock_df['code'].astype(str).str.zfill(6), stock_df['name']))
    df_res['name'] = df_res['code'].map(name_map)
    df_res = df_res[['code', 'name', 'total_return', 'max_drawdown', 'sharpe_ratio', 'trade_count']]
    df_res = df_res.sort_values('total_return', ascending=False)

    # 保存全部回测结果（供参考）
    df_res.to_csv("final_backtest_results.csv", index=False, encoding='utf-8-sig')
    logger.info(f"✅ 全部回测结果已保存至 final_backtest_results.csv（共 {len(df_res)} 只）")

    # 4. 筛选白名单
    whitelist = df_res[
        (df_res['total_return'] > WHITELIST_MIN_RETURN) &
        (df_res['trade_count'] >= WHITELIST_MIN_TRADES) &
        (df_res['max_drawdown'] < WHITELIST_MAX_DRAWDOWN)
    ].copy()
    whitelist = whitelist.sort_values('total_return', ascending=False)

    if whitelist.empty:
        logger.warning("无股票满足白名单条件，可降低筛选阈值")
        return

    whitelist.to_csv(WHITELIST_EXTENDED_FILE, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 最终白名单已生成，共 {len(whitelist)} 只，保存至 {WHITELIST_EXTENDED_FILE}")

    # 打印前10名
    print("\n📋 白名单前10名（按收益率排序）:")
    print(whitelist[['code', 'name', 'total_return', 'sharpe_ratio', 'max_drawdown']].head(10).to_string(index=False, float_format="%.3f"))

    # 统计
    print(f"\n📊 白名单统计:")
    print(f"  平均收益率: {whitelist['total_return'].mean()*100:.2f}%")
    print(f"  平均最大回撤: {whitelist['max_drawdown'].mean()*100:.2f}%")
    print(f"  平均夏普: {whitelist['sharpe_ratio'].mean():.3f}")

if __name__ == "__main__":
    main()