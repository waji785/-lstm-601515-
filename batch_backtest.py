#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并行批量回测系统（增量缓存版 + 限流器）
功能：
1. 数据缓存：首次下载后存本地，避免重复下载
2. 增量更新：每次只下载缺失的新数据
3. 并行加速：多线程同时处理多只股票
4. 限流器：控制请求频率，避免触发数据源反爬
用法: python batch_backtest_optimized.py
"""

import os
import time
import pandas as pd
import numpy as np
import akshare as ak
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# 导入你的核心回测模块（请确保 stock_full2.py 在同一目录）
from stock_full2 import train_and_save_model, run_backtest

# =============================================
# 配置区（你可以自由调整）
# =============================================
# 数据缓存目录
CACHE_DIR = "stock_data_cache"

# 并行线程数（建议 4~8，太多可能触发数据源限流）
MAX_WORKERS = 6

# 限流器：每秒最多请求次数（配合 MAX_WORKERS 一起调）
RATE_LIMIT_CALLS = 8   # 每秒最多 8 次请求

# 全量测试时限制股票数量（设为 None 则测试全部）
MAX_STOCKS_FULL = 100   # 建议先 100 测试，确认可行再改 None

# 白名单文件
WHITELIST_FILE = "whitelist.csv"

# 白名单筛选条件（可调）
WHITELIST_MIN_RETURN = 0.30
WHITELIST_MIN_TRADES = 3
WHITELIST_MAX_DRAWDOWN = 0.50

# 训练轮数（可调，默认 20 轮已足够）
EPOCHS = 20

# 数据起止日期
START_DATE = "2020-01-01"
END_DATE = "2026-07-20"

# =============================================
# 1. 限流器装饰器
# =============================================
def rate_limit(max_calls=10, period=1):
    """
    限流器：控制函数在指定时间段内的最大调用次数
    max_calls: 在 period 秒内最多调用次数
    period: 时间窗口（秒）
    """
    def decorator(func):
        last_called = [0.0]  # 用列表存储以便在闭包中修改
        call_count = [0]     # 当前周期内的调用计数
        window_start = [0.0] # 当前窗口开始时间
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            # 如果窗口已过期，重置计数
            if current_time - window_start[0] > period:
                call_count[0] = 0
                window_start[0] = current_time
            
            # 如果已达到最大调用次数，等待到下一个窗口
            if call_count[0] >= max_calls:
                wait_time = window_start[0] + period - current_time
                if wait_time > 0:
                    time.sleep(wait_time)
                # 重置计数器
                call_count[0] = 0
                window_start[0] = time.time()
            
            # 调用原函数
            ret = func(*args, **kwargs)
            call_count[0] += 1
            return ret
        return wrapper
    return decorator

# =============================================
# 2. 数据缓存（增量更新 + 限流）
# =============================================
@rate_limit(max_calls=RATE_LIMIT_CALLS, period=1)
def get_stock_data_cache(stock_code):
    """
    增量更新缓存：只下载本地缺失的新数据
    自动应用限流器，避免请求过快
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{stock_code}.csv")
    
    # 如果缓存文件存在，读取它并获取最新日期
    if os.path.exists(cache_file):
        df_existing = pd.read_csv(cache_file, parse_dates=['Date'])
        if not df_existing.empty:
            last_date = df_existing['Date'].max()
            if last_date >= pd.to_datetime(END_DATE):
                # 缓存已是最新
                return df_existing
            else:
                new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        else:
            new_start = START_DATE.replace("-", "")
            df_existing = pd.DataFrame(columns=['Date', 'Close', 'Volume'])
    else:
        new_start = START_DATE.replace("-", "")
        df_existing = pd.DataFrame(columns=['Date', 'Close', 'Volume'])
    
    # 下载新数据（从 new_start 到 END_DATE）
    try:
        code = stock_code.replace('.', '')
        df_new = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=new_start,
            end_date=END_DATE.replace("-", ""),
            adjust="qfq"
        )
        if df_new is not None and not df_new.empty:
            df_new.rename(columns={'日期': 'Date', '收盘': 'Close', '成交量': 'Volume'}, inplace=True)
            df_new = df_new[['Date', 'Close', 'Volume']].copy()
            df_new['Date'] = pd.to_datetime(df_new['Date'])
            df_new = df_new.astype({'Close': float, 'Volume': float})
            
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
            df_combined.to_csv(cache_file, index=False)
            return df_combined
        else:
            return df_existing if not df_existing.empty else None
    except Exception as e:
        # 静默处理错误，避免中断主流程
        return df_existing if not df_existing.empty else None

# =============================================
# 3. 获取股票列表（带重试）
# =============================================
def get_stock_list_with_retry(retries=3, delay=2):
    """
    获取 A 股列表，失败时自动重试
    """
    for attempt in range(retries):
        try:
            print(f"📊 获取 A 股列表... (尝试 {attempt+1}/{retries})")
            df = ak.stock_info_a_code_name()
            if df is not None and not df.empty:
                print(f"✅ 获取到 {len(df)} 只")
                return df
        except Exception as e:
            print(f"⚠️ 第 {attempt+1} 次尝试失败: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # 指数退避
            continue
        
        # 备选接口
        try:
            print("📊 尝试备选接口...")
            df = ak.stock_zh_a_spot_em()
            df = df[['代码', '名称']].copy()
            df.columns = ['code', 'name']
            print(f"✅ 获取到 {len(df)} 只（备选）")
            return df
        except Exception as e2:
            print(f"⚠️ 备选接口也失败: {e2}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    
    print("❌ 所有获取股票列表的尝试均失败")
    return None

# =============================================
# 4. 单只股票回测包装器（供并行调用）
# =============================================
def backtest_single_stock_wrapper(stock_code, stock_name=""):
    """
    包装训练和回测，返回结果字典
    """
    result = {
        "code": stock_code,
        "name": stock_name,
        "status": "成功",
        "total_return": np.nan,
        "trade_count": np.nan,
        "max_drawdown": np.nan,
        "sharpe_ratio": np.nan,
        "win_rate": np.nan,
        "error_msg": ""
    }
    
    try:
        # 获取数据（自动利用缓存和限流）
        df = get_stock_data_cache(stock_code)
        if df is None or len(df) < 21:
            result["status"] = "数据不足"
            return result
        
        # 训练模型
        model, scaler_X, scaler_y = train_and_save_model(stock_code)
        if model is None:
            result["status"] = "训练失败"
            return result
        
        # 回测
        backtest_df = run_backtest(stock_code, model, scaler_X, scaler_y)
        if backtest_df is None:
            result["status"] = "回测失败"
            return result
        
        # 提取指标
        df_b = backtest_df
        result["total_return"] = (df_b['Capital'].iloc[-1] - df_b['Capital'].iloc[0]) / df_b['Capital'].iloc[0]
        result["trade_count"] = df_b['Position'].diff().abs().sum() / 2
        
        capital_peak = df_b['Capital'].cummax()
        drawdown = (capital_peak - df_b['Capital']) / capital_peak
        result["max_drawdown"] = drawdown.max()
        
        daily_returns = df_b['Strategy_Return'].dropna()
        if len(daily_returns) > 1 and daily_returns.std() != 0:
            result["sharpe_ratio"] = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            result["sharpe_ratio"] = 0
        
        winning = (df_b['Strategy_Return'] > 0).sum()
        total = (df_b['Strategy_Return'] != 0).sum()
        result["win_rate"] = winning / total if total > 0 else 0
        
    except Exception as e:
        result["status"] = "异常"
        result["error_msg"] = str(e)
    
    return result

# =============================================
# 5. 主函数
# =============================================
def main():
    print("="*60)
    print("🚀 并行批量回测系统 (增量缓存版 + 限流器)")
    print("="*60)
    print(f"📌 配置: 线程数={MAX_WORKERS}, 限流={RATE_LIMIT_CALLS}次/秒, 训练轮数={EPOCHS}")
    
    # 获取股票列表
    stock_df = get_stock_list_with_retry()
    if stock_df is None or len(stock_df) == 0:
        # 尝试从本地白名单加载
        if os.path.exists(WHITELIST_FILE):
            stock_df = pd.read_csv(WHILTELIST_FILE)
            print(f"✅ 从本地白名单加载 {len(stock_df)} 只股票")
        else:
            print("❌ 无法获取股票列表，且没有本地白名单")
            return
    
    # 限制数量
    if MAX_STOCKS_FULL and len(stock_df) > MAX_STOCKS_FULL:
        stock_df = stock_df.head(MAX_STOCKS_FULL)
        print(f"📌 只测试前 {MAX_STOCKS_FULL} 只")
    else:
        print(f"📌 全量测试，共 {len(stock_df)} 只")
    
    # 确认
    confirm = input(f"\n是否开始回测 {len(stock_df)} 只股票？(y/n) ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消")
        return
    
    # 准备任务列表
    tasks = []
    for _, row in stock_df.iterrows():
        code = str(row['code']).replace('sh.', '').replace('sz.', '').strip()
        # 补全为6位数字
        if len(code) < 6 and code.isdigit():
            code = code.zfill(6)
        name = row.get('name', '')
        tasks.append({"code": code, "name": name})
    
    results = []
    start_time = time.time()
    
    # 并行执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}
        for t in tasks:
            future = executor.submit(backtest_single_stock_wrapper, t['code'], t['name'])
            future_map[future] = t
        
        for future in tqdm(as_completed(future_map), total=len(tasks), desc="回测进度"):
            res = future.result()
            results.append(res)
            if res["status"] == "成功":
                print(f"  ✅ {res['code']} {res['name']}: 收益 {res['total_return']*100:.2f}%")
            else:
                print(f"  ⚠️ {res['code']} {res['name']}: {res['status']}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed//60:.0f}分 {elapsed%60:.0f}秒")
    
    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv("batch_results.csv", index=False, encoding='utf-8-sig')
    print("✅ 结果已保存至 batch_results.csv")
    
    # 筛选白名单
    success_df = result_df[result_df["status"] == "成功"].copy()
    if len(success_df) > 0:
        whitelist = success_df[
            (success_df["total_return"] > WHITELIST_MIN_RETURN) &
            (success_df["trade_count"] >= WHITELIST_MIN_TRADES) &
            (success_df["max_drawdown"] < WHITELIST_MAX_DRAWDOWN)
        ].sort_values("total_return", ascending=False)
        
        print(f"\n🌟 白名单股票数: {len(whitelist)}")
        if len(whitelist) > 0:
            whitelist[["code", "name", "total_return", "trade_count", "max_drawdown"]].to_csv(
                WHITELIST_FILE, index=False, encoding='utf-8-sig'
            )
            print("📋 白名单前10:")
            print(whitelist[["code", "name", "total_return", "trade_count"]].head(10).to_string(
                index=False, float_format="%.3f"
            ))
        else:
            print("⚠️ 没有股票满足白名单条件，可调整阈值")
    else:
        print("⚠️ 没有股票成功回测")
    
    print("\n✅ 全部完成！")

if __name__ == "__main__":
    main()