#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
组合回测与绩效分析系统
用法: python pool_backtest_analysis.py
"""

import os
import time
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from stock_full2 import (
    load_from_cache,
    run_trend_backtest,
    DualLSTM,
    FEATURE_COLS,
    set_seed,
    CACHE_DIR,
    fetch_benchmark_data
)

# =============================================
# 1. 加载全市模型
# =============================================
def load_unified_model(model_path="model.pth", 
                       scaler_X_path="scaler_X.pkl", 
                       scaler_y_path="scaler_y.pkl"):
    """
    加载全市训练好的模型和标准化器
    注意：需与保存时模型结构一致（hidden_size=64, num_layers=2）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 与保存的模型结构一致：hidden_size=64, num_layers=2
    model = DualLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    
    print(f"✅ 模型加载成功（hidden_size=64），使用设备: {device}")
    return model, scaler_X, scaler_y

# =============================================
# 2. 组合回测（复用全市模型）
# =============================================
def run_pool_backtest(whitelist_file="whitelist_extended.csv", 
                      max_stocks=None, 
                      initial_capital=100000):
    """
    对白名单中的股票使用全市模型进行组合回测（等权重）
    返回: 组合资金曲线, 各股票收益率列表, 成功股票信息
    """
    print("="*60)
    print("🚀 组合回测系统（复用全市模型）")
    print("="*60)
    
    # 1. 加载全市模型
    model, scaler_X, scaler_y = load_unified_model()
    
    # 2. 加载白名单
    if not os.path.exists(whitelist_file):
        print(f"❌ 未找到白名单文件: {whitelist_file}")
        return None, None, None
    
    df_whitelist = pd.read_csv(whitelist_file)
    if max_stocks is not None and len(df_whitelist) > max_stocks:
        df_whitelist = df_whitelist.head(max_stocks)
    
    print(f"📋 白名单股票数: {len(df_whitelist)}")
    
    # 3. 逐股票回测（使用同一模型）
    all_returns = []
    capital_curves = []
    success_codes = []
    success_names = []
    
    for idx, row in df_whitelist.iterrows():
        code = str(row['code']).zfill(6)
        name = row.get('name', '')
        print(f"\n🔍 回测: {code} {name} ({idx+1}/{len(df_whitelist)})")
        
        # 加载数据
        df = load_from_cache(code)
        if df is None or len(df) < 200:
            print(f"  数据不足，跳过")
            continue
        
        # 使用全市模型回测（不训练）
        backtest_df = run_trend_backtest(df, model, scaler_X, scaler_y, 
                                         initial_capital=initial_capital)
        if backtest_df is None:
            print(f"  回测失败，跳过")
            continue
        
        # 计算收益率
        total_return = (backtest_df['Capital'].iloc[-1] - backtest_df['Capital'].iloc[0]) / backtest_df['Capital'].iloc[0]
        print(f"  收益率: {total_return*100:.2f}%")
        
        all_returns.append(total_return)
        capital_curves.append(backtest_df['Capital'].values)
        success_codes.append(code)
        success_names.append(name)
    
    if not capital_curves:
        print("❌ 没有成功回测的股票")
        return None, None, None
    
    # 4. 组合收益计算（等权重）
    min_len = min(len(curve) for curve in capital_curves)
    n_stocks = len(capital_curves)
    combined_capital = np.zeros(min_len)
    
    for curve in capital_curves:
        combined_capital += curve[:min_len] / n_stocks
    
    combined_return = (combined_capital[-1] - combined_capital[0]) / combined_capital[0]
    
    # 5. 输出结果
    print(f"\n{'='*60}")
    print(f"📊 组合回测结果（{n_stocks} 只股票，等权重）")
    print(f"  组合总收益率: {combined_return*100:.2f}%")
    print(f"  平均单只收益率: {np.mean(all_returns)*100:.2f}%")
    print(f"  正收益股票数: {sum(1 for r in all_returns if r > 0)}/{n_stocks}")
    print(f"  最大单只收益率: {max(all_returns)*100:.2f}%")
    print(f"  最小单只收益率: {min(all_returns)*100:.2f}%")
    print("="*60)
    
    # 保存详细结果
    result_df = pd.DataFrame({
        'code': success_codes,
        'name': success_names,
        'return': all_returns
    })
    result_df.to_csv("pool_backtest_results.csv", index=False, encoding='utf-8-sig')
    print(f"\n💾 详细结果已保存至 pool_backtest_results.csv")
    
    return combined_capital, result_df, df_whitelist

# =============================================
# 3. 绩效指标计算
# =============================================
def compute_metrics(capital_series, risk_free_rate=0.025, trading_days=252):
    """
    计算组合绩效指标
    capital_series: 资金曲线（numpy数组）
    """
    # 日收益率
    daily_ret = np.diff(capital_series) / capital_series[:-1]
    
    # 总收益率
    total_return = (capital_series[-1] - capital_series[0]) / capital_series[0]
    
    # 年化收益率
    n_days = len(capital_series)
    annual_return = (1 + total_return) ** (trading_days / n_days) - 1
    
    # 最大回撤
    peak = np.maximum.accumulate(capital_series)
    drawdown = (peak - capital_series) / peak
    max_drawdown = np.max(drawdown)
    
    # 夏普比率
    excess_ret = daily_ret - risk_free_rate / trading_days
    if np.std(excess_ret) != 0:
        sharpe = np.sqrt(trading_days) * np.mean(excess_ret) / np.std(excess_ret)
    else:
        sharpe = 0
    
    # 胜率（日级别）
    win_days = np.sum(daily_ret > 0)
    total_days = len(daily_ret)
    win_rate = win_days / total_days if total_days > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'n_days': n_days
    }

# =============================================
# 4. 获取沪深300基准（可选）
# =============================================
def get_benchmark_curve(start_date=None, end_date=None):
    """
    获取沪深300指数数据（从缓存或网络），返回价格序列
    """
    df = fetch_benchmark_data()
    if df is None or df.empty:
        print("⚠️ 无法获取沪深300数据，将跳过基准对比")
        return None
    # 如果提供了起止日期，截取相应区间
    if start_date is not None:
        df = df[df['Date'] >= start_date]
    if end_date is not None:
        df = df[df['Date'] <= end_date]
    return df['Close'].values

# =============================================
# 5. 绘图
# =============================================
def plot_pool_performance(combined_capital, benchmark_curve=None, 
                          title="组合资金曲线", save_path="pool_capital_curve.png"):
    """
    绘制组合资金曲线，可选与沪深300对比
    """
    plt.figure(figsize=(12, 6))
    
    # 组合资金曲线（归一化到100000）
    init_cap = combined_capital[0]
    normalized = combined_capital / init_cap * 100000
    plt.plot(normalized, label='组合策略', linewidth=2)
    
    # 如果有基准，绘制基准曲线（归一化）
    if benchmark_curve is not None:
        # 对齐长度（取较短者）
        min_len = min(len(normalized), len(benchmark_curve))
        bench_normalized = benchmark_curve[:min_len] / benchmark_curve[0] * 100000
        plt.plot(bench_normalized, label='沪深300（买入持有）', linestyle='--', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('交易日')
    plt.ylabel('资金 (元)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"📊 资金曲线图已保存至 {save_path}")

# =============================================
# 6. 主程序
# =============================================
def main():
    # 运行组合回测
    combined_capital, result_df, whitelist_df = run_pool_backtest(max_stocks=30)
    
    if combined_capital is None:
        print("❌ 组合回测失败")
        return
    
    # 计算绩效指标
    metrics = compute_metrics(combined_capital)
    
    print("\n" + "="*60)
    print("📈 组合绩效指标")
    print("="*60)
    print(f"  回测区间（交易日）: {metrics['n_days']}")
    print(f"  总收益率:           {metrics['total_return']*100:.2f}%")
    print(f"  年化收益率:         {metrics['annual_return']*100:.2f}%")
    print(f"  最大回撤:           {metrics['max_drawdown']*100:.2f}%")
    print(f"  夏普比率（年化）:   {metrics['sharpe_ratio']:.3f}")
    print(f"  日胜率:             {metrics['win_rate']*100:.2f}%")
    print("="*60)
    
    # 保存绩效指标到CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("pool_metrics.csv", index=False, encoding='utf-8-sig')
    print("💾 绩效指标已保存至 pool_metrics.csv")
    
    # 获取沪深300基准（可选）
    # 注意：资金曲线长度为538个交易日，需要截取对应区间
    # 由于资金曲线是从各股票回测拼接而来，日期难以对齐，这里简单演示
    # 更精确的做法是在回测时记录日期，此处仅作示意
    benchmark = get_benchmark_curve()
    
    # 绘制资金曲线
    plot_pool_performance(combined_capital, benchmark, 
                          title=f"组合策略 vs 沪深300 (30只股票等权重)",
                          save_path="pool_capital_curve.png")
    
    # 额外绘制收益率分布
    plt.figure(figsize=(10, 6))
    plt.hist(result_df['return'], bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', label='盈亏平衡')
    plt.xlabel('单只股票收益率')
    plt.ylabel('股票数量')
    plt.title('白名单股票收益率分布')
    plt.legend()
    plt.grid(True)
    plt.savefig("return_distribution.png", dpi=150)
    plt.show()
    print("📊 收益率分布图已保存至 return_distribution.png")

if __name__ == "__main__":
    main()