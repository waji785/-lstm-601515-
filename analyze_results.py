#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量回测结果深度分析脚本
用法: python analyze_results.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 1. 配置
# =============================================
INPUT_FILE = "batch_results.csv"          # 输入文件名
OUTPUT_DIR = "analysis_output"            # 输出目录
WHITELIST_FILE = "whitelist.csv"          # 白名单文件

# 白名单筛选条件（可调）
WHITELIST_MIN_RETURN = 0.50
WHITELIST_MAX_DRAWDOWN = 0.40
WHITELIST_MIN_TRADES = 3

# =============================================
# 2. 加载数据
# =============================================
def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 文件 {INPUT_FILE} 不存在，请先运行批量回测。")
        return None
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ 加载数据: {len(df)} 条记录")
    return df

# =============================================
# 3. 数据清洗
# =============================================
def clean_data(df):
    # 只保留成功回测的记录
    df_success = df[df["status"] == "成功"].copy()
    print(f"✅ 成功回测: {len(df_success)} 只股票")
    
    # 剔除收益率或回撤为空的记录
    df_success = df_success.dropna(subset=["total_return", "max_drawdown", "trade_count"])
    
    # 剔除夏普比率异常值
    df_success = df_success[df_success["sharpe_ratio"] > -10]
    
    return df_success

# =============================================
# 4. 统计分析
# =============================================
def generate_statistics(df):
    stats = {
        "总测试股票数": len(df),
        "正收益占比": (df["total_return"] > 0).mean(),
        "收益率均值": df["total_return"].mean(),
        "收益率中位数": df["total_return"].median(),
        "收益率标准差": df["total_return"].std(),
        "最大收益率": df["total_return"].max(),
        "最小收益率": df["total_return"].min(),
        "平均最大回撤": df["max_drawdown"].mean(),
        "平均交易次数": df["trade_count"].mean(),
        "平均夏普比率": df["sharpe_ratio"].mean(),
    }
    return pd.DataFrame(stats, index=[0])

# =============================================
# 5. 可视化（已修复分箱错误）
# =============================================
def plot_distributions(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 收益率分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(df["total_return"], bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', label='盈亏平衡线')
    plt.axvline(x=df["total_return"].mean(), color='blue', linestyle='-', label=f'均值: {df["total_return"].mean():.2%}')
    plt.xlabel("总收益率")
    plt.ylabel("股票数量")
    plt.title("策略收益率分布")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "return_distribution.png"), dpi=150)
    plt.close()
    
    # 2. 收益率 vs 最大回撤散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(df["max_drawdown"], df["total_return"], alpha=0.6, s=30)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0.3, color='blue', linestyle='--', alpha=0.5, label='回撤30%参考线')
    plt.xlabel("最大回撤")
    plt.ylabel("总收益率")
    plt.title("收益率 vs 最大回撤")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "return_vs_drawdown.png"), dpi=150)
    plt.close()
    
    # 3. 交易次数分布
    plt.figure(figsize=(10, 6))
    plt.hist(df["trade_count"], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("交易次数")
    plt.ylabel("股票数量")
    plt.title("策略交易次数分布")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "trade_count_distribution.png"), dpi=150)
    plt.close()
    
    # 4. 箱线图：按收益率分组的夏普比率（修复版）
    # 过滤掉收益率为 0 的股票，避免分箱边界重复
    df_nonzero = df[df["total_return"] != 0].copy()
    if len(df_nonzero) > 5:
        try:
            # 尝试用 qcut 分箱
            df_nonzero["return_group"] = pd.qcut(df_nonzero["total_return"], q=5, labels=["很低", "低", "中", "高", "很高"])
            # 合并回原数据
            df["return_group"] = "无交易"
            df.loc[df_nonzero.index, "return_group"] = df_nonzero["return_group"]
        except ValueError:
            # 如果仍然报错，改用手动 cut
            bins = [-1, -0.5, -0.1, 0, 0.5, 1, 10]
            labels = ["<-50%", "-50%~-10%", "-10%~0%", "0%~50%", "50%~100%", ">100%"]
            df["return_group"] = pd.cut(df["total_return"], bins=bins, labels=labels, right=False)
    else:
        df["return_group"] = "无交易"
    
    # 只有当有有效分组时才画箱线图
    if df["return_group"].nunique() > 1:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="return_group", y="sharpe_ratio")
        plt.xlabel("收益率分组")
        plt.ylabel("夏普比率")
        plt.title("不同收益率分组的夏普比率")
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "sharpe_by_return_group.png"), dpi=150)
        plt.close()
    else:
        print("⚠️ 有效分组不足，跳过箱线图")
    
    print(f"✅ 图表已保存至 {OUTPUT_DIR}/")

# =============================================
# 6. 白名单筛选
# =============================================
def generate_whitelist(df):
    whitelist = df[
        (df["total_return"] > WHITELIST_MIN_RETURN) &
        (df["max_drawdown"] < WHITELIST_MAX_DRAWDOWN) &
        (df["trade_count"] >= WHITELIST_MIN_TRADES)
    ].sort_values("total_return", ascending=False)
    
    print(f"✅ 白名单股票数: {len(whitelist)}")
    if len(whitelist) > 0:
        whitelist[["code", "name", "total_return", "trade_count", "max_drawdown", "sharpe_ratio"]].to_csv(
            WHITELIST_FILE, index=False, encoding='utf-8-sig'
        )
        print(f"✅ 白名单已保存至 {WHITELIST_FILE}")
        print("\n📋 白名单前10名:")
        print(whitelist[["code", "name", "total_return", "trade_count", "max_drawdown"]].head(10).to_string(
            index=False, float_format="%.3f"
        ))
    else:
        print("⚠️ 没有股票满足白名单条件，请调整筛选阈值。")
    return whitelist

# =============================================
# 7. 相关性检验
# =============================================
def correlation_test(df):
    corr, p_value = spearmanr(df["total_return"], df["max_drawdown"])
    print(f"\n📊 收益率与最大回撤的秩相关系数: {corr:.3f}")
    print(f"   p值: {p_value:.4f}")
    if p_value < 0.05:
        print("   结论: 两者存在显著相关性（p < 0.05）")
    else:
        print("   结论: 无显著相关性（p ≥ 0.05）")
    return corr, p_value

# =============================================
# 8. 生成摘要报告
# =============================================
def generate_summary(df, stats_df, whitelist):
    # 将白名单数量加入统计表
    stats_df["白名单股票数"] = len(whitelist)
    
    # 格式化百分比显示
    for col in stats_df.columns:
        if "占比" in col or "率" in col:
            stats_df[col] = stats_df[col].apply(lambda x: f"{x:.2%}")
    
    # 保存统计表
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "statistics_summary.csv"), index=False, encoding='utf-8-sig')
    print(f"✅ 统计摘要已保存至 {OUTPUT_DIR}/statistics_summary.csv")
    
    # 额外生成一个更详细的描述性统计
    desc = df[["total_return", "max_drawdown", "trade_count", "sharpe_ratio"]].describe()
    desc.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"), encoding='utf-8-sig')
    print(f"✅ 描述性统计已保存至 {OUTPUT_DIR}/descriptive_stats.csv")
    
    return stats_df

# =============================================
# 9. 打印核心统计（终端输出）
# =============================================
def print_statistics(df):
    print("\n" + "="*60)
    print("📊 核心统计结果")
    print("="*60)
    print(f"总测试股票数:        {len(df)}")
    print(f"正收益占比:          {(df['total_return'] > 0).mean():.2%}")
    print(f"收益率均值:          {df['total_return'].mean():.2%}")
    print(f"收益率中位数:        {df['total_return'].median():.2%}")
    print(f"收益率标准差:        {df['total_return'].std():.2%}")
    print(f"最大收益率:          {df['total_return'].max():.2%}")
    print(f"最小收益率:          {df['total_return'].min():.2%}")
    print(f"平均最大回撤:        {df['max_drawdown'].mean():.2%}")
    print(f"平均交易次数:        {df['trade_count'].mean():.1f}")
    print(f"平均夏普比率:        {df['sharpe_ratio'].mean():.3f}")
    print("="*60)

# =============================================
# 10. 主流程
# =============================================
def main():
    print("="*60)
    print("📊 批量回测结果分析系统")
    print("="*60)
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 清洗
    df_clean = clean_data(df)
    if len(df_clean) == 0:
        print("❌ 无有效数据，退出。")
        return
    
    # 打印核心统计
    print_statistics(df_clean)
    
    # 相关性检验
    correlation_test(df_clean)
    
    # 可视化
    plot_distributions(df_clean)
    
    # 白名单
    whitelist = generate_whitelist(df_clean)
    
    # 摘要报告
    stats_df = generate_statistics(df_clean)
    generate_summary(df_clean, stats_df, whitelist)
    
    print("\n✅ 所有分析任务已完成！")
    print(f"📁 结果保存在: {OUTPUT_DIR}/ 和 {WHITELIST_FILE}")

# =============================================
# 11. 入口
# =============================================
if __name__ == "__main__":
    main()