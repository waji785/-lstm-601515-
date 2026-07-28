import os
import time
import random
import pandas as pd
import numpy as np
import akshare as ak
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import torch
import joblib
warnings.filterwarnings('ignore')

# ----- 导入核心函数 -----
from stock_full2 import (
    FEATURE_COLS,
    load_all_stock_data,
    train_and_save_model,
    run_trend_backtest,
    load_from_cache,
    DualLSTM,
    set_seed
)

# =============================================
# 配置
# =============================================
MAX_WORKERS = 4                     # 并行线程数
TRAIN_STOCKS = 1000                  # 用于训练的股票数量（建议 200~1000）
TEST_STOCKS = None                  # None 表示测试全部，或设置数字如 100
WHITELIST_FILE = "whitelist.csv"
WHITELIST_MIN_RETURN = 0.30
WHITELIST_MIN_TRADES = 3
WHITELIST_MAX_DRAWDOWN = 0.52

# 随机种子（固定以确保可复现）
SEED = 42
set_seed(SEED)

# =============================================
# 单股票回测（使用已训练好的模型）
# =============================================
def backtest_single_stock_with_model(stock_code, stock_name):
    """
    使用全市模型回测单只股票（每个线程独立加载模型）
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
        # 加载数据
        df = load_from_cache(stock_code)
        if df is None or len(df) < 100:
            result["status"] = "数据不足"
            return result
        
        # 独立加载模型（每个线程单独加载，避免共享状态）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DualLSTM(input_size=len(FEATURE_COLS)).to(device)
        model.load_state_dict(torch.load('model.pth', map_location=device))
        model.eval()
        
        # 加载标准化器（也可共享，但为保险也独立加载）
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        
        # 使用全市模型回测
        backtest_df = run_trend_backtest(df, model, scaler_X, scaler_y)
        if backtest_df is None or len(backtest_df) < 20:
            result["status"] = "回测失败"
            return result
        
        # 提取指标
        result["total_return"] = (backtest_df['Capital'].iloc[-1] - backtest_df['Capital'].iloc[0]) / backtest_df['Capital'].iloc[0]
        result["trade_count"] = backtest_df['Position'].diff().abs().sum() / 2
        
        peak = backtest_df['Capital'].cummax()
        drawdown = (peak - backtest_df['Capital']) / peak
        result["max_drawdown"] = drawdown.max()
        
        daily_ret = backtest_df['Capital'].pct_change().fillna(0)
        excess = daily_ret - 0.025 / 252
        std = excess.std()
        result["sharpe_ratio"] = np.sqrt(252) * excess.mean() / std if std > 1e-8 else 0
        
    except Exception as e:
        result["status"] = "异常"
        result["error_msg"] = str(e)
        import traceback
        traceback.print_exc()
    
    return result
# =============================================
# 获取股票列表（过滤北交所）
# =============================================
def get_stock_list(retries=3):
    for attempt in range(retries):
        try:
            print(f"📊 获取 A 股列表... (尝试 {attempt+1}/{retries})")
            df = ak.stock_info_a_code_name()
            if df is not None and not df.empty:
                # 过滤北交所
                beijing_prefixes = ('920', '430', '830', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879')
                before = len(df)
                df = df[~df['code'].astype(str).str.startswith(beijing_prefixes)]
                print(f"✅ 获取到 {len(df)} 只股票（过滤北交所 {before - len(df)} 只）")
                return df
        except Exception as e:
            print(f"⚠️ 第 {attempt+1} 次尝试失败: {e}")
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    try:
        print("📊 尝试备选接口...")
        df = ak.stock_zh_a_spot_em()
        df = df[['代码', '名称']].copy()
        df.columns = ['code', 'name']
        beijing_prefixes = ('920', '430', '830', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879')
        before = len(df)
        df = df[~df['code'].astype(str).str.startswith(beijing_prefixes)]
        print(f"✅ 获取到 {len(df)} 只股票（备选，过滤北交所 {before - len(df)} 只）")
        return df
    except Exception as e:
        print(f"❌ 所有获取股票列表的方式均失败: {e}")
        return None

# =============================================
# 生成白名单
# =============================================
def generate_whitelist(result_df):
    success_df = result_df[result_df["status"] == "成功"].copy()
    if len(success_df) == 0:
        print("⚠️ 没有成功的回测结果，无法生成白名单。")
        return None
    
    whitelist = success_df[
        (success_df["total_return"] > WHITELIST_MIN_RETURN) &
        (success_df["trade_count"] >= WHILTELIST_MIN_TRADES) &
        (success_df["max_drawdown"] < WHILTELIST_MAX_DRAWDOWN)
    ].sort_values("total_return", ascending=False)
    
    if len(whitelist) == 0:
        print(f"⚠️ 没有股票满足白名单条件")
        return None
    
    whitelist[["code", "name", "total_return", "trade_count", "max_drawdown", "sharpe_ratio"]].to_csv(
        WHITELIST_FILE, index=False, encoding='utf-8-sig'
    )
    print(f"\n🌟 白名单已生成，共 {len(whitelist)} 只股票")
    print("📋 白名单前10名：")
    print(whitelist[["code", "name", "total_return", "trade_count", "max_drawdown"]].head(10).to_string(
        index=False, float_format="%.3f"
    ))
    return whitelist

# =============================================
# 主函数
# =============================================
def main():
    print("="*60)
    print("🚀 全市模型批量回测系统")
    print("="*60)
    print(f"📌 训练股票数: {TRAIN_STOCKS}")
    print(f"📌 测试股票数: {TEST_STOCKS or '全量'}")
    print(f"📌 并行线程数: {MAX_WORKERS}")
    print(f"📌 随机种子: {SEED}")
    
    # ========== 第一步：训练全市模型 ==========
    print(f"\n📊 加载 {TRAIN_STOCKS} 只股票数据用于训练全市模型...")
    df_all = load_all_stock_data(max_stocks=TRAIN_STOCKS)
    if df_all is None or len(df_all) < 1000:
        print("❌ 全市数据不足，请检查缓存或增加股票数量")
        return
    
    print(f"✅ 总样本数: {len(df_all)}")
    
    print("\n🚀 开始训练全市模型（这将花费几分钟到十几分钟）...")
    model, scaler_X, scaler_y, _ = train_and_save_model(df=df_all)
    if model is None:
        print("❌ 训练失败")
        return
    
    print("✅ 全市模型训练完成并保存至 model.pth")
    
    # ========== 第二步：获取待测股票列表 ==========
    stock_df = get_stock_list()
    if stock_df is None or len(stock_df) == 0:
        print("❌ 无法获取股票列表")
        return
    
    if TEST_STOCKS:
        stock_df = stock_df.head(TEST_STOCKS)
        print(f"📌 仅测试前 {TEST_STOCKS} 只股票")
    
    # ========== 第三步：确认开始 ==========
    confirm = input(f"\n是否开始回测 {len(stock_df)} 只股票？(y/n) ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    # 准备任务
    tasks = []
    for _, row in stock_df.iterrows():
        code = str(row['code']).zfill(6)
        name = row.get('name', '')
        tasks.append({"code": code, "name": name})
    
    results = []
    start_time = time.time()
    
    # ========== 第四步：并行回测（仅推理，不训练） ==========
    print("\n🚀 开始回测（使用全市模型，仅推理）...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}
        for idx, t in enumerate(tasks):
            future = executor.submit(
                backtest_single_stock_with_model,
                t['code'], t['name']
            )
            future_map[future] = t
            # 控制提交间隔
            if (idx + 1) % 200 == 0:
                print(f"⏳ 已提交 {idx+1} 个任务，暂停 1 秒冷却...")
                time.sleep(1)
        
        for future in tqdm(as_completed(future_map), total=len(tasks), desc="回测进度"):
            res = future.result()
            results.append(res)
            if res["status"] == "成功":
                print(f"  ✅ {res['code']} {res['name']}: 收益 {res['total_return']*100:.2f}%")
            else:
                print(f"  ⚠️ {res['code']} {res['name']}: {res['status']}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ 回测耗时: {elapsed//60:.0f}分 {elapsed%60:.0f}秒")
    
    # ========== 第五步：保存结果与生成白名单 ==========
    result_df = pd.DataFrame(results)
    result_df.to_csv("batch_results_unified.csv", index=False, encoding='utf-8-sig')
    print("✅ 详细结果已保存至 batch_results_unified.csv")
    
    # 统计摘要
    success_df = result_df[result_df["status"] == "成功"]
    print("\n" + "="*60)
    print("📊 回测统计摘要")
    print("="*60)
    print(f"总测试股票数: {len(result_df)}")
    print(f"成功回测: {len(success_df)}")
    print(f"数据不足: {len(result_df[result_df['status'] == '数据不足'])}")
    print(f"回测失败: {len(result_df[result_df['status'] == '回测失败'])}")
    print(f"异常退出: {len(result_df[result_df['status'] == '异常'])}")
    
    if len(success_df) > 0:
        print(f"\n收益率统计:")
        print(f"  平均值: {success_df['total_return'].mean()*100:.2f}%")
        print(f"  中位数: {success_df['total_return'].median()*100:.2f}%")
        print(f"  最大值: {success_df['total_return'].max()*100:.2f}%")
        print(f"  最小值: {success_df['total_return'].min()*100:.2f}%")
    
    # 生成白名单
    generate_whitelist(result_df)
    
    print("\n✅ 全部完成！")

if __name__ == "__main__":
    main()