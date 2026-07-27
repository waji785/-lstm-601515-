import os
import time
import random
import pandas as pd
import numpy as np
import akshare as ak
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ----- 直接从主程序导入核心函数和配置 -----
from stock_full2 import (
    FEATURE_COLS,
    fetch_data_with_fallback,
    train_and_save_model,
    run_backtest,
    fetch_benchmark_data,
    DualLSTM
)

# =============================================
# 配置
# =============================================
# =============================================
# 批量回测系统配置参数
# =============================================

# 最大并行线程数（同时回测的股票数量）
# 说明：控制同时运行的回测任务数，影响整体速度
# 取值范围：1 ~ CPU核心数*2（推荐4~8）
# 建议：
#   - 本地数据已缓存时，可调高（8~16）加速I/O读取
#   - 需要联网下载数据时，建议调低（2~4）避免被数据源限流
#   - 显存较小（<4GB）时，调低以避免CUDA out of memory
# 当前值：16（高性能模式，适合本地缓存充沛的场景,第一次运行建议修改为2）
MAX_WORKERS = 16

# 最大测试股票数量
# 说明：限制批量回测的股票总数，用于快速测试或分批执行
# 取值范围：
#   - None：测试全部股票（全量模式）
#   - 正整数：只测试前N只股票（测试模式）
# 建议：
#   - 首次运行建议设为 20~50 快速验证
#   - 全量跑时设为 None
# 当前值：None（全量模式）
MAX_STOCKS_FULL = None

# 白名单输出文件名
# 说明：满足条件的股票会被保存到此CSV文件中
# 格式：code, name, total_return, trade_count, max_drawdown, sharpe_ratio
WHITELIST_FILE = "whitelist.csv"

# =============================================
# 白名单筛选条件（进入白名单的股票必须同时满足以下条件）
# =============================================

# 最低总收益率阈值
# 说明：策略在该股票上的累计收益率必须超过此比例
# 取值范围：0.0 ~ 1.0（对应 0% ~ 100%）
# 建议值：
#   - 稳健策略：0.30~0.50
#   - 激进策略：0.20~0.30
#   - 熊市可适当降低，牛市可提高
# 当前值：0.30（30%）
WHITELIST_MIN_RETURN = 0.30

# 最少交易次数
# 说明：策略在该股票上必须至少交易N次才会被纳入白名单
# 取值范围：正整数（一般建议 3~10）
# 含义：交易次数太少（如1-2次）可能是运气，缺乏统计显著性
# 当前值：3（最低要求，避免过滤掉低频但高收益的机会）
WHITELIST_MIN_TRADES = 3

# 最大可接受回撤
# 说明：策略在该股票上的最大回撤必须小于此比例
# 取值范围：0.0 ~ 1.0（对应 0% ~ 100%）
# 含义：即使收益很高，如果回撤过大，风险太高，不适合稳健策略
# 建议值：0.30~0.50（30%~50%回撤容忍度）
# 当前值：0.52（52%，相对宽松，允许高波动股票进入）
WHITELIST_MAX_DRAWDOWN = 0.52

# =============================================
# 单只股票重试配置
# =============================================

# 单只股票最大递归重试次数
# 说明：当单只股票的数据获取、训练或回测失败时，最多重试几次
# 取值范围：0 ~ 5（推荐2~3）
# 含义：
#   - 重试次数耗尽后，该股票被标记为"放弃"，不再继续尝试
#   - 每次重试前会等待 RETRY_DELAY 秒
# 当前值：3（平衡成功率与时间成本）
MAX_RETRIES_PER_STOCK = 3

# 重试间隔（秒）
# 说明：每次重试前等待的时间
# 取值范围：1 ~ 60（推荐3~10）
# 含义：
#   - 给网络或服务器足够的恢复时间
#   - 避免高频重试被数据源限流
# 当前值：5（适中的等待时间）
RETRY_DELAY = 5

# =============================================
# 全局变量：记录失败股票
# =============================================
FAILED_STOCKS = []

# =============================================
# 获取股票列表
# =============================================
def get_stock_list(retries=3):
    for attempt in range(retries):
        try:
            print(f"📊 获取 A 股列表... (尝试 {attempt+1}/{retries})")
            df = ak.stock_info_a_code_name()
            if df is not None and not df.empty:
                print(f"✅ 获取到 {len(df)} 只股票")
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
        print(f"✅ 获取到 {len(df)} 只股票（备选）")
        return df
    except Exception as e:
        print(f"❌ 所有获取股票列表的方式均失败: {e}")
        return None

# =============================================
# 单只股票回测包装器（带递归重试，但只在本股票内部）
# =============================================
def backtest_single_stock_wrapper(stock_code, stock_name="", benchmark_df=None, retry_count=0):
    """
    对单只股票执行回测，失败时自动递归重试
    retry_count: 当前已重试次数
    """
    global FAILED_STOCKS
    
    # ----- 检查是否已达到最大重试次数 -----
    if retry_count >= MAX_RETRIES_PER_STOCK:
        result = {
            "code": stock_code,
            "name": stock_name,
            "status": "放弃",
            "total_return": np.nan,
            "trade_count": np.nan,
            "max_drawdown": np.nan,
            "sharpe_ratio": np.nan,
            "win_rate": np.nan,
            "error_msg": f"重试 {MAX_RETRIES_PER_STOCK} 次后失败"
        }
        FAILED_STOCKS.append((stock_code, stock_name, f"放弃: 重试{MAX_RETRIES_PER_STOCK}次失败"))
        return result
    
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
        # ----- 1. 获取数据（内部已支持增量更新） -----
        df = fetch_data_with_fallback(stock_code)
        if df is None or len(df) < 21:
            result["status"] = "数据不足"
            FAILED_STOCKS.append((stock_code, stock_name, "数据不足"))
            if retry_count < MAX_RETRIES_PER_STOCK:
                print(f"  🔄 {stock_code} 数据不足，第 {retry_count+1} 次重试...")
                time.sleep(RETRY_DELAY)
                return backtest_single_stock_wrapper(stock_code, stock_name, benchmark_df, retry_count + 1)
            return result
        
        # ----- 2. 训练模型 -----
        model, scaler_X, scaler_y, _ = train_and_save_model(stock_code)
        if model is None:
            result["status"] = "训练失败"
            FAILED_STOCKS.append((stock_code, stock_name, "训练失败"))
            if retry_count < MAX_RETRIES_PER_STOCK:
                print(f"  🔄 {stock_code} 训练失败，第 {retry_count+1} 次重试...")
                time.sleep(RETRY_DELAY)
                return backtest_single_stock_wrapper(stock_code, stock_name, benchmark_df, retry_count + 1)
            return result
        
        # ----- 3. 回测（传入 benchmark_df） -----
        backtest_df = run_backtest(stock_code, model, scaler_X, scaler_y, df, benchmark_df=benchmark_df)
        if backtest_df is None:
            result["status"] = "回测失败"
            FAILED_STOCKS.append((stock_code, stock_name, "回测失败"))
            if retry_count < MAX_RETRIES_PER_STOCK:
                print(f"  🔄 {stock_code} 回测失败，第 {retry_count+1} 次重试...")
                time.sleep(RETRY_DELAY)
                return backtest_single_stock_wrapper(stock_code, stock_name, benchmark_df, retry_count + 1)
            return result
        
        # ----- 4. 提取指标（成功） -----
        df_b = backtest_df
        result["total_return"] = (df_b['Capital'].iloc[-1] - df_b['Capital'].iloc[0]) / df_b['Capital'].iloc[0]
        result["trade_count"] = df_b['Position'].diff().abs().sum() / 2
        
        peak = df_b['Capital'].cummax()
        drawdown = (peak - df_b['Capital']) / peak
        result["max_drawdown"] = drawdown.max()
        
        daily_ret = df_b['Capital'].pct_change().fillna(0)
        risk_free = 0.025
        excess = daily_ret - risk_free / 252
        std = excess.std()
        
        if std > 1e-8:
            result["sharpe_ratio"] = np.sqrt(252) * excess.mean() / std
        else:
            result["sharpe_ratio"] = 0
        
        if 'Strategy_Return' in df_b.columns:
            winning = (df_b['Strategy_Return'] > 0).sum()
            total = (df_b['Strategy_Return'] != 0).sum()
            result["win_rate"] = winning / total if total > 0 else 0
        else:
            result["win_rate"] = 0
        
        # 成功时从失败列表中移除（如果有）
        FAILED_STOCKS = [f for f in FAILED_STOCKS if f[0] != stock_code]
        
    except Exception as e:
        result["status"] = "异常"
        result["error_msg"] = str(e)
        FAILED_STOCKS.append((stock_code, stock_name, f"异常: {str(e)[:50]}"))
        if retry_count < MAX_RETRIES_PER_STOCK:
            print(f"  🔄 {stock_code} 异常 ({str(e)[:30]})，第 {retry_count+1} 次重试...")
            time.sleep(RETRY_DELAY)
            return backtest_single_stock_wrapper(stock_code, stock_name, benchmark_df, retry_count + 1)
    
    return result

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
        (success_df["trade_count"] >= WHITELIST_MIN_TRADES) &
        (success_df["max_drawdown"] < WHITELIST_MAX_DRAWDOWN)
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
    global FAILED_STOCKS
    
    print("="*60)
    print("🚀 批量回测系统 (无集中重试，仅单股票递归重试)")
    print("="*60)
    print(f"📌 配置: 线程数={MAX_WORKERS}, 测试上限={MAX_STOCKS_FULL or '全量'}")
    print(f"📌 单股票递归重试: 最多 {MAX_RETRIES_PER_STOCK} 次, 间隔 {RETRY_DELAY} 秒")
    
    # ========== 第一步：统一获取沪深300指数 ==========
    print("\n📊 正在获取沪深300指数数据（所有股票共用）...")
    benchmark_df = None
    for attempt in range(3):
        try:
            benchmark_df = fetch_benchmark_data()
            if benchmark_df is not None and not benchmark_df.empty:
                print(f"✅ 沪深300指数数据获取成功，共 {len(benchmark_df)} 条记录")
                break
        except Exception as e:
            print(f"⚠️ 沪深300指数获取失败 (尝试 {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(3)
    
    if benchmark_df is None or benchmark_df.empty:
        print("⚠️ 沪深300指数数据获取失败，将跳过所有对比")
    
    # 获取股票列表
    stock_df = get_stock_list()
    if stock_df is None or len(stock_df) == 0:
        if os.path.exists(WHITELIST_FILE):
            stock_df = pd.read_csv(WHITELIST_FILE)
            print(f"✅ 从本地白名单加载 {len(stock_df)} 只股票")
        else:
            print("❌ 无法获取股票列表，且没有本地白名单")
            return
    
    if MAX_STOCKS_FULL and len(stock_df) > MAX_STOCKS_FULL:
        stock_df = stock_df.head(MAX_STOCKS_FULL)
        print(f"📌 仅测试前 {MAX_STOCKS_FULL} 只股票")
    
    confirm = input(f"\n是否开始回测 {len(stock_df)} 只股票？(y/n) ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消")
        return
    
    # 准备任务
    tasks = []
    for _, row in stock_df.iterrows():
        code = str(row['code']).replace('sh.', '').replace('sz.', '').strip()
        if len(code) < 6 and code.isdigit():
            code = code.zfill(6)
        name = row.get('name', '')
        tasks.append({"code": code, "name": name})
    
    results = []
    start_time = time.time()
    
    # ========== 第一轮：并行执行（单股票内部递归重试） ==========
    print("\n🚀 开始回测（单股票内部最多重试 3 次）...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}
        for t in tasks:
            future = executor.submit(backtest_single_stock_wrapper, t['code'], t['name'], benchmark_df, 0)
            future_map[future] = t
            time.sleep(random.uniform(0.3, 0.8))
        
        for future in tqdm(as_completed(future_map), total=len(tasks), desc="回测进度"):
            res = future.result()
            results.append(res)
            if res["status"] == "成功":
                print(f"  ✅ {res['code']} {res['name']}: 收益 {res['total_return']*100:.2f}%")
            elif res["status"] == "放弃":
                print(f"  ❌ {res['code']} {res['name']}: {res['error_msg']}")
            else:
                print(f"  ⚠️ {res['code']} {res['name']}: {res['status']}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ 回测耗时: {elapsed//60:.0f}分 {elapsed%60:.0f}秒")
    
    # ========== 不再进行集中重试 ==========
    print("\n✅ 回测完成，不再进行集中重试。")
    
    # ========== 最终统计 ==========
    print(f"\n📊 最终失败股票数: {len(FAILED_STOCKS)}")
    if FAILED_STOCKS:
        print("失败股票列表（前10）:")
        for code, name, reason in FAILED_STOCKS[:10]:
            print(f"  ❌ {code} {name}: {reason}")
        if len(FAILED_STOCKS) > 10:
            print(f"  ... 共 {len(FAILED_STOCKS)} 只")
    
    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv("batch_results.csv", index=False, encoding='utf-8-sig')
    print("✅ 详细结果已保存至 batch_results.csv")
    
    # 统计摘要
    success_df = result_df[result_df["status"] == "成功"]
    print("\n" + "="*60)
    print("📊 回测统计摘要")
    print("="*60)
    print(f"总测试股票数: {len(result_df)}")
    print(f"成功回测: {len(success_df)}")
    print(f"放弃: {len(result_df[result_df['status'] == '放弃'])}")
    print(f"数据不足: {len(result_df[result_df['status'] == '数据不足'])}")
    print(f"训练失败: {len(result_df[result_df['status'] == '训练失败'])}")
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