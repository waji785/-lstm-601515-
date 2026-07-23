import os
import sys
import time
import pandas as pd
import numpy as np
import akshare as ak
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入你现有的回测函数（假设你的主文件是 stock_final.py）
from stock_full2 import train_and_save_model, run_backtest

# =============================================
# 配置
# =============================================
# 白名单文件名
WHITELIST_FILE = "whitelist.csv"

# 全量测试时限制数量（设为 None 表示全量，建议先设 20 测试）
MAX_STOCKS_FULL = None  # 全量模式下测试前 N 只，设为 None 则测试全部

# 白名单筛选条件
WHITELIST_MIN_RETURN = 0.30      # 最低收益率 30%
WHITELIST_MIN_TRADES = 3          # 最少交易次数
WHITELIST_MAX_DRAWDOWN = 0.50     # 最大回撤不超过 50%

# =============================================
# 1. 获取 A 股全列表
# =============================================
def get_all_a_stocks():
    """获取所有 A 股股票代码及名称"""
    print("📊 正在获取 A 股全列表...")
    try:
        df = ak.stock_info_a_code_name()
        print(f"✅ 获取到 {len(df)} 只股票")
        return df
    except Exception as e:
        print(f"❌ 获取列表失败: {e}")
        try:
            df = ak.stock_zh_a_spot_em()
            df = df[['代码', '名称']].copy()
            df.columns = ['code', 'name']
            print(f"✅ 获取到 {len(df)} 只股票（备选接口）")
            return df
        except Exception as e2:
            print(f"❌ 备选方案也失败: {e2}")
            return None

# =============================================
# 2. 加载白名单
# =============================================
def load_whitelist():
    """加载白名单文件，返回股票代码列表"""
    if not os.path.exists(WHITELIST_FILE):
        return None
    
    try:
        df = pd.read_csv(WHITELIST_FILE)
        codes = df['code'].astype(str).str.replace('sh.', '').str.replace('sz.', '').str.strip().tolist()
        print(f"📋 已加载白名单，共 {len(codes)} 只股票")
        return codes
    except Exception as e:
        print(f"⚠️ 读取白名单失败: {e}")
        return None

# =============================================
# 3. 单只股票回测包装函数（同前）
# =============================================
def backtest_single_stock(stock_code, stock_name=""):
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
        model, scaler_X, scaler_y = train_and_save_model(stock_code)
        if model is None:
            result["status"] = "训练失败"
            return result
        
        backtest_df = run_backtest(stock_code, model, scaler_X, scaler_y)
        if backtest_df is None:
            result["status"] = "回测失败"
            return result
        
        df = backtest_df
        result["total_return"] = (df['Capital'].iloc[-1] - df['Capital'].iloc[0]) / df['Capital'].iloc[0]
        result["trade_count"] = df['Position'].diff().abs().sum() / 2
        
        capital_peak = df['Capital'].cummax()
        drawdown = (capital_peak - df['Capital']) / capital_peak
        result["max_drawdown"] = drawdown.max()
        
        daily_returns = df['Strategy_Return'].dropna()
        if len(daily_returns) > 1 and daily_returns.std() != 0:
            result["sharpe_ratio"] = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            result["sharpe_ratio"] = 0
        
        winning = (df['Strategy_Return'] > 0).sum()
        total = (df['Strategy_Return'] != 0).sum()
        result["win_rate"] = winning / total if total > 0 else 0
        
    except Exception as e:
        result["status"] = "异常"
        result["error_msg"] = str(e)
    
    return result

# =============================================
# 4. 生成白名单
# =============================================
def generate_whitelist(result_df, min_return=WHITELIST_MIN_RETURN, min_trades=WHITELIST_MIN_TRADES, max_drawdown=WHITELIST_MAX_DRAWDOWN):
    """从回测结果中筛选出符合条件的好股票，生成白名单"""
    # 只筛选成功的股票
    success_df = result_df[result_df["status"] == "成功"].copy()
    
    if len(success_df) == 0:
        print("⚠️ 没有成功的回测结果，无法生成白名单。")
        return None
    
    # 筛选条件
    whitelist = success_df[
        (success_df['total_return'] > min_return) &
        (success_df['trade_count'] >= min_trades) &
        (success_df['max_drawdown'] < max_drawdown) &
        (~success_df['name'].str.contains('ST|退', na=False, case=False))
    ].copy()
    
    # 按收益率排序
    whitelist = whitelist.sort_values('total_return', ascending=False)
    
    if len(whitelist) == 0:
        print(f"⚠️ 没有股票满足白名单条件（收益>{min_return*100}%，交易次数>={min_trades}，回撤<{max_drawdown*100}%）")
        return None
    
    # 保存白名单（只保留 code 和 name）
    whitelist[['code', 'name']].to_csv(WHITELIST_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n🌟 已生成白名单，共 {len(whitelist)} 只股票（保存至 {WHITELIST_FILE}）")
    print("📋 白名单前10名：")
    print(whitelist[['code', 'name', 'total_return', 'trade_count', 'max_drawdown']].head(10).to_string(index=False, float_format="%.3f"))
    
    return whitelist

# =============================================
# 5. 批量回测主函数（含模式选择）
# =============================================
def batch_backtest():
    """批量回测主流程"""
    print("\n" + "="*60)
    print("🚀 批量回测系统 v2.0")
    print("="*60)
    
    # ---------- 模式选择 ----------
    print("\n请选择运行模式：")
    print("  1. 全量回测（遍历所有A股，生成新白名单）")
    print("  2. 仅跑白名单（只回测白名单中的股票）")
    print("  3. 退出")
    
    mode = input("\n请输入数字 (1/2/3)：").strip()
    
    if mode == "3":
        print("👋 已退出。")
        return
    
    # ---------- 模式2：仅跑白名单 ----------
    if mode == "2":
        whitelist_codes = load_whitelist()
        if whitelist_codes is None or len(whitelist_codes) == 0:
            print("❌ 白名单为空或不存在，请先运行全量回测生成白名单。")
            return
        stock_list = pd.DataFrame({'code': whitelist_codes, 'name': [''] * len(whitelist_codes)})
        print(f"📌 使用白名单，共 {len(stock_list)} 只股票")
    
    # ---------- 模式1：全量回测 ----------
    elif mode == "1":
        stock_df = get_all_a_stocks()
        if stock_df is None or len(stock_df) == 0:
            print("❌ 无法获取股票列表，程序退出。")
            return
        
        if MAX_STOCKS_FULL and len(stock_df) > MAX_STOCKS_FULL:
            stock_df = stock_df.head(MAX_STOCKS_FULL)
            print(f"📌 仅测试前 {MAX_STOCKS_FULL} 只股票（可修改 MAX_STOCKS_FULL 变量）")
        else:
            print(f"📌 全量测试，共 {len(stock_df)} 只股票")
        stock_list = stock_df
    
    else:
        print("❌ 无效输入，请重新运行。")
        return
    
    # ---------- 确认开始 ----------
    confirm = input(f"\n是否开始回测 {len(stock_list)} 只股票？(y/n) ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消操作。")
        return
    
    # ---------- 执行回测 ----------
    results = []
    total = len(stock_list)
    
    print(f"\n📊 开始回测，共 {total} 只股票\n")
    
    for idx, row in tqdm(stock_list.iterrows(), total=total, desc="回测进度"):
        code = str(row['code']).replace('sh.', '').replace('sz.', '').strip()
        if len(code) < 6 and code.isdigit():
            code = code.zfill(6)
        name = row.get('name', '')
        
        result = backtest_single_stock(code, name)
        results.append(result)
        
        # 实时显示结果
        if result["status"] == "成功":
            print(f"  ✅ {code} {name}: 收益 {result['total_return']*100:.2f}%")
        else:
            print(f"  ⚠️ {code} {name}: {result['status']}")
    
    # ---------- 结果汇总 ----------
    result_df = pd.DataFrame(results)
    result_df_sorted = result_df.sort_values("total_return", ascending=False)
    
    # 打印排行榜
    print("\n" + "="*60)
    print("🏆 回测排行榜（前20）")
    print("="*60)
    success_df = result_df_sorted[result_df_sorted["status"] == "成功"].copy()
    if len(success_df) > 0:
        display_cols = ["code", "name", "total_return", "trade_count", "win_rate", "max_drawdown"]
        print(success_df[display_cols].head(20).to_string(index=False, float_format="%.3f"))
    
    # 统计摘要
    print("\n" + "="*60)
    print("📊 统计摘要")
    print("="*60)
    print(f"总测试: {len(result_df)}")
    print(f"成功: {len(success_df)}")
    print(f"训练失败: {len(result_df[result_df['status'] == '训练失败'])}")
    print(f"回测失败: {len(result_df[result_df['status'] == '回测失败'])}")
    print(f"异常: {len(result_df[result_df['status'] == '异常'])}")
    
    # ---------- 生成白名单（仅全量模式） ----------
    if mode == "1" and len(success_df) > 0:
        print("\n" + "="*60)
        print("🌟 正在生成白名单...")
        print("="*60)
        whitelist = generate_whitelist(result_df_sorted)
        
        if whitelist is not None:
            print(f"\n✅ 白名单已生成，共 {len(whitelist)} 只股票")
            print("下次运行可选择「仅跑白名单」模式，只测试这些股票。")
    
    # 保存详细结果
    OUTPUT_CSV = "batch_results.csv"
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ 详细结果已保存至: {OUTPUT_CSV}")
    
    return result_df

# =============================================
# 入口
# =============================================
if __name__ == "__main__":
    batch_backtest()