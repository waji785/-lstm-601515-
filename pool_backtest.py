import pandas as pd
import numpy as np
from stock_full2 import load_from_cache, run_trend_backtest, train_and_save_model, set_seed

def run_pool_backtest(whitelist_file="whitelist_extended.csv", max_stocks=10, seed=42):
    """
    对白名单中的股票运行趋势跟踪策略，返回组合收益
    """
    set_seed(seed)
    
    # 加载白名单
    df_whitelist = pd.read_csv(whitelist_file)
    if max_stocks:
        df_whitelist = df_whitelist.head(max_stocks)
    
    print(f"📊 白名单股票数: {len(df_whitelist)}")
    
    all_results = []
    for idx, row in df_whitelist.iterrows():
        code = str(row['code']).zfill(6)
        name = row.get('name', '')
        print(f"\n🔍 正在回测: {code} {name} ({idx+1}/{len(df_whitelist)})")
        
        # 加载数据
        df = load_from_cache(code)
        if df is None or len(df) < 200:
            print(f"  数据不足，跳过")
            continue
        
        # 训练模型
        model, scaler_X, scaler_y, _ = train_and_save_model(stock_code=code)
        if model is None:
            print(f"  训练失败，跳过")
            continue
        
        # 趋势跟踪回测
        backtest_df = run_trend_backtest(df, model, scaler_X, scaler_y)
        if backtest_df is None:
            print(f"  回测失败，跳过")
            continue
        
        total_return = (backtest_df['Capital'].iloc[-1] - backtest_df['Capital'].iloc[0]) / backtest_df['Capital'].iloc[0]
        print(f"  收益率: {total_return*100:.2f}%")
        
        all_results.append({
            'code': code,
            'name': name,
            'return': total_return,
            'capital': backtest_df['Capital'].values
        })
    
    if not all_results:
        print("❌ 没有有效结果")
        return None
    
    # ----- 组合收益（等权重）-----
    # 将所有股票的资金曲线对齐到相同长度
    min_len = min(len(r['capital']) for r in all_results)
    combined_capital = np.ones(min_len) * 100000 / len(all_results)
    
    for r in all_results:
        combined_capital += r['capital'][:min_len] / len(all_results)
    
    combined_return = (combined_capital[-1] - combined_capital[0]) / combined_capital[0]
    
    print(f"\n{'='*60}")
    print(f"📊 组合回测结果 ({len(all_results)} 只股票)")
    print(f"  组合总收益率: {combined_return*100:.2f}%")
    print(f"  平均单只收益率: {np.mean([r['return'] for r in all_results])*100:.2f}%")
    print(f"  正收益股票数: {sum(1 for r in all_results if r['return'] > 0)}/{len(all_results)}")
    print(f"{'='*60}")
    
    return combined_capital, all_results

if __name__ == "__main__":
    run_pool_backtest(max_stocks=10)