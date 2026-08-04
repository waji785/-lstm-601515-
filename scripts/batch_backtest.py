# scripts/batch_backtest.py
import sys
import os
# 获取项目根目录（假设 scripts 在根目录下）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
print(f"项目根目录已添加至 sys.path: {_PROJECT_ROOT}")

import time
import pandas as pd
import numpy as np
import torch
import joblib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config.settings import *
from core.model import DualLSTM
from core.trainer import train_and_save_model
from core.metrics import compute_metrics
from core.features import construct_features, clean_data
from core.data_loader import load_all_stock_data, load_from_cache
from core.backtest_engine import run_backtest
from utils.common import set_seed
from utils.logger import setup_logger
from utils.common import create_sequences

logger = setup_logger(__name__)

# ----- 配置（可覆盖 settings.py 中的值） -----
MAX_WORKERS = 1
TRAIN_STOCKS = 6000          # 训练股票数（用于扩展窗口验证）
TEST_STOCKS = None            # 回测股票数（None 表示全部）
TRAIN_START_DATE = "2018-01-01"
WINDOW_LENGTH_YEARS = 3
TEST_LENGTH_YEARS = 1
NUM_WINDOWS = 6              # 窗口数量（可根据数据长度调整）

def backtest_single_stock(code, name, model, scaler_X, scaler_y, test_start, test_end):
    """使用已加载的模型回测单只股票（指定测试日期范围）"""
    try:
        df = load_from_cache(code)
        if df is None or len(df) < 200:
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        # 过滤测试期
        df_test = df[(df['Date'] >= pd.to_datetime(test_start)) & 
                     (df['Date'] <= pd.to_datetime(test_end))].copy()
        if len(df_test) < 50:
            return None

        # -------- 数据质量检查（在 df_test 定义后）--------
        if df_test.isnull().any().any():
            logger.warning(f"{code} 数据含 NaN，跳过")
            return None
        # 检查价格是否长时间持平（停牌）
        close = df_test['Close']
        if (close == close.shift(1)).sum() > len(close) * 0.3:
            logger.warning(f"{code} 价格长时间持平，跳过")
            return None
        # 检查波动率是否过低
        if close.pct_change().std() < 0.001:
            logger.warning(f"{code} 波动率过低，跳过")
            return None
        # --------------------------------------------

        if 'Target_Price' not in df_test.columns:
            df_test = construct_features(df_test)
            df_test = clean_data(df_test)
        backtest_df = run_backtest(df_test, model, scaler_X, scaler_y)
        if backtest_df is None or len(backtest_df) < 10:
            return None
        metrics = compute_metrics(backtest_df['Capital'].values)
        trades = (backtest_df['Position'].diff().abs() > 0.01).sum() / 2
        return {
            'code': code,
            'name': name,
            'total_return': metrics.get('total_return', np.nan),
            'max_drawdown': metrics.get('max_drawdown', np.nan),
            'sharpe_ratio': metrics.get('sharpe_ratio', np.nan),
            'trade_count': trades,
            'test_start': test_start,
            'test_end': test_end
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"{code} 回测异常: {e}")
        return None

def run_window_backtest(window_idx, train_end_date, test_start, test_end, 
                        df_all, stock_list, model_path_prefix):
    logger.info(f"\n{'='*60}")
    logger.info(f"窗口 {window_idx+1}: 训练截止 {train_end_date}, 测试期 {test_start} ~ {test_end}")
    logger.info(f"{'='*60}")

    model_path = f"{model_path_prefix}_window{window_idx+1}.pth"
    scaler_x_path = f"scaler_X_window{window_idx+1}.pkl"
    scaler_y_path = f"scaler_Y_window{window_idx+1}.pkl"

    # 1. 训练模型
    model, scaler_X, scaler_y, _ = train_and_save_model(
        df=df_all,
        train_end_date=train_end_date,
        model_save_path=model_path,
        scaler_x_path=scaler_x_path,
        scaler_y_path=scaler_y_path
    )
    if model is None:
        logger.error(f"窗口 {window_idx+1} 训练失败")
        return None

    logger.info(f"窗口 {window_idx+1} 训练完成，模型保存至 {model_path}")

    # 2. 重新加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model_reloaded = torch.load(model_path, map_location=device, weights_only=False)
        model_reloaded.eval()
        if hasattr(model_reloaded.lstm, 'flatten_parameters'):
            model_reloaded.lstm.flatten_parameters()
        logger.info("✅ 模型加载成功")
        model = model_reloaded
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 3. 准备边跑边写的 CSV 文件
    csv_file = f"window_{window_idx+1}_results.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)
    header_written = False
    results = []  # 用于汇总统计

    logger.info(f"窗口 {window_idx+1} 开始回测 {len(stock_list)} 只股票...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}
        for idx, row in stock_list.iterrows():
            code = str(row['code']).zfill(6)
            name = row.get('name', '')
            future = executor.submit(backtest_single_stock, code, name, model, 
                                     scaler_X, scaler_y, test_start, test_end)
            future_map[future] = (code, name)
            time.sleep(0.05)

        for future in tqdm(as_completed(future_map), total=len(future_map), desc=f"窗口{window_idx+1}回测"):
            res = future.result()
            if res:
                results.append(res)
                df_row = pd.DataFrame([res])
                if not header_written or not os.path.exists(csv_file):
                    df_row.to_csv(csv_file, index=False, mode='w', encoding='utf-8-sig')
                    header_written = True
                else:
                    df_row.to_csv(csv_file, index=False, mode='a', header=False, encoding='utf-8-sig')

    # 4. 清理模型文件（可选）
    for f in [model_path, scaler_x_path, scaler_y_path]:
        if os.path.exists(f):
            os.remove(f)
            logger.debug(f"已删除 {f}")

    # 5. 如果没有结果，记录并返回
    if not results:
        logger.warning(f"窗口 {window_idx+1} 无有效回测结果")
        # 仍然写入一个空汇总行（但跳过本次统计）
        summary_row = {
            'window': window_idx + 1,
            'train_end': train_end_date,
            'test_start': test_start,
            'test_end': test_end,
            'avg_return': np.nan,
            'win_ratio': np.nan,
            'avg_sharpe': np.nan,
            'avg_max_drawdown': np.nan,
            'num_stocks': 0
        }
        # 写入汇总文件（追加空行）
        summary_file = "expanding_window_summary.csv"
        new_row_df = pd.DataFrame([summary_row])
        if os.path.exists(summary_file):
            df_summary = pd.read_csv(summary_file)
            # 检查是否已存在相同窗口，若存在则删除旧行
            mask = (df_summary['window'] == window_idx + 1) & (df_summary['train_end'] == train_end_date)
            if mask.any():
                df_summary = df_summary[~mask]
            df_summary = pd.concat([df_summary, new_row_df], ignore_index=True)
        else:
            df_summary = new_row_df
        df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        return []

    # 6. 计算汇总统计
    df_res = pd.DataFrame(results)
    avg_ret = df_res['total_return'].mean()
    win_ratio = (df_res['total_return'] > 0).mean()
    avg_sharpe = df_res['sharpe_ratio'].mean()
    avg_dd = df_res['max_drawdown'].mean()

    summary_row = {
        'window': window_idx + 1,
        'train_end': train_end_date,
        'test_start': test_start,
        'test_end': test_end,
        'avg_return': avg_ret,
        'win_ratio': win_ratio,
        'avg_sharpe': avg_sharpe,
        'avg_max_drawdown': avg_dd,
        'num_stocks': len(df_res)
    }

    # 7. 写入汇总文件（安全追加，避免类型错误）
    summary_file = "expanding_window_summary.csv"
    new_row_df = pd.DataFrame([summary_row])
    if os.path.exists(summary_file):
        df_summary = pd.read_csv(summary_file)
        # 检查是否已存在相同窗口，若存在则删除旧行
        mask = (df_summary['window'] == window_idx + 1) & (df_summary['train_end'] == train_end_date)
        if mask.any():
            df_summary = df_summary[~mask]
        df_summary = pd.concat([df_summary, new_row_df], ignore_index=True)
    else:
        df_summary = new_row_df

    df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    logger.info(f"窗口 {window_idx+1} 汇总已保存至 {summary_file}")

    # 打印本窗口汇总
    print(f"\n窗口 {window_idx+1} 汇总:")
    print(f"  平均收益率: {avg_ret*100:.2f}%")
    print(f"  胜率: {win_ratio*100:.2f}%")
    print(f"  平均夏普: {avg_sharpe:.3f}")
    print(f"  平均最大回撤: {avg_dd*100:.2f}%")
    print(f"  有效股票数: {len(df_res)}")

    return results
def main():
    set_seed(SEED)

    # 1. 加载全量数据
    logger.info("加载全市股票数据...")
    df_all = load_all_stock_data(max_stocks=TRAIN_STOCKS, min_days=200)
    if df_all is None or len(df_all) < 1000:
        logger.error("全市数据不足")
        return

    # 2. 获取待测股票列表
    # ----- 2. 获取待测股票列表（用于回测） -----
    import akshare as ak
    stock_df = ak.stock_info_a_code_name()

    # 1. 过滤北交所（与 data_loader 一致）
    stock_df = stock_df[~stock_df['code'].str.startswith(('920','430','830','870','871','872','873','874','875','876','877','878','879'))]

    # 2. 过滤 ST / *ST（与 data_loader 的 exclude_st=True 一致）
    before = len(stock_df)
    stock_df = stock_df[~stock_df['name'].str.contains('ST|\\*ST', na=False, case=False)]
    logger.info(f"过滤 ST 股票 {before - len(stock_df)} 只")

    # 3. 限制测试数量
    if TEST_STOCKS:
        stock_df = stock_df.head(TEST_STOCKS)

    logger.info(f"待回测股票数: {len(stock_df)}")

    # 3. 生成扩展窗口计划
    start_dt = pd.to_datetime(TRAIN_START_DATE)
    data_end_dt = df_all['Date'].max()

    windows = []
    for i in range(NUM_WINDOWS):
        train_end = start_dt + pd.DateOffset(years=WINDOW_LENGTH_YEARS + i * TEST_LENGTH_YEARS)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = train_end + pd.DateOffset(years=TEST_LENGTH_YEARS)
        if test_start > data_end_dt:
            logger.warning(f"窗口 {i+1} 测试起始日期 {test_start} 超出数据范围，跳过")
            break
        if test_end > data_end_dt:
            test_end = data_end_dt
        windows.append({
            'idx': i,
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d')
        })

    if not windows:
        logger.error("无有效窗口，请检查数据范围或调整参数")
        return

    logger.info(f"共生成 {len(windows)} 个窗口")
    for w in windows:
        logger.info(f"窗口 {w['idx']+1}: 训练截止 {w['train_end']}, 测试 {w['test_start']} ~ {w['test_end']}")

    # 4. 执行扩展窗口交叉验证
    all_window_results = []
    model_prefix = "model"
    for w in windows:
        res = run_window_backtest(
            window_idx=w['idx'],
            train_end_date=w['train_end'],
            test_start=w['test_start'],
            test_end=w['test_end'],
            df_all=df_all,
            stock_list=stock_df,
            model_path_prefix=model_prefix
        )
        if res is not None:
            all_window_results.append(res)
        else:
            logger.warning(f"窗口 {w['idx']+1} 无结果")

    if all_window_results:
        # 汇总各窗口结果
        summary = []
        for idx, res_list in enumerate(all_window_results):
            if not res_list:
                continue
            df_res = pd.DataFrame(res_list)
            avg_ret = df_res['total_return'].mean()
            win_ratio = (df_res['total_return'] > 0).mean()
            avg_sharpe = df_res['sharpe_ratio'].mean()
            avg_dd = df_res['max_drawdown'].mean()
            summary.append({
                'window': idx+1,
                'train_end': windows[idx]['train_end'],
                'test_start': windows[idx]['test_start'],
                'test_end': windows[idx]['test_end'],
                'avg_return': avg_ret,
                'win_ratio': win_ratio,
                'avg_sharpe': avg_sharpe,
                'avg_max_drawdown': avg_dd,
                'num_stocks': len(df_res)
            })
            df_res.to_csv(f"window_{idx+1}_results.csv", index=False, encoding='utf-8-sig')

        df_summary = pd.DataFrame(summary)
        df_summary.to_csv("expanding_window_summary.csv", index=False, encoding='utf-8-sig')

        print("\n" + "="*60)
        print("📊 扩展窗口交叉验证汇总")
        print("="*60)
        print(df_summary.to_string(index=False, float_format="%.3f"))
        print("="*60)

        overall_avg = df_summary[['avg_return', 'win_ratio', 'avg_sharpe', 'avg_max_drawdown']].mean()
        print("\n📈 整体平均绩效（各窗口平均）:")
        print(f"  平均收益率: {overall_avg['avg_return']*100:.2f}%")
        print(f"  胜率: {overall_avg['win_ratio']*100:.2f}%")
        print(f"  平均夏普: {overall_avg['avg_sharpe']:.3f}")
        print(f"  平均最大回撤: {overall_avg['avg_max_drawdown']*100:.2f}%")
    else:
        logger.warning("所有窗口均无回测结果，跳过汇总")

    # ========== 5. 训练最终生产模型（使用全部数据，包含最新日期） ==========
    logger.info("\n" + "="*60)
    logger.info("🚀 训练最终生产模型（使用全部历史数据 2018 ~ 今天）")
    logger.info("="*60)

    final_model, final_scaler_X, final_scaler_Y, _ = train_and_save_model(
        df=df_all,
        train_end_date=TODAY,          # 训练截止到今天
        model_save_path="model_final.pth",
        scaler_x_path="scaler_X_final.pkl",
        scaler_y_path="scaler_Y_final.pkl"
    )
    if final_model is not None:
        logger.info(f"✅ 最终模型已保存为 model_final.pth，包含数据截止 {TODAY}")
    else:
        logger.error("最终模型训练失败")

if __name__ == "__main__":
    main()