# core/data_loader.py
import os
import time
import pandas as pd
import numpy as np
import akshare as ak
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import CACHE_DIR, FEATURE_COLS, TODAY
from core.features import construct_features, clean_data
from utils.logger import setup_logger

logger = setup_logger(__name__)

def get_cache_path(stock_code):
    return os.path.join(CACHE_DIR, f"{stock_code}.parquet")

def load_from_cache(stock_code):
    cache_file = get_cache_path(stock_code)
    if os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            required_cols = set(['Date'] + FEATURE_COLS + ['Target_Price', 'Target_Direction'])
            if required_cols.issubset(set(df.columns)):
                return df
            else:
                logger.warning(f"{stock_code} 缓存缺少特征列，删除并重新下载")
                os.remove(cache_file)
        except Exception as e:
            logger.error(f"读取缓存失败 {stock_code}: {e}")
            os.remove(cache_file)
    return None

def save_to_cache(stock_code, df):
    try:
        df.to_parquet(get_cache_path(stock_code), index=False)
    except Exception as e:
        logger.error(f"保存缓存失败 {stock_code}: {e}")

def fetch_data_akshare(stock_code, start="2018-01-01", end=TODAY):
    code = stock_code.zfill(6)
    if code.startswith('6'):
        symbol = f"sh{code}"
    elif code.startswith(('0', '3')):
        symbol = f"sz{code}"
    else:
        return None
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.replace('-', ''),
            end_date=end.replace('-', ''),
            adjust="qfq"
        )
        if df.empty:
            return None
        df.rename(columns={
            '日期': 'Date', '开盘': 'Open', '收盘': 'Close',
            '最高': 'High', '最低': 'Low',
            '成交量': 'Volume', '成交额': 'Amount',
            '涨跌幅': 'PctChg', '换手率': 'Turn'
        }, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['TradeStatus'] = 1
        df['PeTTM'] = 0
        df['PbMRQ'] = 0
        df['PsTTM'] = 0
        df['PcfNcfTTM'] = 0
        logger.info(f"akshare 获取成功 {code}，共 {len(df)} 条")
        return df
    except Exception as e:
        logger.error(f"akshare 获取失败 {code}: {e}")
        return None

def fetch_data_with_fallback(stock_code, start="2018-01-01", end=TODAY):
    code = stock_code.zfill(6)
    if code.startswith('920'):
        return None
    cached = load_from_cache(code)
    if cached is not None:
        return cached
    logger.info(f"下载 {code} ...")
    df = fetch_data_akshare(code, start, end)
    if df is not None:
        df = construct_features(df)
        df = clean_data(df)
        save_to_cache(code, df)
        return df
    return None

def download_stock_worker(stock_code, start, end):
    try:
        return fetch_data_with_fallback(stock_code, start, end)
    except Exception as e:
        logger.error(f"{stock_code} 下载异常: {e}")
        return None

def load_all_stock_data(max_stocks=None, min_days=100, download_missing=True,
                        start="2018-01-01", end=TODAY, max_workers=1,
                        force_full=False, exclude_st=True):
    logger.info("获取 A 股列表...")
    try:
        stock_df = ak.stock_info_a_code_name()
        stock_df = stock_df[~stock_df['code'].str.startswith(('920','430','830','870','871','872','873','874','875','876','877','878','879'))]
        if exclude_st:
            before = len(stock_df)
            stock_df = stock_df[~stock_df['name'].str.contains('ST|\\*ST', na=False, case=False)]
            logger.info(f"过滤 ST 股票 {before - len(stock_df)} 只")
        codes = stock_df['code'].astype(str).str.zfill(6).tolist()
        logger.info(f"共获取 {len(codes)} 只股票（已过滤北交所和ST）")
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return None

    if max_stocks:
        codes = codes[:max_stocks]
        logger.info(f"限制为前 {max_stocks} 只")

    existing_codes = []
    for code in codes:
        df = load_from_cache(code)
        if df is not None and len(df) >= min_days:
            existing_codes.append(code)

    logger.info(f"已缓存且满足天数（≥{min_days}）的股票: {len(existing_codes)} 只")
    missing_codes = [c for c in codes if c not in existing_codes]
    if force_full:
        missing_codes = codes
    logger.info(f"需下载/更新的股票: {len(missing_codes)} 只")

    if download_missing and missing_codes:
        logger.info(f"开始下载 {len(missing_codes)} 只股票...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_stock_worker, code, start, end): code for code in missing_codes}
            for future in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
                code = futures[future]
                result = future.result()
                if result is not None:
                    existing_codes.append(code)

    all_dfs = []
    for code in existing_codes:
        df = load_from_cache(code)
        if df is not None and len(df) >= min_days:
            df['stock_code'] = code
            all_dfs.append(df)

    if not all_dfs:
        logger.error("未加载到任何符合条件的股票，请检查网络或降低 min_days")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(['stock_code', 'Date']).reset_index(drop=True)
    logger.info(f"最终加载 {len(all_dfs)} 只股票，总样本数 {len(combined)}")
    return combined