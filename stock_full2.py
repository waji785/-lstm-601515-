import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import baostock as bs
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import warnings
import joblib
import matplotlib.pyplot as plt
import os
import time
import datetime
import random
from torch.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

# =============================================
# 固定随机种子函数
# =============================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def generate_random_seeds(n=5, min_seed=1, max_seed=10000):
    try:
        random.seed(int(time.time() * 1000))
        if max_seed - min_seed + 1 < n:
            max_seed = min_seed + n + 10
        seeds = random.sample(range(min_seed, max_seed), n)
        return seeds
    except Exception as e:
        print(f"⚠️ 生成随机种子失败: {e}")
        return [42, 123, 2024, 999, 777]

# =============================================
# 解决中文显示问题
# =============================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =============================================
# 获取当前日期
# =============================================
TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
print(f"📅 数据截止日期: {TODAY}")

# =============================================
# 全局策略参数
# =============================================
BUY_THRESHOLD = 0.45
SELL_THRESHOLD = 0.43
STOP_LOSS = -0.08
TAKE_PROFIT = 0.2
MAX_POSITION = 0.8  # 最大仓位

# =============================================
# 数据缓存配置
# =============================================
CACHE_DIR = "stock_data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================
# 核心特征列表（17个）
# =============================================
FEATURE_COLS = [
    'Close', 'Volume',
    'Momentum_5', 'Momentum_10',
    'Return_1d',
    'Volatility_5', 'Volatility_10',
    'MA_5', 'MA_10', 'MA_20',
    'RSI_14',
    'Volume_Ratio',
    'BB_position',
    'Price_MA_5_Ratio', 'Price_MA_20_Ratio',
    'High_Low_Ratio',
    'MA_5_20_diff',
]

# =============================================
# 模型定义
# =============================================
class DualLSTM(nn.Module):
    def __init__(self, input_size=None, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        if input_size is None:
            input_size = len(FEATURE_COLS)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.reg_head(last_out), self.cls_head(last_out)

# =============================================
# 技术指标计算函数
# =============================================
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=20, num_std=2):
    middle = data.rolling(window).mean()
    std = data.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return (data - lower) / (upper - lower), (upper - lower) / middle

def calculate_volume_ratios(volume):
    ma5 = volume.rolling(5).mean()
    ma10 = volume.rolling(10).mean()
    return volume / ma5.replace(0, np.nan), ma5, ma10

def construct_features(df):
    df = df.copy()
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']

    df['Momentum_5'] = close.pct_change(5)
    df['Momentum_10'] = close.pct_change(10)

    df['Return_1d'] = close.pct_change()
    df['Volatility_5'] = df['Return_1d'].rolling(5).std()
    df['Volatility_10'] = df['Return_1d'].rolling(10).std()

    for w in [5, 10, 20]:
        df[f'MA_{w}'] = close.rolling(w).mean()
    df['Price_MA_5_Ratio'] = close / df['MA_5'] - 1
    df['Price_MA_20_Ratio'] = close / df['MA_20'] - 1
    df['MA_5_20_diff'] = df['MA_5'] - df['MA_20']

    df['RSI_14'] = calculate_rsi(close, 14)
    bb_pos, _ = calculate_bollinger_bands(close)
    df['BB_position'] = bb_pos

    vol_ratio, _, _ = calculate_volume_ratios(volume)
    df['Volume_Ratio'] = vol_ratio

    df['High_Low_Ratio'] = (high - low) / close

    df['Target_Price'] = close.shift(-1)
    df['Target_Direction'] = (close.shift(-1) > close).astype(int)
    return df.dropna()

def clean_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

# =============================================
# 数据缓存函数
# =============================================
def get_cache_path(stock_code):
    return os.path.join(CACHE_DIR, f"{stock_code}.csv")

def load_from_cache(stock_code):
    cache_file = get_cache_path(stock_code)
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=['Date'])
            if not df.empty:
                return df
        except:
            pass
    return None

def save_to_cache(stock_code, df):
    try:
        df.to_csv(get_cache_path(stock_code), index=False)
    except:
        pass

def load_all_stock_data(cache_dir="stock_data_cache", max_stocks=None, min_days=100):
    all_dfs = []
    csv_files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]
    if max_stocks:
        csv_files = csv_files[:max_stocks]
    total = len(csv_files)
    print(f"📊 开始加载本地股票数据，共 {total} 个文件...")
    for i, file in enumerate(csv_files):
        code = file.replace('.csv', '')
        try:
            df = pd.read_csv(os.path.join(cache_dir, file), parse_dates=['Date'])
            if len(df) >= min_days:
                # ----- 保留 Date 列用于排序 -----
                cols = ['Date'] + FEATURE_COLS + ['Target_Price', 'Target_Direction']
                df = df[cols].copy()
                all_dfs.append(df)
                if (i+1) % 200 == 0:
                    print(f"  已加载 {i+1}/{total} 个文件")
        except:
            pass
    if not all_dfs:
        print("❌ 未加载到任何有效数据")
        return None
    combined = pd.concat(all_dfs, ignore_index=True)
    # ----- 按日期排序 -----
    combined = combined.sort_values('Date').reset_index(drop=True)
    print(f"✅ 共加载 {len(all_dfs)} 只股票，总样本数: {len(combined)}")
    return combined
def run_trend_backtest(df, model, scaler_X, scaler_y, initial_capital=100000):
    """
    趋势跟踪回测（含趋势过滤器 + 动态仓位）
    """
    if df is None or len(df) < 21:
        return None
    if 'Target_Price' not in df.columns:
        df = construct_features(df)
        df = clean_data(df)
    
    scaled_features = scaler_X.transform(df[FEATURE_COLS].values)
    X, _, _ = create_sequences(scaled_features, df['Target_Price'].values, df['Target_Direction'].values)
    if len(X) == 0:
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dates = df['Date'].values[20:]
    close_prices = df['Close'].values[20:]
    
    # ----- 计算技术指标（用于趋势过滤）-----
    ma_50 = df['Close'].rolling(50).mean().values[20:]   # 50日均线
    ma_200 = df['Close'].rolling(200).mean().values[20:] # 200日均线
    atr = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
    atr = atr.values[20:] / df['Close'].values[20:] * 100  # ATR百分比
    
    # ----- 策略参数 -----
    TREND_THRESHOLD = 0.02  # 趋势强度阈值（价格相对MA200的偏离）
    MIN_VOLATILITY = 0.5    # 最低波动率（ATR% > 0.5% 才交易）
    MAX_POSITION = 0.8      # 最大仓位 80%
    
    positions = []
    capital = float(initial_capital)
    holdings = 0.0
    entry_price = 0.0
    trade_log = []
    up_probs = []
    
    with torch.no_grad():
        for i in range(len(X_tensor)):
            current_price = close_prices[i]
            x_sample = X_tensor[i].unsqueeze(0)
            _, dir_logits = model(x_sample)
            prob = torch.softmax(dir_logits, dim=1).squeeze().cpu().numpy()
            up_prob = prob[1]
            up_probs.append(up_prob)
            
            # ----- 1. 趋势过滤器（决定是否允许开仓）-----
            # 价格相对200日均线的偏离
            price_ma200_ratio = (current_price - ma_200[i]) / ma_200[i] if ma_200[i] > 0 else 0
            # 短期均线方向
            ma_5 = df['Close'].rolling(5).mean().values[20+i] if i+20 < len(df) else 0
            ma_20 = df['Close'].rolling(20).mean().values[20+i] if i+20 < len(df) else 0
            trend_up = ma_5 > ma_20  # 短期均线上穿中期均线
            
            # 只有满足以下条件才允许开多仓：
            # (1) 价格在MA200之上（牛市环境）或 (2) 趋势强度足够（偏离 > 2%）
            allow_long = (price_ma200_ratio > 0.05) or (trend_up and price_ma200_ratio > -0.02)
            
            # ----- 2. 波动率过滤器（低波动时减少风险暴露）-----
            vol_factor = min(atr[i] / MIN_VOLATILITY, 1.0) if i < len(atr) else 0.5
            
            # ----- 3. 持仓管理 -----
            if holdings > 0:
                profit = (current_price - entry_price) / entry_price
                
                # 跟踪止损：当盈利超过10%后，回撤超过5%则止盈
                if profit > 0.10:
                    trailing_stop = max(0.05, profit * 0.5)
                    if profit - (current_price / entry_price - 1) + 1 < trailing_stop:
                        capital = holdings * current_price
                        trade_log.append(('跟踪止盈', dates[i], current_price, profit))
                        holdings = 0
                        entry_price = 0
                        positions.append(0)
                        continue
                
                # 硬止损/止盈
                if profit >= TAKE_PROFIT:
                    capital = holdings * current_price
                    trade_log.append(('止盈', dates[i], current_price, profit))
                    holdings = 0
                    entry_price = 0
                    positions.append(0)
                    continue
                elif profit <= STOP_LOSS:
                    capital = holdings * current_price
                    trade_log.append(('止损', dates[i], current_price, profit))
                    holdings = 0
                    entry_price = 0
                    positions.append(0)
                    continue
            
            # ----- 4. 开仓信号（趋势 + 概率 + 波动）-----
            if holdings == 0 and allow_long and up_prob > BUY_THRESHOLD and vol_factor > 0.3:
                # 动态仓位：概率越高仓位越重，且受波动率限制
                position_ratio = min((up_prob - 0.45) * 3, MAX_POSITION) * vol_factor
                position_ratio = max(position_ratio, 0.1)  # 最低10%仓位
                buy_amount = capital * position_ratio
                holdings = buy_amount / current_price
                entry_price = current_price
                capital = capital - buy_amount
                trade_log.append(('买入', dates[i], current_price, None, position_ratio))
                positions.append(position_ratio)  # 记录仓位比例
            else:
                positions.append(1 if holdings > 0 else 0)
    
    # ----- 诊断输出 -----
    if up_probs:
        print(f"📊 预测概率统计: 均值={np.mean(up_probs):.4f}, 最大值={np.max(up_probs):.4f}, 最小值={np.min(up_probs):.4f}")
    
    if not positions:
        return None
    
    # 将仓位比例转换为实际持仓
    backtest_df = pd.DataFrame({
        'Date': dates[:len(positions)],
        'Close': close_prices[:len(positions)].astype(float),
        'Position': positions,
        'Capital': float(initial_capital)
    }, dtype=float)
    
    # 计算策略收益（考虑动态仓位）
    backtest_df['Position_float'] = backtest_df['Position'].apply(lambda x: x if isinstance(x, float) else (1.0 if x > 0 else 0.0))
    backtest_df['Capital'] = initial_capital * (1 + backtest_df['Close'].pct_change().fillna(0) * backtest_df['Position_float'].shift(1).fillna(0)).cumprod()
    backtest_df['Capital'] = backtest_df['Capital'].astype(float)
    
    return backtest_df
# =============================================
# 序列生成
# =============================================
def create_sequences(features, price_targets, dir_targets, seq_len=20):
    X, yp, yd = [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        yp.append(price_targets[i])
        yd.append(dir_targets[i])
    return np.array(X, dtype=np.float32), np.array(yp, dtype=np.float32), np.array(yd, dtype=np.float32)

# =============================================
# baostock 数据获取
# =============================================
def fetch_data_baostock(stock_code, start="2020-01-01", end=TODAY, retries=5):
    print(f"📊 正在从 baostock 获取 {stock_code} 数据...")
    code = stock_code.replace('.', '').replace('sh', '').replace('sz', '')
    if len(code) < 6 and code.isdigit():
        code = code.zfill(6)
    if code.startswith(('0', '3')):
        bs_code = f"sz.{code}"
    else:
        bs_code = f"sh.{code}"
    for attempt in range(retries):
        try:
            try:
                bs.logout()
            except:
                pass
            time.sleep(0.5)
            lg = bs.login()
            if lg.error_code != '0':
                print(f"⚠️ 登录失败 (尝试 {attempt+1}/{retries}): {lg.error_msg}")
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 3
                    time.sleep(wait_time)
                    continue
                return None
            rs = bs.query_history_k_data_plus(
                bs_code,
                fields="date,open,high,low,close,volume",
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag="3"
            )
            if rs is None or rs.error_code != '0':
                error_msg = rs.error_msg if rs else "未知错误"
                print(f"⚠️ 查询失败 (尝试 {attempt+1}/{retries}): {error_msg}")
                bs.logout()
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 3
                    time.sleep(wait_time)
                    continue
                return None
            data_list = []
            while (rs.error_code == '0') & rs.next():
                row = rs.get_row_data()
                if row is not None:
                    data_list.append(row)
            bs.logout()
            if not data_list:
                print(f"⚠️ 未获取到数据 (尝试 {attempt+1}/{retries})")
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 3
                    time.sleep(wait_time)
                    continue
                return None
            df = pd.DataFrame(data_list, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df = df.replace('', np.nan).dropna()
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            df = df.astype({
                'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            print(f"✅ baostock 获取成功，共 {len(df)} 个交易日")
            return df
        except Exception as e:
            error_msg = str(e)
            if "WinError 10038" in error_msg or "non-socket" in error_msg:
                print(f"⚠️ 套接字错误 (尝试 {attempt+1}/{retries}): 连接已关闭，正在重置并重试...")
                try:
                    bs.logout()
                except:
                    pass
                time.sleep(5)
            elif "invalid distance" in error_msg or "invalid start byte" in error_msg:
                print(f"⚠️ 数据损坏 (尝试 {attempt+1}/{retries})")
            else:
                print(f"⚠️ 第 {attempt+1} 次尝试失败: {e}")
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 4
                print(f"⏳ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"❌ 失败 {retries} 次，放弃该股票")
                return None
    return None

def fetch_data_with_fallback(stock_code, start="2020-01-01", end=TODAY):
    print(f"\n🔍 开始获取 {stock_code} 数据...")
    code = stock_code.replace('.', '').replace('sh', '').replace('sz', '')
    if len(code) < 6 and code.isdigit():
        code = code.zfill(6)
    cached_df = load_from_cache(code)
    if cached_df is not None:
        required_cols = set(FEATURE_COLS + ['Date', 'Target_Price', 'Target_Direction'])
        if required_cols.issubset(set(cached_df.columns)):
            print(f"✅ 缓存数据完整，共 {len(cached_df)} 个交易日")
            return clean_data(cached_df)
        else:
            print("⚠️ 缓存数据不完整，重新下载...")
    df = update_cache_incremental(code, start_date=start)
    if df is not None:
        if 'Target_Price' not in df.columns:
            df = construct_features(df)
        df = clean_data(df)
        save_to_cache(code, df)
        return df
    else:
        for attempt in range(3):
            print(f"📡 尝试全量下载 baostock... (第 {attempt+1} 次)")
            df_raw = fetch_data_baostock(code, start, end)
            if df_raw is not None:
                df = construct_features(df_raw)
                df = clean_data(df)
                save_to_cache(code, df)
                print(f"✅ 全量下载成功，共 {len(df)} 个交易日")
                return df
            else:
                if attempt < 2:
                    time.sleep(3)
        print("❌ 所有数据获取方式均失败")
        return None

def update_cache_incremental(stock_code, start_date=None):
    cache_file = get_cache_path(stock_code)
    old_df = load_from_cache(stock_code)
    if old_df is not None and not old_df.empty:
        last_date = old_df['Date'].max().strftime("%Y-%m-%d")
        print(f"📂 缓存最新日期: {last_date}")
        if start_date is None:
            start_date = last_date
        else:
            start_date = max(start_date, last_date)
    else:
        old_df = pd.DataFrame()
        if start_date is None:
            start_date = "2020-01-01"
    print(f"📡 正在获取 {stock_code} 从 {start_date} 之后的数据...")
    new_df = fetch_data_baostock(stock_code, start=start_date)
    if new_df is None or new_df.empty:
        print(f"⚠️ {stock_code} 无新数据，返回已有缓存")
        return old_df if not old_df.empty else None
    if not old_df.empty:
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined['Date'] = pd.to_datetime(combined['Date'])
    combined = combined.drop_duplicates(subset=['Date']).sort_values('Date')
    combined = combined.reset_index(drop=True)
    save_to_cache(stock_code, combined)
    print(f"✅ {stock_code} 缓存更新成功，共 {len(combined)} 条记录")
    return combined

def fetch_benchmark_data(start="2020-01-01", end=TODAY):
    cache_file = os.path.join(CACHE_DIR, "benchmark_300.csv")
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=['Date'])
            if not df.empty:
                print(f"📂 从缓存加载沪深300数据，共 {len(df)} 条记录")
                return df
        except:
            pass
    print("📊 正在获取沪深300指数数据...")
    try:
        lg = bs.login()
        if lg.error_code != '0':
            return None
        rs = bs.query_history_k_data_plus("sh.000300",
            fields="date,close", start_date=start, end_date=end,
            frequency="d", adjustflag="3")
        if rs.error_code != '0':
            bs.logout()
            return None
        data = []
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            if row:
                data.append(row)
        bs.logout()
        if not data:
            return None
        df = pd.DataFrame(data, columns=['Date', 'Close'])
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df.to_csv(cache_file, index=False)
        print(f"💾 沪深300数据已缓存，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"⚠️ 获取沪深300失败: {e}")
        return None

# =============================================
# 训练函数
# =============================================
def train_and_save_model(stock_code=None, df=None, batch_size=1024, epochs=100, train_ratio=0.7):
    """
    训练模型并保存

    参数:
        stock_code: 单只股票代码（与 df 二选一）
        df: 合并后的 DataFrame（与 stock_code 二选一）
        batch_size: 训练批次大小
        epochs: 最大训练轮数
        train_ratio: 训练集比例（前 train_ratio 为训练，后 1-train_ratio 为测试）
    """
    # ==================== 数据加载 ====================
    if df is None and stock_code is not None:
        df = load_from_cache(stock_code)
        if df is None:
            print(f"❌ 未找到 {stock_code} 缓存，请先下载")
            return None, None, None, None
        if 'Target_Price' not in df.columns:
            df = construct_features(df)
            df = clean_data(df)
    elif df is None and stock_code is None:
        # 全市数据训练（默认加载前200只，可通过外部参数调整）
        df = load_all_stock_data(max_stocks=200)
        if df is None:
            return None, None, None, None
    elif df is not None:
        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)
        else:
            print("⚠️ 传入的 DataFrame 没有 Date 列，无法保证时间顺序")
    else:
        return None, None, None, None

    if df is None or len(df) < 100:
        print("❌ 数据不足")
        return None, None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 训练使用设备: {device}")
    print(f"📊 总样本数: {len(df)}")

    # ==================== 特征标准化 ====================
    scaler_X = StandardScaler()
    scaled_features = scaler_X.fit_transform(df[FEATURE_COLS].values)
    price_targets = df['Target_Price'].values
    dir_targets = df['Target_Direction'].values

    # ==================== 构建序列 ====================
    X, y_price, y_dir = create_sequences(scaled_features, price_targets, dir_targets)

    # ==================== 时间分割 ====================
    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_price_train, y_price_test = y_price[:split], y_price[split:]
    y_dir_train, y_dir_test = y_dir[:split], y_dir[split:]
    print(f"📊 时间分割: 训练 {split} 条 ({train_ratio*100:.0f}%), 测试 {len(X)-split} 条 ({(1-train_ratio)*100:.0f}%)")

    # ==================== 标签标准化（仅使用训练集拟合） ====================
    scaler_y = StandardScaler()
    y_price_train_scaled = scaler_y.fit_transform(y_price_train.reshape(-1, 1)).ravel()
    y_price_test_scaled = scaler_y.transform(y_price_test.reshape(-1, 1)).ravel()

    # ==================== 转换为 Tensor（保留在 CPU） ====================
    X_train_t = torch.tensor(X_train, dtype=torch.float32)          # CPU
    X_test_t = torch.tensor(X_test, dtype=torch.float32)            # CPU
    y_price_train_t = torch.tensor(y_price_train_scaled, dtype=torch.float32).reshape(-1, 1)  # CPU
    y_price_test_t = torch.tensor(y_price_test_scaled, dtype=torch.float32).reshape(-1, 1)    # CPU
    y_dir_train_t = torch.tensor(y_dir_train, dtype=torch.long)     # CPU
    y_dir_test_t = torch.tensor(y_dir_test, dtype=torch.long)       # CPU

    print(f"✅ 训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

    # ==================== 创建 DataLoader（数据保持在 CPU，训练时再转移） ====================
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X_train_t, y_price_train_t, y_dir_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=2)    # pin_memory 有效加速 CPU->GPU 传输

    test_dataset = TensorDataset(X_test_t, y_price_test_t, y_dir_test_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=2)

    # ==================== 模型 ====================
    model = DualLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    # 混合精度（仅当 CUDA 可用时启用）
    from torch.amp import autocast, GradScaler
    if torch.cuda.is_available():
        scaler = GradScaler('cuda')
    else:
        scaler = None

    # ==================== 训练循环 ====================
    print("🚀 开始训练...")
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y_price, batch_y_dir in train_loader:
            # 将数据移至设备
            batch_X = batch_X.to(device)
            batch_y_price = batch_y_price.to(device)
            batch_y_dir = batch_y_dir.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast('cuda'):
                    price_pred, dir_pred = model(batch_X)
                    loss_reg = criterion_reg(price_pred, batch_y_price)
                    loss_cls = criterion_cls(dir_pred, batch_y_dir)
                    loss = loss_reg + loss_cls
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                price_pred, dir_pred = model(batch_X)
                loss_reg = criterion_reg(price_pred, batch_y_price)
                loss_cls = criterion_cls(dir_pred, batch_y_dir)
                loss = loss_reg + loss_cls
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ==================== 验证（分批进行） ====================
        model.eval()
        val_loss_total = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch_X, batch_y_price, _ in test_loader:
                batch_X = batch_X.to(device)
                batch_y_price = batch_y_price.to(device)
                if scaler is not None:
                    with autocast('cuda'):
                        val_price_pred, _ = model(batch_X)
                        val_loss_reg = criterion_reg(val_price_pred, batch_y_price)
                else:
                    val_price_pred, _ = model(batch_X)
                    val_loss_reg = criterion_reg(val_price_pred, batch_y_price)

                val_loss_total += val_loss_reg.item()
                val_steps += 1

        val_loss = val_loss_total / val_steps

        # 早停
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"轮次 [{epoch+1}/{epochs}] | 训练损失: {avg_train_loss:.4f} | 验证损失: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"⏹️ 早停于 epoch {epoch+1}")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ==================== 最终评估 MAE ====================
    model.eval()
    mae_total = 0.0
    mae_steps = 0

    with torch.no_grad():
        for batch_X, batch_y_price, _ in test_loader:
            batch_X = batch_X.to(device)
            batch_y_price = batch_y_price.to(device)
            if scaler is not None:
                with autocast('cuda'):
                    pred_price_scaled, _ = model(batch_X)
            else:
                pred_price_scaled, _ = model(batch_X)

            pred_price_real = scaler_y.inverse_transform(pred_price_scaled.cpu().numpy())
            mae_total += np.mean(np.abs(pred_price_real - batch_y_price.cpu().numpy()))
            mae_steps += 1

    mae = mae_total / mae_steps
    print(f"📈 测试集 MAE: {mae:.2f} 元")

    # ==================== 保存模型 ====================
    torch.save(model.cpu().state_dict(), 'model.pth')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("✅ 模型和Scaler已保存")

    model.to(device)  # 移回原设备
    return model, scaler_X, scaler_y, df
# =============================================
# 回测函数（单股票）
# =============================================
def run_backtest(stock_code, model, scaler_X, scaler_y, df, initial_capital=100000, benchmark_df=None):
    print(f"\n📊 开始回测: {stock_code}")
    if df is None:
        df = load_from_cache(stock_code)
        if df is None:
            return None
    if len(df) < 21:
        return None
    if 'Target_Price' not in df.columns:
        df = construct_features(df)
        df = clean_data(df)

    scaled_features = scaler_X.transform(df[FEATURE_COLS].values)
    X, y_price, y_dir = create_sequences(scaled_features, df['Target_Price'].values, df['Target_Direction'].values)
    if len(X) == 0:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dates = df['Date'].values[20:]
    close_prices = df['Close'].values[20:]

    positions = []
    capital = float(initial_capital)
    holdings = 0.0
    entry_price = 0.0
    trade_log = []

    with torch.no_grad():
        for i in range(len(X_tensor)):
            current_price = close_prices[i]
            x_sample = X_tensor[i].unsqueeze(0)
            _, dir_logits = model(x_sample)
            prob = torch.softmax(dir_logits, dim=1).squeeze().cpu().numpy()
            up_prob = prob[1]

            if holdings > 0:
                profit = (current_price - entry_price) / entry_price
                if profit >= TAKE_PROFIT:
                    capital = holdings * current_price
                    trade_log.append(('止盈', dates[i], current_price, profit))
                    holdings = 0
                    entry_price = 0
                    positions.append(0)
                    continue
                elif profit <= STOP_LOSS:
                    capital = holdings * current_price
                    trade_log.append(('止损', dates[i], current_price, profit))
                    holdings = 0
                    entry_price = 0
                    positions.append(0)
                    continue

            if holdings == 0 and up_prob > BUY_THRESHOLD:
                holdings = capital / current_price
                entry_price = current_price
                capital = 0.0
                trade_log.append(('买入', dates[i], current_price, None))
                positions.append(1)
            elif holdings > 0 and up_prob < SELL_THRESHOLD:
                capital = holdings * current_price
                trade_log.append(('卖出', dates[i], current_price, None))
                holdings = 0
                entry_price = 0
                positions.append(0)
            else:
                positions.append(1 if holdings > 0 else 0)

    if not positions:
        return None

    print(f"📊 共生成 {len(positions)} 个持仓日")
    print("\n📋 交易明细:")
    for action, date, price, pct in trade_log:
        date_str = str(date)[:10]
        if pct is not None:
            print(f"  {action}: {date_str} 价格: {price:.2f} 盈亏: {pct*100:.2f}%")
        else:
            print(f"  {action}: {date_str} 价格: {price:.2f}")

    backtest_df = pd.DataFrame({
        'Date': dates[:len(positions)],
        'Close': close_prices[:len(positions)].astype(float),
        'Position': positions,
        'Capital': float(initial_capital)
    }, dtype=float)
    
    backtest_df['Capital'] = initial_capital * (1 + backtest_df['Close'].pct_change().fillna(0) * backtest_df['Position'].shift(1).fillna(0)).cumprod()
    backtest_df['Capital'] = backtest_df['Capital'].astype(float)

    trades = backtest_df['Position'].diff().abs().sum() / 2
    print(f"\n📊 交易次数: {trades:.0f}")

    peak = backtest_df['Capital'].cummax()
    drawdown = (peak - backtest_df['Capital']) / peak
    max_drawdown = drawdown.max()
    print(f"📉 最大回撤: {max_drawdown*100:.2f}%")

    # ----- 夏普比率（添加安全保护） -----
    daily_ret = backtest_df['Capital'].pct_change().fillna(0)
    risk_free = 0.025
    excess = daily_ret - risk_free / 252
    std = excess.std()
    if std > 1e-8:
        sharpe = np.sqrt(252) * excess.mean() / std
    else:
        sharpe = 0
    print(f"📊 夏普比率 (年化): {sharpe:.3f}")

    if benchmark_df is None:
        try:
            benchmark_df = pd.read_csv(os.path.join(CACHE_DIR, "benchmark_300.csv"), parse_dates=['Date'])
        except:
            benchmark_df = None

    if benchmark_df is not None and not benchmark_df.empty:
        try:
            backtest_df['Date'] = pd.to_datetime(backtest_df['Date'])
            benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
            merged = pd.merge(backtest_df[['Date', 'Capital']], benchmark_df, on='Date', how='inner')
            if len(merged) > 0:
                init_price = merged['Close'].iloc[0]
                merged['Benchmark_Capital'] = initial_capital * (merged['Close'] / init_price)
                backtest_df = pd.merge(backtest_df, merged[['Date', 'Benchmark_Capital']], on='Date', how='left')
                backtest_df['Benchmark_Capital'] = backtest_df['Benchmark_Capital'].fillna(method='ffill')
                final_cap = backtest_df['Capital'].iloc[-1]
                final_bench = backtest_df['Benchmark_Capital'].iloc[-1]
                excess_return = (final_cap - final_bench) / initial_capital * 100
                print(f"📊 超额收益 (vs 沪深300): {excess_return:.2f}%")
        except:
            pass

    return backtest_df

# =============================================
# 扩展窗口测试函数
# =============================================
def expanding_window_test(stock_code, train_years=3, test_years=0.5, seeds=[42]):
    df = load_from_cache(stock_code)
    if df is None:
        print(f"❌ 未找到 {stock_code} 缓存")
        return None
    df = df.sort_values('Date').reset_index(drop=True)
    if 'Target_Price' not in df.columns:
        df = construct_features(df)
        df = clean_data(df)
    
    total_days = len(df)
    train_days = train_years * 252
    test_days = test_years * 252
    
    if total_days < train_days + test_days:
        print("❌ 数据不足，无法进行扩展窗口测试")
        return None
    
    results = []
    train_start = 0
    train_end = train_days
    test_end = train_days + test_days
    round_num = 1
    
    print(f"\n{'='*60}")
    print(f"🔬 扩展窗口测试: {stock_code}")
    print(f"初始训练: {train_years}年 ({train_days}天) → 测试: {test_years}年 ({test_days}天)")
    print(f"总交易日: {total_days}")
    print(f"{'='*60}")
    
    while test_end <= total_days:
        print(f"\n📍 第 {round_num} 轮")
        print(f"  训练区间: {df['Date'].iloc[train_start].strftime('%Y-%m-%d')} ~ {df['Date'].iloc[train_end-1].strftime('%Y-%m-%d')} ({len(df[train_start:train_end])} 天)")
        print(f"  测试区间: {df['Date'].iloc[train_end].strftime('%Y-%m-%d')} ~ {df['Date'].iloc[test_end-1].strftime('%Y-%m-%d')} ({len(df[train_end:test_end])} 天)")
        
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        
        seed = seeds[0]
        set_seed(seed)
        for f in ['model.pth', 'scaler_X.pkl', 'scaler_y.pkl']:
            if os.path.exists(f):
                os.remove(f)
        model, scaler_X, scaler_y, _ = train_and_save_model(df=train_df)
        if model is None:
            print(f"  训练失败，跳过")
            train_end = test_end
            test_end = train_end + test_days
            round_num += 1
            continue
        
        # 在测试集上回测
        backtest_df = run_backtest_on_df(test_df, model, scaler_X, scaler_y)
        if backtest_df is None:
            print(f"  回测失败，跳过")
        else:
            total_return = (backtest_df['Capital'].iloc[-1] - backtest_df['Capital'].iloc[0]) / backtest_df['Capital'].iloc[0]
            peak = backtest_df['Capital'].cummax()
            drawdown = (peak - backtest_df['Capital']) / peak
            max_dd = drawdown.max()
            # ----- 夏普比率（添加安全保护） -----
            daily_ret = backtest_df['Capital'].pct_change().fillna(0)
            excess = daily_ret - 0.025/252
            std = excess.std()
            if std > 1e-8:
                sharpe = np.sqrt(252) * excess.mean() / std
            else:
                sharpe = 0
            results.append({
                'round': round_num,
                'train_start': df['Date'].iloc[train_start],
                'train_end': df['Date'].iloc[train_end-1],
                'test_start': df['Date'].iloc[train_end],
                'test_end': df['Date'].iloc[test_end-1],
                'return': total_return,
                'max_drawdown': max_dd,
                'sharpe': sharpe
            })
            print(f"  测试集收益率: {total_return*100:.2f}%")
            print(f"  最大回撤: {max_dd*100:.2f}%")
            print(f"  夏普比率: {sharpe:.3f}")
        
        # 扩展窗口
        train_end = test_end
        test_end = train_end + test_days
        round_num += 1
    
    if not results:
        print("❌ 没有成功完成任何轮次")
        return None
    
    df_results = pd.DataFrame(results)
    avg_return = df_results['return'].mean()
    std_return = df_results['return'].std()
    avg_sharpe = df_results['sharpe'].mean()
    avg_dd = df_results['max_drawdown'].mean()
    
    print(f"\n{'='*60}")
    print(f"📊 扩展窗口测试汇总 ({len(results)} 轮)")
    print(f"  平均收益率: {avg_return*100:.2f}%")
    print(f"  收益率标准差: {std_return*100:.2f}%")
    print(f"  平均最大回撤: {avg_dd*100:.2f}%")
    print(f"  平均夏普比率: {avg_sharpe:.3f}")
    print(f"  正收益轮数: {(df_results['return'] > 0).sum()}/{len(results)}")
    print(f"{'='*60}")
    
    df_results.to_csv("expanding_window_results.csv", index=False)
    print("💾 详细结果已保存至 expanding_window_results.csv")
    
    return df_results

# =============================================
# 回测函数（直接使用传入的 DataFrame，含诊断）
# =============================================
def run_backtest_on_df(df, model, scaler_X, scaler_y, initial_capital=100000):
    if df is None or len(df) < 21:
        return None
    if 'Target_Price' not in df.columns:
        df = construct_features(df)
        df = clean_data(df)
    
    scaled_features = scaler_X.transform(df[FEATURE_COLS].values)
    X, y_price, y_dir = create_sequences(scaled_features, price_targets, dir_targets)
    if len(X) == 0:
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dates = df['Date'].values[20:]
    close_prices = df['Close'].values[20:]
    
    positions = []
    capital = float(initial_capital)
    holdings = 0.0
    entry_price = 0.0
    trade_log = []
    up_probs = []  # 用于诊断
    
    with torch.no_grad():
        for i in range(len(X_tensor)):
            current_price = close_prices[i]
            x_sample = X_tensor[i].unsqueeze(0)
            _, dir_logits = model(x_sample)
            prob = torch.softmax(dir_logits, dim=1).squeeze().cpu().numpy()
            up_prob = prob[1]
            up_probs.append(up_prob)
            
            if holdings > 0:
                profit = (current_price - entry_price) / entry_price
                if profit >= TAKE_PROFIT:
                    capital = holdings * current_price
                    trade_log.append(('止盈', dates[i], current_price, profit))
                    holdings = 0
                    entry_price = 0
                    positions.append(0)
                    continue
                elif profit <= STOP_LOSS:
                    capital = holdings * current_price
                    trade_log.append(('止损', dates[i], current_price, profit))
                    holdings = 0
                    entry_price = 0
                    positions.append(0)
                    continue
            
            if holdings == 0 and up_prob > BUY_THRESHOLD:
                holdings = capital / current_price
                entry_price = current_price
                capital = 0.0
                trade_log.append(('买入', dates[i], current_price, None))
                positions.append(1)
            elif holdings > 0 and up_prob < SELL_THRESHOLD:
                capital = holdings * current_price
                trade_log.append(('卖出', dates[i], current_price, None))
                holdings = 0
                entry_price = 0
                positions.append(0)
            else:
                positions.append(1 if holdings > 0 else 0)
    
    # ----- 诊断输出 -----
    if up_probs:
        print(f"📊 预测概率统计: 均值={np.mean(up_probs):.4f}, 最大值={np.max(up_probs):.4f}, 最小值={np.min(up_probs):.4f}")
        print(f"📊 超过买入阈值 {BUY_THRESHOLD} 的比例: {sum(1 for p in up_probs if p > BUY_THRESHOLD)/len(up_probs)*100:.2f}%")
    
    if not positions:
        print("⚠️ 没有产生任何交易信号，收益率为 0%")
        # 返回一个空但有效的 DataFrame，以便后续处理
        empty_df = pd.DataFrame({
            'Date': dates,
            'Close': close_prices,
            'Position': [0] * len(dates),
            'Capital': [initial_capital] * len(dates)
        })
        return empty_df
    
    backtest_df = pd.DataFrame({
        'Date': dates[:len(positions)],
        'Close': close_prices[:len(positions)].astype(float),
        'Position': positions,
        'Capital': float(initial_capital)
    }, dtype=float)
    
    backtest_df['Capital'] = initial_capital * (1 + backtest_df['Close'].pct_change().fillna(0) * backtest_df['Position'].shift(1).fillna(0)).cumprod()
    backtest_df['Capital'] = backtest_df['Capital'].astype(float)
    
    return backtest_df

# =============================================
# 多种子测试（支持全市数据）
# =============================================
def test_multi_seeds(stock_code=None, seeds=None, use_all_stocks=False, max_stocks=200):
    if seeds is None:
        seeds = generate_random_seeds(5)
        print(f"📌 自动生成种子: {seeds}")
    else:
        print(f"📌 使用种子列表: {seeds}")

    if use_all_stocks:
        print(f"📊 使用全市数据（前 {max_stocks} 只股票）...")
        df = load_all_stock_data(max_stocks=max_stocks)
    elif stock_code is not None:
        print(f"📊 使用单股票: {stock_code}")
        df = load_from_cache(stock_code)
        if df is None:
            print("❌ 缓存不存在，请先下载")
            return None
    else:
        print("❌ 请指定股票代码或启用全市数据")
        return None

    if df is None or len(df) < 100:
        print("❌ 数据不足")
        return None

    print(f"\n{'='*60}")
    print(f"🔬 多种子稳健性测试")
    print(f"{'='*60}")
    
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n📍 种子 [{i+1}/{len(seeds)}]: {seed}")
        set_seed(seed)
        for f in ['model.pth', 'scaler_X.pkl', 'scaler_y.pkl']:
            if os.path.exists(f):
                os.remove(f)
        
        model, scaler_X, scaler_y, _ = train_and_save_model(df=df)
        if model is None:
            print(f"❌ 种子 {seed} 训练失败")
            continue
        
        test_code = stock_code if stock_code is not None else "601515"
        backtest_df = run_backtest(test_code, model, scaler_X, scaler_y, None)
        if backtest_df is None:
            print(f"❌ 种子 {seed} 回测失败")
            continue
        
        total_return = (backtest_df['Capital'].iloc[-1] - backtest_df['Capital'].iloc[0]) / backtest_df['Capital'].iloc[0]
        results.append(total_return)
        print(f"  收益率: {total_return*100:.2f}%")
    
    if len(results) < 2:
        print("❌ 有效结果不足")
        return None
    
    results = np.array(results)
    avg = results.mean()
    std = results.std()
    
    print(f"\n{'='*60}")
    print(f"📊 多种子统计结果")
    print(f"  种子数: {len(results)}")
    print(f"  平均收益率: {avg*100:.2f}%")
    print(f"  标准差: {std*100:.2f}%")
    print(f"  最大值: {results.max()*100:.2f}%")
    print(f"  最小值: {results.min()*100:.2f}%")
    
    if std < 0.10:
        print(f"  稳健性: ✅ 优良 (标准差 < 10%)")
    elif std < 0.20:
        print(f"  稳健性: ⚠️ 一般 (标准差 10%~20%)")
    else:
        print(f"  稳健性: ❌ 较差 (标准差 > 20%)")
    print(f"{'='*60}")
    
    return results

# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("📈 A股量化回测系统 (扩展窗口测试版)")
    print("="*50)

    while True:
        print("\n请选择模式：")
        print("  1. 输入股票代码（如 601515）→ 单股票回测")
        print("  2. 输入 'exp' → 扩展窗口测试（输入股票代码）")
        print("  3. 输入 'all' → 全市数据训练 + 固定种子测试")
        print("  4. 输入 'all_random' → 全市数据训练 + 随机种子测试")
        print("  5. 输入 'test' → 单股票 + 固定种子测试")
        print("  6. 输入 'random' → 单股票 + 随机种子测试")
        
        user_input = input("\n请输入：").strip()
        if not user_input:
            continue
        
        if user_input.lower() == "exp":
            stock_code = input("请输入股票代码（如 601515）：").strip()
            code = stock_code.replace('sh.', '').replace('sz.', '').replace('.', '').strip()
            if code.isdigit():
                expanding_window_test(code, train_years=3, test_years=1, seeds=[42])
            else:
                print("❌ 代码格式错误")
            continue
        
        elif user_input.lower() == "all":
            test_multi_seeds(use_all_stocks=True, max_stocks=200, seeds=[42, 123, 2024, 999, 777])
            continue
        elif user_input.lower() == "all_random":
            test_multi_seeds(use_all_stocks=True, max_stocks=200, seeds=None)
            continue
        elif user_input.lower() == "test":
            test_multi_seeds(stock_code="601515", seeds=[42, 123, 2024, 999, 777])
            continue
        elif user_input.lower() == "random":
            test_multi_seeds(stock_code="601515", seeds=None)
            continue
        else:
            code = user_input.replace('sh.', '').replace('sz.', '').replace('.', '').strip()
            if code.isdigit():
                STOCK_CODE = code
                print(f"✅ 已识别股票代码：{STOCK_CODE}")
                break
            else:
                print("❌ 格式错误，请重新输入")

    # 单股票回测
    confirm = input(f"\n是否开始回测 {STOCK_CODE}？(y/n) ").strip().lower()
    if confirm != 'y':
        exit()

    print(f"\n🚀 开始处理股票 {STOCK_CODE} ...")
    for f in ['model.pth', 'scaler_X.pkl', 'scaler_y.pkl']:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑️ 已删除旧文件: {f}")

    model, scaler_X, scaler_y, df = train_and_save_model(stock_code=STOCK_CODE)

    if model is not None:
        backtest_df = run_backtest(STOCK_CODE, model, scaler_X, scaler_y, df)
        if backtest_df is not None:
            total_return = (backtest_df['Capital'].iloc[-1] - backtest_df['Capital'].iloc[0]) / backtest_df['Capital'].iloc[0] * 100
            print(f"\n📊 总收益率: {total_return:.2f}%")

            plt.figure(figsize=(12, 6))
            plt.plot(backtest_df['Date'], backtest_df['Capital'], label='策略资金曲线', linewidth=2)
            if 'Benchmark_Capital' in backtest_df.columns:
                plt.plot(backtest_df['Date'], backtest_df['Benchmark_Capital'], 
                        label='沪深300 (买入持有)', linestyle='--', alpha=0.7)
            plt.title(f'策略 vs 沪深300 资金曲线 ({STOCK_CODE})')
            plt.xlabel('日期')
            plt.ylabel('资金 (元)')
            plt.legend()
            plt.grid(True)
            plt.savefig('backtest_result.png')
            print("📊 资金曲线图已保存为 backtest_result.png")
            plt.show()
        else:
            print("❌ 回测失败")
    else:
        print("❌ 训练失败")