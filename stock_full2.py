import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import baostock as bs
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
import matplotlib.pyplot as plt
import os
import time
warnings.filterwarnings('ignore')

# =============================================
# 解决中文显示问题
# =============================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =============================================
# 全局策略参数
# =============================================
BUY_THRESHOLD = 0.52
SELL_THRESHOLD = 0.48
STOP_LOSS = -0.10
TAKE_PROFIT = 0.30

# =============================================
# 数据缓存配置
# =============================================
CACHE_DIR = "stock_data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================
# 特征列表
# =============================================
FEATURE_COLS = [
    'Close', 'Volume',
    'Momentum_5', 'Momentum_10', 'Momentum_20',
    'Return_1d', 'Volatility_5', 'Volatility_10', 'Volatility_20', 'Volatility_change',
    'MA_5', 'MA_10', 'MA_20', 'MA_60',
    'Price_MA_5_Ratio', 'Price_MA_20_Ratio', 'Price_MA_60_Ratio',
    'MA_5_20_diff', 'MA_10_60_diff',
    'RSI_7', 'RSI_14', 'RSI_21',
    'BB_position', 'BB_width',
    'Volume_Ratio', 'Volume_MA_5', 'Volume_MA_10', 'Volume_MA_10_Ratio',
    'Volume_Price_Signal',
    'High_Low_Ratio', 'Close_Open_Ratio',
    'Upper_Shadow', 'Lower_Shadow',
    'Price_Position', 'New_High', 'New_Low'
]

# =============================================
# 模型定义
# =============================================
class DualLSTM(nn.Module):
    def __init__(self, input_size=None, hidden_size=64, num_layers=2):
        super().__init__()
        if input_size is None:
            input_size = len(FEATURE_COLS)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.reg_head(last_out), self.cls_head(last_out)

# =============================================
# 技术指标函数
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

# =============================================
# 特征构造
# =============================================
def construct_features(df):
    df = df.copy()
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']

    df['Momentum_5'] = close.pct_change(5)
    df['Momentum_10'] = close.pct_change(10)
    df['Momentum_20'] = close.pct_change(20)

    df['Return_1d'] = close.pct_change()
    df['Volatility_5'] = df['Return_1d'].rolling(5).std()
    df['Volatility_10'] = df['Return_1d'].rolling(10).std()
    df['Volatility_20'] = df['Return_1d'].rolling(20).std()
    df['Volatility_change'] = df['Volatility_5'].pct_change()

    for w in [5, 10, 20, 60]:
        df[f'MA_{w}'] = close.rolling(w).mean()
    df['Price_MA_5_Ratio'] = close / df['MA_5'] - 1
    df['Price_MA_20_Ratio'] = close / df['MA_20'] - 1
    df['Price_MA_60_Ratio'] = close / df['MA_60'] - 1
    df['MA_5_20_diff'] = df['MA_5'] - df['MA_20']
    df['MA_10_60_diff'] = df['MA_10'] - df['MA_60']

    df['RSI_7'] = calculate_rsi(close, 7)
    df['RSI_14'] = calculate_rsi(close, 14)
    df['RSI_21'] = calculate_rsi(close, 21)

    bb_pos, bb_width = calculate_bollinger_bands(close)
    df['BB_position'] = bb_pos
    df['BB_width'] = bb_width

    vol_ratio, vol_ma5, vol_ma10 = calculate_volume_ratios(volume)
    df['Volume_Ratio'] = vol_ratio
    df['Volume_MA_5'] = vol_ma5
    df['Volume_MA_10'] = vol_ma10
    df['Volume_MA_10_Ratio'] = volume / vol_ma10.replace(0, np.nan)
    df['Volume_Price_Signal'] = ((df['Return_1d'] > 0) & (df['Volume_Ratio'] > 1.2)).astype(int)

    df['High_Low_Ratio'] = (high - low) / close
    df['Close_Open_Ratio'] = close / df['Open'] - 1
    df['Upper_Shadow'] = (high - close) / (high - low + 0.001)
    df['Lower_Shadow'] = (close - low) / (high - low + 0.001)

    df['Highest_20'] = close.rolling(20).max()
    df['Lowest_20'] = close.rolling(20).min()
    df['Price_Position'] = (close - df['Lowest_20']) / (df['Highest_20'] - df['Lowest_20'] + 0.001)
    df['New_High'] = (close == df['Highest_20']).astype(int)
    df['New_Low'] = (close == df['Lowest_20']).astype(int)

    df['Target_Price'] = close.shift(-1)
    df['Target_Direction'] = (close.shift(-1) > close).astype(int)
    return df.dropna()

# =============================================
# 数据缓存函数（增量更新核心）
# =============================================
def get_cache_path(stock_code):
    return os.path.join(CACHE_DIR, f"{stock_code}.csv")

def load_from_cache(stock_code):
    cache_file = get_cache_path(stock_code)
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=['Date'])
            if not df.empty:
                print(f"📂 从缓存加载 {stock_code} 数据，共 {len(df)} 个交易日")
                return df
        except Exception as e:
            print(f"⚠️ 缓存读取失败: {e}")
    return None

def save_to_cache(stock_code, df):
    try:
        cache_file = get_cache_path(stock_code)
        df.to_csv(cache_file, index=False)
        print(f"💾 数据已保存至缓存: {cache_file}")
    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")

def update_cache_incremental(stock_code, start_date=None):
    """增量更新缓存：读取现有缓存的最后日期，下载新数据并追加"""
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

# =============================================
# baostock 数据获取（修复连接泄露版）
# =============================================
def fetch_data_baostock(stock_code, start="2020-01-01", end="2026-07-20", retries=5):
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

# =============================================
# 数据获取（主入口：优先缓存，增量更新）
# =============================================
def fetch_data_with_fallback(stock_code, start="2020-01-01", end="2026-07-20"):
    print(f"\n🔍 开始获取 {stock_code} 数据...")
    code = stock_code.replace('.', '').replace('sh', '').replace('sz', '')
    if len(code) < 6 and code.isdigit():
        code = code.zfill(6)
    
    cached_df = load_from_cache(code)
    if cached_df is not None:
        required_cols = set(FEATURE_COLS + ['Date', 'Target_Price', 'Target_Direction'])
        if required_cols.issubset(set(cached_df.columns)):
            print(f"✅ 缓存数据完整，共 {len(cached_df)} 个交易日")
            return cached_df
        else:
            print("⚠️ 缓存数据不完整，重新下载...")
    
    df = update_cache_incremental(code, start_date=start)
    if df is not None:
        df = construct_features(df)
        return df
    else:
        for attempt in range(3):
            print(f"📡 尝试全量下载 baostock... (第 {attempt+1} 次)")
            df_raw = fetch_data_baostock(code, start, end)
            if df_raw is not None:
                df = construct_features(df_raw)
                save_to_cache(code, df)
                print(f"✅ 全量下载成功，共 {len(df)} 个交易日")
                return df
            else:
                if attempt < 2:
                    time.sleep(3)
        print("❌ 所有数据获取方式均失败")
        return None

# =============================================
# 沪深300数据获取（带缓存）
# =============================================
def fetch_benchmark_data(start="2020-01-01", end="2026-07-20"):
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
# 训练函数（GPU支持）
# =============================================
def train_and_save_model(stock_code):
    df = fetch_data_with_fallback(stock_code)
    if df is None:
        return None, None, None, None

    # ----- 设备检测 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 训练使用设备: {device}")

    scaler_X = StandardScaler()
    scaled_features = scaler_X.fit_transform(df[FEATURE_COLS].values)
    price_targets = df['Target_Price'].values
    dir_targets = df['Target_Direction'].values

    X, y_price, y_dir = create_sequences(scaled_features, price_targets, dir_targets)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_price_train, y_price_test = y_price[:split], y_price[split:]
    y_dir_train, y_dir_test = y_dir[:split], y_dir[split:]

    scaler_y = StandardScaler()
    y_price_train_scaled = scaler_y.fit_transform(y_price_train.reshape(-1, 1)).ravel()
    y_price_test_scaled = scaler_y.transform(y_price_test.reshape(-1, 1)).ravel()

    # ----- 张量迁移到设备 -----
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_price_train_t = torch.tensor(y_price_train_scaled, dtype=torch.float32).reshape(-1, 1).to(device)
    y_price_test_t = torch.tensor(y_price_test_scaled, dtype=torch.float32).reshape(-1, 1).to(device)
    y_dir_train_t = torch.tensor(y_dir_train, dtype=torch.long).to(device)
    y_dir_test_t = torch.tensor(y_dir_test, dtype=torch.long).to(device)

    print(f"✅ 训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

    model = DualLSTM(input_size=len(FEATURE_COLS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    print("🚀 开始训练...")
    for epoch in range(50):
        price_pred, dir_pred = model(X_train_t)
        loss_reg = criterion_reg(price_pred, y_price_train_t)
        loss_cls = criterion_cls(dir_pred, y_dir_train_t)
        loss = loss_reg + loss_cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"轮次 [{epoch+1}/50] | 总损失: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_price_scaled, _ = model(X_test_t)
        # 将结果移回 CPU 以便 numpy 处理
        pred_price_real = scaler_y.inverse_transform(pred_price_scaled.cpu().numpy())
        mae = np.mean(np.abs(pred_price_real - y_price_test))
        print(f"📈 测试集 MAE: {mae:.2f} 元")

    # 保存模型和 Scaler（模型参数在 CPU 上保存）
    torch.save(model.cpu().state_dict(), 'model.pth')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("✅ 模型和Scaler已保存")
    return model, scaler_X, scaler_y, df

# =============================================
# 回测函数（接收 benchmark_df）
# =============================================
def run_backtest(stock_code, model, scaler_X, scaler_y, df, initial_capital=100000, benchmark_df=None):
    print(f"\n📊 开始回测: {stock_code}")
    if df is None:
        cache_file = f"stock_data_cache/{stock_code}.csv"
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, parse_dates=['Date'])
        else:
            return None

    if len(df) < 21:
        return None

    scaled_features = scaler_X.transform(df[FEATURE_COLS].values)
    X, y_price, y_dir = create_sequences(scaled_features, df['Target_Price'].values, df['Target_Direction'].values)
    if len(X) == 0:
        return None

    # 回测推理时可以使用 CPU（减少显存占用）
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

    print("🚀 开始生成交易信号...")
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

    # 最大回撤
    peak = backtest_df['Capital'].cummax()
    drawdown = (peak - backtest_df['Capital']) / peak
    max_drawdown = drawdown.max()
    print(f"📉 最大回撤: {max_drawdown*100:.2f}%")

    # 夏普比率
    daily_ret = backtest_df['Capital'].pct_change().fillna(0)
    risk_free = 0.025
    excess = daily_ret - risk_free / 252
    std = excess.std()
    
    if std > 1e-8:
        sharpe = np.sqrt(252) * excess.mean() / std
    else:
        sharpe = 0
    print(f"📊 夏普比率 (年化): {sharpe:.3f}")

    # 沪深300对比
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
            else:
                backtest_df['Benchmark_Capital'] = initial_capital
        except Exception as e:
            print(f"⚠️ 沪深300对比失败（不影响回测结果）: {e}")
            backtest_df['Benchmark_Capital'] = initial_capital
    else:
        backtest_df['Benchmark_Capital'] = initial_capital

    return backtest_df

# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("📈 A股量化回测系统 (GPU支持 + 增量缓存)")
    print("="*50)

    while True:
        user_input = input("\n请输入股票代码（如 601515 或 sh.601515）：").strip()
        if not user_input:
            continue
        code = user_input.replace('sh.', '').replace('sz.', '').replace('.', '').strip()
        if code.isdigit():
            STOCK_CODE = code
            print(f"✅ 已识别股票代码：{STOCK_CODE}")
            break
        else:
            print("❌ 格式错误，请重新输入")

    confirm = input(f"\n是否开始回测 {STOCK_CODE}？(y/n) ").strip().lower()
    if confirm != 'y':
        exit()

    print(f"\n🚀 开始处理股票 {STOCK_CODE} ...")
    for f in ['model.pth', 'scaler_X.pkl', 'scaler_y.pkl']:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑️ 已删除旧文件: {f}")

    model, scaler_X, scaler_y, df = train_and_save_model(STOCK_CODE)

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