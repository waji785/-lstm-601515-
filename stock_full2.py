import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import akshare as ak
import baostock as bs
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
import matplotlib.pyplot as plt
import os
import time
import random
import requests
warnings.filterwarnings('ignore')

# =============================================
# 解决中文显示问题
# =============================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =============================================
# 全局策略参数（你可以在这里调参）
# =============================================
BUY_THRESHOLD = 0.52      # 上涨概率高于此值买入
SELL_THRESHOLD = 0.48     # 上涨概率低于此值卖出
STOP_LOSS = -0.10         # 亏损 10% 时强制止损
TAKE_PROFIT = 0.30        # 盈利 30% 时强制止盈

# =============================================
# 特征列表（必须在模型定义之前定义）
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
        price_pred = self.reg_head(last_out)
        dir_pred = self.cls_head(last_out)
        return price_pred, dir_pred

# =============================================
# 技术指标计算函数
# =============================================
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20, num_std=2):
    middle = data.rolling(window).mean()
    std = data.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    position = (data - lower) / (upper - lower)
    width = (upper - lower) / middle
    return position, width

def calculate_volume_ratios(volume):
    volume_ma_5 = volume.rolling(5).mean()
    volume_ma_10 = volume.rolling(10).mean()
    volume_ratio = volume / volume_ma_5.replace(0, np.nan)
    return volume_ratio, volume_ma_5, volume_ma_10

# =============================================
# 代理辅助函数
# =============================================
def load_proxies(filename="available_proxies.txt"):
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        proxies = [line.strip() for line in f if line.strip()]
    valid_proxies = []
    for p in proxies:
        if not p.startswith(('http://', 'https://')):
            p = 'http://' + p
        valid_proxies.append(p)
    return valid_proxies

# =============================================
# 特征构造函数（复用）
# =============================================
def construct_features(df):
    """从原始 K 线数据构造所有技术指标特征"""
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    df['Momentum_5'] = close.pct_change(5)
    df['Momentum_10'] = close.pct_change(10)
    df['Momentum_20'] = close.pct_change(20)
    
    df['Return_1d'] = close.pct_change()
    df['Volatility_5'] = df['Return_1d'].rolling(5).std()
    df['Volatility_10'] = df['Return_1d'].rolling(10).std()
    df['Volatility_20'] = df['Return_1d'].rolling(20).std()
    df['Volatility_change'] = df['Volatility_5'].pct_change()
    
    df['MA_5'] = close.rolling(5).mean()
    df['MA_10'] = close.rolling(10).mean()
    df['MA_20'] = close.rolling(20).mean()
    df['MA_60'] = close.rolling(60).mean()
    
    df['Price_MA_5_Ratio'] = close / df['MA_5'] - 1
    df['Price_MA_20_Ratio'] = close / df['MA_20'] - 1
    df['Price_MA_60_Ratio'] = close / df['MA_60'] - 1
    df['MA_5_20_diff'] = df['MA_5'] - df['MA_20']
    df['MA_10_60_diff'] = df['MA_10'] - df['MA_60']
    
    df['RSI_14'] = calculate_rsi(close, 14)
    df['RSI_21'] = calculate_rsi(close, 21)
    df['RSI_7'] = calculate_rsi(close, 7)
    
    bb_position, bb_width = calculate_bollinger_bands(close, window=20, num_std=2)
    df['BB_position'] = bb_position
    df['BB_width'] = bb_width
    
    volume_ratio, volume_ma_5, volume_ma_10 = calculate_volume_ratios(volume)
    df['Volume_Ratio'] = volume_ratio
    df['Volume_MA_5'] = volume_ma_5
    df['Volume_MA_10'] = volume_ma_10
    df['Volume_MA_10_Ratio'] = volume / volume_ma_10.replace(0, np.nan)
    df['Volume_Price_Signal'] = (df['Return_1d'] > 0) & (df['Volume_Ratio'] > 1.2)
    df['Volume_Price_Signal'] = df['Volume_Price_Signal'].astype(int)
    
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
    
    df = df.dropna()
    return df

# =============================================
# baostock 数据获取（备选数据源）
# =============================================
def fetch_data_baostock(stock_code, start="2020-01-01", end="2026-07-20"):
    """使用 baostock 获取数据（国内稳定）"""
    print(f"📊 正在从 baostock 获取 {stock_code} 数据...")
    
    code = stock_code.replace('.', '').replace('sh', '').replace('sz', '')
    if len(code) < 6 and code.isdigit():
        code = code.zfill(6)
    if code.startswith(('0', '3')):
        bs_code = f"sz.{code}"
    else:
        bs_code = f"sh.{code}"
    
    try:
        lg = bs.login()
        if lg.error_code != '0':
            print(f"⚠️ baostock 登录失败: {lg.error_msg}")
            return None
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            fields="date,open,high,low,close,volume",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3"
        )
        
        if rs.error_code != '0':
            print(f"⚠️ baostock 查询失败: {rs.error_msg}")
            bs.logout()
            return None
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            if row is not None:
                data_list.append(row)
        
        bs.logout()
        
        if not data_list:
            print(f"⚠️ baostock 未获取到 {stock_code} 的数据")
            return None
        
        df = pd.DataFrame(data_list, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df = df.replace('', np.nan)
        df = df.dropna()
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        df = df.astype({
            'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"✅ baostock 获取成功，共 {len(df)} 个交易日")
        return df
        
    except Exception as e:
        print(f"❌ baostock 获取失败: {e}")
        return None

# =============================================
# 处理 akshare 原始数据
# =============================================
def _process_akshare_data(df):
    """处理 akshare 返回的原始数据"""
    df.rename(columns={
        '日期': 'Date', '开盘': 'Open', '收盘': 'Close',
        '最高': 'High', '最低': 'Low', '成交量': 'Volume'
    }, inplace=True)
    
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    return construct_features(df)

# =============================================
# 四级降级数据获取函数（核心）
# =============================================
def fetch_data_with_fallback(stock_code, start="2020-01-01", end="2026-07-20"):
    """
    数据获取顺序：
    1. akshare 直连
    2. baostock 直连
    3. 读取代理文件，用代理重试 akshare
    4. 全部失败则退出
    """
    print(f"\n🔍 开始获取 {stock_code} 数据...")
    
    code = stock_code.replace('.', '').replace('sh', '').replace('sz', '')
    if len(code) < 6 and code.isdigit():
        code = code.zfill(6)
    
    # ========== 第一步：akshare 直连 ==========
    print("\n📡 [1/4] 尝试 akshare 直连...")
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq"
        )
        if not df.empty:
            df = _process_akshare_data(df)
            print(f"✅ akshare 直连成功，共 {len(df)} 个交易日")
            return df
        else:
            print("⚠️ akshare 直连返回空数据")
    except Exception as e:
        print(f"⚠️ akshare 直连失败: {e}")
    
    # ========== 第二步：baostock 直连 ==========
    print("\n📡 [2/4] 尝试 baostock 直连...")
    df_raw = fetch_data_baostock(stock_code, start, end)
    if df_raw is not None:
        df = construct_features(df_raw)
        print(f"✅ baostock 直连成功，共 {len(df)} 个交易日")
        return df
    else:
        print("⚠️ baostock 直连失败")
    
    # ========== 第三步：读取代理文件 ==========
    print("\n📡 [3/4] 读取代理文件...")
    proxies_list = load_proxies()
    if not proxies_list:
        print("❌ 未找到代理文件或代理列表为空，放弃重试")
        return None
    
    print(f"✅ 加载到 {len(proxies_list)} 个代理，开始重试 akshare...")
    
    # ========== 第四步：用代理重试 akshare ==========
    for attempt, proxy in enumerate(proxies_list, 1):
        print(f"\n🔀 [4/4] akshare 代理尝试 {attempt}/{len(proxies_list)}: {proxy}")
        try:
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
            os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
            
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq"
            )
            
            if not df.empty:
                df = _process_akshare_data(df)
                print(f"✅ 代理 {proxy} 成功，共 {len(df)} 个交易日")
                return df
            else:
                print(f"⚠️ 代理 {proxy} 返回空数据")
        except Exception as e:
            print(f"⚠️ 代理 {proxy} 失败: {e}")
        
        time.sleep(1)
    
    print("\n❌ 所有数据源尝试均失败（akshare直连、baostock直连、代理重试全部失败）")
    return None

# =============================================
# 序列生成函数
# =============================================
def create_sequences(features, price_targets, dir_targets, seq_len=20):
    X, y_price, y_dir = [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        y_price.append(price_targets[i])
        y_dir.append(dir_targets[i])
    return np.array(X, dtype=np.float32), np.array(y_price, dtype=np.float32), np.array(y_dir, dtype=np.float32)

# =============================================
# 训练函数
# =============================================
def train_and_save_model(stock_code):
    df = fetch_data_with_fallback(stock_code)
    if df is None:
        return None, None, None, None

    scaler_X = StandardScaler()
    scaled_features = scaler_X.fit_transform(df[FEATURE_COLS].values)
    price_targets = df['Target_Price'].values
    dir_targets = df['Target_Direction'].values

    X, y_price, y_dir = create_sequences(scaled_features, price_targets, dir_targets, seq_len=20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_price_train, y_price_test = y_price[:split_idx], y_price[split_idx:]
    y_dir_train, y_dir_test = y_dir[:split_idx], y_dir[split_idx:]

    scaler_y = StandardScaler()
    y_price_train_scaled = scaler_y.fit_transform(y_price_train.reshape(-1, 1)).ravel()
    y_price_test_scaled = scaler_y.transform(y_price_test.reshape(-1, 1)).ravel()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_price_train_t = torch.tensor(y_price_train_scaled, dtype=torch.float32).reshape(-1, 1)
    y_price_test_t = torch.tensor(y_price_test_scaled, dtype=torch.float32).reshape(-1, 1)
    y_dir_train_t = torch.tensor(y_dir_train, dtype=torch.long)
    y_dir_test_t = torch.tensor(y_dir_test, dtype=torch.long)

    print(f"✅ 训练样本: {len(X_train)}，测试样本: {len(X_test)}")

    model = DualLSTM(input_size=len(FEATURE_COLS))
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 开始训练...")
    for epoch in range(50):
        price_pred, dir_pred = model(X_train_t)
        loss_reg = criterion_reg(price_pred, y_price_train_t)
        loss_cls = criterion_cls(dir_pred, y_dir_train_t)
        loss = loss_reg + loss_cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"轮次 [{epoch+1}/50] | 总损失: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_price_scaled, _ = model(X_test_t)
        pred_price_real = scaler_y.inverse_transform(pred_price_scaled.numpy())
        mae = np.mean(np.abs(pred_price_real - y_price_test))
        print(f"📈 测试集 MAE: {mae:.2f} 元")

    torch.save(model.state_dict(), 'model.pth')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("✅ 模型和Scaler已保存")
    
    return model, scaler_X, scaler_y, df

# =============================================
# 回测函数
# =============================================
def run_backtest(stock_code, model, scaler_X, scaler_y, df, initial_capital=100000):
    print(f"\n📊 开始回测: {stock_code}")
    
    if df is None:
        cache_file = f"stock_data_cache/{stock_code}.csv"
        if os.path.exists(cache_file):
            print(f"📂 从缓存读取 {stock_code} 数据...")
            df = pd.read_csv(cache_file, parse_dates=['Date'])
        else:
            print("❌ 未提供数据且缓存不存在")
            return None

    if len(df) < 21:
        print(f"❌ 数据量不足（仅 {len(df)} 条），无法进行回测。")
        return None

    scaled_features = scaler_X.transform(df[FEATURE_COLS].values)
    X, y_price, y_dir = create_sequences(scaled_features, df['Target_Price'].values, df['Target_Direction'].values, seq_len=20)
    
    if len(X) == 0:
        print("❌ 序列生成失败，没有生成任何样本。")
        return None
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dates = df['Date'].values[20:]
    close_prices = df['Close'].values[20:]

    positions = []
    capital = float(initial_capital)
    holdings = 0.0
    entry_price = 0.0

    BUY = BUY_THRESHOLD
    SELL = SELL_THRESHOLD
    SL = STOP_LOSS
    TP = TAKE_PROFIT

    model.eval()
    print("🚀 开始生成交易信号（含止盈止损）...")
    trade_log = []

    with torch.no_grad():
        for i in range(len(X_tensor)):
            current_price = close_prices[i]
            x_sample = X_tensor[i].unsqueeze(0)
            
            _, dir_logits = model(x_sample)
            prob = torch.softmax(dir_logits, dim=1).squeeze().numpy()
            up_prob = prob[1]

            if holdings > 0:
                profit_pct = (current_price - entry_price) / entry_price
                if profit_pct >= TP:
                    capital = holdings * current_price
                    trade_log.append(('止盈', dates[i], current_price, profit_pct))
                    holdings = 0.0
                    entry_price = 0.0
                    positions.append(0)
                    continue
                elif profit_pct <= SL:
                    capital = holdings * current_price
                    trade_log.append(('止损', dates[i], current_price, profit_pct))
                    holdings = 0.0
                    entry_price = 0.0
                    positions.append(0)
                    continue

            if holdings == 0 and up_prob > BUY:
                holdings = capital / current_price
                entry_price = current_price
                capital = 0.0
                trade_log.append(('买入', dates[i], current_price, None))
                positions.append(1)
            elif holdings > 0 and up_prob < SELL:
                capital = holdings * current_price
                trade_log.append(('卖出', dates[i], current_price, None))
                holdings = 0.0
                entry_price = 0.0
                positions.append(0)
            else:
                positions.append(1 if holdings > 0 else 0)

    if not positions:
        print("❌ 没有生成任何交易信号，回测失败。")
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
    
    backtest_df['Close'] = backtest_df['Close'].astype(float)
    backtest_df['Position'] = backtest_df['Position'].astype(float)
    backtest_df['Capital'] = backtest_df['Capital'].astype(float)
    
    backtest_df['Daily_Return'] = backtest_df['Close'].pct_change().fillna(0)
    backtest_df['Position_shifted'] = backtest_df['Position'].shift(1).fillna(0)
    backtest_df['Strategy_Return'] = backtest_df['Daily_Return'] * backtest_df['Position_shifted']
    backtest_df['Capital'] = initial_capital * (1 + backtest_df['Strategy_Return']).cumprod()
    backtest_df['Capital'] = backtest_df['Capital'].astype(float)
    
    trades = backtest_df['Position'].diff().abs().sum() / 2
    print(f"\n📊 交易次数: {trades:.0f}")
    
    # 计算最大回撤
    peak = backtest_df['Capital'].cummax()
    drawdown = (peak - backtest_df['Capital']) / peak
    max_drawdown = drawdown.max()
    print(f"📉 最大回撤: {max_drawdown*100:.2f}%")
    
    return backtest_df

# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("📈 欢迎使用 A股量化回测系统（双数据源版）")
    print("="*50)
    
    while True:
        user_input = input("\n请输入股票代码（如 601515 或 sh.601515）：").strip()
        if not user_input:
            print("❌ 输入不能为空，请重新输入。")
            continue
        code = user_input.replace('sh.', '').replace('sz.', '').replace('.', '').strip()
        if code.isdigit():
            STOCK_CODE = code
            print(f"✅ 已识别股票代码：{STOCK_CODE}")
            break
        else:
            print("❌ 输入格式有误，请重新输入（仅支持数字或 'sh.' 前缀）。")
    
    confirm = input(f"\n是否开始回测 {STOCK_CODE}？(y/n) ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消操作。")
        exit()
    
    print(f"\n🚀 开始处理股票 {STOCK_CODE} ...")

    for f in ['model.pth', 'scaler_X.pkl', 'scaler_y.pkl']:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑️ 已删除旧文件: {f}")

    print("🚀 开始全新训练...")
    model, scaler_X, scaler_y, df = train_and_save_model(STOCK_CODE)

    if model is not None:
        print("\n✅ 训练完成，开始回测...")
        backtest_df = run_backtest(STOCK_CODE, model, scaler_X, scaler_y, df)
        if backtest_df is not None:
            total_return = (backtest_df['Capital'].iloc[-1] - backtest_df['Capital'].iloc[0]) / backtest_df['Capital'].iloc[0]
            print(f"\n📊 总收益率: {total_return*100:.2f}%")
            
            # 确保日期列是 datetime 类型
            backtest_df['Date'] = pd.to_datetime(backtest_df['Date'])
            
            plt.figure(figsize=(10, 5))
            plt.plot(backtest_df['Date'], backtest_df['Capital'])
            plt.title(f'资金曲线 ({STOCK_CODE})')
            plt.xlabel('日期')
            plt.ylabel('资金 (元)')
            plt.grid(True)
            plt.savefig('backtest_result.png')
            print("📊 资金曲线图已保存为 backtest_result.png")
            plt.show()
        else:
            print("❌ 回测失败，请检查回测函数。")
    else:
        print("❌ 训练失败，请检查网络或股票代码。")