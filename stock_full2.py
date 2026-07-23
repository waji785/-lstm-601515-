import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import akshare as ak
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
import matplotlib.pyplot as plt
import os
warnings.filterwarnings('ignore')

# =============================================
# 全局策略参数（你可以在这里调参）
# =============================================
BUY_THRESHOLD = 0.52      # 上涨概率高于此值买入
SELL_THRESHOLD = 0.48     # 上涨概率低于此值卖出
STOP_LOSS = -0.10         # 亏损 10% 时强制止损
TAKE_PROFIT = 0.30        # 盈利 30% 时强制止盈

# =============================================
# 模型定义
# =============================================
class DualLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super().__init__()
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
# 数据获取与处理
# =============================================
def fetch_data_akshare(stock_code, start="2020-01-01", end="2026-07-20"):
    print(f"📊 正在从 akshare 获取 {stock_code} 数据...")
    code = stock_code.replace('.', '')
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq"
        )
        if df.empty:
            print("⚠️ 未获取到数据，请检查股票代码。")
            return None
        df.rename(columns={'日期': 'Date', '收盘': 'Close', '成交量': 'Volume'}, inplace=True)
        df = df[['Date', 'Close', 'Volume']].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.astype({'Close': float, 'Volume': float})
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        return None

    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Volatility'] = df['Close'].rolling(5).std()
    df['Return_1d'] = df['Close'].pct_change()
    df['Target_Price'] = df['Close'].shift(-1)
    df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    print(f"✅ 数据下载成功，共 {len(df)} 个有效交易日")
    return df

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
    df = fetch_data_akshare(stock_code)
    if df is None:
        return None, None, None

    feature_cols = ['Close', 'Volume', 'MA_5', 'MA_20', 'Volatility', 'Return_1d']
    scaler_X = StandardScaler()
    scaled_features = scaler_X.fit_transform(df[feature_cols].values)
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

    model = DualLSTM()
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
    return model, scaler_X, scaler_y

# =============================================
# 回测函数（已修复所有问题）
# =============================================
def run_backtest(stock_code, model, scaler_X, scaler_y, initial_capital=100000):
    print(f"\n📊 开始回测: {stock_code}")
    df = fetch_data_akshare(stock_code)
    if df is None:
        return None

    if len(df) < 21:
        print(f"❌ 数据量不足（仅 {len(df)} 条），无法进行回测。")
        return None

    feature_cols = ['Close', 'Volume', 'MA_5', 'MA_20', 'Volatility', 'Return_1d']
    scaled_features = scaler_X.transform(df[feature_cols].values)
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
    
    # ----- 关键修复：日期直接转字符串 -----
    print("\n📋 交易明细:")
    for action, date, price, pct in trade_log:
        date_str = str(date)[:10]  # 直接转换，不调用 strftime
        if pct is not None:
            print(f"  {action}: {date_str} 价格: {price:.2f} 盈亏: {pct*100:.2f}%")
        else:
            print(f"  {action}: {date_str} 价格: {price:.2f}")
    # ------------------------------------

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
    
    return backtest_df

# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    # ========== 交互模块 ==========
    print("\n" + "="*50)
    print("📈 欢迎使用 A股量化回测系统")
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
    # ================================

    # 清理旧文件
    for f in ['model.pth', 'scaler_X.pkl', 'scaler_y.pkl']:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑️ 已删除旧文件: {f}")

    print("🚀 开始全新训练...")
    model, scaler_X, scaler_y = train_and_save_model(STOCK_CODE)

    if model is not None:
        print("\n✅ 训练完成，开始回测...")
        backtest_df = run_backtest(STOCK_CODE, model, scaler_X, scaler_y)
        if backtest_df is not None:
            total_return = (backtest_df['Capital'].iloc[-1] - backtest_df['Capital'].iloc[0]) / backtest_df['Capital'].iloc[0]
            print(f"\n📊 总收益率: {total_return*100:.2f}%")
            
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
    STOCK_CODE = "601515"

    for f in ['model.pth', 'scaler_X.pkl', 'scaler_y.pkl']:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑️ 已删除旧文件: {f}")

    print("🚀 开始全新训练...")
    model, scaler_X, scaler_y = train_and_save_model(STOCK_CODE)

    if model is not None:
        print("\n✅ 训练完成，开始回测...")
        backtest_df = run_backtest(STOCK_CODE, model, scaler_X, scaler_y)
        if backtest_df is not None:
            total_return = (backtest_df['Capital'].iloc[-1] - backtest_df['Capital'].iloc[0]) / backtest_df['Capital'].iloc[0]
            print(f"\n📊 总收益率: {total_return*100:.2f}%")
            
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