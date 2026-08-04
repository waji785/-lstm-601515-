# core/backtest_engine.py
import pandas as pd
import numpy as np
import torch
from config.settings import (
    FEATURE_COLS, BUY_THRESHOLD, SELL_THRESHOLD,
    STOP_LOSS, TAKE_PROFIT, MAX_POSITION, MIN_VOLATILITY,
    COMMISSION_RATE, MIN_COMMISSION, STAMP_DUTY_RATE, SLIPPAGE
)
from utils.common import create_sequences
from utils.logger import setup_logger

logger = setup_logger(__name__)

def calc_trade_cost(price, shares, is_buy, commission_rate=COMMISSION_RATE,
                    min_commission=MIN_COMMISSION, stamp_duty_rate=STAMP_DUTY_RATE,
                    slippage=SLIPPAGE):
    """
    计算交易成本（佣金、印花税、滑点）
    返回: (总成本, 实际成交价格)
    """
    if shares <= 0 or price <= 0:
        return 0, price

    if is_buy:
        exec_price = price * (1 + slippage)
    else:
        exec_price = price * (1 - slippage)

    turnover = exec_price * shares
    commission = max(turnover * commission_rate, min_commission)
    stamp_duty = turnover * stamp_duty_rate if not is_buy else 0.0
    total_cost = commission + stamp_duty
    return total_cost, exec_price

def run_backtest(df, model, scaler_X, scaler_y, initial_capital=100000,
                 return_log=False, start_date=None, end_date=None):
    """
    统一回测函数：使用已训练好的模型对指定股票数据进行回测
    支持交易成本，返回资金曲线和交易明细
    """
    if df is None or len(df) < 21:
        return None

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]
    if len(df) < 21:
        return None

    if 'Target_Price' not in df.columns:
        from core.features import construct_features, clean_data
        df = construct_features(df)
        df = clean_data(df)

    scaled = scaler_X.transform(df[FEATURE_COLS].values)
    X, _, _ = create_sequences(scaled, df['Target_Price'].values,
                               df['Target_Direction'].values, seq_len=20)
    if len(X) == 0:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if hasattr(model.lstm, 'flatten_parameters'):
         model.lstm.flatten_parameters()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dates = df['Date'].values[20:]
    close_prices = df['Close'].values[20:]

    ma_200 = df['Close'].rolling(200).mean().values[20:]
    atr = (df['High'].rolling(14).max() - df['Low'].rolling(14).min()).values[20:] / close_prices * 100

    capital = float(initial_capital)
    holdings = 0.0
    entry_price = 0.0
    entry_idx = 0
    trade_log = []
    positions = []

    with torch.no_grad():
        for i in range(len(X_tensor)):
            current_price = close_prices[i]
            x_sample = X_tensor[i].unsqueeze(0)
            _, dir_logits = model(x_sample)
            prob = torch.softmax(dir_logits, dim=1).squeeze().cpu().numpy()
            up_prob = prob[1]

            # 趋势过滤
            if i < len(ma_200) and ma_200[i] > 0:
                price_ma200_ratio = (current_price - ma_200[i]) / ma_200[i]
            else:
                price_ma200_ratio = 0
            ma5 = df['Close'].rolling(5).mean().values[20+i] if (20+i) < len(df) else current_price
            ma20 = df['Close'].rolling(20).mean().values[20+i] if (20+i) < len(df) else current_price
            trend_up = ma5 > ma20
            allow_long = (price_ma200_ratio > 0.05) or (trend_up and price_ma200_ratio > -0.02)
            vol_factor = min(atr[i] / MIN_VOLATILITY, 1.0) if i < len(atr) else 0.5

            total_asset = capital + holdings * current_price
            current_pos_ratio = (holdings * current_price) / total_asset if total_asset > 0 else 0

            # --- 卖出逻辑 ---
            if holdings > 0:
                profit = (current_price - entry_price) / entry_price if entry_price > 0 else 0
                hold_days = i - entry_idx

                # 跟踪止损
                if profit > 0.10:
                    trailing_stop = max(0.05, profit * 0.5)
                    if profit - (current_price / entry_price - 1) + 1 < trailing_stop:
                        cost, exec_price = calc_trade_cost(current_price, holdings, is_buy=False)
                        capital += holdings * exec_price - cost
                        trade_log.append(('跟踪止盈', dates[i], exec_price, profit))
                        holdings = 0
                        entry_price = 0
                        positions.append(0)
                        continue

                # 硬止损/止盈
                if profit >= TAKE_PROFIT:
                    cost, exec_price = calc_trade_cost(current_price, holdings, is_buy=False)
                    capital += holdings * exec_price - cost
                    trade_log.append(('止盈', dates[i], exec_price, profit))
                    holdings = 0
                    entry_price = 0
                    positions.append(0)
                    continue
                elif profit <= STOP_LOSS:
                    cost, exec_price = calc_trade_cost(current_price, holdings, is_buy=False)
                    capital += holdings * exec_price - cost
                    trade_log.append(('止损', dates[i], exec_price, profit))
                    holdings = 0
                    entry_price = 0
                    positions.append(0)
                    continue

                # 持有超过60天且盈利<5%，减仓一半
                if hold_days > 60 and profit < 0.05:
                    sell_shares = holdings * 0.5
                    cost, exec_price = calc_trade_cost(current_price, sell_shares, is_buy=False)
                    capital += sell_shares * exec_price - cost
                    holdings -= sell_shares
                    trade_log.append(('减仓', dates[i], exec_price, profit))
                    total_asset = capital + holdings * current_price
                    current_pos_ratio = (holdings * current_price) / total_asset if total_asset > 0 else 0

            # --- 买入逻辑 ---
            if holdings == 0 and allow_long and up_prob > BUY_THRESHOLD and vol_factor > 0.3:
                position_ratio = min((up_prob - 0.45) * 1.0, MAX_POSITION) * vol_factor
                position_ratio = max(position_ratio, 0.1)
                buy_amount = capital * position_ratio
                cost, exec_price = calc_trade_cost(current_price, 1, is_buy=True)  # 估算单位成本
                max_shares = int((buy_amount - cost) / exec_price) if exec_price > 0 else 0
                if max_shares > 0:
                    cost_actual, exec_price_actual = calc_trade_cost(current_price, max_shares, is_buy=True)
                    total_cost = max_shares * exec_price_actual + cost_actual
                    if total_cost <= capital:
                        holdings = max_shares
                        entry_price = exec_price_actual
                        entry_idx = i
                        capital -= total_cost
                        trade_log.append(('买入', dates[i], exec_price_actual, None, position_ratio))
                        total_asset = capital + holdings * current_price
                        current_pos_ratio = (holdings * current_price) / total_asset if total_asset > 0 else 0

            positions.append(current_pos_ratio)

    if not positions:
        return None

    # 强制长度一致
    target_len = len(positions)
    dates = dates[:target_len]
    close_prices = close_prices[:target_len]

    # 计算资金曲线
    close = np.array(close_prices)
    pos = np.array(positions)
    returns = np.diff(close) / close[:-1]
    # 确保 pos 与 returns 长度一致
    if len(pos) == len(returns) + 1:
        pos = pos[:-1]
    elif len(pos) < len(returns):
        returns = returns[:len(pos)]
    portfolio_returns = pos * returns
    capital_curve = [initial_capital]
    for r in portfolio_returns:
        capital_curve.append(capital_curve[-1] * (1 + r))
    while len(capital_curve) < target_len:
        capital_curve.append(capital_curve[-1])
    capital_curve = capital_curve[:target_len]

    # 断言长度一致
    assert len(dates) == len(close_prices) == len(positions) == len(capital_curve), \
        f"长度不一致: dates={len(dates)}, close={len(close_prices)}, positions={len(positions)}, capital={len(capital_curve)}"

    backtest_df = pd.DataFrame({
        'Date': dates,
        'Close': close_prices,
        'Position': positions,
        'Capital': capital_curve
    })

    if return_log:
        return backtest_df, trade_log
    else:
        return backtest_df