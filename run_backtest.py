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

    # 使用全局策略参数
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

            # 止盈止损检查
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

            # 买卖信号
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
    
    # 打印交易明细（修复日期格式）
    print("\n📋 交易明细:")
    for action, date, price, pct in trade_log:
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        if pct is not None:
            print(f"  {action}: {date_str} 价格: {price:.2f} 盈亏: {pct*100:.2f}%")
        else:
            print(f"  {action}: {date_str} 价格: {price:.2f}")

    # 构建回测DataFrame
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