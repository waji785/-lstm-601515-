import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# 从主程序导入必要函数
from stock_full2 import (
    load_from_cache,
    train_and_save_model,
    run_trend_backtest,
    construct_features,
    clean_data,
    FEATURE_COLS,
    set_seed
)

# =============================================
# 配置
# =============================================
WHITELIST_INPUT = "whitelist.csv"           
WHITELIST_OUTPUT = "whitelist_extended.csv" 

TARGET_STOCKS = 30          
MAX_WORKERS = 4             
MIN_TRADING_DAYS = 200      
TRAIN_RATIO = 0.7           

# 筛选条件（更宽松）
MIN_RETURN = 0.15           
MIN_TRADES = 2              
MAX_DRAWDOWN = 0.65         

# ----- 训练参数（损失保护）-----
EPOCHS = 50                 # 从100减少到50，减少过拟合风险
PATIENCE = 10               # 早停耐心值
LOSS_CLIP = 100.0           # 损失裁剪阈值（防止爆炸）
REGRESSION_WEIGHT = 0.1     # 回归损失权重（原为1.0，降低到0.1）

# =============================================
# 单只股票回测（严格时间分割 + 损失保护）
# =============================================
def backtest_stock(code, name=""):
    """
    对单只股票进行时间分割回测，含损失保护
    """
    try:
        # 加载数据
        df = load_from_cache(code)
        if df is None or len(df) < MIN_TRADING_DAYS:
            return None

        # 按日期排序
        df = df.sort_values('Date').reset_index(drop=True)

        # 时间分割
        split_idx = int(len(df) * TRAIN_RATIO)
        if split_idx < 100 or len(df) - split_idx < 50:
            return None

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # 确保特征完整
        if 'Target_Price' not in train_df.columns:
            train_df = construct_features(train_df)
            train_df = clean_data(train_df)
        if 'Target_Price' not in test_df.columns:
            test_df = construct_features(test_df)
            test_df = clean_data(test_df)

        if len(train_df) < 100 or len(test_df) < 20:
            return None

        # ----- 使用修改后的训练函数（带损失保护）-----
        model, scaler_X, scaler_y, _ = train_and_save_model_with_protection(
            df=train_df, 
            epochs=EPOCHS, 
            patience=PATIENCE,
            loss_clip=LOSS_CLIP,
            regression_weight=REGRESSION_WEIGHT
        )
        if model is None:
            return None

        # 在测试集上回测
        backtest_df = run_trend_backtest(test_df, model, scaler_X, scaler_y)
        if backtest_df is None or len(backtest_df) < 20:
            return None

        # 计算指标
        total_return = (backtest_df['Capital'].iloc[-1] - backtest_df['Capital'].iloc[0]) / backtest_df['Capital'].iloc[0]

        # 最大回撤
        peak = backtest_df['Capital'].cummax()
        drawdown = (peak - backtest_df['Capital']) / peak
        max_drawdown = drawdown.max()

        # 夏普比率
        daily_ret = backtest_df['Capital'].pct_change().fillna(0)
        excess = daily_ret - 0.025 / 252
        std = excess.std()
        sharpe = np.sqrt(252) * excess.mean() / std if std > 1e-8 else 0

        # 交易次数
        if 'Position_float' in backtest_df.columns:
            trades = (backtest_df['Position_float'].diff().abs() > 0.01).sum() / 2
        else:
            trades = (backtest_df['Position'].diff().abs() > 0.01).sum() / 2

        return {
            'code': code,
            'name': name,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'trade_count': trades,
            'capital': backtest_df['Capital'].values
        }
    except Exception as e:
        # 静默跳过
        return None

# =============================================
# 带损失保护的训练函数（专门用于扩展白名单）
# =============================================
def train_and_save_model_with_protection(df, epochs=50, patience=10, loss_clip=100.0, regression_weight=0.1):
    """
    带损失保护的训练函数
    1. 降低回归损失权重（减少对价格预测的依赖）
    2. 梯度裁剪（防止损失爆炸）
    3. 更激进的早停
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import TensorDataset, DataLoader
    import joblib
    
    if df is None or len(df) < 100:
        return None, None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 标准化
    scaler_X = StandardScaler()
    scaled_features = scaler_X.fit_transform(df[FEATURE_COLS].values)
    price_targets = df['Target_Price'].values
    dir_targets = df['Target_Direction'].values

    # 构建序列
    X, y_price, y_dir = create_sequences(scaled_features, price_targets, dir_targets)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_price_train, y_price_test = y_price[:split], y_price[split:]
    y_dir_train, y_dir_test = y_dir[:split], y_dir[split:]

    # 标签标准化
    scaler_y = StandardScaler()
    y_price_train_scaled = scaler_y.fit_transform(y_price_train.reshape(-1, 1)).ravel()
    y_price_test_scaled = scaler_y.transform(y_price_test.reshape(-1, 1)).ravel()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_price_train_t = torch.tensor(y_price_train_scaled, dtype=torch.float32).reshape(-1, 1).to(device)
    y_price_test_t = torch.tensor(y_price_test_scaled, dtype=torch.float32).reshape(-1, 1).to(device)
    y_dir_train_t = torch.tensor(y_dir_train, dtype=torch.long).to(device)
    y_dir_test_t = torch.tensor(y_dir_test, dtype=torch.long).to(device)

    # 模型
    from stock_full2 import DualLSTM
    model = DualLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_price_train_t, y_dir_train_t)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y_price, batch_y_dir in train_loader:
            price_pred, dir_pred = model(batch_X)
            loss_reg = criterion_reg(price_pred, batch_y_price)
            loss_cls = criterion_cls(dir_pred, batch_y_dir)
            # ----- 关键：降低回归损失权重 -----
            loss = regression_weight * loss_reg + loss_cls
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（防止爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_price_pred, _ = model(X_test_t)
            val_loss_reg = criterion_reg(val_price_pred, y_price_test_t).item()
            # 使用同样的权重计算验证损失
            val_loss = regression_weight * val_loss_reg + 0.0  # 分类损失在验证时忽略

        # 裁剪验证损失，防止显示爆炸
        val_loss_clipped = min(val_loss, loss_clip)

        if val_loss_clipped < best_loss:
            best_loss = val_loss_clipped
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  轮次 [{epoch+1}/{epochs}] | 训练损失: {avg_train_loss:.4f} | 验证损失: {val_loss_clipped:.4f}")

        if patience_counter >= patience:
            print(f"  ⏹️ 早停于 epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 保存模型
    torch.save(model.cpu().state_dict(), 'model.pth')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')

    model.to(device)
    return model, scaler_X, scaler_y, df

# =============================================
# 辅助函数：构建序列
# =============================================
def create_sequences(features, price_targets, dir_targets, seq_len=20):
    X, yp, yd = [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        yp.append(price_targets[i])
        yd.append(dir_targets[i])
    return np.array(X, dtype=np.float32), np.array(yp, dtype=np.float32), np.array(yd, dtype=np.float32)

# =============================================
# 扩展白名单主函数
# =============================================
def expand_whitelist():
    print("=" * 60)
    print("🚀 白名单扩展工具（损失保护版）")
    print("=" * 60)

    if not os.path.exists(WHITELIST_INPUT):
        print(f"❌ 未找到 {WHITELIST_INPUT}，请先运行批量回测")
        return None

    df_original = pd.read_csv(WHITELIST_INPUT)
    print(f"📋 原始白名单: {len(df_original)} 只股票")

    # 筛选候选（更宽松）
    candidates = df_original[
        (df_original['total_return'] > MIN_RETURN) &
        (df_original['trade_count'] >= MIN_TRADES) &
        (df_original['max_drawdown'] < MAX_DRAWDOWN)
    ].sort_values('total_return', ascending=False)

    if len(candidates) < TARGET_STOCKS:
        candidates = df_original.head(TARGET_STOCKS * 2)

    print(f"📊 候选股票: {len(candidates)} 只")

    print(f"\n🔄 开始回测 {len(candidates)} 只股票（并行 {MAX_WORKERS} 线程）...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}
        for idx, row in candidates.iterrows():
            code = str(row['code']).zfill(6)
            name = row.get('name', '')
            future = executor.submit(backtest_stock, code, name)
            future_map[future] = (code, name)
            time.sleep(0.2)

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="回测进度"):
            res = future.result()
            if res is not None:
                results.append(res)

    if not results:
        print("❌ 所有股票回测失败")
        return None

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('total_return', ascending=False)
    df_final = df_results.head(TARGET_STOCKS)

    df_final.to_csv(WHITELIST_OUTPUT, index=False, encoding='utf-8-sig')

    print(f"\n{'=' * 60}")
    print(f"📊 扩展白名单完成")
    print(f"  候选股票数: {len(results)}")
    print(f"  最终白名单: {len(df_final)} 只")
    print(f"  平均收益率: {df_final['total_return'].mean() * 100:.2f}%")
    print(f"  平均夏普比率: {df_final['sharpe_ratio'].mean():.3f}")
    print(f"  平均最大回撤: {df_final['max_drawdown'].mean() * 100:.2f}%")
    print(f"{'=' * 60}")

    print("\n📋 白名单前10名：")
    print(df_final[['code', 'name', 'total_return', 'sharpe_ratio', 'max_drawdown']].head(10).to_string(
        index=False, float_format="%.3f"
    ))

    print(f"\n✅ 扩展白名单已保存至: {WHITELIST_OUTPUT}")
    return df_final

if __name__ == "__main__":
    expand_whitelist()