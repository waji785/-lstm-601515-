# core/trainer.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import akshare as ak
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from config.settings import *
from core.model import DualLSTM
from utils.common import create_sequences, set_seed
from utils.logger import setup_logger

logger = setup_logger(__name__)

def train_and_save_model(df, train_end_date=None, model_save_path=MODEL_PATH,
                         scaler_x_path=SCALER_X_PATH, scaler_y_path=SCALER_Y_PATH,
                         val_ratio=0.2, seed=42, exclude_st=True):
    """
    训练全市模型并保存（增强版）
    
    参数:
        df: 必须包含 stock_code 列，且已有特征和 target
        train_end_date: 训练截止日期（字符串），若为 None 则自动取数据最后日期前推 1 年
        model_save_path: 模型保存路径
        scaler_x_path, scaler_y_path: 标准化器保存路径
        val_ratio: 若外部验证集不足，从训练集划分的比例
        seed: 随机种子（用于划分验证集）
        exclude_st: 是否剔除 ST/*ST 股票（默认 True）
    """
    set_seed(seed)
    
    if df is None or len(df) < 100:
        logger.error("数据不足")
        return None, None, None, None

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # ========== 剔除 ST 股票（只在训练时） ==========
    if exclude_st and 'stock_code' in df.columns:
        try:
            logger.info("正在获取股票名称列表以剔除 ST 股票...")
            stock_name_df = ak.stock_info_a_code_name()
            name_map = dict(zip(stock_name_df['code'].astype(str).str.zfill(6), stock_name_df['name']))
            def is_st(code):
                name = name_map.get(code, '')
                return 'ST' in name.upper() or '*ST' in name.upper()
            before = df['stock_code'].nunique()
            df = df[~df['stock_code'].apply(is_st)].copy()
            after = df['stock_code'].nunique()
            if after < before:
                logger.info(f"已剔除 ST 股票 {before - after} 只，剩余 {after} 只用于训练")
            else:
                logger.info("未发现 ST 股票")
        except Exception as e:
            logger.warning(f"获取股票名称失败，跳过 ST 过滤: {e}")
    # ==================================================

    # 自动确定训练截止日期
    if train_end_date is None:
        max_date = df['Date'].max()
        train_end_date = (max_date - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        logger.info(f"自动设置训练截止日期: {train_end_date}")
    else:
        train_end_date = pd.to_datetime(train_end_date)
    
    train_end = pd.to_datetime(train_end_date)

    # 按股票分组生成序列
    X_train_list, y_price_train_list, y_dir_train_list = [], [], []
    X_val_list, y_price_val_list, y_dir_val_list = [], [], []
    train_raw_features = []

    for code, group in df.groupby('stock_code'):
        group = group.sort_values('Date').reset_index(drop=True)
        if len(group) < SEQ_LEN + 1:
            continue

        train_group = group[group['Date'] <= train_end].copy()
        val_group = group[group['Date'] > train_end].copy()

        if len(train_group) < SEQ_LEN + 1:
            continue

        # 训练集
        train_feat = train_group[FEATURE_COLS].values
        train_tp = train_group['Target_Price'].values
        train_td = train_group['Target_Direction'].values
        X_t, yp_t, yd_t = create_sequences(train_feat, train_tp, train_td, SEQ_LEN)
        if len(X_t) > 0:
            X_train_list.append(X_t)
            y_price_train_list.append(yp_t)
            y_dir_train_list.append(yd_t)
            train_raw_features.append(train_feat)

        # 验证集（外部时间分割）
        if len(val_group) >= SEQ_LEN + 1:
            val_feat = val_group[FEATURE_COLS].values
            val_tp = val_group['Target_Price'].values
            val_td = val_group['Target_Direction'].values
            X_v, yp_v, yd_v = create_sequences(val_feat, val_tp, val_td, SEQ_LEN)
            if len(X_v) > 0:
                X_val_list.append(X_v)
                y_price_val_list.append(yp_v)
                y_dir_val_list.append(yd_v)

    if not X_train_list:
        logger.error("无训练序列")
        return None, None, None, None

    X_train = np.concatenate(X_train_list, axis=0)
    y_price_train = np.concatenate(y_price_train_list, axis=0)
    y_dir_train = np.concatenate(y_dir_train_list, axis=0)
    logger.info(f"训练序列总数: {len(X_train)}")

    # 标准化器拟合
    if train_raw_features:
        all_raw = np.concatenate(train_raw_features, axis=0)
    else:
        n_train, seq_len, n_feat = X_train.shape
        all_raw = X_train.reshape(-1, n_feat)
    scaler_X = StandardScaler()
    scaler_X.fit(all_raw)

    # 转换训练序列
    n_train, seq_len, n_feat = X_train.shape
    X_train_flat = X_train.reshape(-1, n_feat)
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(n_train, seq_len, n_feat)

    # 处理验证集
    if X_val_list:
        X_val = np.concatenate(X_val_list, axis=0)
        y_price_val = np.concatenate(y_price_val_list, axis=0)
        y_dir_val = np.concatenate(y_dir_val_list, axis=0)
        n_val, seq_len, n_feat = X_val.shape
        X_val_flat = X_val.reshape(-1, n_feat)
        X_val_scaled = scaler_X.transform(X_val_flat).reshape(n_val, seq_len, n_feat)
        has_external_val = True
    else:
        X_val_scaled = None
        y_price_val = None
        y_dir_val = None
        has_external_val = False
        logger.warning("无外部验证集，将从训练集划分验证集")

    # 价格标签标准化
    scaler_y = StandardScaler()
    y_price_train_scaled = scaler_y.fit_transform(y_price_train.reshape(-1, 1)).ravel()
    if has_external_val:
        y_price_val_scaled = scaler_y.transform(y_price_val.reshape(-1, 1)).ravel()
    else:
        y_price_val_scaled = None

    # 转 Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_price_train_t = torch.tensor(y_price_train_scaled, dtype=torch.float32).reshape(-1, 1)
    y_dir_train_t = torch.tensor(y_dir_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_t, y_price_train_t, y_dir_train_t)

    # 创建 DataLoader
    if has_external_val:
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_price_val_t = torch.tensor(y_price_val_scaled, dtype=torch.float32).reshape(-1, 1)
        y_dir_val_t = torch.tensor(y_dir_val, dtype=torch.long)
        val_dataset = TensorDataset(X_val_t, y_price_val_t, y_dir_val_t)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  pin_memory=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                pin_memory=True, num_workers=0)
    else:
        # 从训练集划分验证集
        train_size = int((1 - val_ratio) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        if val_size == 0:
            logger.warning("训练集太小，无法划分验证集，将使用全部训练数据")
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      pin_memory=True, num_workers=0)
            val_loader = None
        else:
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      pin_memory=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    pin_memory=True, num_workers=0)
            logger.info(f"从训练集划分 {val_size} 个样本作为验证集")

    # 模型
    model = DualLSTM(input_size=len(FEATURE_COLS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_X, by_price, by_dir in train_loader:
            batch_X = batch_X.to(device)
            by_price = by_price.to(device)
            by_dir = by_dir.to(device)

            optimizer.zero_grad()
            price_pred, dir_pred = model(batch_X)
            loss_reg = criterion_reg(price_pred, by_price)
            loss_cls = criterion_cls(dir_pred, by_dir)
            loss = REGRESSION_WEIGHT * loss_reg + loss_cls
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # 验证
        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, by_price, by_dir in val_loader:
                    batch_X = batch_X.to(device)
                    by_price = by_price.to(device)
                    by_dir = by_dir.to(device)
                    price_pred, dir_pred = model(batch_X)
                    loss_reg = criterion_reg(price_pred, by_price)
                    loss_cls = criterion_cls(dir_pred, by_dir)
                    val_loss = REGRESSION_WEIGHT * loss_reg + loss_cls
                    val_loss_total += val_loss.item()
                    _, preds = torch.max(dir_pred, 1)
                    correct += (preds == by_dir).sum().item()
                    total += by_dir.size(0)
            avg_val_loss = val_loss_total / len(val_loader)
            val_acc = correct / total if total > 0 else 0.0

            # 多指标早停：损失下降且准确率提升
            if avg_val_loss < best_loss and val_acc > best_acc:
                best_loss = avg_val_loss
                best_acc = val_acc
                patience_counter = 0
                best_state = model.state_dict().copy()
                logger.info(f"Epoch {epoch+1}: 训练损失 {avg_train_loss:.4f}, 验证损失 {avg_val_loss:.4f}, 准确率 {val_acc:.4f} (改善)")
            else:
                patience_counter += 1

            if (epoch+1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{EPOCHS} 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, 准确率: {val_acc:.4f}")

            if patience_counter >= PATIENCE:
                logger.info(f"早停于 epoch {epoch+1}")
                break
        else:
            # 无验证集，每10轮保存一次
            if (epoch+1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{EPOCHS} 训练损失: {avg_train_loss:.4f}")
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        logger.warning("未保存最佳模型，使用最后一个 epoch 的模型")

    # 保存模型和 scaler
    model.cpu()
    torch.save(model.cpu(), model_save_path)
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    logger.info(f"模型和 scaler 已保存至 {model_save_path}")
    model.to(device)
    return model, scaler_X, scaler_y, df