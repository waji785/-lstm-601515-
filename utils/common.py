# utils/common.py
import random
import numpy as np
import torch
import time

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
        return random.sample(range(min_seed, max_seed), n)
    except Exception:
        return [42, 123, 2024, 999, 777]

def create_sequences(features, price_targets, dir_targets, seq_len=20):
    """
    从特征矩阵生成时间序列样本
    """
    X, yp, yd = [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        yp.append(price_targets[i])
        yd.append(dir_targets[i])
    return (np.array(X, dtype=np.float32),
            np.array(yp, dtype=np.float32),
            np.array(yd, dtype=np.float32))