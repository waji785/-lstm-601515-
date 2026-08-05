# config/settings.py
import os
import datetime

# ------------------- 时间范围 -------------------
TRAIN_END_DATE = "2025-12-31"
BACKTEST_START_DATE = "2025-01-01"
TODAY = datetime.datetime.now().strftime("%Y-%m-%d")   # 必须定义

# ------------------- 路径 -------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "stock_data_cache")  # 必须定义
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
SCALER_X_PATH = os.path.join(BASE_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y.pkl")
WHITELIST_FILE = os.path.join(BASE_DIR, "whitelist.csv")
WHITELIST_EXTENDED_FILE = os.path.join(BASE_DIR, "whitelist_extended.csv")

# ------------------- 特征列（17+ 个） -------------------
FEATURE_COLS = [
    'Close', 'Volume',
    'Momentum_5', 'Momentum_10', 'Return_1d',
    'Volatility_5', 'Volatility_10',
    'MA_5', 'MA_10', 'MA_20',
    'Price_MA_5_Ratio', 'Price_MA_20_Ratio',
    'MA_5_20_diff',
    'RSI_14', 'BB_position', 'Volume_Ratio', 'High_Low_Ratio',
    'Amount_Ratio', 'PctChg', 'TradeStatus',
    'PeTTM', 'PbMRQ', 'PsTTM', 'PcfNcfTTM', 'Turnover'
]

# ------------------- 模型超参数 -------------------
SEQ_LEN = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

# ------------------- 训练参数 -------------------
BATCH_SIZE = 4096
EPOCHS = 30
PATIENCE = 15
REGRESSION_WEIGHT = 0.1
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0

# ------------------- 回测参数 -------------------
BUY_THRESHOLD = 0.45
SELL_THRESHOLD = 0.43
STOP_LOSS = -0.08
TAKE_PROFIT = 0.20
MAX_POSITION = 0.6
MIN_VOLATILITY = 1.0

# ------------------- 回撤控制 -------------------
DRAWDOWN_THRESHOLD = 0.025
RECOVERY_RATIO = 0.25

# ------------------- 白名单筛选条件 -------------------
WHITELIST_MIN_RETURN = 0.15
WHITELIST_MIN_TRADES = 2
WHITELIST_MAX_DRAWDOWN = 0.65

# ------------------- 日志配置 -------------------
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "logs", "quant.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ------------------- 交易成本参数 -------------------
COMMISSION_RATE = 0.00025        # 佣金费率（万2.5）
MIN_COMMISSION = 5.0             # 最低佣金（元）
STAMP_DUTY_RATE = 0.001          # 印花税率（仅卖出，千1）
SLIPPAGE = 0.001                 # 滑点（买卖各0.1%）

# ------------------- 扩展窗口交叉验证参数 -------------------
TRAIN_START_DATE = "2018-01-01"        # 训练起始日期（固定）
WINDOW_LENGTH_YEARS = 3                # 每个窗口的训练集长度（年）
TEST_LENGTH_YEARS = 1                  # 每个窗口的测试集长度（年）
NUM_WINDOWS = 1                        # 窗口数量（最多不超过数据总长度）


# ----- 配置（覆盖 settings.py 默认值） -----
MAX_WORKERS = 1
TRAIN_STOCKS = 10          # 参与训练的股票数量
TEST_STOCKS = 6000            # 回测股票数（None 表示全部）
WHITELIST_MIN_RETURN = 0.15
WHITELIST_MIN_TRADES = 2
WHITELIST_MAX_DRAWDOWN = 0.65

SEED = 42