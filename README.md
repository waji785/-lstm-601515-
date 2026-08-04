📈 A股LSTM量化交易系统
基于 LSTM 神经网络 的 A 股市场量化策略回测与交易系统，支持全市场选股、多窗口交叉验证、组合回测与绩效分析。

🚀 核心特性
全流程自动化：数据下载 → 特征工程 → 模型训练 → 回测 → 绩效评估，一键运行。
时间分割交叉验证：扩展窗口（Expanding Window）验证策略在不同年份的稳健性，避免过拟合。
高透明度：回测结果实时写入 CSV，支持逐只股票回溯，方便分析。
实战化支持：交易成本（佣金、印花税、滑点）、仓位管理、止盈止损机制。
模型集成：支持生成最终生产模型，用于实盘（或模拟盘）预测。

## 🛠️ 技术栈

| 类别 | 技术 |
| :--- | :--- |
| **深度学习框架** | PyTorch 2.x（CUDA） |
| **数据源** | baostock（本地缓存） |
| **数据处理** | Pandas、NumPy、Scikit-learn |
| **可视化** | Matplotlib、Seaborn |
| **模型持久化** | joblib |

## 📁 项目结构

```text
.
├── config/                      # 全局配置
│   ├── __init__.py
│   └── settings.py              # 所有参数（阈值、路径、特征列表）
├── core/                        # 核心功能模块
│   ├── __init__.py
│   ├── data_loader.py           # 数据下载、缓存、增量更新
│   ├── features.py              # 特征构造（技术指标、目标变量）
│   ├── model.py                 # LSTM 模型定义
│   ├── trainer.py               # 模型训练（早停、验证集划分）
│   ├── backtest_engine.py       # 回测引擎（交易成本、信号生成）
│   └── metrics.py               # 绩效指标（夏普、回撤等）
├── scripts/                     # 可执行脚本
│   ├── batch_backtest.py        # 扩展窗口交叉验证 + 最终模型训练
│   ├── train_final_model.py     # 单独训练最终模型
│   └── pool_backtest_analysis.py # 组合回测与绩效分析
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── common.py                # 随机种子、序列生成
│   └── logger.py                # 日志配置
├── logs/                        # 日志文件（自动生成）
├── stock_data_cache/            # 股票数据缓存（Parquet 格式）
├── requirements.txt             # 依赖列表
├── .gitignore
└── README.md
```

```text
📊 运行流程
1. 首次运行：下载数据并训练模型
bash
pip install -r requirements.txt
python scripts/batch_backtest.py
自动下载 A 股列表，过滤北交所和 ST 股票。
下载每只股票的历史数据（优先缓存，支持增量更新）。
执行 扩展窗口交叉验证（默认 3 个窗口，可调整 NUM_WINDOWS）。
最后训练 最终模型（使用全部历史数据）。

2. 仅训练最终模型（跳过窗口验证）
bash
python scripts/train_final_model.py --stocks 6000 --end-date 2026-08-04
3. 组合回测（对白名单股票进行等权重组合）
bash
python scripts/pool_backtest_analysis.py
加载 whitelist_extended.csv 中的股票。
使用最终模型进行组合回测，计算组合收益、夏普、最大回撤。
生成资金曲线图和绩效指标 CSV。
```

```text
🔍 结果文件说明
window_1_results.csv ~ window_N_results.csv：每个窗口的逐股票回测结果（包含收益率、夏普、回撤、交易次数等）。
expanding_window_summary.csv：各窗口的汇总统计（平均收益、胜率、夏普、回撤、有效股票数）。
model_final.pth：最终生产模型（包含全部历史数据）。
scaler_X_final.pkl、scaler_Y_final.pkl：标准化器（用于新数据预处理）。
pool_backtest_results.csv：组合回测的股票贡献明细。
pool_metrics.csv：组合的绩效指标。
```

```text
⚙️ 关键参数调优建议
参数	含义	调优方向
BUY_THRESHOLD	买入阈值（概率）	提高可减少假信号，但可能错过机会
STOP_LOSS	止损比例	收紧可控制单笔亏损，但可能过早离场
TAKE_PROFIT	止盈比例	降低可更快锁定利润，提高胜率
MAX_POSITION	单只股票最大仓位	控制集中度，降低组合风险
SEQ_LEN	序列长度（天数）	影响模型时间窗口，可尝试 10~30
HIDDEN_SIZE	LSTM 隐藏层维度	64 或 128，越大模型容量越大
NUM_LAYERS	LSTM 层数	2~3 层足够，过深易过拟合
```

```text
📖 常见问题
1. 数据下载失败或超时？
检查网络，或改用 akshare 作为主数据源（在 core/data_loader.py 中调整优先级）。
若频繁超时，可降低 max_workers 或增加 timeout。

2. 模型训练太慢？
减少 TRAIN_STOCKS（训练股票数）。
启用混合精度训练（在 trainer.py 中启用 autocast）。
使用 GPU（CUDA）加速。

3. zip() argument 2 is longer than argument 1 错误？
确保模型定义（core/model.py）与保存的权重一致（HIDDEN_SIZE、NUM_LAYERS）。
在 run_window_backtest 中已采用保存整个模型对象的方式，该错误不应再出现。

4. 如何查看单只股票的回测详情？
运行 test.py（需自行创建），加载 model_final.pth，对指定股票进行回测并输出交易明细。

5. 如何扩展新特征？
在 config/settings.py 的 FEATURE_COLS 中添加新列名
在 core/features.py 的 construct_features 函数中实现计算逻辑。
运行 reconstruct_all_features()（在 data_loader.py 中）重建缓存特征，无需重新下载数据。
```

**⚠️ 免责声明：本项目仅供量化研究与学习使用，不构成任何投资建议。实盘交易需谨慎，风险自负。**

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.