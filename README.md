**A股量化回测系统 — 基于LSTM的多任务学习策略**
## 📖 项目简介
本项目构建了一个**完整的 A 股量化回测系统**，核心是一个基于 **LSTM 的多任务学习模型**，同时预测股票次日**收盘价**（回归任务）和**涨跌方向**（分类任务）。系统实现了从**全市数据训练**到**精选股票池组合策略**的完整链路。

---

## 🎯 核心特性

- ✅ **多任务学习架构**：共享 LSTM 底层，同时优化回归和分类任务
- ✅ **40+ 技术指标特征**：包含 RSI、布林带、动量、成交量等
- ✅ **趋势跟踪策略**：基于 MA200 趋势过滤、动态仓位、移动止盈
- ✅ **全市数据训练**：用全市场数据训练统一模型，提升泛化能力
- ✅ **组合策略**：白名单股票等权重持仓，分散风险
- ✅ **严格时间验证**：扩展窗口测试（Walk-Forward），杜绝未来函数
- ✅ **完整评估体系**：收益率、最大回撤、夏普比率、超额收益（vs 沪深300）
- ✅ **GPU 加速**：支持 NVIDIA GPU 训练，大幅提升效率
---

## 🏆 核心回测结果（601515）

回测中


为什么是601515？因为这是我持有的股票。
### 全市场批量回测（4340 只股票）

回测中



---

### 📊 白名单前 10 名

回测中

---

## 📂 项目结构

```text
.
├── stock_full2.py              # 核心引擎（数据、模型、回测）
├── batch_backtest.py           # 全市模型批量回测 + 白名单生成
├── expand_whitelist.py         # 白名单扩展（严格时间分割验证）
├── pool_backtest.py            # 组合策略回测
├── analyze_results.py          # 结果深度分析（统计 + 可视化）
├── requirements.txt            # 依赖清单
├── README.md                   # 项目说明
├── .gitignore                  # Git 忽略文件
│
├── stock_data_cache/           # 数据缓存（自动生成）
├── model.pth                   # 模型权重
├── scaler_X.pkl                # 特征标准化器
├── scaler_y.pkl                # 标签标准化器
├── whitelist.csv               # 白名单（自动生成）
├── whitelist_extended.csv      # 扩展白名单（自动生成）
├── batch_results_unified.csv   # 全市模型批量回测结果
└── backtest_result.png         # 资金曲线图（自动生成）



🚀 快速开始
1. 安装依赖
pip install -r requirements.txt

2. 单只股票回测
python stock_full2.py
程序会交互式询问股票代码，自动完成：
数据获取（baostock，支持重试）
特征构造（40+ 技术指标）
LSTM 训练（50 轮）
策略回测（含止盈止损）
生成资金曲线图（含沪深300对比）

3. 批量回测与白名单生成
python batch_backtest.py
程序会：
统一获取沪深300指数数据（所有股票共用）
并行回测指定数量的股票
自动重试失败股票（最多3次）
生成 batch_results.csv 和 whitelist.csv
💡 你可以在 batch_backtest.py 顶部修改 MAX_STOCKS_FULL 控制测试数量，首次建议设为 20 快速验证。

4. 回测结果深度分析
python analyze_results.py
程序会：
收益率分布直方图
收益 vs 回撤散点图
交易次数分布图
夏普比率分组箱线图
统计摘要报告（CSV）

🔧 策略参数调优
你可以在 stock_full2.py 顶部自由调整核心参数：
BUY_THRESHOLD = 0.52      # 上涨概率高于此值买入
SELL_THRESHOLD = 0.48     # 上涨概率低于此值卖出
STOP_LOSS = -0.10         # 亏损 10% 时强制止损
TAKE_PROFIT = 0.30        # 盈利 30% 时强制止盈
白名单筛选参数（在 batch_backtest.py 中）：
WHITELIST_MIN_RETURN = 0.30    # 最低收益率（30%）
WHITELIST_MIN_TRADES = 3       # 最少交易次数（3次）
WHITELIST_MAX_DRAWDOWN = 0.50  # 最大可接受回撤（50%）

📊 结果可视化
运行结束后自动生成：
资金曲线对比图 (backtest_result.png)：策略 vs 沪深300
交易明细（终端打印）
最大回撤 / 夏普比率 / 超额收益（终端打印）




**⚠️ 免责声明：本项目仅供量化研究与学习使用，不构成任何投资建议。实盘交易需谨慎，风险自负。**

