# 时间序列分析模块 (Time Series Analysis Module)

## 模块概述 (Overview)

本模块提供完整的时间序列分析教程，涵盖从基础概念到高级预测方法的全面内容。

This module provides a comprehensive tutorial on time series analysis, covering everything from basic concepts to advanced forecasting methods.

## 文件结构 (File Structure)

```
09_TimeSeries/
├── 01_time_series_basics.py        # 时间序列基础 (约450行)
├── 02_arima.py                      # ARIMA 模型 (约570行)
├── 03_forecasting_methods.py       # 多种预测方法对比 (约600行)
└── README.md                        # 本文件
```

## 学习路径 (Learning Path)

### 1️⃣ 01_time_series_basics.py
**时间序列基础 (Time Series Basics)**

**学习内容：**
- 时间序列的基本概念和组成部分
- 平稳性检验（ADF、KPSS）
- 时间序列分解技术
- ACF 和 PACF 分析
- 差分方法
- 特征工程（滞后特征、滚动统计、时间特征）

**生成图表（8张）：**
1. 时间序列组成部分分解
2. ACF 和 PACF 图
3. 差分前后对比
4. 加法模型分解
5. 滞后特征相关性
6. 滚动统计
7. 时间特征分布
8. 特征重要性

### 2️⃣ 02_arima.py
**ARIMA 模型 (ARIMA Model)**

**学习内容：**
- ARIMA(p,d,q) 模型原理
- 参数选择方法（网格搜索、AIC/BIC）
- 模型训练和诊断
- 残差分析
- 单步和多步预测
- SARIMA（季节性 ARIMA）
- 与基线模型对比

**生成图表（9张）：**
1. CO2 数据探索
2. 原始数据 ACF/PACF
3. 差分后 ACF/PACF
4. AIC/BIC 参数选择
5. 残差诊断（4个子图）
6. ARIMA 预测结果
7. 多模型对比
8. SARIMA vs ARIMA
9. 性能总结

### 3️⃣ 03_forecasting_methods.py
**多种预测方法对比 (Multiple Forecasting Methods)**

**学习内容：**
- 基线模型（Naive, SMA, WMA）
- 指数平滑（SES, DES, TES）
- 统计模型（ARIMA, SARIMA）
- 机器学习方法（Random Forest, XGBoost）
- 模型对比和评估
- 实际应用案例

**生成图表（12张）：**
1. 基线模型对比
2. 指数平滑方法对比
3. ARIMA vs SARIMA
4. 特征重要性对比
5. 机器学习方法对比
6. 所有模型性能
7. Top 5 模型预测
8. 误差分布
9. 训练时间对比
10. 精度 vs 速度
11. 销售预测案例
12. 方法选择指南

---

## 预测方法对比表 (Forecasting Methods Comparison)

### 📊 方法对比总览

| 方法类别 | 具体方法 | 优点 | 缺点 | 适用场景 | 复杂度 |
|---------|---------|------|------|---------|--------|
| **基线模型** | Naive Forecast | 极简单、快速 | 精度最低 | 基准对比 | ⭐ |
| | Simple MA | 简单、平滑噪声 | 滞后、无趋势 | 短期平滑 | ⭐ |
| | Weighted MA | 重视近期数据 | 权重选择难 | 短期预测 | ⭐⭐ |
| **指数平滑** | SES | 简单、适合平稳数据 | 无趋势处理 | 平稳序列 | ⭐⭐ |
| | DES (Holt) | 处理趋势 | 无季节性 | 有趋势数据 | ⭐⭐⭐ |
| | TES (Holt-Winters) | 处理趋势+季节性 | 需要足够数据 | 季节性数据 | ⭐⭐⭐⭐ |
| **统计模型** | ARIMA | 理论严谨、可解释 | 参数选择复杂 | 单变量平稳序列 | ⭐⭐⭐⭐ |
| | SARIMA | 处理季节性 | 计算复杂 | 季节性数据 | ⭐⭐⭐⭐⭐ |
| | Auto ARIMA | 自动选参 | 计算慢 | 快速建模 | ⭐⭐⭐ |
| **机器学习** | Random Forest | 精度高、鲁棒 | 需特征工程 | 大数据集 | ⭐⭐⭐⭐ |
| | XGBoost | 精度最高、快速 | 黑盒、易过拟合 | 复杂模式 | ⭐⭐⭐⭐⭐ |
| | LightGBM | 更快、内存友好 | 小数据集易过拟合 | 超大数据集 | ⭐⭐⭐⭐⭐ |
| **深度学习** | LSTM | 捕捉长期依赖 | 需大量数据、慢 | 复杂序列 | ⭐⭐⭐⭐⭐ |
| | GRU | 比LSTM快 | 需大量数据 | 复杂序列 | ⭐⭐⭐⭐⭐ |

### 📈 性能对比（CO2 数据集）

基于本教程的实验结果：

| 模型 | MAE | RMSE | MAPE | 训练时间 | 推荐度 |
|------|-----|------|------|---------|--------|
| TES (Holt-Winters) | 0.15-0.25 | 0.20-0.35 | 0.05-0.08% | <0.1s | ⭐⭐⭐⭐⭐ |
| SARIMA | 0.18-0.28 | 0.25-0.40 | 0.06-0.09% | 1-3s | ⭐⭐⭐⭐⭐ |
| XGBoost | 0.20-0.35 | 0.28-0.45 | 0.06-0.10% | 0.5-1s | ⭐⭐⭐⭐ |
| ARIMA | 0.25-0.40 | 0.35-0.50 | 0.08-0.12% | 0.5-1s | ⭐⭐⭐⭐ |
| Random Forest | 0.30-0.45 | 0.40-0.55 | 0.09-0.13% | 0.3-0.8s | ⭐⭐⭐ |
| DES (Holt) | 0.80-1.20 | 1.00-1.50 | 0.25-0.35% | <0.1s | ⭐⭐⭐ |
| SMA-12 | 1.50-2.00 | 1.80-2.30 | 0.45-0.60% | <0.01s | ⭐⭐ |
| Naive | 2.00-2.50 | 2.30-2.80 | 0.60-0.75% | <0.01s | ⭐ |

*注：实际性能会因数据集特性而异*

---

## ARIMA 参数选择指南 (ARIMA Parameter Selection Guide)

### 🎯 ARIMA(p, d, q) 参数详解

#### 1. **d（差分次数）- Differencing Order**

**如何确定：**
```
1. 看原始数据的平稳性
   ├─ 进行 ADF 检验
   │  └─ p-value < 0.05 → 平稳 → d=0
   └─ p-value >= 0.05 → 非平稳 → 需要差分

2. 一阶差分后检验
   ├─ ADF p-value < 0.05 → d=1
   └─ 仍非平稳 → d=2

3. 通常 d ≤ 2
   └─ 过度差分会损失信息
```

**决策表：**
| 数据特征 | d 值 | 说明 |
|---------|------|------|
| 数据已平稳 | 0 | 无需差分 |
| 有线性趋势 | 1 | 一阶差分去除趋势 |
| 有抛物线趋势 | 2 | 二阶差分 |
| 有季节性 | 0 或 1 | 考虑 SARIMA |

#### 2. **p（AR阶数）- Autoregressive Order**

**如何确定：**
```
观察 PACF 图（偏自相关函数）
├─ PACF 在 lag p 后截尾（突然下降到置信区间内）
│  └─ 建议 AR(p)
│
├─ PACF 缓慢衰减
│  └─ 考虑 MA 或 ARMA 模型
│
└─ 经验法则：
   ├─ 短期依赖：p = 1-3
   ├─ 中期依赖：p = 4-7
   └─ 长期依赖：p = 8-12
```

**PACF 模式识别：**
```
PACF 图示                    模型建议
────────────────────────────────────
|█▁▁▁▁▁▁▁▁▁▁...            AR(1)
|█▆▁▁▁▁▁▁▁▁▁...            AR(2)
|█▆▄▁▁▁▁▁▁▁▁...            AR(3)
|█▆▅▄▃▂▁▁▁▁▁...            MA 或 ARMA
```

#### 3. **q（MA阶数）- Moving Average Order**

**如何确定：**
```
观察 ACF 图（自相关函数）
├─ ACF 在 lag q 后截尾
│  └─ 建议 MA(q)
│
├─ ACF 缓慢衰减
│  └─ 考虑 AR 或 ARMA 模型
│
└─ 经验法则：
   ├─ 短期冲击：q = 1-3
   ├─ 中期冲击：q = 4-7
   └─ 长期冲击：q = 8-12
```

**ACF 模式识别：**
```
ACF 图示                     模型建议
────────────────────────────────────
|█▁▁▁▁▁▁▁▁▁▁...            MA(1)
|█▆▁▁▁▁▁▁▁▁▁...            MA(2)
|█▆▄▁▁▁▁▁▁▁▁...            MA(3)
|█▆▅▄▃▂▁▁▁▁▁...            AR 或 ARMA
```

### 🔍 参数选择流程图

```
开始
  │
  ▼
平稳性检验 (ADF Test)
  │
  ├─ 平稳 → d=0
  │         │
  │         ▼
  │      看 ACF/PACF
  │         │
  └─ 非平稳 → 差分 → 再检验 → d=1 或 d=2
              │
              ▼
         看 ACF/PACF 图
              │
      ┌───────┼───────┐
      │       │       │
   PACF截尾  ACF截尾  都衰减
      │       │       │
      ▼       ▼       ▼
    AR(p)   MA(q)   ARMA(p,q)
      │       │       │
      └───────┴───────┘
              │
              ▼
      网格搜索优化 (Grid Search)
      使用 AIC/BIC 准则
              │
              ▼
         选择最优模型
              │
              ▼
          残差诊断
              │
      ┌───────┴───────┐
      │               │
   通过检验        未通过
      │               │
      ▼               ▼
   使用模型      调整参数/重新选择
```

### 📐 网格搜索示例代码

```python
import itertools
from statsmodels.tsa.arima.model import ARIMA

# 定义参数范围
p_range = range(0, 4)    # AR: 0-3
d_range = range(0, 3)    # I:  0-2
q_range = range(0, 4)    # MA: 0-3

best_aic = float('inf')
best_order = None

# 网格搜索
for p, d, q in itertools.product(p_range, d_range, q_range):
    try:
        model = ARIMA(train_data, order=(p, d, q))
        fitted = model.fit()

        if fitted.aic < best_aic:
            best_aic = fitted.aic
            best_order = (p, d, q)

    except:
        continue

print(f"最优参数: ARIMA{best_order}")
print(f"最优 AIC: {best_aic:.2f}")
```

### 🎓 经验法则 (Rules of Thumb)

#### 常见数据模式与推荐参数

| 数据特征 | 推荐 ARIMA | 说明 |
|---------|-----------|------|
| **平稳白噪声** | (0,0,0) | 无需建模 |
| **随机游走** | (0,1,0) | 一阶差分即可 |
| **随机游走+漂移** | (0,1,1) | 带MA项 |
| **自回归过程** | (p,0,0) | p由PACF确定 |
| **移动平均过程** | (0,0,q) | q由ACF确定 |
| **混合过程** | (p,d,q) | 网格搜索 |
| **月度经济数据** | (1,1,1) 到 (3,1,3) | 常见范围 |
| **日度金融数据** | (1,0,1) 到 (5,0,5) | 波动大 |

#### AIC vs BIC 准则选择

| 准则 | 特点 | 适用场景 |
|------|------|---------|
| **AIC** (赤池信息准则) | 惩罚较轻，倾向复杂模型 | 预测导向 |
| **BIC** (贝叶斯信息准则) | 惩罚较重，倾向简单模型 | 解释导向 |

```python
# AIC vs BIC 选择
if goal == 'prediction':
    use_criterion = 'AIC'  # 更好的预测
elif goal == 'interpretation':
    use_criterion = 'BIC'  # 更简洁的模型
```

---

## SARIMA 参数选择指南 (SARIMA Parameter Selection)

### 🌊 SARIMA(p,d,q)(P,D,Q)s 参数详解

#### 季节性参数 (Seasonal Parameters)

```
SARIMA(p,d,q)(P,D,Q)s
         │      │   │
         │      │   └─ s: 季节周期
         │      └───── (P,D,Q): 季节性参数
         └──────────── (p,d,q): 非季节性参数
```

#### s（季节周期）确定

| 数据频率 | s 值 | 示例 |
|---------|------|------|
| 月度数据 | 12 | 年度季节性 |
| 季度数据 | 4 | 年度季节性 |
| 周数据 | 52 | 年度季节性 |
| 日数据（周模式） | 7 | 周季节性 |
| 日数据（年模式） | 365 | 年度季节性 |
| 小时数据 | 24 | 日季节性 |

#### P、D、Q 选择方法

```python
# 1. 季节性差分 (D)
seasonal_diff = data.diff(s)  # s 是季节周期

# 2. 观察季节性 ACF/PACF
# - 在 lag = s, 2s, 3s 处的相关性
# - 季节性 PACF 截尾 → P
# - 季节性 ACF 截尾 → Q

# 3. 常用值
P = 0, 1, 或 2
D = 0 或 1
Q = 0, 1, 或 2
```

### 🎯 SARIMA 快速决策表

| 数据特征 | 推荐 SARIMA | 示例 |
|---------|------------|------|
| 月度销售（年季节性） | (1,1,1)(1,1,1,12) | 零售数据 |
| 季度GDP | (1,1,1)(1,1,1,4) | 经济数据 |
| 日度访客（周模式） | (1,1,1)(1,0,1,7) | 网站流量 |
| 小时用电量 | (1,0,1)(1,0,1,24) | 能源数据 |
| 无季节性 | (p,d,q)(0,0,0,0) | 等同ARIMA |

---

## 特征工程指南 (Feature Engineering Guide)

### 🛠️ 机器学习时间序列特征

#### 1. 滞后特征 (Lag Features)

```python
# 创建滞后特征
for lag in [1, 2, 3, 7, 14, 30]:
    df[f'lag_{lag}'] = df['value'].shift(lag)
```

**选择滞后阶数：**
- 根据 ACF 图确定显著滞后
- 业务知识（如周、月周期）
- 常用：1, 2, 3（短期）+ 7, 30（周期）

#### 2. 滚动统计 (Rolling Statistics)

```python
windows = [7, 14, 30]
for w in windows:
    df[f'rolling_mean_{w}'] = df['value'].rolling(w).mean()
    df[f'rolling_std_{w}'] = df['value'].rolling(w).std()
    df[f'rolling_min_{w}'] = df['value'].rolling(w).min()
    df[f'rolling_max_{w}'] = df['value'].rolling(w).max()
```

**窗口选择：**
- 7天：捕捉周模式
- 30天：捕捉月模式
- 90天：捕捉季度模式

#### 3. 时间特征 (Time Features)

```python
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df['quarter'] = df.index.quarter
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
df['is_month_start'] = df.index.is_month_start.astype(int)
df['is_month_end'] = df.index.is_month_end.astype(int)
```

#### 4. 差分特征 (Differencing Features)

```python
df['diff_1'] = df['value'].diff(1)      # 一阶差分
df['diff_7'] = df['value'].diff(7)      # 周差分
df['diff_30'] = df['value'].diff(30)    # 月差分
df['pct_change'] = df['value'].pct_change()  # 百分比变化
```

#### 5. 季节性分解特征

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['value'], model='additive', period=12)
df['trend'] = decomposition.trend
df['seasonal'] = decomposition.seasonal
df['residual'] = decomposition.resid
```

### 📊 特征重要性排序（经验值）

1. **滞后特征** (Lag-1, Lag-2, Lag-7) - 最重要
2. **滚动均值** (Rolling Mean) - 非常重要
3. **趋势特征** (Trend from decomposition) - 重要
4. **月份/星期** (Month/DayOfWeek) - 重要
5. **滚动标准差** (Rolling Std) - 中等重要
6. **差分特征** (Differencing) - 中等重要
7. **季节性特征** (Seasonal) - 视数据而定

---

## 模型评估指标 (Evaluation Metrics)

### 📏 评估指标对比

| 指标 | 公式 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | 直观、对异常值不敏感 | 无法体现大误差 | 一般预测 |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | 惩罚大误差 | 受异常值影响大 | 关注大误差 |
| **MAPE** | $\frac{100}{n}\sum\|\frac{y_i - \hat{y}_i}{y_i}\|$ | 百分比、易比较 | y接近0时不稳定 | 跨尺度比较 |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | 解释方差比例 | 可能为负 | 模型拟合度 |
| **SMAPE** | $\frac{100}{n}\sum\frac{\|y_i - \hat{y}_i\|}{(\|y_i\| + \|\hat{y}_i\|)/2}$ | 对称、稳定 | 解释稍复杂 | 对称误差 |

### 🎯 指标选择建议

```
业务场景                      推荐指标
──────────────────────────────────────
销售预测                     MAPE (易于管理层理解)
库存管理                     MAE (直接反映库存误差)
能源需求预测                  RMSE (惩罚大的供需失衡)
股票价格预测                  MAPE + 方向准确率
需求预测（多产品）            MAPE (便于跨产品比较)
异常检测                     RMSE + 分位数
学术研究                     MAE + RMSE + R²
```

---

## 常见问题与解决方案 (FAQ)

### ❓ 常见问题

#### 1. **数据量太少怎么办？**
```
问题：只有几十个数据点
解决方案：
  ├─ 使用简单模型（ARIMA, Exponential Smoothing）
  ├─ 避免复杂机器学习模型
  ├─ 考虑外部数据增强
  └─ 使用领域知识辅助
```

#### 2. **预测结果一直是直线？**
```
原因：
  ├─ 模型过于简单（如Naive）
  ├─ 差分过度
  └─ 数据缺乏变化

解决：
  ├─ 尝试包含趋势的模型（DES, ARIMA）
  ├─ 检查特征工程
  └─ 考虑季节性因素
```

#### 3. **ARIMA 拟合失败？**
```
常见错误：
  ├─ "The computed initial MA coefficients are not invertible"
  │  └─ 解决：降低 q 值或使用不同的参数组合
  │
  ├─ "The computed initial AR coefficients are not stationary"
  │  └─ 解决：检查数据平稳性，调整 p 和 d
  │
  └─ 收敛问题
     └─ 解决：增加最大迭代次数，尝试不同初始值
```

#### 4. **残差不是白噪声？**
```
诊断：
  ├─ Ljung-Box 检验 p-value < 0.05
  └─ 残差 ACF 图显示显著相关

改进：
  ├─ 增加 AR 或 MA 阶数
  ├─ 考虑季节性成分（SARIMA）
  ├─ 检查是否有遗漏的特征
  └─ 尝试非线性模型
```

#### 5. **机器学习模型过拟合？**
```
症状：
  └─ 训练集表现好，测试集差

解决：
  ├─ 减少特征数量（特征选择）
  ├─ 增加正则化强度
  ├─ 使用时间序列交叉验证
  ├─ 减小模型复杂度
  └─ 收集更多数据
```

---

## 最佳实践 (Best Practices)

### ✅ 推荐做法

1. **始终从简单开始**
   - 先建立基线模型（Naive, SMA）
   - 逐步增加复杂度
   - 每次改进都要验证

2. **充分的探索性分析（EDA）**
   - 可视化时间序列
   - 检查平稳性
   - 识别趋势和季节性
   - 检测异常值

3. **正确的数据分割**
   ```python
   # ✓ 正确：保持时间顺序
   train = data[:int(len(data)*0.8)]
   test = data[int(len(data)*0.8):]

   # ✗ 错误：随机分割
   from sklearn.model_selection import train_test_split
   train, test = train_test_split(data)  # 破坏时间依赖
   ```

4. **使用时间序列交叉验证**
   ```python
   from sklearn.model_selection import TimeSeriesSplit

   tscv = TimeSeriesSplit(n_splits=5)
   for train_idx, test_idx in tscv.split(data):
       # 训练和评估
   ```

5. **多模型集成**
   ```python
   # 简单平均
   ensemble = (pred_arima + pred_xgb + pred_rf) / 3

   # 加权平均（基于验证集性能）
   ensemble = 0.4*pred_arima + 0.35*pred_xgb + 0.25*pred_rf
   ```

6. **监控预测不确定性**
   - 计算预测区间
   - 使用置信区间
   - 考虑概率预测

7. **定期重新训练**
   - 设置重训练周期
   - 监控模型性能下降
   - 自动化重训练流程

### ❌ 避免陷阱

1. **数据泄露**
   - 不要使用未来信息
   - 特征工程要在分割后进行
   - 注意滚动窗口特征的边界

2. **忽略业务知识**
   - 纯统计模型可能忽略重要因素
   - 结合领域专家意见
   - 考虑外部事件影响

3. **过度依赖历史**
   - 市场/环境可能发生结构性变化
   - 考虑加入外部特征
   - 定期评估模型假设

---

## 进阶资源 (Advanced Resources)

### 📚 推荐阅读

1. **书籍**
   - "Forecasting: Principles and Practice" - Rob J Hyndman
   - "Time Series Analysis and Its Applications" - Robert H. Shumway
   - "Introduction to Time Series and Forecasting" - Peter J. Brockwell

2. **在线课程**
   - Coursera: "Practical Time Series Analysis"
   - Udacity: "Time Series Forecasting"

3. **Python 库**
   - `statsmodels`: 统计模型
   - `pmdarima`: Auto ARIMA
   - `prophet`: Facebook Prophet
   - `sktime`: Scikit-learn 风格的时间序列
   - `darts`: 现代时间序列库

### 🚀 进阶主题

- **多变量时间序列**: VAR, VARMA
- **深度学习方法**: LSTM, GRU, Transformer
- **概率预测**: 预测分布而非点估计
- **异常检测**: 时间序列中的异常识别
- **因果推断**: 时间序列因果分析
- **在线学习**: 实时模型更新

---

## 快速参考卡 (Quick Reference Card)

### 🔍 问题诊断流程

```
1. 数据是否平稳？
   NO → 差分 / 去趋势 / ARIMA
   YES → AR / MA / ARMA

2. 是否有季节性？
   YES → SARIMA / TES
   NO → ARIMA / DES

3. 数据量如何？
   小 (<500) → 统计模型
   大 (>1000) → 机器学习

4. 需要可解释性？
   YES → ARIMA / SARIMA
   NO → XGBoost / 深度学习

5. 实时性要求？
   高 → Naive / SMA / SES
   低 → 任意复杂模型
```

### ⚡ 快速命令

```python
# 平稳性检验
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(data)
print(f"p-value: {adf_result[1]}")

# ARIMA 快速建模
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(1,1,1))
fitted = model.fit()
forecast = fitted.forecast(steps=10)

# 指数平滑
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
fitted = model.fit()
forecast = fitted.forecast(steps=10)

# 评估
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
```

---

## 运行说明 (Running Instructions)

### 环境要求
```bash
# 安装依赖
pip install numpy pandas matplotlib
pip install statsmodels scikit-learn xgboost
pip install scipy
```

### 运行顺序
```bash
# 1. 基础概念
python 01_time_series_basics.py

# 2. ARIMA 模型
python 02_arima.py

# 3. 方法对比
python 03_forecasting_methods.py
```

### 输出文件
所有图表将保存到 `output/` 目录，文件名格式：
- `01_*.png` - 来自 01_time_series_basics.py
- `09_*.png` - 来自 02_arima.py
- `18_*.png` - 来自 03_forecasting_methods.py

---

## 联系与反馈 (Contact & Feedback)

如有问题或建议，欢迎：
- 提交 Issue
- 发起 Pull Request
- 联系项目维护者

---

**最后更新**: 2025-11-18

**许可证**: MIT License

**贡献者**: machineLearning-basics 团队
