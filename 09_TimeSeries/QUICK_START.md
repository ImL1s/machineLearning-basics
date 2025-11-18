# 时间序列分析 - 快速入门指南
# Time Series Analysis - Quick Start Guide

## 🚀 5分钟快速上手

### 第一步：安装依赖

```bash
# 进入项目目录
cd /home/user/machineLearning-basics

# 安装基础依赖
pip install numpy pandas matplotlib scipy

# 安装统计模型库
pip install statsmodels

# 安装机器学习库
pip install scikit-learn xgboost
```

### 第二步：运行示例

```bash
# 按顺序运行三个脚本
cd 09_TimeSeries

# 1. 时间序列基础（约 30-60 秒）
python 01_time_series_basics.py

# 2. ARIMA 模型（约 2-3 分钟，包含网格搜索）
python 02_arima.py

# 3. 多种方法对比（约 1-2 分钟）
python 03_forecasting_methods.py
```

### 第三步：查看结果

```bash
# 所有图表保存在 output/ 目录
ls -lh ../output/*.png

# 查看评估结果
cat ../output/forecasting_methods_results.csv
```

---

## 📖 学习路线图

### 🎯 初学者路线（1-2 小时）

```
第1步：理解基础概念（30分钟）
├─ 运行 01_time_series_basics.py
├─ 阅读输出，理解四大组成部分
├─ 查看图表 01-08
└─ 学习平稳性检验

第2步：掌握 ARIMA（45分钟）
├─ 运行 02_arima.py
├─ 理解 p, d, q 参数含义
├─ 学习如何读 ACF/PACF 图
└─ 查看残差诊断

第3步：对比多种方法（30分钟）
├─ 运行 03_forecasting_methods.py
├─ 对比不同方法的优缺点
└─ 理解如何选择模型
```

### 🎓 进阶用户路线（3-4 小时）

```
第1步：深入代码（1小时）
├─ 阅读源代码，理解实现细节
├─ 修改参数，观察结果变化
└─ 尝试自己的数据

第2步：参数调优（1.5小时）
├─ 网格搜索 ARIMA 参数
├─ 调整机器学习超参数
└─ 对比不同配置的性能

第3步：实战应用（1.5小时）
├─ 准备自己的时间序列数据
├─ 应用学到的技术
└─ 生成预测报告
```

---

## 🎨 快速示例代码

### 示例1：最简单的预测

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 生成示例数据
dates = pd.date_range('2020-01-01', periods=100, freq='D')
data = pd.Series(np.random.randn(100).cumsum(), index=dates)

# 分割数据
train = data[:80]
test = data[80:]

# 训练模型（三指数平滑）
model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7)
fitted = model.fit()

# 预测
forecast = fitted.forecast(steps=20)

# 评估
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test, forecast)
print(f'MAE: {mae:.2f}')
```

### 示例2：ARIMA 建模

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# 平稳性检验
def check_stationarity(data):
    result = adfuller(data)
    return result[1] < 0.05  # p-value < 0.05 表示平稳

# 如果非平稳，进行差分
if not check_stationarity(train):
    train_diff = train.diff().dropna()
    d = 1
else:
    train_diff = train
    d = 0

# 训练 ARIMA
model = ARIMA(train, order=(1, d, 1))
fitted = model.fit()

# 预测
forecast = fitted.forecast(steps=20)
```

### 示例3：机器学习方法

```python
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# 创建特征
def create_features(data):
    df = pd.DataFrame({'value': data.values})

    # 滞后特征
    for lag in [1, 2, 3, 7]:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    # 滚动均值
    df['rolling_mean_7'] = df['value'].rolling(7).mean()

    return df.dropna()

# 准备数据
train_features = create_features(train)
X_train = train_features.drop('value', axis=1)
y_train = train_features['value']

# 训练模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测（需要递归预测）
# 详见 03_forecasting_methods.py
```

---

## 🔧 常用代码片段

### 1. 数据加载和准备

```python
import pandas as pd

# 从 CSV 加载
data = pd.read_csv('data.csv', index_col='date', parse_dates=True)

# 重采样
data_monthly = data.resample('M').mean()

# 处理缺失值
data_filled = data.fillna(method='ffill')  # 前向填充
data_interpolated = data.interpolate()     # 线性插值
```

### 2. 平稳性检验

```python
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity_test(timeseries):
    # ADF 检验
    adf_result = adfuller(timeseries)
    print(f'ADF p-value: {adf_result[1]:.4f}')

    # KPSS 检验
    kpss_result = kpss(timeseries)
    print(f'KPSS p-value: {kpss_result[1]:.4f}')

    return adf_result[1] < 0.05  # 平稳
```

### 3. ACF/PACF 可视化

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data, lags=40, ax=axes[0])
plot_pacf(data, lags=40, ax=axes[1])
plt.show()
```

### 4. 模型评估

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f'MAE:  {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAPE: {mape:.2f}%')

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
```

### 5. 网格搜索 ARIMA

```python
import itertools
from statsmodels.tsa.arima.model import ARIMA

def grid_search_arima(data, p_range, d_range, q_range):
    best_aic = float('inf')
    best_order = None

    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(data, order=(p, d, q))
            fitted = model.fit()

            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
        except:
            continue

    return best_order, best_aic

# 使用
best_order, best_aic = grid_search_arima(
    train,
    p_range=range(0, 4),
    d_range=range(0, 3),
    q_range=range(0, 4)
)
print(f'Best ARIMA{best_order}, AIC={best_aic:.2f}')
```

---

## 📊 决策流程图

### 选择预测方法的简化流程

```
开始预测任务
    │
    ▼
有多少数据？
    │
    ├─ 少于 100 个点
    │   └─→ 使用简单方法
    │       ├─ 有季节性？→ TES (三指数平滑)
    │       └─ 无季节性？→ DES (双指数平滑)
    │
    └─ 超过 100 个点
        │
        ▼
    需要可解释性？
        │
        ├─ 是 →─→ 统计方法
        │         ├─ 有季节性？→ SARIMA
        │         └─ 无季节性？→ ARIMA
        │
        └─ 否 →─→ 机器学习
                  ├─ 追求速度？→ Random Forest
                  └─ 追求精度？→ XGBoost
```

---

## ⚡ 性能优化技巧

### 1. 加速 ARIMA 训练

```python
# 使用较小的参数范围
p_range = range(0, 3)  # 而不是 range(0, 6)
d_range = range(0, 2)  # 通常 d 不超过 2
q_range = range(0, 3)

# 并行化（如果可能）
# 某些实现支持并行网格搜索
```

### 2. 减少特征数量

```python
# 只使用最重要的滞后特征
important_lags = [1, 2, 7, 30]  # 而不是 range(1, 50)

# 特征选择
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=15)
X_selected = selector.fit_transform(X_train, y_train)
```

### 3. 数据采样

```python
# 对于超大数据集，考虑采样
if len(data) > 10000:
    data_sampled = data.resample('W').mean()  # 周采样
```

---

## 🐛 常见错误及解决

### 错误1：内存不足

```python
# 问题：数据集太大
# 解决：
# 1. 减少特征数量
# 2. 使用数据采样
# 3. 分批处理
```

### 错误2：ARIMA 收敛失败

```python
# 问题：模型无法收敛
# 解决：
model = ARIMA(data, order=(p, d, q))
fitted = model.fit(
    method='innovations_mle',  # 尝试不同的优化方法
    maxiter=500,               # 增加最大迭代次数
    start_params=None          # 或提供初始参数
)
```

### 错误3：预测结果全是 NaN

```python
# 问题：差分过度或数据质量问题
# 解决：
# 1. 检查原始数据
# 2. 减少差分次数
# 3. 处理缺失值
data_clean = data.fillna(method='ffill').dropna()
```

---

## 📈 实战检查清单

在开始时间序列项目前，检查：

### 数据准备 ✓
- [ ] 数据已按时间排序
- [ ] 时间索引格式正确
- [ ] 缺失值已处理
- [ ] 异常值已识别
- [ ] 数据频率一致

### 探索分析 ✓
- [ ] 绘制原始时间序列图
- [ ] 检查平稳性
- [ ] 识别趋势和季节性
- [ ] 查看 ACF/PACF 图
- [ ] 统计描述性分析

### 模型选择 ✓
- [ ] 定义评估指标
- [ ] 选择候选模型
- [ ] 正确分割训练/测试集
- [ ] 设置基线模型
- [ ] 考虑业务约束

### 模型训练 ✓
- [ ] 参数调优
- [ ] 交叉验证
- [ ] 残差诊断
- [ ] 过拟合检查
- [ ] 记录训练时间

### 模型评估 ✓
- [ ] 计算评估指标
- [ ] 可视化预测结果
- [ ] 分析误差分布
- [ ] 对比多个模型
- [ ] 业务验证

### 部署准备 ✓
- [ ] 模型保存
- [ ] 文档完善
- [ ] 代码优化
- [ ] 监控方案
- [ ] 重训练计划

---

## 🎓 学习资源

### 📚 推荐书籍
1. **"Forecasting: Principles and Practice"** - Hyndman & Athanasopoulos
   - 免费在线：https://otexts.com/fpp3/
   - 最佳时间序列入门书

2. **"Time Series Analysis and Its Applications"** - Shumway & Stoffer
   - 理论深入，适合进阶

3. **"Practical Time Series Analysis"** - Nielsen
   - 实战导向，代码示例丰富

### 🎥 在线课程
- Coursera: "Sequences, Time Series and Prediction"
- Udacity: "Time Series Forecasting"
- DataCamp: "Time Series with Python" 路径

### 🔗 有用链接
- Statsmodels 文档: https://www.statsmodels.org/
- XGBoost 文档: https://xgboost.readthedocs.io/
- Sklearn 时间序列: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble

---

## 💡 小贴士

### 🎯 提高预测精度
1. **特征工程是关键** - 投入 80% 时间在特征上
2. **集成多个模型** - 结合统计和机器学习方法
3. **外部数据** - 考虑相关的外部因素
4. **定期重训练** - 适应数据分布变化
5. **业务知识** - 结合领域专家意见

### ⚠️ 避免常见陷阱
1. **不要过度拟合** - 简单模型往往更好
2. **注意数据泄露** - 严格的时间分割
3. **考虑季节性** - 不要忽视周期模式
4. **检查假设** - ARIMA 有严格的假设条件
5. **验证预测** - 不要盲目信任模型

---

## 🆘 获取帮助

### 遇到问题？

1. **查看文档**
   - README.md - 详细教程
   - VISUALIZATION_INDEX.md - 图表说明
   - 本文件 - 快速参考

2. **检查代码**
   - 阅读源代码注释
   - 运行示例代码
   - 修改参数实验

3. **社区支持**
   - Stack Overflow: 搜索 [time-series] [python]
   - GitHub Issues: 提交问题
   - 统计论坛: Cross Validated

---

## 🚀 下一步

完成本模块后，你可以：

1. **深入学习**
   - 学习 Prophet（Facebook 的时间序列库）
   - 探索深度学习方法（LSTM, GRU）
   - 研究多变量时间序列（VAR）

2. **实战项目**
   - 股票价格预测
   - 销售预测系统
   - 网站流量预测
   - 能源需求预测

3. **高级主题**
   - 异常检测
   - 因果推断
   - 在线学习
   - 概率预测

---

**祝你学习愉快！** 🎉

**Happy Learning!** 🚀

---

*最后更新: 2025-11-18*
*版本: 1.0*
