"""
时间序列基础 (Time Series Basics)
====================================

学习目标 / Learning Objectives:
1. 理解时间序列的基本概念和组成部分
2. 掌握平稳性检验方法（ADF、KPSS）
3. 学习时间序列分解技术
4. 实践时间序列特征工程

Time Series Analysis:
- Understanding components (trend, seasonality, noise)
- Stationarity testing
- Time series decomposition
- Feature engineering for time series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.datasets import co2
import warnings
warnings.filterwarnings('ignore')

from utils import RANDOM_STATE, setup_chinese_fonts, create_subplots, save_figure, get_output_path

# 设置中文字体
setup_chinese_fonts()

print("=" * 80)
print("时间序列基础教程 (Time Series Basics Tutorial)".center(80))
print("=" * 80)

# ============================================================================
# 第一部分：时间序列基础概念
# Part 1: Time Series Basic Concepts
# ============================================================================

print("\n" + "=" * 80)
print("第一部分：时间序列基础概念".center(70))
print("Part 1: Time Series Basic Concepts".center(80))
print("=" * 80)

print("""
时间序列 (Time Series)：按时间顺序排列的数据点序列

时间序列的四大组成部分：
1. 趋势 (Trend)：数据的长期增长或下降模式
2. 季节性 (Seasonality)：固定周期的重复模式（如每年、每月、每周）
3. 周期性 (Cycle)：不固定周期的波动（通常由经济因素引起）
4. 随机性/噪声 (Noise/Residual)：不可预测的随机波动

平稳性 (Stationarity)：
- 统计特性（均值、方差）不随时间改变
- 大多数时间序列模型要求数据平稳
- 非平稳数据可通过差分、对数转换等方法处理
""")

# ============================================================================
# 第二部分：数据生成和加载
# Part 2: Data Generation and Loading
# ============================================================================

print("\n" + "=" * 80)
print("第二部分：生成模拟时间序列".center(70))
print("Part 2: Generate Simulated Time Series".center(80))
print("=" * 80)

# 设置随机种子以保证可重复性
np.random.seed(RANDOM_STATE)

# 生成时间索引（3年的日数据）
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
n = len(dates)

# 1. 趋势组件 (Trend Component)
# 线性趋势：逐渐增长
trend = np.linspace(10, 50, n)

# 2. 季节性组件 (Seasonal Component)
# 年度季节性（365天周期）和周度季节性（7天周期）
seasonal_yearly = 10 * np.sin(2 * np.pi * np.arange(n) / 365)
seasonal_weekly = 5 * np.sin(2 * np.pi * np.arange(n) / 7)
seasonal = seasonal_yearly + seasonal_weekly

# 3. 周期性组件 (Cyclical Component)
# 不规则周期（约500天）
cyclical = 8 * np.sin(2 * np.pi * np.arange(n) / 500)

# 4. 随机噪声 (Random Noise)
noise = np.random.normal(0, 3, n)

# 组合所有组件
time_series = trend + seasonal + cyclical + noise

# 创建 DataFrame
df = pd.DataFrame({
    'date': dates,
    'value': time_series,
    'trend': trend,
    'seasonal': seasonal,
    'cyclical': cyclical,
    'noise': noise
})
df.set_index('date', inplace=True)

print(f"✓ 生成了 {n} 个数据点的时间序列")
print(f"✓ 时间范围: {df.index[0]} 到 {df.index[-1]}")
print(f"\n前5行数据:")
print(df.head())

# ============================================================================
# 可视化1：时间序列组成部分
# Visualization 1: Time Series Components
# ============================================================================

fig, axes = create_subplots(5, 1, figsize=(16, 14))

# 原始时间序列
axes[0].plot(df.index, df['value'], color='#2E86AB', linewidth=1.5)
axes[0].set_title('完整时间序列 (Complete Time Series)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('数值 (Value)', fontsize=11)
axes[0].grid(True, alpha=0.3)

# 趋势
axes[1].plot(df.index, df['trend'], color='#A23B72', linewidth=2)
axes[1].set_title('趋势组件 (Trend Component)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('趋势 (Trend)', fontsize=11)
axes[1].grid(True, alpha=0.3)

# 季节性
axes[2].plot(df.index, df['seasonal'], color='#F18F01', linewidth=1.5)
axes[2].set_title('季节性组件 (Seasonal Component)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('季节性 (Seasonal)', fontsize=11)
axes[2].grid(True, alpha=0.3)

# 周期性
axes[3].plot(df.index, df['cyclical'], color='#C73E1D', linewidth=1.5)
axes[3].set_title('周期性组件 (Cyclical Component)', fontsize=14, fontweight='bold')
axes[3].set_ylabel('周期性 (Cyclical)', fontsize=11)
axes[3].grid(True, alpha=0.3)

# 噪声
axes[4].plot(df.index, df['noise'], color='#6A994E', linewidth=0.8, alpha=0.7)
axes[4].set_title('随机噪声 (Random Noise)', fontsize=14, fontweight='bold')
axes[4].set_ylabel('噪声 (Noise)', fontsize=11)
axes[4].set_xlabel('日期 (Date)', fontsize=11)
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('01_time_series_components.png'))
plt.close()

# ============================================================================
# 第三部分：平稳性检验
# Part 3: Stationarity Testing
# ============================================================================

print("\n" + "=" * 80)
print("第三部分：平稳性检验".center(70))
print("Part 3: Stationarity Testing".center(80))
print("=" * 80)

def check_stationarity(timeseries, name='Time Series'):
    """
    执行平稳性检验 (Perform Stationarity Tests)

    使用两种统计检验：
    1. ADF 检验 (Augmented Dickey-Fuller Test)：原假设为非平稳
    2. KPSS 检验 (Kwiatkowski-Phillips-Schmidt-Shin Test)：原假设为平稳

    Parameters:
    -----------
    timeseries : array-like
        时间序列数据
    name : str
        时间序列名称

    Returns:
    --------
    dict : 包含检验结果的字典
    """
    print(f"\n{'=' * 60}")
    print(f"平稳性检验: {name}")
    print(f"{'=' * 60}")

    # 1. ADF 检验 (ADF Test)
    # 原假设 H0: 时间序列是非平稳的（有单位根）
    # 如果 p-value < 0.05，拒绝原假设，认为序列平稳
    adf_result = adfuller(timeseries, autolag='AIC')

    print("\n【ADF 检验结果】")
    print(f"  ADF 统计量 (ADF Statistic): {adf_result[0]:.6f}")
    print(f"  p-值 (p-value): {adf_result[1]:.6f}")
    print(f"  使用的滞后阶数 (Lags Used): {adf_result[2]}")
    print(f"  观测数 (Observations): {adf_result[3]}")
    print("  临界值 (Critical Values):")
    for key, value in adf_result[4].items():
        print(f"    {key}: {value:.3f}")

    if adf_result[1] < 0.05:
        print("  ✓ 结论: 拒绝原假设，序列是平稳的 (Stationary)")
        adf_stationary = True
    else:
        print("  ✗ 结论: 不能拒绝原假设，序列是非平稳的 (Non-stationary)")
        adf_stationary = False

    # 2. KPSS 检验 (KPSS Test)
    # 原假设 H0: 时间序列是平稳的
    # 如果 p-value < 0.05，拒绝原假设，认为序列非平稳
    kpss_result = kpss(timeseries, regression='c', nlags='auto')

    print("\n【KPSS 检验结果】")
    print(f"  KPSS 统计量 (KPSS Statistic): {kpss_result[0]:.6f}")
    print(f"  p-值 (p-value): {kpss_result[1]:.6f}")
    print(f"  使用的滞后阶数 (Lags Used): {kpss_result[2]}")
    print("  临界值 (Critical Values):")
    for key, value in kpss_result[3].items():
        print(f"    {key}: {value:.3f}")

    if kpss_result[1] >= 0.05:
        print("  ✓ 结论: 不能拒绝原假设，序列是平稳的 (Stationary)")
        kpss_stationary = True
    else:
        print("  ✗ 结论: 拒绝原假设，序列是非平稳的 (Non-stationary)")
        kpss_stationary = False

    print("\n" + "=" * 60)
    if adf_stationary and kpss_stationary:
        print("✓ 综合结论: 序列是平稳的")
    elif not adf_stationary and not kpss_stationary:
        print("✗ 综合结论: 序列是非平稳的")
    else:
        print("⚠ 综合结论: 检验结果不一致，需要进一步分析")
    print("=" * 60)

    return {
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'adf_stationary': adf_stationary,
        'kpss_statistic': kpss_result[0],
        'kpss_pvalue': kpss_result[1],
        'kpss_stationary': kpss_stationary
    }

# 检验原始时间序列的平稳性
stationarity_original = check_stationarity(df['value'], '原始时间序列')

# ============================================================================
# 可视化2：ACF 和 PACF 图
# Visualization 2: ACF and PACF Plots
# ============================================================================

print("\n生成 ACF 和 PACF 图...")

fig, axes = create_subplots(2, 1, figsize=(14, 10))

# ACF 图 (Autocorrelation Function)
# 显示时间序列与其滞后版本的相关性
plot_acf(df['value'], lags=50, ax=axes[0], alpha=0.05)
axes[0].set_title('自相关函数 ACF (Autocorrelation Function)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('滞后阶数 (Lag)', fontsize=11)
axes[0].set_ylabel('相关系数 (Correlation)', fontsize=11)
axes[0].grid(True, alpha=0.3)

# PACF 图 (Partial Autocorrelation Function)
# 显示在移除中间滞后影响后的相关性
plot_pacf(df['value'], lags=50, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('偏自相关函数 PACF (Partial Autocorrelation Function)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('滞后阶数 (Lag)', fontsize=11)
axes[1].set_ylabel('相关系数 (Correlation)', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('02_acf_pacf_plots.png'))
plt.close()

# ============================================================================
# 差分使序列平稳
# Differencing to Make Series Stationary
# ============================================================================

print("\n" + "=" * 80)
print("使用差分处理非平稳序列".center(70))
print("Using Differencing to Handle Non-stationary Series".center(80))
print("=" * 80)

# 一阶差分 (First-order differencing)
df['diff_1'] = df['value'].diff()

# 二阶差分 (Second-order differencing)
df['diff_2'] = df['diff_1'].diff()

# 季节性差分 (Seasonal differencing, period=7 for weekly)
df['seasonal_diff'] = df['value'].diff(7)

print("✓ 完成差分处理")

# 检验差分后的平稳性
stationarity_diff1 = check_stationarity(df['diff_1'].dropna(), '一阶差分序列')

# ============================================================================
# 可视化3：差分前后对比
# Visualization 3: Before and After Differencing
# ============================================================================

fig, axes = create_subplots(4, 1, figsize=(16, 14))

# 原始序列
axes[0].plot(df.index, df['value'], color='#2E86AB', linewidth=1.5)
axes[0].set_title('原始时间序列 (Original Time Series)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('数值 (Value)', fontsize=11)
axes[0].grid(True, alpha=0.3)

# 一阶差分
axes[1].plot(df.index, df['diff_1'], color='#A23B72', linewidth=1.5)
axes[1].set_title('一阶差分 (First-order Differencing)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('差分值 (Diff)', fontsize=11)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1].grid(True, alpha=0.3)

# 二阶差分
axes[2].plot(df.index, df['diff_2'], color='#F18F01', linewidth=1.5)
axes[2].set_title('二阶差分 (Second-order Differencing)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('差分值 (Diff)', fontsize=11)
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[2].grid(True, alpha=0.3)

# 季节性差分
axes[3].plot(df.index, df['seasonal_diff'], color='#6A994E', linewidth=1.5)
axes[3].set_title('季节性差分 (Seasonal Differencing, period=7)', fontsize=14, fontweight='bold')
axes[3].set_ylabel('差分值 (Diff)', fontsize=11)
axes[3].set_xlabel('日期 (Date)', fontsize=11)
axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('03_differencing_comparison.png'))
plt.close()

# ============================================================================
# 第四部分：时间序列分解
# Part 4: Time Series Decomposition
# ============================================================================

print("\n" + "=" * 80)
print("第四部分：时间序列分解".center(70))
print("Part 4: Time Series Decomposition".center(80))
print("=" * 80)

print("""
时间序列分解方法：
1. 加法模型 (Additive Model): Y = T + S + R
   - 当季节性波动的幅度不随趋势变化时使用

2. 乘法模型 (Multiplicative Model): Y = T × S × R
   - 当季节性波动的幅度随趋势增长时使用

其中：
T = 趋势 (Trend)
S = 季节性 (Seasonality)
R = 残差 (Residual)
""")

# 加法分解 (Additive Decomposition)
decomposition_add = seasonal_decompose(df['value'], model='additive', period=365)

# 乘法分解 (Multiplicative Decomposition)
# 确保数据为正值
df_positive = df['value'] - df['value'].min() + 1
decomposition_mul = seasonal_decompose(df_positive, model='multiplicative', period=365)

print("✓ 完成时间序列分解")

# ============================================================================
# 可视化4：加法模型分解
# Visualization 4: Additive Decomposition
# ============================================================================

fig, axes = create_subplots(4, 1, figsize=(16, 14))

# 原始数据
axes[0].plot(df.index, df['value'], color='#2E86AB', linewidth=1.5)
axes[0].set_title('原始序列 (Original Series)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('数值 (Value)', fontsize=11)
axes[0].grid(True, alpha=0.3)

# 趋势
axes[1].plot(decomposition_add.trend.index, decomposition_add.trend, color='#A23B72', linewidth=2)
axes[1].set_title('趋势 (Trend)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('趋势 (Trend)', fontsize=11)
axes[1].grid(True, alpha=0.3)

# 季节性
axes[2].plot(decomposition_add.seasonal.index, decomposition_add.seasonal, color='#F18F01', linewidth=1.5)
axes[2].set_title('季节性 (Seasonality)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('季节性 (Seasonal)', fontsize=11)
axes[2].grid(True, alpha=0.3)

# 残差
axes[3].plot(decomposition_add.resid.index, decomposition_add.resid, color='#6A994E', linewidth=1, alpha=0.7)
axes[3].set_title('残差 (Residual)', fontsize=14, fontweight='bold')
axes[3].set_ylabel('残差 (Residual)', fontsize=11)
axes[3].set_xlabel('日期 (Date)', fontsize=11)
axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[3].grid(True, alpha=0.3)

fig.suptitle('加法模型分解 (Additive Model Decomposition)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
save_figure(fig, get_output_path('04_additive_decomposition.png'))
plt.close()

# ============================================================================
# 第五部分：特征工程
# Part 5: Feature Engineering
# ============================================================================

print("\n" + "=" * 80)
print("第五部分：时间序列特征工程".center(70))
print("Part 5: Time Series Feature Engineering".center(80))
print("=" * 80)

# 创建特征工程DataFrame
df_features = df[['value']].copy()

# 1. 滞后特征 (Lag Features)
print("\n1. 创建滞后特征...")
for i in [1, 2, 3, 7, 14, 30]:
    df_features[f'lag_{i}'] = df_features['value'].shift(i)

# 2. 滚动统计 (Rolling Statistics)
print("2. 创建滚动统计特征...")
windows = [7, 14, 30]
for window in windows:
    # 滚动均值
    df_features[f'rolling_mean_{window}'] = df_features['value'].rolling(window=window).mean()
    # 滚动标准差
    df_features[f'rolling_std_{window}'] = df_features['value'].rolling(window=window).std()
    # 滚动最小值
    df_features[f'rolling_min_{window}'] = df_features['value'].rolling(window=window).min()
    # 滚动最大值
    df_features[f'rolling_max_{window}'] = df_features['value'].rolling(window=window).max()

# 3. 时间特征 (Time Features)
print("3. 创建时间特征...")
df_features['year'] = df_features.index.year
df_features['month'] = df_features.index.month
df_features['day'] = df_features.index.day
df_features['dayofweek'] = df_features.index.dayofweek
df_features['dayofyear'] = df_features.index.dayofyear
df_features['quarter'] = df_features.index.quarter
df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)

# 4. 差分特征 (Difference Features)
print("4. 创建差分特征...")
df_features['diff_1'] = df_features['value'].diff(1)
df_features['diff_7'] = df_features['value'].diff(7)
df_features['diff_30'] = df_features['value'].diff(30)

print(f"\n✓ 特征工程完成，共创建 {len(df_features.columns)} 个特征")
print(f"\n特征列表:")
for i, col in enumerate(df_features.columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# 可视化5：滞后特征相关性
# Visualization 5: Lag Feature Correlation
# ============================================================================

fig, axes = create_subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

lag_periods = [1, 2, 3, 7, 14, 30]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']

for idx, (lag, color) in enumerate(zip(lag_periods, colors)):
    # 移除 NaN 值
    temp_df = df_features[['value', f'lag_{lag}']].dropna()

    axes[idx].scatter(temp_df[f'lag_{lag}'], temp_df['value'],
                     alpha=0.5, s=10, color=color)

    # 计算相关系数
    corr = temp_df['value'].corr(temp_df[f'lag_{lag}'])

    axes[idx].set_xlabel(f'滞后 {lag} 期 (Lag {lag})', fontsize=10)
    axes[idx].set_ylabel('当前值 (Current Value)', fontsize=10)
    axes[idx].set_title(f'Lag-{lag} 相关性 (Correlation: {corr:.3f})',
                       fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

    # 添加对角线
    lims = [
        np.min([axes[idx].get_xlim(), axes[idx].get_ylim()]),
        np.max([axes[idx].get_xlim(), axes[idx].get_ylim()]),
    ]
    axes[idx].plot(lims, lims, 'k--', alpha=0.3, zorder=0)

fig.suptitle('滞后特征与当前值的相关性 (Lag Features Correlation)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
save_figure(fig, get_output_path('05_lag_features_correlation.png'))
plt.close()

# ============================================================================
# 可视化6：滚动统计
# Visualization 6: Rolling Statistics
# ============================================================================

fig, axes = create_subplots(3, 1, figsize=(16, 12))

# 滚动均值
axes[0].plot(df_features.index, df_features['value'],
            label='原始值 (Original)', color='gray', alpha=0.5, linewidth=1)
axes[0].plot(df_features.index, df_features['rolling_mean_7'],
            label='7天均值 (7-day MA)', color='#2E86AB', linewidth=2)
axes[0].plot(df_features.index, df_features['rolling_mean_30'],
            label='30天均值 (30-day MA)', color='#A23B72', linewidth=2)
axes[0].set_title('滚动均值 (Rolling Mean)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('数值 (Value)', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 滚动标准差
axes[1].plot(df_features.index, df_features['rolling_std_7'],
            label='7天标准差 (7-day Std)', color='#F18F01', linewidth=2)
axes[1].plot(df_features.index, df_features['rolling_std_30'],
            label='30天标准差 (30-day Std)', color='#C73E1D', linewidth=2)
axes[1].set_title('滚动标准差 (Rolling Standard Deviation)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('标准差 (Std)', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# 滚动最大值和最小值
axes[2].plot(df_features.index, df_features['value'],
            label='原始值 (Original)', color='gray', alpha=0.5, linewidth=1)
axes[2].plot(df_features.index, df_features['rolling_max_30'],
            label='30天最大值 (30-day Max)', color='#6A994E', linewidth=2, linestyle='--')
axes[2].plot(df_features.index, df_features['rolling_min_30'],
            label='30天最小值 (30-day Min)', color='#BC4B51', linewidth=2, linestyle='--')
axes[2].fill_between(df_features.index,
                     df_features['rolling_min_30'],
                     df_features['rolling_max_30'],
                     alpha=0.2, color='#6A994E')
axes[2].set_title('滚动最大/最小值 (Rolling Max/Min)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('数值 (Value)', fontsize=11)
axes[2].set_xlabel('日期 (Date)', fontsize=11)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('06_rolling_statistics.png'))
plt.close()

# ============================================================================
# 可视化7：时间特征分布
# Visualization 7: Time Feature Distribution
# ============================================================================

fig, axes = create_subplots(2, 3, figsize=(18, 10))

# 按月份统计
monthly_avg = df_features.groupby('month')['value'].mean()
axes[0, 0].bar(monthly_avg.index, monthly_avg.values, color='#2E86AB', alpha=0.7)
axes[0, 0].set_title('月度平均值 (Monthly Average)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('月份 (Month)', fontsize=10)
axes[0, 0].set_ylabel('平均值 (Avg Value)', fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 按星期统计
weekly_avg = df_features.groupby('dayofweek')['value'].mean()
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[0, 1].bar(range(7), weekly_avg.values, color='#A23B72', alpha=0.7)
axes[0, 1].set_xticks(range(7))
axes[0, 1].set_xticklabels(day_names)
axes[0, 1].set_title('星期平均值 (Day of Week Average)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('星期 (Day)', fontsize=10)
axes[0, 1].set_ylabel('平均值 (Avg Value)', fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 按季度统计
quarterly_avg = df_features.groupby('quarter')['value'].mean()
axes[0, 2].bar(quarterly_avg.index, quarterly_avg.values, color='#F18F01', alpha=0.7)
axes[0, 2].set_title('季度平均值 (Quarterly Average)', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('季度 (Quarter)', fontsize=10)
axes[0, 2].set_ylabel('平均值 (Avg Value)', fontsize=10)
axes[0, 2].grid(True, alpha=0.3, axis='y')

# 工作日 vs 周末
weekend_avg = df_features.groupby('is_weekend')['value'].mean()
axes[1, 0].bar(['工作日\n(Weekday)', '周末\n(Weekend)'], weekend_avg.values,
              color=['#C73E1D', '#6A994E'], alpha=0.7)
axes[1, 0].set_title('工作日 vs 周末 (Weekday vs Weekend)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('平均值 (Avg Value)', fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 月度箱线图
df_features.boxplot(column='value', by='month', ax=axes[1, 1])
axes[1, 1].set_title('月度分布 (Monthly Distribution)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('月份 (Month)', fontsize=10)
axes[1, 1].set_ylabel('数值 (Value)', fontsize=10)
axes[1, 1].get_figure().suptitle('')  # 移除自动标题

# 星期箱线图
df_features.boxplot(column='value', by='dayofweek', ax=axes[1, 2])
axes[1, 2].set_title('星期分布 (Day of Week Distribution)', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('星期 (Day of Week)', fontsize=10)
axes[1, 2].set_ylabel('数值 (Value)', fontsize=10)
axes[1, 2].set_xticklabels(day_names)
axes[1, 2].get_figure().suptitle('')  # 移除自动标题

fig.suptitle('时间特征分布分析 (Time Feature Distribution Analysis)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
save_figure(fig, get_output_path('07_time_features_distribution.png'))
plt.close()

# ============================================================================
# 可视化8：特征重要性（基于相关性）
# Visualization 8: Feature Importance (Based on Correlation)
# ============================================================================

# 计算所有数值特征与目标值的相关性
numeric_features = df_features.select_dtypes(include=[np.number]).columns
correlations = df_features[numeric_features].corr()['value'].drop('value').abs().sort_values(ascending=False)

# 选取前20个最相关的特征
top_20_features = correlations.head(20)

fig, ax = create_subplots(1, 1, figsize=(12, 10))

colors = plt.cm.RdYlGn(top_20_features.values)
bars = ax.barh(range(len(top_20_features)), top_20_features.values, color=colors, alpha=0.8)

ax.set_yticks(range(len(top_20_features)))
ax.set_yticklabels(top_20_features.index, fontsize=10)
ax.set_xlabel('绝对相关系数 (Absolute Correlation)', fontsize=11)
ax.set_title('Top 20 特征重要性（基于相关性）\nTop 20 Feature Importance (Based on Correlation)',
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# 在柱子上添加数值
for i, (bar, val) in enumerate(zip(bars, top_20_features.values)):
    ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
save_figure(fig, get_output_path('08_feature_importance.png'))
plt.close()

# ============================================================================
# 总结和最佳实践
# Summary and Best Practices
# ============================================================================

print("\n" + "=" * 80)
print("总结和最佳实践".center(70))
print("Summary and Best Practices".center(80))
print("=" * 80)

print("""
✓ 时间序列分析关键要点：

1. 【理解数据】
   - 可视化是第一步，观察趋势、季节性和异常值
   - 了解数据的业务背景和收集方式

2. 【平稳性检验】
   - 使用 ADF 和 KPSS 检验互相验证
   - 大多数模型要求数据平稳
   - 非平稳数据可通过差分、对数转换等方法处理

3. 【时间序列分解】
   - 加法模型：季节性波动幅度固定
   - 乘法模型：季节性波动幅度随趋势变化
   - 分解有助于理解数据结构

4. 【特征工程】
   - 滞后特征：捕捉时间依赖性
   - 滚动统计：平滑噪声，提取趋势
   - 时间特征：捕捉周期性模式
   - 差分特征：处理非平稳性

5. 【ACF 和 PACF】
   - ACF：识别 MA 阶数
   - PACF：识别 AR 阶数
   - 帮助选择 ARIMA 模型参数

6. 【常见陷阱】
   - 忽略数据的季节性
   - 过度差分导致信息丢失
   - 使用未来信息（数据泄露）
   - 忽略异常值的影响

7. 【实用建议】
   - 始终进行探索性数据分析（EDA）
   - 尝试多种模型并比较
   - 使用交叉验证评估模型
   - 关注业务指标，不只是统计指标
""")

print("\n" + "=" * 80)
print("✓ 时间序列基础教程完成！")
print("✓ Time Series Basics Tutorial Completed!")
print("=" * 80)
print(f"\n生成的可视化图表:")
print(f"  1. 01_time_series_components.png - 时间序列组成部分")
print(f"  2. 02_acf_pacf_plots.png - ACF 和 PACF 图")
print(f"  3. 03_differencing_comparison.png - 差分前后对比")
print(f"  4. 04_additive_decomposition.png - 加法模型分解")
print(f"  5. 05_lag_features_correlation.png - 滞后特征相关性")
print(f"  6. 06_rolling_statistics.png - 滚动统计")
print(f"  7. 07_time_features_distribution.png - 时间特征分布")
print(f"  8. 08_feature_importance.png - 特征重要性")
print("=" * 80)
