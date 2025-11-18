"""
ARIMA 模型 (ARIMA Model)
========================

学习目标 / Learning Objectives:
1. 理解 ARIMA 模型的组成（AR、I、MA）
2. 掌握 ARIMA 参数选择方法
3. 学习模型诊断和残差分析
4. 实践单步和多步预测
5. 了解 SARIMA（季节性 ARIMA）

ARIMA (AutoRegressive Integrated Moving Average):
- AR(p): 自回归，使用过去 p 个观测值
- I(d): 差分，进行 d 次差分使序列平稳
- MA(q): 移动平均，使用过去 q 个预测误差
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.datasets import co2
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import warnings
warnings.filterwarnings('ignore')

from utils import RANDOM_STATE, setup_chinese_fonts, create_subplots, save_figure, get_output_path

# 设置中文字体
setup_chinese_fonts()

print("=" * 80)
print("ARIMA 模型教程 (ARIMA Model Tutorial)".center(80))
print("=" * 80)

# ============================================================================
# 第一部分：ARIMA 原理
# Part 1: ARIMA Principles
# ============================================================================

print("\n" + "=" * 80)
print("第一部分：ARIMA 模型原理".center(70))
print("Part 1: ARIMA Model Principles".center(80))
print("=" * 80)

print("""
ARIMA(p, d, q) 模型详解：

1. 【AR - 自回归 (AutoRegressive)】
   - 参数 p：自回归阶数
   - 使用过去 p 个时间点的值来预测当前值
   - 公式: y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t
   - PACF 图帮助确定 p 值

2. 【I - 差分 (Integrated)】
   - 参数 d：差分次数
   - 将非平稳序列转换为平稳序列
   - 通常 d = 0, 1, 或 2
   - ADF 检验帮助确定 d 值

3. 【MA - 移动平均 (Moving Average)】
   - 参数 q：移动平均阶数
   - 使用过去 q 个预测误差来预测当前值
   - 公式: y_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θₑε_{t-q}
   - ACF 图帮助确定 q 值

4. 【参数选择指南】
   - ACF 缓慢衰减 → AR 模型
   - PACF 缓慢衰减 → MA 模型
   - 两者都缓慢衰减 → ARMA 模型
   - 使用 AIC/BIC 准则选择最优参数
""")

# ============================================================================
# 第二部分：数据准备
# Part 2: Data Preparation
# ============================================================================

print("\n" + "=" * 80)
print("第二部分：数据加载和探索".center(70))
print("Part 2: Data Loading and Exploration".center(80))
print("=" * 80)

# 加载 CO2 数据集（大气中 CO2 浓度）
# 这是一个经典的时间序列数据集，具有明显的趋势和季节性
data = co2.load_pandas().data
data = data.fillna(data.interpolate())  # 插值填补缺失值

# 重采样为月度数据
data = data.resample('M').mean()

print(f"✓ 数据加载完成")
print(f"  数据点数: {len(data)}")
print(f"  时间范围: {data.index[0]} 到 {data.index[-1]}")
print(f"  数据类型: 大气 CO2 浓度 (ppm)")
print(f"\n前5行数据:")
print(data.head())
print(f"\n基本统计:")
print(data.describe())

# ============================================================================
# 可视化1：原始数据和基本分析
# Visualization 1: Original Data and Basic Analysis
# ============================================================================

fig, axes = create_subplots(2, 2, figsize=(16, 10))

# 原始时间序列
axes[0, 0].plot(data.index, data.values, color='#2E86AB', linewidth=1.5)
axes[0, 0].set_title('大气 CO2 浓度时间序列\nAtmospheric CO2 Concentration',
                     fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('年份 (Year)', fontsize=10)
axes[0, 0].set_ylabel('CO2 浓度 (ppm)', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 直方图
axes[0, 1].hist(data.values, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('CO2 浓度分布\nCO2 Concentration Distribution',
                     fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('CO2 浓度 (ppm)', fontsize=10)
axes[0, 1].set_ylabel('频数 (Frequency)', fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 年度箱线图
data_with_year = pd.DataFrame({'co2': data.values.flatten(), 'year': data.index.year})
yearly_data = [data_with_year[data_with_year['year'] == year]['co2'].values
               for year in range(1958, 2002, 5)]
axes[1, 0].boxplot(yearly_data, labels=range(1958, 2002, 5))
axes[1, 0].set_title('年度 CO2 浓度分布\nYearly CO2 Distribution',
                     fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('年份 (Year)', fontsize=10)
axes[1, 0].set_ylabel('CO2 浓度 (ppm)', fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 月度季节性模式
monthly_avg = data.groupby(data.index.month).mean()
axes[1, 1].bar(range(1, 13), monthly_avg.values.flatten(),
              color='#F18F01', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('月度平均 CO2 浓度\nMonthly Average CO2',
                     fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('月份 (Month)', fontsize=10)
axes[1, 1].set_ylabel('CO2 浓度 (ppm)', fontsize=10)
axes[1, 1].set_xticks(range(1, 13))
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_figure(fig, get_output_path('09_co2_data_exploration.png'))
plt.close()

# ============================================================================
# 平稳性检验
# Stationarity Testing
# ============================================================================

print("\n" + "=" * 80)
print("平稳性检验".center(70))
print("Stationarity Testing".center(80))
print("=" * 80)

def adf_test(timeseries, name=''):
    """ADF 检验"""
    result = adfuller(timeseries, autolag='AIC')
    print(f'\n【ADF 检验结果 - {name}】')
    print(f'  ADF 统计量: {result[0]:.6f}')
    print(f'  p-值: {result[1]:.6f}')
    print(f'  临界值:')
    for key, value in result[4].items():
        print(f'    {key}: {value:.3f}')
    if result[1] <= 0.05:
        print("  ✓ 序列是平稳的 (p < 0.05)")
        return True
    else:
        print("  ✗ 序列是非平稳的 (p >= 0.05)")
        return False

# 检验原始数据
is_stationary = adf_test(data.values.flatten(), '原始数据')

# 一阶差分
data_diff1 = data.diff().dropna()
is_stationary_diff1 = adf_test(data_diff1.values.flatten(), '一阶差分')

# ============================================================================
# 可视化2：ACF 和 PACF 图（原始数据）
# Visualization 2: ACF and PACF Plots (Original Data)
# ============================================================================

fig, axes = create_subplots(2, 1, figsize=(14, 10))

# ACF 图
plot_acf(data.values.flatten(), lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title('原始数据的自相关函数 (ACF)\nAutocorrelation Function of Original Data',
                 fontsize=13, fontweight='bold')
axes[0].set_xlabel('滞后阶数 (Lag)', fontsize=10)
axes[0].set_ylabel('相关系数 (Correlation)', fontsize=10)
axes[0].grid(True, alpha=0.3)

# PACF 图
plot_pacf(data.values.flatten(), lags=40, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('原始数据的偏自相关函数 (PACF)\nPartial Autocorrelation Function of Original Data',
                 fontsize=13, fontweight='bold')
axes[1].set_xlabel('滞后阶数 (Lag)', fontsize=10)
axes[1].set_ylabel('相关系数 (Correlation)', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('10_acf_pacf_original.png'))
plt.close()

# ============================================================================
# 可视化3：差分后的 ACF 和 PACF
# Visualization 3: ACF and PACF after Differencing
# ============================================================================

fig, axes = create_subplots(2, 1, figsize=(14, 10))

# ACF 图
plot_acf(data_diff1.values.flatten(), lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title('一阶差分的自相关函数 (ACF)\nACF of First-order Differenced Data',
                 fontsize=13, fontweight='bold')
axes[0].set_xlabel('滞后阶数 (Lag)', fontsize=10)
axes[0].set_ylabel('相关系数 (Correlation)', fontsize=10)
axes[0].grid(True, alpha=0.3)

# PACF 图
plot_pacf(data_diff1.values.flatten(), lags=40, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('一阶差分的偏自相关函数 (PACF)\nPACF of First-order Differenced Data',
                 fontsize=13, fontweight='bold')
axes[1].set_xlabel('滞后阶数 (Lag)', fontsize=10)
axes[1].set_ylabel('相关系数 (Correlation)', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('11_acf_pacf_differenced.png'))
plt.close()

# ============================================================================
# 分割训练集和测试集
# Split Train and Test Sets
# ============================================================================

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

print(f"\n✓ 数据分割完成")
print(f"  训练集大小: {len(train)} ({len(train)/len(data)*100:.1f}%)")
print(f"  测试集大小: {len(test)} ({len(test)/len(data)*100:.1f}%)")
print(f"  训练集范围: {train.index[0]} 到 {train.index[-1]}")
print(f"  测试集范围: {test.index[0]} 到 {test.index[-1]}")

# ============================================================================
# 第三部分：模型选择（网格搜索）
# Part 3: Model Selection (Grid Search)
# ============================================================================

print("\n" + "=" * 80)
print("第三部分：ARIMA 参数选择（网格搜索）".center(70))
print("Part 3: ARIMA Parameter Selection (Grid Search)".center(80))
print("=" * 80)

print("\n正在进行网格搜索以找到最优 ARIMA 参数...")
print("这可能需要几分钟时间...")

# 定义参数范围
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

# 存储结果
results = []

# 网格搜索
for p, d, q in itertools.product(p_values, d_values, q_values):
    try:
        model = ARIMA(train, order=(p, d, q))
        fitted_model = model.fit()
        aic = fitted_model.aic
        bic = fitted_model.bic
        results.append({
            'order': (p, d, q),
            'AIC': aic,
            'BIC': bic
        })
        print(f"  ARIMA{(p, d, q)} - AIC: {aic:.2f}, BIC: {bic:.2f}")
    except Exception as e:
        continue

# 转换为 DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AIC')

print(f"\n✓ 网格搜索完成")
print(f"\nTop 5 模型（按 AIC 排序）:")
print(results_df.head())

# 选择最优模型
best_order = results_df.iloc[0]['order']
print(f"\n✓ 最优 ARIMA 参数: {best_order}")

# ============================================================================
# 可视化4：AIC 和 BIC 对比
# Visualization 4: AIC and BIC Comparison
# ============================================================================

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# AIC
results_df_sorted_aic = results_df.sort_values('AIC').head(10)
colors_aic = plt.cm.viridis(np.linspace(0, 1, len(results_df_sorted_aic)))
bars1 = axes[0].barh(range(len(results_df_sorted_aic)),
                     results_df_sorted_aic['AIC'].values,
                     color=colors_aic, alpha=0.8)
axes[0].set_yticks(range(len(results_df_sorted_aic)))
axes[0].set_yticklabels([str(order) for order in results_df_sorted_aic['order'].values])
axes[0].set_xlabel('AIC 值 (AIC Value)', fontsize=11)
axes[0].set_title('Top 10 模型 - AIC 准则\nTop 10 Models - AIC Criterion',
                 fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].invert_yaxis()

# 标注数值
for i, (bar, val) in enumerate(zip(bars1, results_df_sorted_aic['AIC'].values)):
    axes[0].text(val + 5, i, f'{val:.1f}', va='center', fontsize=9)

# BIC
results_df_sorted_bic = results_df.sort_values('BIC').head(10)
colors_bic = plt.cm.plasma(np.linspace(0, 1, len(results_df_sorted_bic)))
bars2 = axes[1].barh(range(len(results_df_sorted_bic)),
                     results_df_sorted_bic['BIC'].values,
                     color=colors_bic, alpha=0.8)
axes[1].set_yticks(range(len(results_df_sorted_bic)))
axes[1].set_yticklabels([str(order) for order in results_df_sorted_bic['order'].values])
axes[1].set_xlabel('BIC 值 (BIC Value)', fontsize=11)
axes[1].set_title('Top 10 模型 - BIC 准则\nTop 10 Models - BIC Criterion',
                 fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].invert_yaxis()

# 标注数值
for i, (bar, val) in enumerate(zip(bars2, results_df_sorted_bic['BIC'].values)):
    axes[1].text(val + 5, i, f'{val:.1f}', va='center', fontsize=9)

plt.tight_layout()
save_figure(fig, get_output_path('12_aic_bic_comparison.png'))
plt.close()

# ============================================================================
# 第四部分：模型训练和诊断
# Part 4: Model Training and Diagnostics
# ============================================================================

print("\n" + "=" * 80)
print("第四部分：模型训练和诊断".center(70))
print("Part 4: Model Training and Diagnostics".center(80))
print("=" * 80)

# 使用最优参数训练模型
model = ARIMA(train, order=best_order)
fitted_model = model.fit()

print(f"\n✓ ARIMA{best_order} 模型训练完成")
print("\n模型摘要:")
print(fitted_model.summary())

# 残差分析
residuals = fitted_model.resid

print(f"\n残差统计:")
print(f"  均值: {residuals.mean():.6f}")
print(f"  标准差: {residuals.std():.6f}")
print(f"  偏度: {stats.skew(residuals):.6f}")
print(f"  峰度: {stats.kurtosis(residuals):.6f}")

# Ljung-Box 检验（检验残差是否为白噪声）
lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
print(f"\nLjung-Box 检验（残差白噪声检验）:")
print(lb_test)

# ============================================================================
# 可视化5：残差诊断
# Visualization 5: Residual Diagnostics
# ============================================================================

fig, axes = create_subplots(2, 2, figsize=(16, 10))

# 残差时序图
axes[0, 0].plot(residuals.index, residuals.values, color='#2E86AB', linewidth=1)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_title('残差时序图\nResiduals over Time', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('时间 (Time)', fontsize=10)
axes[0, 0].set_ylabel('残差 (Residuals)', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 残差直方图
axes[0, 1].hist(residuals, bins=30, color='#A23B72', alpha=0.7, edgecolor='black', density=True)
# 添加正态分布曲线
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='正态分布')
axes[0, 1].set_title('残差分布\nResidual Distribution', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('残差 (Residuals)', fontsize=10)
axes[0, 1].set_ylabel('密度 (Density)', fontsize=10)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Q-Q 图
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q 图\nQ-Q Plot', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 残差 ACF
plot_acf(residuals, lags=40, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('残差自相关图\nResidual ACF', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('滞后阶数 (Lag)', fontsize=10)
axes[1, 1].set_ylabel('相关系数 (Correlation)', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(f'ARIMA{best_order} 残差诊断\nARIMA{best_order} Residual Diagnostics',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
save_figure(fig, get_output_path('13_residual_diagnostics.png'))
plt.close()

# ============================================================================
# 第五部分：预测和评估
# Part 5: Forecasting and Evaluation
# ============================================================================

print("\n" + "=" * 80)
print("第五部分：预测和评估".center(70))
print("Part 5: Forecasting and Evaluation".center(80))
print("=" * 80)

# 进行预测
forecast_steps = len(test)
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()

# 评估指标
def evaluate_forecast(actual, predicted, model_name='Model'):
    """计算评估指标"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"\n【{model_name} 评估结果】")
    print(f"  MAE  (平均绝对误差): {mae:.4f}")
    print(f"  RMSE (均方根误差): {rmse:.4f}")
    print(f"  MAPE (平均绝对百分比误差): {mape:.4f}%")

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

metrics = evaluate_forecast(test.values.flatten(), forecast, f'ARIMA{best_order}')

# ============================================================================
# 可视化6：预测结果（含置信区间）
# Visualization 6: Forecast Results with Confidence Intervals
# ============================================================================

fig, ax = create_subplots(1, 1, figsize=(16, 8))

# 绘制训练数据
ax.plot(train.index, train.values, label='训练数据 (Training Data)',
       color='#2E86AB', linewidth=2)

# 绘制测试数据
ax.plot(test.index, test.values, label='实际值 (Actual)',
       color='#A23B72', linewidth=2)

# 绘制预测
ax.plot(test.index, forecast, label=f'ARIMA{best_order} 预测 (Forecast)',
       color='#F18F01', linewidth=2, linestyle='--')

# 绘制置信区间
ax.fill_between(test.index,
               forecast_ci.iloc[:, 0],
               forecast_ci.iloc[:, 1],
               color='#F18F01', alpha=0.2,
               label='95% 置信区间 (95% CI)')

ax.set_title(f'ARIMA{best_order} 预测结果\nARIMA{best_order} Forecast Results',
            fontsize=14, fontweight='bold')
ax.set_xlabel('时间 (Time)', fontsize=11)
ax.set_ylabel('CO2 浓度 (ppm)', fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

# 添加文本框显示评估指标
textstr = f'MAE: {metrics["MAE"]:.4f}\nRMSE: {metrics["RMSE"]:.4f}\nMAPE: {metrics["MAPE"]:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
       verticalalignment='top', bbox=props)

plt.tight_layout()
save_figure(fig, get_output_path('14_arima_forecast.png'))
plt.close()

# ============================================================================
# 基线模型对比
# Baseline Model Comparison
# ============================================================================

print("\n" + "=" * 80)
print("与基线模型对比".center(70))
print("Comparison with Baseline Models".center(80))
print("=" * 80)

# 1. Naive Forecast（使用最后一个观测值）
naive_forecast = np.repeat(train.values[-1], len(test))
naive_metrics = evaluate_forecast(test.values.flatten(), naive_forecast, 'Naive Forecast')

# 2. Simple Moving Average
window = 12
sma_forecast = np.repeat(train.values[-window:].mean(), len(test))
sma_metrics = evaluate_forecast(test.values.flatten(), sma_forecast, f'SMA-{window}')

# ============================================================================
# 可视化7：多模型对比
# Visualization 7: Multiple Models Comparison
# ============================================================================

fig, axes = create_subplots(2, 1, figsize=(16, 12))

# 预测对比
axes[0].plot(test.index, test.values, label='实际值 (Actual)',
            color='black', linewidth=2.5, marker='o', markersize=3)
axes[0].plot(test.index, forecast, label=f'ARIMA{best_order}',
            color='#2E86AB', linewidth=2, linestyle='--', marker='s', markersize=3)
axes[0].plot(test.index, naive_forecast, label='Naive',
            color='#A23B72', linewidth=2, linestyle='--', marker='^', markersize=3)
axes[0].plot(test.index, sma_forecast, label=f'SMA-{window}',
            color='#F18F01', linewidth=2, linestyle='--', marker='d', markersize=3)

axes[0].set_title('多模型预测对比\nMultiple Models Forecast Comparison',
                 fontsize=14, fontweight='bold')
axes[0].set_xlabel('时间 (Time)', fontsize=11)
axes[0].set_ylabel('CO2 浓度 (ppm)', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 评估指标对比
models = ['ARIMA', 'Naive', 'SMA']
mae_values = [metrics['MAE'], naive_metrics['MAE'], sma_metrics['MAE']]
rmse_values = [metrics['RMSE'], naive_metrics['RMSE'], sma_metrics['RMSE']]
mape_values = [metrics['MAPE'], naive_metrics['MAPE'], sma_metrics['MAPE']]

x = np.arange(len(models))
width = 0.25

bars1 = axes[1].bar(x - width, mae_values, width, label='MAE', color='#2E86AB', alpha=0.8)
bars2 = axes[1].bar(x, rmse_values, width, label='RMSE', color='#A23B72', alpha=0.8)
bars3 = axes[1].bar(x + width, mape_values, width, label='MAPE', color='#F18F01', alpha=0.8)

axes[1].set_title('模型性能对比\nModel Performance Comparison',
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('模型 (Model)', fontsize=11)
axes[1].set_ylabel('误差值 (Error Value)', fontsize=11)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
save_figure(fig, get_output_path('15_models_comparison.png'))
plt.close()

# ============================================================================
# 第六部分：SARIMA（季节性 ARIMA）
# Part 6: SARIMA (Seasonal ARIMA)
# ============================================================================

print("\n" + "=" * 80)
print("第六部分：SARIMA（季节性 ARIMA）".center(70))
print("Part 6: SARIMA (Seasonal ARIMA)".center(80))
print("=" * 80)

print("""
SARIMA(p,d,q)(P,D,Q)s 模型：

非季节性部分:
- p: AR 阶数
- d: 差分次数
- q: MA 阶数

季节性部分:
- P: 季节性 AR 阶数
- D: 季节性差分次数
- Q: 季节性 MA 阶数
- s: 季节周期（如 12 表示月度数据的年度季节性）

CO2 数据具有明显的年度季节性，适合使用 SARIMA 模型
""")

# SARIMA 参数
# 非季节性: (p, d, q)
# 季节性: (P, D, Q, s) - s=12 表示年度季节性
sarima_order = (1, 1, 1)
sarima_seasonal_order = (1, 1, 1, 12)

print(f"\n训练 SARIMA{sarima_order}x{sarima_seasonal_order} 模型...")

# 训练 SARIMA 模型
sarima_model = SARIMAX(train,
                       order=sarima_order,
                       seasonal_order=sarima_seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
sarima_fitted = sarima_model.fit(disp=False)

print(f"✓ SARIMA 模型训练完成")

# SARIMA 预测
sarima_forecast = sarima_fitted.forecast(steps=forecast_steps)
sarima_forecast_ci = sarima_fitted.get_forecast(steps=forecast_steps).conf_int()

# 评估 SARIMA
sarima_metrics = evaluate_forecast(test.values.flatten(), sarima_forecast,
                                   f'SARIMA{sarima_order}x{sarima_seasonal_order}')

# ============================================================================
# 可视化8：SARIMA vs ARIMA
# Visualization 8: SARIMA vs ARIMA
# ============================================================================

fig, axes = create_subplots(2, 1, figsize=(16, 12))

# 预测对比
axes[0].plot(train.index[-50:], train.values[-50:], label='训练数据 (Training)',
            color='gray', linewidth=1.5, alpha=0.7)
axes[0].plot(test.index, test.values, label='实际值 (Actual)',
            color='black', linewidth=2.5, marker='o', markersize=4)
axes[0].plot(test.index, forecast, label=f'ARIMA{best_order}',
            color='#2E86AB', linewidth=2, linestyle='--', marker='s', markersize=3)
axes[0].plot(test.index, sarima_forecast, label=f'SARIMA{sarima_order}x{sarima_seasonal_order}',
            color='#A23B72', linewidth=2, linestyle='--', marker='^', markersize=3)

# SARIMA 置信区间
axes[0].fill_between(test.index,
                     sarima_forecast_ci.iloc[:, 0],
                     sarima_forecast_ci.iloc[:, 1],
                     color='#A23B72', alpha=0.2,
                     label='SARIMA 95% CI')

axes[0].set_title('SARIMA vs ARIMA 预测对比\nSARIMA vs ARIMA Forecast Comparison',
                 fontsize=14, fontweight='bold')
axes[0].set_xlabel('时间 (Time)', fontsize=11)
axes[0].set_ylabel('CO2 浓度 (ppm)', fontsize=11)
axes[0].legend(fontsize=10, loc='upper left')
axes[0].grid(True, alpha=0.3)

# 误差对比
arima_errors = test.values.flatten() - forecast
sarima_errors = test.values.flatten() - sarima_forecast

axes[1].plot(test.index, arima_errors, label='ARIMA 误差',
            color='#2E86AB', linewidth=2, marker='o', markersize=3)
axes[1].plot(test.index, sarima_errors, label='SARIMA 误差',
            color='#A23B72', linewidth=2, marker='s', markersize=3)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)

axes[1].set_title('预测误差对比\nForecast Error Comparison',
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('时间 (Time)', fontsize=11)
axes[1].set_ylabel('误差 (Error)', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('16_sarima_vs_arima.png'))
plt.close()

# ============================================================================
# 可视化9：性能总结
# Visualization 9: Performance Summary
# ============================================================================

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# 所有模型的性能指标
all_models = ['Naive', 'SMA-12', f'ARIMA{best_order}', 'SARIMA']
all_mae = [naive_metrics['MAE'], sma_metrics['MAE'], metrics['MAE'], sarima_metrics['MAE']]
all_rmse = [naive_metrics['RMSE'], sma_metrics['RMSE'], metrics['RMSE'], sarima_metrics['RMSE']]
all_mape = [naive_metrics['MAPE'], sma_metrics['MAPE'], metrics['MAPE'], sarima_metrics['MAPE']]

x = np.arange(len(all_models))
width = 0.25

# MAE 和 RMSE
bars1 = axes[0].bar(x - width/2, all_mae, width, label='MAE', color='#2E86AB', alpha=0.8)
bars2 = axes[0].bar(x + width/2, all_rmse, width, label='RMSE', color='#A23B72', alpha=0.8)

axes[0].set_title('MAE 和 RMSE 对比\nMAE and RMSE Comparison',
                 fontsize=13, fontweight='bold')
axes[0].set_xlabel('模型 (Model)', fontsize=11)
axes[0].set_ylabel('误差值 (Error Value)', fontsize=11)
axes[0].set_xticks(x)
axes[0].set_xticklabels(all_models, rotation=15, ha='right')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# MAPE
colors = ['#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
bars3 = axes[1].bar(all_models, all_mape, color=colors, alpha=0.8)

axes[1].set_title('MAPE 对比\nMAPE Comparison',
                 fontsize=13, fontweight='bold')
axes[1].set_xlabel('模型 (Model)', fontsize=11)
axes[1].set_ylabel('MAPE (%)', fontsize=11)
axes[1].set_xticklabels(all_models, rotation=15, ha='right')
axes[1].grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars3:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

fig.suptitle('所有模型性能总结\nAll Models Performance Summary',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
save_figure(fig, get_output_path('17_performance_summary.png'))
plt.close()

# ============================================================================
# 总结和最佳实践
# Summary and Best Practices
# ============================================================================

print("\n" + "=" * 80)
print("总结和最佳实践".center(70))
print("Summary and Best Practices".center(80))
print("=" * 80)

print(f"""
✓ ARIMA 模型分析总结：

1. 【模型选择】
   - 最优 ARIMA 模型: {best_order}
   - 最优 SARIMA 模型: {sarima_order}x{sarima_seasonal_order}
   - SARIMA 通常在有明显季节性时表现更好

2. 【性能对比】
   模型                     MAE      RMSE     MAPE
   {'─' * 55}
   Naive Forecast          {naive_metrics['MAE']:6.3f}   {naive_metrics['RMSE']:6.3f}   {naive_metrics['MAPE']:6.2f}%
   SMA-12                  {sma_metrics['MAE']:6.3f}   {sma_metrics['RMSE']:6.3f}   {sma_metrics['MAPE']:6.2f}%
   ARIMA{best_order}              {metrics['MAE']:6.3f}   {metrics['RMSE']:6.3f}   {metrics['MAPE']:6.2f}%
   SARIMA                  {sarima_metrics['MAE']:6.3f}   {sarima_metrics['RMSE']:6.3f}   {sarima_metrics['MAPE']:6.2f}%

3. 【参数选择指南】
   a) 差分次数 d:
      - 使用 ADF 检验确定
      - 一般 d = 0, 1, 或 2
      - 过度差分会损失信息

   b) AR 阶数 p (PACF):
      - PACF 在 lag p 后截尾 → AR(p)
      - PACF 缓慢衰减 → 考虑 MA 或 ARMA

   c) MA 阶数 q (ACF):
      - ACF 在 lag q 后截尾 → MA(q)
      - ACF 缓慢衰减 → 考虑 AR 或 ARMA

   d) 季节性参数 (P, D, Q, s):
      - s = 数据的季节周期（12=月度年度周期，7=日度周周期）
      - 使用季节性 ACF/PACF 确定 P 和 Q
      - 通常 P, D, Q 取较小值（0, 1, 2）

4. 【模型诊断要点】
   ✓ 残差应该是白噪声（Ljung-Box 检验 p > 0.05）
   ✓ 残差应该服从正态分布（Q-Q 图）
   ✓ 残差 ACF 应该不显著（在置信区间内）
   ✓ 残差均值应该接近 0

5. 【实用建议】
   - 始终从数据可视化和平稳性检验开始
   - 使用网格搜索找到最优参数
   - 对比多个模型，不要只依赖一个
   - 关注业务指标，不只是统计指标
   - SARIMA 适合有明显季节性的数据
   - 定期重新训练模型以适应数据变化

6. 【常见陷阱】
   ✗ 忽略季节性成分
   ✗ 过度拟合（参数过多）
   ✗ 使用未来信息进行预测
   ✗ 忽略残差诊断
   ✗ 不考虑预测不确定性
""")

print("\n" + "=" * 80)
print("✓ ARIMA 模型教程完成！")
print("✓ ARIMA Model Tutorial Completed!")
print("=" * 80)
print(f"\n生成的可视化图表:")
print(f"  9.  09_co2_data_exploration.png - CO2 数据探索")
print(f"  10. 10_acf_pacf_original.png - 原始数据 ACF/PACF")
print(f"  11. 11_acf_pacf_differenced.png - 差分后 ACF/PACF")
print(f"  12. 12_aic_bic_comparison.png - AIC/BIC 参数选择")
print(f"  13. 13_residual_diagnostics.png - 残差诊断")
print(f"  14. 14_arima_forecast.png - ARIMA 预测结果")
print(f"  15. 15_models_comparison.png - 多模型对比")
print(f"  16. 16_sarima_vs_arima.png - SARIMA vs ARIMA")
print(f"  17. 17_performance_summary.png - 性能总结")
print("=" * 80)
