"""
多种预测方法对比 (Multiple Forecasting Methods Comparison)
==========================================================

学习目标 / Learning Objectives:
1. 掌握多种时间序列预测方法
2. 理解不同方法的优缺点和适用场景
3. 学习如何选择合适的预测方法
4. 实践模型对比和评估

包含的方法:
- 基线模型: Naive, Moving Average, Exponential Smoothing
- 统计模型: ARIMA, SARIMA, Auto ARIMA
- 机器学习: Random Forest, XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.datasets import co2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

from utils import RANDOM_STATE, setup_chinese_fonts, create_subplots, save_figure, get_output_path

# 设置中文字体
setup_chinese_fonts()

print("=" * 80)
print("多种预测方法对比教程 (Multiple Forecasting Methods Tutorial)".center(80))
print("=" * 80)

# ============================================================================
# 第一部分：预测方法概述
# Part 1: Forecasting Methods Overview
# ============================================================================

print("\n" + "=" * 80)
print("第一部分：预测方法概述".center(70))
print("Part 1: Forecasting Methods Overview".center(80))
print("=" * 80)

print("""
时间序列预测方法分类：

1. 【基线模型 (Baseline Models)】
   - Naive Forecast: 使用最后一个观测值
   - Simple Moving Average (SMA): 简单移动平均
   - Weighted Moving Average (WMA): 加权移动平均
   - Exponential Smoothing: 指数平滑（单/双/三）

2. 【统计模型 (Statistical Models)】
   - ARIMA: 自回归积分移动平均
   - SARIMA: 季节性 ARIMA
   - Auto ARIMA: 自动参数选择
   - Prophet: Facebook 开发的预测工具

3. 【机器学习方法 (Machine Learning Methods)】
   - Random Forest: 随机森林回归
   - XGBoost: 梯度提升树
   - LightGBM: 轻量级梯度提升
   - Neural Networks: 神经网络

4. 【深度学习方法 (Deep Learning Methods)】
   - LSTM: 长短期记忆网络
   - GRU: 门控循环单元
   - Transformer: 注意力机制模型

本教程重点介绍前三类方法
""")

# ============================================================================
# 数据准备
# Data Preparation
# ============================================================================

print("\n" + "=" * 80)
print("数据准备".center(70))
print("Data Preparation".center(80))
print("=" * 80)

# 加载 CO2 数据集
data = co2.load_pandas().data
data = data.fillna(data.interpolate())
data = data.resample('M').mean()

print(f"✓ 数据加载完成")
print(f"  数据点数: {len(data)}")
print(f"  时间范围: {data.index[0]} 到 {data.index[-1]}")

# 分割数据集（80% 训练，20% 测试）
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

print(f"\n✓ 数据分割完成")
print(f"  训练集: {len(train)} 个月 ({train.index[0]} 到 {train.index[-1]})")
print(f"  测试集: {len(test)} 个月 ({test.index[0]} 到 {test.index[-1]})")

# ============================================================================
# 评估函数
# Evaluation Functions
# ============================================================================

def evaluate_forecast(actual, predicted, model_name='Model'):
    """
    计算预测评估指标

    Parameters:
    -----------
    actual : array-like
        实际值
    predicted : array-like
        预测值
    model_name : str
        模型名称

    Returns:
    --------
    dict : 包含 MAE, RMSE, MAPE, R² 的字典
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)

    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }

# 存储所有模型的结果
all_results = []
all_forecasts = {}
all_training_times = {}

# ============================================================================
# 第二部分：基线模型
# Part 2: Baseline Models
# ============================================================================

print("\n" + "=" * 80)
print("第二部分：基线模型".center(70))
print("Part 2: Baseline Models".center(80))
print("=" * 80)

# ------------------------------------------------------------------------
# 1. Naive Forecast
# ------------------------------------------------------------------------
print("\n1. Naive Forecast（朴素预测）")
print("   使用最后一个观测值作为所有未来预测")

start_time = time.time()
naive_forecast = np.repeat(train.values[-1], len(test))
naive_time = time.time() - start_time

all_forecasts['Naive'] = naive_forecast
all_training_times['Naive'] = naive_time
result = evaluate_forecast(test.values.flatten(), naive_forecast, 'Naive')
all_results.append(result)

print(f"   MAE: {result['MAE']:.4f}, RMSE: {result['RMSE']:.4f}, MAPE: {result['MAPE']:.2f}%")
print(f"   训练时间: {naive_time:.4f} 秒")

# ------------------------------------------------------------------------
# 2. Simple Moving Average (SMA)
# ------------------------------------------------------------------------
print("\n2. Simple Moving Average（简单移动平均）")
print("   使用过去 N 个观测值的平均值")

windows = [3, 6, 12]
for window in windows:
    start_time = time.time()
    sma_forecast = np.repeat(train.values[-window:].mean(), len(test))
    sma_time = time.time() - start_time

    model_name = f'SMA-{window}'
    all_forecasts[model_name] = sma_forecast
    all_training_times[model_name] = sma_time
    result = evaluate_forecast(test.values.flatten(), sma_forecast, model_name)
    all_results.append(result)

    print(f"   SMA-{window}: MAE={result['MAE']:.4f}, RMSE={result['RMSE']:.4f}, MAPE={result['MAPE']:.2f}%")

# ------------------------------------------------------------------------
# 3. Weighted Moving Average (WMA)
# ------------------------------------------------------------------------
print("\n3. Weighted Moving Average（加权移动平均）")
print("   给予最近的观测值更高的权重")

start_time = time.time()
window = 12
weights = np.arange(1, window + 1)
weights = weights / weights.sum()
wma_value = np.sum(train.values[-window:].flatten() * weights)
wma_forecast = np.repeat(wma_value, len(test))
wma_time = time.time() - start_time

all_forecasts['WMA-12'] = wma_forecast
all_training_times['WMA-12'] = wma_time
result = evaluate_forecast(test.values.flatten(), wma_forecast, 'WMA-12')
all_results.append(result)

print(f"   MAE: {result['MAE']:.4f}, RMSE: {result['RMSE']:.4f}, MAPE: {result['MAPE']:.2f}%")

# ------------------------------------------------------------------------
# 4. Exponential Smoothing（指数平滑）
# ------------------------------------------------------------------------
print("\n4. Exponential Smoothing（指数平滑）")

# 4.1 Simple Exponential Smoothing (SES)
print("   4.1 Simple Exponential Smoothing（单指数平滑）")
start_time = time.time()
ses_model = ExponentialSmoothing(train, trend=None, seasonal=None)
ses_fitted = ses_model.fit()
ses_forecast = ses_fitted.forecast(steps=len(test))
ses_time = time.time() - start_time

all_forecasts['SES'] = ses_forecast.values
all_training_times['SES'] = ses_time
result = evaluate_forecast(test.values.flatten(), ses_forecast.values, 'SES')
all_results.append(result)

print(f"       MAE: {result['MAE']:.4f}, RMSE: {result['RMSE']:.4f}, MAPE: {result['MAPE']:.2f}%")

# 4.2 Double Exponential Smoothing (Holt's Method)
print("   4.2 Double Exponential Smoothing（双指数平滑 - Holt 方法）")
start_time = time.time()
des_model = ExponentialSmoothing(train, trend='add', seasonal=None)
des_fitted = des_model.fit()
des_forecast = des_fitted.forecast(steps=len(test))
des_time = time.time() - start_time

all_forecasts['DES'] = des_forecast.values
all_training_times['DES'] = des_time
result = evaluate_forecast(test.values.flatten(), des_forecast.values, 'DES')
all_results.append(result)

print(f"       MAE: {result['MAE']:.4f}, RMSE: {result['RMSE']:.4f}, MAPE: {result['MAPE']:.2f}%")

# 4.3 Triple Exponential Smoothing (Holt-Winters Method)
print("   4.3 Triple Exponential Smoothing（三指数平滑 - Holt-Winters 方法）")
start_time = time.time()
tes_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
tes_fitted = tes_model.fit()
tes_forecast = tes_fitted.forecast(steps=len(test))
tes_time = time.time() - start_time

all_forecasts['TES'] = tes_forecast.values
all_training_times['TES'] = tes_time
result = evaluate_forecast(test.values.flatten(), tes_forecast.values, 'TES')
all_results.append(result)

print(f"       MAE: {result['MAE']:.4f}, RMSE: {result['RMSE']:.4f}, MAPE: {result['MAPE']:.2f}%")

# ============================================================================
# 可视化1：基线模型对比
# Visualization 1: Baseline Models Comparison
# ============================================================================

fig, axes = create_subplots(2, 2, figsize=(18, 12))

baseline_models = ['Naive', 'SMA-12', 'WMA-12', 'SES']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for idx, (model, color) in enumerate(zip(baseline_models, colors)):
    row, col = idx // 2, idx % 2

    axes[row, col].plot(test.index, test.values, label='实际值 (Actual)',
                       color='black', linewidth=2.5, marker='o', markersize=4)
    axes[row, col].plot(test.index, all_forecasts[model], label=f'{model} 预测',
                       color=color, linewidth=2, linestyle='--', marker='s', markersize=3)

    axes[row, col].set_title(f'{model} 预测结果', fontsize=12, fontweight='bold')
    axes[row, col].set_xlabel('时间 (Time)', fontsize=10)
    axes[row, col].set_ylabel('CO2 浓度 (ppm)', fontsize=10)
    axes[row, col].legend(fontsize=9)
    axes[row, col].grid(True, alpha=0.3)

    # 添加指标文本框
    model_result = [r for r in all_results if r['Model'] == model][0]
    textstr = f'MAE: {model_result["MAE"]:.2f}\nRMSE: {model_result["RMSE"]:.2f}\nMAPE: {model_result["MAPE"]:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    axes[row, col].text(0.02, 0.98, textstr, transform=axes[row, col].transAxes,
                       fontsize=9, verticalalignment='top', bbox=props)

fig.suptitle('基线模型预测对比\nBaseline Models Forecast Comparison',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
save_figure(fig, get_output_path('18_baseline_models_comparison.png'))
plt.close()

# ============================================================================
# 可视化2：指数平滑方法对比
# Visualization 2: Exponential Smoothing Methods Comparison
# ============================================================================

fig, ax = create_subplots(1, 1, figsize=(16, 8))

ax.plot(train.index[-50:], train.values[-50:], label='训练数据 (Training)',
       color='gray', linewidth=1.5, alpha=0.5)
ax.plot(test.index, test.values, label='实际值 (Actual)',
       color='black', linewidth=2.5, marker='o', markersize=4)
ax.plot(test.index, all_forecasts['SES'], label='单指数平滑 (SES)',
       color='#2E86AB', linewidth=2, linestyle='--', marker='s', markersize=3)
ax.plot(test.index, all_forecasts['DES'], label='双指数平滑 (DES)',
       color='#A23B72', linewidth=2, linestyle='--', marker='^', markersize=3)
ax.plot(test.index, all_forecasts['TES'], label='三指数平滑 (TES)',
       color='#F18F01', linewidth=2, linestyle='--', marker='d', markersize=3)

ax.set_title('指数平滑方法对比\nExponential Smoothing Methods Comparison',
            fontsize=14, fontweight='bold')
ax.set_xlabel('时间 (Time)', fontsize=11)
ax.set_ylabel('CO2 浓度 (ppm)', fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('19_exponential_smoothing_comparison.png'))
plt.close()

# ============================================================================
# 第三部分：统计模型
# Part 3: Statistical Models
# ============================================================================

print("\n" + "=" * 80)
print("第三部分：统计模型（ARIMA、SARIMA）".center(70))
print("Part 3: Statistical Models (ARIMA, SARIMA)".center(80))
print("=" * 80)

# ------------------------------------------------------------------------
# ARIMA
# ------------------------------------------------------------------------
print("\n1. ARIMA 模型")

start_time = time.time()
arima_model = ARIMA(train, order=(1, 1, 1))
arima_fitted = arima_model.fit()
arima_forecast = arima_fitted.forecast(steps=len(test))
arima_time = time.time() - start_time

all_forecasts['ARIMA'] = arima_forecast.values
all_training_times['ARIMA'] = arima_time
result = evaluate_forecast(test.values.flatten(), arima_forecast.values, 'ARIMA(1,1,1)')
all_results.append(result)

print(f"   MAE: {result['MAE']:.4f}, RMSE: {result['RMSE']:.4f}, MAPE: {result['MAPE']:.2f}%")
print(f"   训练时间: {arima_time:.4f} 秒")

# ------------------------------------------------------------------------
# SARIMA
# ------------------------------------------------------------------------
print("\n2. SARIMA 模型（季节性 ARIMA）")

start_time = time.time()
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fitted = sarima_model.fit(disp=False)
sarima_forecast = sarima_fitted.forecast(steps=len(test))
sarima_time = time.time() - start_time

all_forecasts['SARIMA'] = sarima_forecast.values
all_training_times['SARIMA'] = sarima_time
result = evaluate_forecast(test.values.flatten(), sarima_forecast.values, 'SARIMA')
all_results.append(result)

print(f"   MAE: {result['MAE']:.4f}, RMSE: {result['RMSE']:.4f}, MAPE: {result['MAPE']:.2f}%")
print(f"   训练时间: {sarima_time:.4f} 秒")

# ============================================================================
# 可视化3：ARIMA vs SARIMA
# Visualization 3: ARIMA vs SARIMA
# ============================================================================

fig, ax = create_subplots(1, 1, figsize=(16, 8))

ax.plot(train.index[-50:], train.values[-50:], label='训练数据 (Training)',
       color='gray', linewidth=1.5, alpha=0.5)
ax.plot(test.index, test.values, label='实际值 (Actual)',
       color='black', linewidth=2.5, marker='o', markersize=4)
ax.plot(test.index, all_forecasts['ARIMA'], label='ARIMA(1,1,1)',
       color='#2E86AB', linewidth=2, linestyle='--', marker='s', markersize=3)
ax.plot(test.index, all_forecasts['SARIMA'], label='SARIMA(1,1,1)(1,1,1,12)',
       color='#A23B72', linewidth=2, linestyle='--', marker='^', markersize=3)

ax.set_title('ARIMA vs SARIMA 预测对比\nARIMA vs SARIMA Forecast Comparison',
            fontsize=14, fontweight='bold')
ax.set_xlabel('时间 (Time)', fontsize=11)
ax.set_ylabel('CO2 浓度 (ppm)', fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('20_arima_vs_sarima.png'))
plt.close()

# ============================================================================
# 第四部分：机器学习方法
# Part 4: Machine Learning Methods
# ============================================================================

print("\n" + "=" * 80)
print("第四部分：机器学习方法".center(70))
print("Part 4: Machine Learning Methods".center(80))
print("=" * 80)

print("\n准备机器学习特征...")

# ------------------------------------------------------------------------
# 特征工程
# Feature Engineering
# ------------------------------------------------------------------------

def create_features(data, lags=[1, 2, 3, 6, 12], rolling_windows=[3, 6, 12]):
    """
    为机器学习创建时间序列特征

    Parameters:
    -----------
    data : Series
        时间序列数据
    lags : list
        滞后特征列表
    rolling_windows : list
        滚动窗口大小列表

    Returns:
    --------
    DataFrame : 包含所有特征的数据框
    """
    df = pd.DataFrame({'value': data.values.flatten()}, index=data.index)

    # 1. 滞后特征
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    # 2. 滚动统计
    for window in rolling_windows:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['value'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['value'].rolling(window=window).max()

    # 3. 时间特征
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year

    # 4. 差分特征
    df['diff_1'] = df['value'].diff(1)
    df['diff_12'] = df['value'].diff(12)

    return df

# 创建特征
train_features = create_features(train)
test_features = create_features(pd.concat([train, test]))

# 对齐训练和测试特征
test_features = test_features.loc[test.index]

# 移除 NaN 值
train_features = train_features.dropna()
test_features = test_features.dropna()

# 分离特征和目标
feature_cols = [col for col in train_features.columns if col != 'value']
X_train = train_features[feature_cols]
y_train = train_features['value']
X_test = test_features[feature_cols]
y_test = test_features['value']

print(f"✓ 特征工程完成")
print(f"  训练样本数: {len(X_train)}")
print(f"  测试样本数: {len(X_test)}")
print(f"  特征数量: {len(feature_cols)}")
print(f"  特征列表: {feature_cols[:10]}... (显示前10个)")

# ------------------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------------------
print("\n1. Random Forest Regressor（随机森林回归）")

start_time = time.time()
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_forecast = rf_model.predict(X_test)
rf_time = time.time() - start_time

all_forecasts['Random Forest'] = rf_forecast
all_training_times['Random Forest'] = rf_time
result = evaluate_forecast(y_test.values, rf_forecast, 'Random Forest')
all_results.append(result)

print(f"   MAE: {result['MAE']:.4f}, RMSE: {result['RMSE']:.4f}, MAPE: {result['MAPE']:.2f}%")
print(f"   训练时间: {rf_time:.4f} 秒")

# 特征重要性
feature_importance_rf = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 5 重要特征:")
for idx, row in feature_importance_rf.head().iterrows():
    print(f"     {row['feature']}: {row['importance']:.4f}")

# ------------------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------------------
print("\n2. XGBoost Regressor（梯度提升树）")

start_time = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_forecast = xgb_model.predict(X_test)
xgb_time = time.time() - start_time

all_forecasts['XGBoost'] = xgb_forecast
all_training_times['XGBoost'] = xgb_time
result = evaluate_forecast(y_test.values, xgb_forecast, 'XGBoost')
all_results.append(result)

print(f"   MAE: {result['MAE']:.4f}, RMSE: {result['RMSE']:.4f}, MAPE: {result['MAPE']:.2f}%")
print(f"   训练时间: {xgb_time:.4f} 秒")

# 特征重要性
feature_importance_xgb = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 5 重要特征:")
for idx, row in feature_importance_xgb.head().iterrows():
    print(f"     {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# 可视化4：特征重要性对比
# Visualization 4: Feature Importance Comparison
# ============================================================================

fig, axes = create_subplots(1, 2, figsize=(18, 8))

# Random Forest 特征重要性
top_n = 15
top_features_rf = feature_importance_rf.head(top_n)
colors_rf = plt.cm.viridis(np.linspace(0, 1, len(top_features_rf)))
bars1 = axes[0].barh(range(len(top_features_rf)), top_features_rf['importance'].values,
                     color=colors_rf, alpha=0.8)
axes[0].set_yticks(range(len(top_features_rf)))
axes[0].set_yticklabels(top_features_rf['feature'].values, fontsize=9)
axes[0].set_xlabel('重要性 (Importance)', fontsize=11)
axes[0].set_title('Random Forest 特征重要性\nRandom Forest Feature Importance',
                 fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].invert_yaxis()

# XGBoost 特征重要性
top_features_xgb = feature_importance_xgb.head(top_n)
colors_xgb = plt.cm.plasma(np.linspace(0, 1, len(top_features_xgb)))
bars2 = axes[1].barh(range(len(top_features_xgb)), top_features_xgb['importance'].values,
                     color=colors_xgb, alpha=0.8)
axes[1].set_yticks(range(len(top_features_xgb)))
axes[1].set_yticklabels(top_features_xgb['feature'].values, fontsize=9)
axes[1].set_xlabel('重要性 (Importance)', fontsize=11)
axes[1].set_title('XGBoost 特征重要性\nXGBoost Feature Importance',
                 fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].invert_yaxis()

plt.tight_layout()
save_figure(fig, get_output_path('21_feature_importance_comparison.png'))
plt.close()

# ============================================================================
# 可视化5：机器学习方法对比
# Visualization 5: Machine Learning Methods Comparison
# ============================================================================

fig, ax = create_subplots(1, 1, figsize=(16, 8))

# 对齐测试数据索引
test_aligned = test.loc[y_test.index]

ax.plot(y_test.index, test_aligned.values, label='实际值 (Actual)',
       color='black', linewidth=2.5, marker='o', markersize=4)
ax.plot(y_test.index, rf_forecast, label='Random Forest',
       color='#2E86AB', linewidth=2, linestyle='--', marker='s', markersize=3)
ax.plot(y_test.index, xgb_forecast, label='XGBoost',
       color='#A23B72', linewidth=2, linestyle='--', marker='^', markersize=3)

ax.set_title('机器学习方法预测对比\nMachine Learning Methods Forecast Comparison',
            fontsize=14, fontweight='bold')
ax.set_xlabel('时间 (Time)', fontsize=11)
ax.set_ylabel('CO2 浓度 (ppm)', fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('22_ml_methods_comparison.png'))
plt.close()

# ============================================================================
# 第五部分：所有模型对比
# Part 5: All Models Comparison
# ============================================================================

print("\n" + "=" * 80)
print("第五部分：所有模型性能对比".center(70))
print("Part 5: All Models Performance Comparison".center(80))
print("=" * 80)

# 创建结果 DataFrame
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('MAE')

print("\n所有模型评估结果（按 MAE 排序）:")
print("=" * 85)
print(f"{'模型':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10} {'训练时间(秒)':>15}")
print("=" * 85)

for _, row in results_df.iterrows():
    model_name = row['Model']
    train_time = all_training_times.get(model_name, 0)
    print(f"{model_name:<20} {row['MAE']:>10.4f} {row['RMSE']:>10.4f} "
          f"{row['MAPE']:>9.2f}% {row['R²']:>10.4f} {train_time:>15.4f}")

print("=" * 85)

# 最佳模型
best_model = results_df.iloc[0]
print(f"\n✓ 最佳模型: {best_model['Model']}")
print(f"  MAE: {best_model['MAE']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.4f}")
print(f"  MAPE: {best_model['MAPE']:.2f}%")
print(f"  R²: {best_model['R²']:.4f}")

# ============================================================================
# 可视化6：所有模型性能对比（柱状图）
# Visualization 6: All Models Performance Comparison (Bar Chart)
# ============================================================================

fig, axes = create_subplots(2, 2, figsize=(18, 12))

# MAE
mae_sorted = results_df.sort_values('MAE', ascending=True).head(10)
colors = plt.cm.viridis(np.linspace(0, 1, len(mae_sorted)))
bars = axes[0, 0].barh(range(len(mae_sorted)), mae_sorted['MAE'].values,
                       color=colors, alpha=0.8)
axes[0, 0].set_yticks(range(len(mae_sorted)))
axes[0, 0].set_yticklabels(mae_sorted['Model'].values, fontsize=9)
axes[0, 0].set_xlabel('MAE', fontsize=10)
axes[0, 0].set_title('平均绝对误差 (MAE) - Top 10', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')
axes[0, 0].invert_yaxis()

# RMSE
rmse_sorted = results_df.sort_values('RMSE', ascending=True).head(10)
colors = plt.cm.plasma(np.linspace(0, 1, len(rmse_sorted)))
bars = axes[0, 1].barh(range(len(rmse_sorted)), rmse_sorted['RMSE'].values,
                       color=colors, alpha=0.8)
axes[0, 1].set_yticks(range(len(rmse_sorted)))
axes[0, 1].set_yticklabels(rmse_sorted['Model'].values, fontsize=9)
axes[0, 1].set_xlabel('RMSE', fontsize=10)
axes[0, 1].set_title('均方根误差 (RMSE) - Top 10', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')
axes[0, 1].invert_yaxis()

# MAPE
mape_sorted = results_df.sort_values('MAPE', ascending=True).head(10)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(mape_sorted)))
bars = axes[1, 0].barh(range(len(mape_sorted)), mape_sorted['MAPE'].values,
                       color=colors, alpha=0.8)
axes[1, 0].set_yticks(range(len(mape_sorted)))
axes[1, 0].set_yticklabels(mape_sorted['Model'].values, fontsize=9)
axes[1, 0].set_xlabel('MAPE (%)', fontsize=10)
axes[1, 0].set_title('平均绝对百分比误差 (MAPE) - Top 10', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')
axes[1, 0].invert_yaxis()

# R²
r2_sorted = results_df.sort_values('R²', ascending=False).head(10)
colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(r2_sorted)))
bars = axes[1, 1].barh(range(len(r2_sorted)), r2_sorted['R²'].values,
                       color=colors, alpha=0.8)
axes[1, 1].set_yticks(range(len(r2_sorted)))
axes[1, 1].set_yticklabels(r2_sorted['Model'].values, fontsize=9)
axes[1, 1].set_xlabel('R² Score', fontsize=10)
axes[1, 1].set_title('决定系数 (R²) - Top 10', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')
axes[1, 1].invert_yaxis()

fig.suptitle('所有模型性能指标对比\nAll Models Performance Metrics Comparison',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
save_figure(fig, get_output_path('23_all_models_performance.png'))
plt.close()

# ============================================================================
# 可视化7：预测结果对比（折线图）
# Visualization 7: Forecast Results Comparison (Line Chart)
# ============================================================================

# 选择表现最好的几个模型
top_models = results_df.head(5)['Model'].values

fig, ax = create_subplots(1, 1, figsize=(18, 8))

# 绘制实际值
ax.plot(test.index, test.values, label='实际值 (Actual)',
       color='black', linewidth=3, marker='o', markersize=5, zorder=10)

# 绘制各模型预测
colors_top = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
markers = ['s', '^', 'd', 'v', 'p']

for idx, (model, color, marker) in enumerate(zip(top_models, colors_top, markers)):
    # 对于机器学习模型，需要对齐索引
    if model in ['Random Forest', 'XGBoost']:
        forecast_values = all_forecasts[model]
        forecast_index = y_test.index
    else:
        forecast_values = all_forecasts[model]
        forecast_index = test.index

    ax.plot(forecast_index, forecast_values, label=model,
           color=color, linewidth=2, linestyle='--', marker=marker,
           markersize=3, alpha=0.8)

ax.set_title('Top 5 模型预测结果对比\nTop 5 Models Forecast Comparison',
            fontsize=14, fontweight='bold')
ax.set_xlabel('时间 (Time)', fontsize=11)
ax.set_ylabel('CO2 浓度 (ppm)', fontsize=11)
ax.legend(fontsize=10, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('24_top5_models_forecast.png'))
plt.close()

# ============================================================================
# 可视化8：误差分布
# Visualization 8: Error Distribution
# ============================================================================

fig, axes = create_subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# 选择6个代表性模型
selected_models = ['TES', 'SARIMA', 'Random Forest', 'XGBoost', 'Naive', 'SMA-12']
colors_err = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']

for idx, (model, color) in enumerate(zip(selected_models, colors_err)):
    # 计算误差
    if model in ['Random Forest', 'XGBoost']:
        test_aligned = test.loc[y_test.index]
        errors = test_aligned.values.flatten() - all_forecasts[model]
    else:
        errors = test.values.flatten() - all_forecasts[model]

    # 绘制直方图
    axes[idx].hist(errors, bins=20, color=color, alpha=0.7, edgecolor='black')
    axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[idx].set_title(f'{model} 误差分布', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('误差 (Error)', fontsize=9)
    axes[idx].set_ylabel('频数 (Frequency)', fontsize=9)
    axes[idx].grid(True, alpha=0.3, axis='y')

    # 添加统计信息
    textstr = f'均值: {errors.mean():.2f}\n标准差: {errors.std():.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    axes[idx].text(0.02, 0.98, textstr, transform=axes[idx].transAxes,
                  fontsize=9, verticalalignment='top', bbox=props)

fig.suptitle('模型预测误差分布\nModel Forecast Error Distribution',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
save_figure(fig, get_output_path('25_error_distribution.png'))
plt.close()

# ============================================================================
# 可视化9：训练时间对比
# Visualization 9: Training Time Comparison
# ============================================================================

fig, ax = create_subplots(1, 1, figsize=(14, 8))

# 准备数据
models_for_time = list(all_training_times.keys())
times = list(all_training_times.values())

# 按时间排序
sorted_indices = np.argsort(times)
models_sorted = [models_for_time[i] for i in sorted_indices]
times_sorted = [times[i] for i in sorted_indices]

# 绘制柱状图
colors_time = plt.cm.viridis(np.linspace(0, 1, len(models_sorted)))
bars = ax.barh(range(len(models_sorted)), times_sorted, color=colors_time, alpha=0.8)

ax.set_yticks(range(len(models_sorted)))
ax.set_yticklabels(models_sorted, fontsize=9)
ax.set_xlabel('训练时间 (秒) / Training Time (seconds)', fontsize=11)
ax.set_title('模型训练时间对比\nModel Training Time Comparison',
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, times_sorted)):
    ax.text(val + 0.01, i, f'{val:.4f}s', va='center', fontsize=9)

plt.tight_layout()
save_figure(fig, get_output_path('26_training_time_comparison.png'))
plt.close()

# ============================================================================
# 可视化10：精度 vs 速度权衡
# Visualization 10: Accuracy vs Speed Trade-off
# ============================================================================

fig, ax = create_subplots(1, 1, figsize=(14, 10))

# 准备数据
models_list = results_df['Model'].values
mae_list = results_df['MAE'].values
time_list = [all_training_times.get(model, 0) for model in models_list]

# 创建散点图
colors_scatter = plt.cm.rainbow(np.linspace(0, 1, len(models_list)))
scatter = ax.scatter(time_list, mae_list, s=200, c=colors_scatter,
                    alpha=0.7, edgecolors='black', linewidth=2)

# 添加标签
for model, time_val, mae_val, color in zip(models_list, time_list, mae_list, colors_scatter):
    ax.annotate(model, (time_val, mae_val),
               xytext=(5, 5), textcoords='offset points',
               fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor=color, alpha=0.3))

ax.set_xlabel('训练时间 (秒) / Training Time (seconds)', fontsize=11)
ax.set_ylabel('MAE (平均绝对误差)', fontsize=11)
ax.set_title('模型精度 vs 速度权衡\nModel Accuracy vs Speed Trade-off',
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 添加理想区域标注（低 MAE，低训练时间）
ax.axhline(y=mae_list.mean(), color='red', linestyle='--', alpha=0.3, label='平均 MAE')
ax.axvline(x=np.median(time_list), color='blue', linestyle='--', alpha=0.3, label='中位训练时间')
ax.legend(fontsize=10)

plt.tight_layout()
save_figure(fig, get_output_path('27_accuracy_vs_speed.png'))
plt.close()

# ============================================================================
# 可视化11：实际应用案例 - 销售预测示例
# Visualization 11: Real Application - Sales Forecasting Example
# ============================================================================

print("\n生成实际应用案例可视化...")

# 模拟销售数据（带有趋势和季节性）
np.random.seed(RANDOM_STATE)
dates_sales = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
n_sales = len(dates_sales)

# 趋势 + 季节性 + 噪声
trend_sales = np.linspace(100, 300, n_sales)
seasonal_sales = 50 * np.sin(2 * np.pi * np.arange(n_sales) / 365)
weekly_sales = 20 * np.sin(2 * np.pi * np.arange(n_sales) / 7)
noise_sales = np.random.normal(0, 15, n_sales)
sales = trend_sales + seasonal_sales + weekly_sales + noise_sales

sales_df = pd.Series(sales, index=dates_sales)

# 分割数据
train_sales = sales_df[:'2023-06-30']
test_sales = sales_df['2023-07-01':]

# 使用 TES 模型预测
tes_sales_model = ExponentialSmoothing(train_sales, trend='add', seasonal='add', seasonal_periods=365)
tes_sales_fitted = tes_sales_model.fit()
tes_sales_forecast = tes_sales_fitted.forecast(steps=len(test_sales))

fig, axes = create_subplots(2, 1, figsize=(16, 10))

# 完整时间序列
axes[0].plot(sales_df.index, sales_df.values, color='#2E86AB',
            linewidth=1, alpha=0.7, label='历史销售数据')
axes[0].axvline(x=train_sales.index[-1], color='red', linestyle='--',
               linewidth=2, label='训练/测试分割点')
axes[0].set_title('销售数据时间序列\nSales Data Time Series',
                 fontsize=13, fontweight='bold')
axes[0].set_xlabel('日期 (Date)', fontsize=10)
axes[0].set_ylabel('销售额 (Sales)', fontsize=10)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 预测结果
axes[1].plot(train_sales.index[-100:], train_sales.values[-100:],
            color='gray', linewidth=1.5, alpha=0.5, label='训练数据')
axes[1].plot(test_sales.index, test_sales.values,
            color='black', linewidth=2, marker='o', markersize=3,
            label='实际销售')
axes[1].plot(test_sales.index, tes_sales_forecast.values,
            color='#F18F01', linewidth=2, linestyle='--', marker='s',
            markersize=3, label='预测销售')

# 计算预测指标
mae_sales = mean_absolute_error(test_sales.values, tes_sales_forecast.values)
rmse_sales = np.sqrt(mean_squared_error(test_sales.values, tes_sales_forecast.values))
mape_sales = np.mean(np.abs((test_sales.values - tes_sales_forecast.values) / test_sales.values)) * 100

axes[1].set_title('销售预测结果（三指数平滑法）\nSales Forecast (Triple Exponential Smoothing)',
                 fontsize=13, fontweight='bold')
axes[1].set_xlabel('日期 (Date)', fontsize=10)
axes[1].set_ylabel('销售额 (Sales)', fontsize=10)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# 添加指标文本框
textstr = f'MAE: {mae_sales:.2f}\nRMSE: {rmse_sales:.2f}\nMAPE: {mape_sales:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
axes[1].text(0.02, 0.98, textstr, transform=axes[1].transAxes,
            fontsize=10, verticalalignment='top', bbox=props)

fig.suptitle('实际应用案例：销售预测\nReal Application: Sales Forecasting',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
save_figure(fig, get_output_path('28_sales_forecasting_example.png'))
plt.close()

# ============================================================================
# 可视化12：方法选择决策树
# Visualization 12: Method Selection Decision Tree
# ============================================================================

fig, ax = create_subplots(1, 1, figsize=(16, 12))

# 隐藏坐标轴
ax.axis('off')

# 创建决策流程图文本
decision_text = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                       时间序列预测方法选择指南                              ║
║                  Time Series Forecasting Method Selection Guide            ║
╚═══════════════════════════════════════════════════════════════════════════╝

                            开始 (Start)
                                 │
                 ┌───────────────┴───────────────┐
                 │  数据量如何？(Data Size?)      │
                 └───────┬───────────────┬───────┘
                         │               │
                    数据量小          数据量大
                    (Small)          (Large)
                         │               │
             ┌───────────┴─────┐         │
             │  有季节性？      │         │
             │  (Seasonal?)    │         │
             └────┬──────┬─────┘         │
                  │      │               │
                 是     否               │
                 │      │               │
            ┌────┘      └────┐          │
            │                │          │
       【SARIMA】      【ARIMA】         │
     季节性数据最佳    简单趋势数据       │
            │                │          │
            │                │          │
            └────────┬───────┘          │
                     │                  │
                     │          ┌───────┴──────────┐
                     │          │  需要可解释性？    │
                     │          │  (Interpretable?) │
                     │          └───┬──────────┬───┘
                     │              │          │
                     │             是         否
                     │              │          │
                     │         ┌────┘          └────┐
                     │         │                    │
                     │    【SARIMA】          【XGBoost】
                     │    统计模型            高精度预测
                     │    可解释性强           速度快
                     │         │                    │
                     └─────────┴────────────────────┘
                                 │
                          最终预测模型
                          (Final Model)

╔═══════════════════════════════════════════════════════════════════════════╗
║                          快速选择建议                                      ║
║                      Quick Selection Guide                                ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  1. 快速原型/基线    →  Naive, SMA                                        ║
║     Quick Prototype                                                       ║
║                                                                           ║
║  2. 有明显季节性     →  SARIMA, TES (三指数平滑)                          ║
║     Clear Seasonality                                                     ║
║                                                                           ║
║  3. 简单趋势数据     →  ARIMA, DES (双指数平滑)                           ║
║     Simple Trend                                                          ║
║                                                                           ║
║  4. 大数据集         →  XGBoost, Random Forest                            ║
║     Large Dataset                                                         ║
║                                                                           ║
║  5. 需要高精度       →  SARIMA + XGBoost 集成                             ║
║     Need High Accuracy   (Ensemble)                                       ║
║                                                                           ║
║  6. 实时预测         →  Naive, SES (计算快速)                             ║
║     Real-time Forecast                                                    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

ax.text(0.5, 0.5, decision_text, ha='center', va='center',
       fontsize=10, family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

ax.set_title('时间序列预测方法选择决策树\nTime Series Forecasting Method Selection',
            fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
save_figure(fig, get_output_path('29_method_selection_guide.png'))
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
✓ 多种预测方法对比总结：

1. 【最佳模型】
   - 整体最佳: {best_model['Model']} (MAE: {best_model['MAE']:.4f})
   - 最快训练: {min(all_training_times, key=all_training_times.get)}
   - 最高 R²: {results_df.loc[results_df['R²'].idxmax(), 'Model']}

2. 【方法分类总结】

   基线模型 (Baseline Models):
   ✓ 优点: 简单快速，适合作为基准
   ✗ 缺点: 精度较低，无法捕捉复杂模式
   适用: 快速原型、基准对比

   指数平滑 (Exponential Smoothing):
   ✓ 优点: 实现简单，适合短期预测
   ✗ 缺点: 对长期趋势预测能力有限
   适用: 有季节性的中短期预测

   ARIMA/SARIMA:
   ✓ 优点: 统计基础扎实，可解释性强
   ✗ 缺点: 参数选择复杂，计算较慢
   适用: 有明确趋势和季节性的数据

   机器学习方法 (Random Forest, XGBoost):
   ✓ 优点: 精度高，能捕捉复杂非线性关系
   ✗ 缺点: 需要特征工程，可解释性差
   适用: 大数据集，有多个特征

3. 【方法选择决策】

   问题特征                      推荐方法
   ────────────────────────────────────────────
   数据量小 (<500)              ARIMA, SARIMA
   数据量大 (>1000)             XGBoost, RF
   有明显季节性                 SARIMA, TES
   简单趋势                     ARIMA, DES
   需要可解释性                 ARIMA, SARIMA
   追求最高精度                 集成方法
   实时预测需求                 Naive, SES
   多变量预测                   XGBoost, RF

4. 【特征工程要点】
   ✓ 滞后特征 (Lag Features): 捕捉时间依赖性
   ✓ 滚动统计 (Rolling Stats): 平滑噪声
   ✓ 时间特征 (Time Features): 捕捉周期性
   ✓ 差分特征 (Diff Features): 处理非平稳性
   ✓ 交互特征: 捕捉复杂关系

5. 【评估指标选择】
   - MAE: 对异常值不敏感，易于解释
   - RMSE: 对大误差惩罚更重
   - MAPE: 百分比误差，便于跨数据集对比
   - R²: 解释方差比例

6. 【实用建议】
   ✓ 始终从简单模型开始（Naive, SMA）建立基线
   ✓ 可视化是关键，理解数据特征
   ✓ 尝试多种方法并对比
   ✓ 考虑集成方法以提高精度
   ✓ 关注业务需求，不只是统计指标
   ✓ 定期重新训练模型
   ✓ 使用交叉验证评估模型稳定性

7. 【常见陷阱】
   ✗ 过度拟合训练数据
   ✗ 忽略数据的季节性和趋势
   ✗ 使用未来信息（数据泄露）
   ✗ 不进行模型诊断
   ✗ 忽略预测不确定性
   ✗ 不考虑计算成本和实时性

8. 【进阶方向】
   - 深度学习: LSTM, GRU, Transformer
   - 集成学习: Stacking, Blending
   - 概率预测: 预测分布而非点估计
   - 多变量预测: VAR, VARMA
   - 在线学习: 实时模型更新
""")

print("\n" + "=" * 80)
print("✓ 多种预测方法对比教程完成！")
print("✓ Multiple Forecasting Methods Tutorial Completed!")
print("=" * 80)
print(f"\n生成的可视化图表:")
print(f"  18. 18_baseline_models_comparison.png - 基线模型对比")
print(f"  19. 19_exponential_smoothing_comparison.png - 指数平滑方法对比")
print(f"  20. 20_arima_vs_sarima.png - ARIMA vs SARIMA")
print(f"  21. 21_feature_importance_comparison.png - 特征重要性对比")
print(f"  22. 22_ml_methods_comparison.png - 机器学习方法对比")
print(f"  23. 23_all_models_performance.png - 所有模型性能")
print(f"  24. 24_top5_models_forecast.png - Top 5 模型预测")
print(f"  25. 25_error_distribution.png - 误差分布")
print(f"  26. 26_training_time_comparison.png - 训练时间对比")
print(f"  27. 27_accuracy_vs_speed.png - 精度 vs 速度")
print(f"  28. 28_sales_forecasting_example.png - 销售预测案例")
print(f"  29. 29_method_selection_guide.png - 方法选择指南")
print("=" * 80)

# 保存结果到 CSV
results_df.to_csv(get_output_path('forecasting_methods_results.csv'), index=False)
print(f"\n✓ 评估结果已保存到: forecasting_methods_results.csv")
