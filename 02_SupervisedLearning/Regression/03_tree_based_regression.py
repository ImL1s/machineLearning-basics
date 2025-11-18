"""
基於樹的回歸（Tree-based Regression）
使用決策樹及其集成方法進行回歸

包含：
- 決策樹回歸（Decision Tree Regressor）
- 隨機森林回歸（Random Forest Regressor）
- 梯度提升回歸（Gradient Boosting Regressor）
- XGBoost 回歸
- LightGBM 回歸
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import time
import warnings
warnings.filterwarnings('ignore')

# XGBoost 和 LightGBM（可選）
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠ XGBoost 未安裝，相關功能將被跳過")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("⚠ LightGBM 未安裝，相關功能將被跳過")

# 導入工具模塊 / Import utility modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import RANDOM_STATE, TEST_SIZE, setup_chinese_fonts, create_subplots, get_output_path, save_figure

# 設置中文字體 / Setup Chinese fonts
setup_chinese_fonts()

print("=" * 80)
print("基於樹的回歸（Tree-based Regression）教程".center(80))
print("=" * 80)

# ============================================================================
# 第一部分：基於樹的回歸概述
# Part 1: Tree-based Regression Overview
# ============================================================================
print("\n【第一部分】基於樹的回歸概述")
print("-" * 80)
print("""
決策樹回歸原理：
Decision Tree Regression Principle:

決策樹通過遞歸地分割特徵空間，在每個葉節點上用平均值進行預測。
Decision trees recursively split the feature space and predict using the mean value
at each leaf node.

分裂準則（Splitting Criteria）：
• MSE (Mean Squared Error)：最小化均方誤差
• MAE (Mean Absolute Error)：最小化絕對誤差

集成方法（Ensemble Methods）：
1. Bagging（Bootstrap Aggregating）
   - 隨機森林（Random Forest）
   - 通過降低方差提高性能

2. Boosting（提升）
   - 梯度提升（Gradient Boosting）
   - XGBoost, LightGBM
   - 通過降低偏差提高性能

適用場景：
✓ 非線性關係
✓ 特徵交互複雜
✓ 不需要特徵縮放
✓ 可處理混合類型數據
✓ 自動特徵選擇
""")

# ============================================================================
# 評估函數 / Evaluation Functions
# ============================================================================
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name='Model'):
    """
    全面評估回歸模型
    Comprehensive evaluation of regression model

    Args:
        model: 訓練好的模型
        X_train, X_test: 訓練和測試特徵
        y_train, y_test: 訓練和測試標籤
        model_name: 模型名稱

    Returns:
        dict: 評估指標字典
    """
    # 預測
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 計算指標
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\n{model_name}:")
    print(f"  訓練集 R²: {train_r2:.4f}")
    print(f"  測試集 R²: {test_r2:.4f}")
    print(f"  測試集 RMSE: {test_rmse:.4f}")
    print(f"  測試集 MAE: {test_mae:.4f}")

    # 檢查過擬合
    if abs(train_r2 - test_r2) > 0.1:
        print(f"  ⚠ 可能過擬合（R² 差異: {abs(train_r2 - test_r2):.4f}）")

    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse,
        'mae': test_mae,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

# ============================================================================
# 數據準備 / Data Preparation
# ============================================================================
print("\n【數據準備】加載糖尿病數據集")
print("-" * 80)

# 加載數據
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print(f"數據集大小：{X.shape}")
print(f"特徵名稱：{diabetes.feature_names}")
print(f"目標變量範圍：{y.min():.1f} - {y.max():.1f}")

# 數據分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"訓練集大小：{X_train.shape}")
print(f"測試集大小：{X_test.shape}")

# ============================================================================
# 第二部分：決策樹回歸（Decision Tree Regressor）
# Part 2: Decision Tree Regressor
# ============================================================================
print("\n【第二部分】決策樹回歸（Decision Tree Regressor）")
print("-" * 80)
print("""
決策樹回歸原理：
Decision Tree Regression Principle:

1. 選擇最佳分裂點，使得分裂後的 MSE 最小
2. 遞歸地對子節點重複此過程
3. 直到滿足停止條件（如最大深度、最小樣本數）

關鍵參數：
Key Parameters:
• max_depth：最大深度（控制過擬合）
• min_samples_split：分裂所需的最小樣本數
• min_samples_leaf：葉節點最小樣本數
• max_features：分裂時考慮的最大特徵數

優點：
✓ 易於理解和解釋
✓ 不需要特徵縮放
✓ 可處理非線性關係
✓ 自動特徵選擇

缺點：
✗ 容易過擬合
✗ 對數據變化敏感
✗ 預測精度較低
""")

# ============================================================================
# 2.1 基礎決策樹
# ============================================================================
print("\n【2.1】基礎決策樹")

# 訓練基礎決策樹
dt_basic = DecisionTreeRegressor(random_state=RANDOM_STATE)
dt_basic.fit(X_train, y_train)
dt_basic_results = evaluate_model(dt_basic, X_train, X_test, y_train, y_test, "基礎決策樹（無限制）")

# ============================================================================
# 2.2 參數影響分析 - max_depth
# ============================================================================
print("\n【2.2】參數影響分析 - max_depth（最大深度）")
print("-" * 80)

max_depths = [2, 3, 5, 7, 10, 15, None]
depth_results = {}

for depth in max_depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)

    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    depth_name = str(depth) if depth is not None else 'None'
    depth_results[depth_name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'model': dt,
        'y_test_pred': y_test_pred
    }

    print(f"max_depth={depth_name:>4}: 訓練R²={train_r2:.4f}, 測試R²={test_r2:.4f}")

# ============================================================================
# 2.3 可視化：決策樹深度影響（2x2子圖）
# ============================================================================
print("\n【2.3】生成可視化：決策樹深度影響")

fig, axes = create_subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# 繪製 4 個不同深度的決策樹
depths_to_plot = [2, 3, 5, 10]

for idx, depth in enumerate(depths_to_plot):
    plot_tree(depth_results[str(depth)]['model'],
             feature_names=diabetes.feature_names,
             filled=True,
             rounded=True,
             fontsize=8,
             ax=axes[idx])
    axes[idx].set_title(f'決策樹 (max_depth={depth})\n訓練R²={depth_results[str(depth)]["train_r2"]:.3f}, '
                       f'測試R²={depth_results[str(depth)]["test_r2"]:.3f}',
                       fontsize=12, fontweight='bold')

plt.tight_layout()
save_figure(fig, get_output_path('decision_tree_depth_visualization.png', 'Regression'))

# ============================================================================
# 2.4 可視化：深度對性能的影響
# ============================================================================
print("\n【2.4】生成可視化：深度參數影響曲線")

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# 準備數據（排除 None）
numeric_depths = [2, 3, 5, 7, 10, 15]
train_r2_list = [depth_results[str(d)]['train_r2'] for d in numeric_depths]
test_r2_list = [depth_results[str(d)]['test_r2'] for d in numeric_depths]

# 左圖：R² vs 深度
axes[0].plot(numeric_depths, train_r2_list, 'o-', linewidth=2, markersize=8, label='訓練集 R²', color='blue')
axes[0].plot(numeric_depths, test_r2_list, 's-', linewidth=2, markersize=8, label='測試集 R²', color='red')
axes[0].axvline(x=5, color='green', linestyle='--', alpha=0.5, label='推薦值')
axes[0].set_xlabel('最大深度 / Max Depth', fontsize=12)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('決策樹深度對 R² 的影響\nEffect of Tree Depth on R²', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 右圖：過擬合程度
overfitting = [abs(train_r2_list[i] - test_r2_list[i]) for i in range(len(numeric_depths))]
axes[1].bar(numeric_depths, overfitting, color='coral', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('最大深度 / Max Depth', fontsize=12)
axes[1].set_ylabel('過擬合程度 (|訓練R² - 測試R²|)', fontsize=12)
axes[1].set_title('決策樹過擬合程度分析\nOverfitting Analysis', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_figure(fig, get_output_path('decision_tree_depth_analysis.png', 'Regression'))

# ============================================================================
# 2.5 特徵重要性
# ============================================================================
print("\n【2.5】決策樹特徵重要性")
print("-" * 80)

# 使用最佳深度的樹
best_dt = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)
best_dt.fit(X_train, y_train)

feature_importance = best_dt.feature_importances_
feature_names = diabetes.feature_names

# 排序
indices = np.argsort(feature_importance)[::-1]

print("特徵重要性排名：")
for i, idx in enumerate(indices[:5]):
    print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")

# ============================================================================
# 第三部分：隨機森林回歸（Random Forest Regressor）
# Part 3: Random Forest Regressor
# ============================================================================
print("\n【第三部分】隨機森林回歸（Random Forest Regressor）")
print("-" * 80)
print("""
隨機森林原理：
Random Forest Principle:

集成多個決策樹，通過 Bagging（Bootstrap Aggregating）提高性能。
Ensemble of multiple decision trees using Bagging to improve performance.

關鍵機制：
Key Mechanisms:
1. Bootstrap 採樣：有放回地隨機抽取訓練樣本
2. 特徵隨機選擇：每次分裂時隨機選擇特徵子集
3. 平均預測：對所有樹的預測取平均

關鍵參數：
Key Parameters:
• n_estimators：樹的數量（越多越好，但計算成本增加）
• max_depth：每棵樹的最大深度
• max_features：分裂時考慮的最大特徵數
• min_samples_split：分裂所需的最小樣本數
• bootstrap：是否使用 bootstrap 採樣

優點：
✓ 準確度高，泛化能力強
✓ 自動處理過擬合
✓ 可評估特徵重要性
✓ 可並行訓練
✓ 對缺失值不敏感

缺點：
✗ 模型大，預測速度慢
✗ 難以解釋
✗ 訓練時間較長
""")

# ============================================================================
# 3.1 基礎隨機森林
# ============================================================================
print("\n【3.1】基礎隨機森林")

rf_basic = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_basic.fit(X_train, y_train)
rf_basic_results = evaluate_model(rf_basic, X_train, X_test, y_train, y_test, "基礎隨機森林")

# ============================================================================
# 3.2 參數調優 - n_estimators
# ============================================================================
print("\n【3.2】參數調優 - n_estimators（樹的數量）")
print("-" * 80)

n_estimators_list = [10, 50, 100, 200, 300, 500]
n_estimators_results = {}

for n_est in n_estimators_list:
    rf = RandomForestRegressor(n_estimators=n_est, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_test_pred = rf.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)

    n_estimators_results[n_est] = test_r2
    print(f"n_estimators={n_est:>3}: 測試R²={test_r2:.4f}")

# ============================================================================
# 3.3 參數調優 - max_depth
# ============================================================================
print("\n【3.3】參數調優 - max_depth")
print("-" * 80)

rf_depths = [5, 10, 15, 20, None]
rf_depth_results = {}

for depth in rf_depths:
    rf = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    depth_name = str(depth) if depth is not None else 'None'
    rf_depth_results[depth_name] = {
        'train_r2': train_r2,
        'test_r2': test_r2
    }

    print(f"max_depth={depth_name:>4}: 訓練R²={train_r2:.4f}, 測試R²={test_r2:.4f}")

# ============================================================================
# 3.4 可視化：隨機森林參數影響
# ============================================================================
print("\n【3.4】生成可視化：隨機森林參數影響")

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# 左圖：n_estimators 影響
n_est_keys = list(n_estimators_results.keys())
n_est_values = list(n_estimators_results.values())

axes[0].plot(n_est_keys, n_est_values, 'o-', linewidth=2, markersize=8, color='green')
axes[0].set_xlabel('樹的數量 / Number of Trees (n_estimators)', fontsize=12)
axes[0].set_ylabel('測試集 R² Score', fontsize=12)
axes[0].set_title('樹的數量對性能的影響\nEffect of Number of Trees on Performance',
                  fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=max(n_est_values), color='red', linestyle='--', alpha=0.5, label='最佳性能')
axes[0].legend(fontsize=11)

# 右圖：max_depth 影響
rf_depth_numeric = [5, 10, 15, 20]
rf_train_r2 = [rf_depth_results[str(d)]['train_r2'] for d in rf_depth_numeric]
rf_test_r2 = [rf_depth_results[str(d)]['test_r2'] for d in rf_depth_numeric]

axes[1].plot(rf_depth_numeric, rf_train_r2, 'o-', linewidth=2, markersize=8, label='訓練集 R²', color='blue')
axes[1].plot(rf_depth_numeric, rf_test_r2, 's-', linewidth=2, markersize=8, label='測試集 R²', color='red')
axes[1].set_xlabel('最大深度 / Max Depth', fontsize=12)
axes[1].set_ylabel('R² Score', fontsize=12)
axes[1].set_title('樹深度對性能的影響\nEffect of Tree Depth on Performance', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('random_forest_parameter_tuning.png', 'Regression'))

# ============================================================================
# 3.5 特徵重要性分析
# ============================================================================
print("\n【3.5】隨機森林特徵重要性")
print("-" * 80)

# 內置特徵重要性
rf_importance = rf_basic.feature_importances_
rf_indices = np.argsort(rf_importance)[::-1]

print("特徵重要性排名（基於不純度）：")
for i, idx in enumerate(rf_indices[:5]):
    print(f"  {i+1}. {feature_names[idx]}: {rf_importance[idx]:.4f}")

# Permutation Importance
print("\n計算 Permutation Importance（可能需要一些時間）...")
perm_importance = permutation_importance(rf_basic, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
perm_indices = np.argsort(perm_importance.importances_mean)[::-1]

print("\n特徵重要性排名（基於 Permutation）：")
for i, idx in enumerate(perm_indices[:5]):
    print(f"  {i+1}. {feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f}")

# ============================================================================
# 3.6 可視化：特徵重要性對比
# ============================================================================
print("\n【3.6】生成可視化：特徵重要性")

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# 左圖：內置特徵重要性
axes[0].barh(range(len(feature_names)), rf_importance[rf_indices], color='steelblue', alpha=0.7)
axes[0].set_yticks(range(len(feature_names)))
axes[0].set_yticklabels([feature_names[i] for i in rf_indices], fontsize=10)
axes[0].set_xlabel('重要性分數 / Importance Score', fontsize=12)
axes[0].set_title('隨機森林特徵重要性（基於不純度）\nRandom Forest Feature Importance (Impurity-based)',
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# 右圖：Permutation Importance
perm_means = perm_importance.importances_mean[perm_indices]
perm_stds = perm_importance.importances_std[perm_indices]
axes[1].barh(range(len(feature_names)), perm_means, xerr=perm_stds, color='coral', alpha=0.7)
axes[1].set_yticks(range(len(feature_names)))
axes[1].set_yticklabels([feature_names[i] for i in perm_indices], fontsize=10)
axes[1].set_xlabel('重要性分數 / Importance Score', fontsize=12)
axes[1].set_title('隨機森林特徵重要性（基於 Permutation）\nRandom Forest Feature Importance (Permutation)',
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
save_figure(fig, get_output_path('random_forest_feature_importance.png', 'Regression'))

# ============================================================================
# 第四部分：梯度提升回歸（Gradient Boosting Regressor）
# Part 4: Gradient Boosting Regressor
# ============================================================================
print("\n【第四部分】梯度提升回歸（Gradient Boosting Regressor）")
print("-" * 80)
print("""
梯度提升原理：
Gradient Boosting Principle:

通過順序地訓練多個弱學習器（通常是淺決策樹），每個新樹擬合前一輪的殘差。
Sequentially train multiple weak learners (usually shallow trees), with each new tree
fitting the residuals of the previous round.

核心思想：
Core Idea:
1. 初始化：使用簡單模型（如均值）
2. 迭代：訓練新樹擬合殘差（負梯度）
3. 更新：按學習率加權添加新樹
4. 重複直到達到指定樹數量

關鍵參數：
Key Parameters:
• n_estimators：樹的數量（迭代次數）
• learning_rate：學習率，控制每棵樹的貢獻
• max_depth：樹的最大深度（通常較小，3-5）
• subsample：樣本採樣比例（引入隨機性）
• min_samples_split/leaf：控制樹的複雜度

優點：
✓ 預測精度高
✓ 可處理非線性和交互作用
✓ 自動特徵選擇
✓ 魯棒性強

缺點：
✗ 容易過擬合（需調參）
✗ 訓練慢（順序訓練）
✗ 對參數敏感
""")

# ============================================================================
# 4.1 基礎梯度提升
# ============================================================================
print("\n【4.1】基礎梯度提升")

gb_basic = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)
gb_basic.fit(X_train, y_train)
gb_basic_results = evaluate_model(gb_basic, X_train, X_test, y_train, y_test, "基礎梯度提升")

# ============================================================================
# 4.2 參數調優 - learning_rate vs n_estimators
# ============================================================================
print("\n【4.2】參數調優 - learning_rate vs n_estimators")
print("-" * 80)

learning_rates = [0.01, 0.05, 0.1, 0.2]
gb_lr_results = {}

for lr in learning_rates:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, max_depth=3, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)

    y_test_pred = gb.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)

    gb_lr_results[lr] = test_r2
    print(f"learning_rate={lr:.2f}: 測試R²={test_r2:.4f}")

# ============================================================================
# 4.3 學習曲線分析
# ============================================================================
print("\n【4.3】學習曲線分析")
print("-" * 80)

# 訓練一個梯度提升模型並記錄每一步的性能
gb_staged = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)
gb_staged.fit(X_train, y_train)

# 獲取每一步的預測
train_scores = []
test_scores = []

for y_train_pred, y_test_pred in zip(gb_staged.staged_predict(X_train), gb_staged.staged_predict(X_test)):
    train_scores.append(r2_score(y_train, y_train_pred))
    test_scores.append(r2_score(y_test, y_test_pred))

print(f"最終訓練R²: {train_scores[-1]:.4f}")
print(f"最終測試R²: {test_scores[-1]:.4f}")

# ============================================================================
# 4.4 可視化：梯度提升學習曲線
# ============================================================================
print("\n【4.4】生成可視化：梯度提升學習曲線")

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# 左圖：學習曲線
axes[0].plot(range(1, len(train_scores) + 1), train_scores, label='訓練集 R²', linewidth=2, color='blue')
axes[0].plot(range(1, len(test_scores) + 1), test_scores, label='測試集 R²', linewidth=2, color='red')
axes[0].set_xlabel('迭代次數 / Number of Iterations', fontsize=12)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('梯度提升學習曲線\nGradient Boosting Learning Curve', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 右圖：learning_rate 影響
lr_keys = list(gb_lr_results.keys())
lr_values = list(gb_lr_results.values())

axes[1].plot(lr_keys, lr_values, 'o-', linewidth=2, markersize=10, color='green')
axes[1].set_xlabel('學習率 / Learning Rate', fontsize=12)
axes[1].set_ylabel('測試集 R² Score', fontsize=12)
axes[1].set_title('學習率對性能的影響\nEffect of Learning Rate on Performance',
                  fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('gradient_boosting_learning_curve.png', 'Regression'))

# ============================================================================
# 第五部分：XGBoost 回歸
# Part 5: XGBoost Regressor
# ============================================================================
if XGB_AVAILABLE:
    print("\n【第五部分】XGBoost 回歸")
    print("-" * 80)
    print("""
XGBoost 原理和優勢：
XGBoost Principle and Advantages:

XGBoost（Extreme Gradient Boosting）是梯度提升的優化實現。
XGBoost is an optimized implementation of gradient boosting.

創新點：
Innovations:
1. 正則化：L1/L2 正則化防止過擬合
2. 二階導數：使用泰勒展開的二階近似
3. 並行處理：特徵並行化加速訓練
4. 列採樣：類似隨機森林的特徵採樣
5. 缺失值處理：自動學習缺失值的最優方向

關鍵參數：
Key Parameters:
• n_estimators：樹的數量
• learning_rate (eta)：學習率
• max_depth：樹的最大深度
• colsample_bytree：列採樣比例
• subsample：行採樣比例
• reg_alpha：L1 正則化
• reg_lambda：L2 正則化
    """)

    # ============================================================================
    # 5.1 基礎 XGBoost
    # ============================================================================
    print("\n【5.1】基礎 XGBoost")

    xgb_basic = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_basic.fit(X_train, y_train)
    xgb_basic_results = evaluate_model(xgb_basic, X_train, X_test, y_train, y_test, "基礎 XGBoost")

    # ============================================================================
    # 5.2 XGBoost 參數調優
    # ============================================================================
    print("\n【5.2】XGBoost 參數調優")
    print("-" * 80)

    # 測試不同的 max_depth
    xgb_depths = [3, 5, 7, 9]
    xgb_depth_results = {}

    for depth in xgb_depths:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=depth, learning_rate=0.1,
                                random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)

        xgb_depth_results[depth] = test_r2
        print(f"max_depth={depth}: 測試R²={test_r2:.4f}")

    # ============================================================================
    # 5.3 特徵重要性（3種類型）
    # ============================================================================
    print("\n【5.3】XGBoost 特徵重要性")
    print("-" * 80)

    # 獲取不同類型的特徵重要性
    importance_types = ['weight', 'gain', 'cover']
    xgb_importances = {}

    for imp_type in importance_types:
        importance = xgb_basic.get_booster().get_score(importance_type=imp_type)
        # 轉換為數組形式
        imp_array = np.zeros(len(feature_names))
        for key, value in importance.items():
            feature_idx = int(key[1:])  # 'f0' -> 0
            imp_array[feature_idx] = value
        xgb_importances[imp_type] = imp_array

        print(f"\n特徵重要性（{imp_type}）排名：")
        indices = np.argsort(imp_array)[::-1]
        for i, idx in enumerate(indices[:5]):
            print(f"  {i+1}. {feature_names[idx]}: {imp_array[idx]:.2f}")

    # ========================================================================
    # 5.4 可視化：XGBoost 特徵重要性（3種）
    # ========================================================================
    print("\n【5.4】生成可視化：XGBoost 特徵重要性")

    fig, axes = create_subplots(1, 3, figsize=(18, 6))

    for idx, (imp_type, imp_array) in enumerate(xgb_importances.items()):
        indices = np.argsort(imp_array)[::-1]
        axes[idx].barh(range(len(feature_names)), imp_array[indices], color='teal', alpha=0.7)
        axes[idx].set_yticks(range(len(feature_names)))
        axes[idx].set_yticklabels([feature_names[i] for i in indices], fontsize=9)
        axes[idx].set_xlabel('重要性分數', fontsize=11)
        axes[idx].set_title(f'XGBoost 特徵重要性\n({imp_type.upper()})', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_figure(fig, get_output_path('xgboost_feature_importance.png', 'Regression'))

else:
    print("\n⚠ XGBoost 未安裝，跳過 XGBoost 相關內容")
    xgb_basic_results = None

# ============================================================================
# 第六部分：LightGBM 回歸
# Part 6: LightGBM Regressor
# ============================================================================
if LGB_AVAILABLE:
    print("\n【第六部分】LightGBM 回歸")
    print("-" * 80)
    print("""
LightGBM 原理：
LightGBM Principle:

LightGBM 是微軟開發的高效梯度提升框架。
LightGBM is an efficient gradient boosting framework developed by Microsoft.

核心技術：
Core Technologies:
1. Histogram-based 算法：使用直方圖加速
2. Leaf-wise 生長：按葉子節點生長而非層級
3. GOSS（Gradient-based One-Side Sampling）：基於梯度的採樣
4. EFB（Exclusive Feature Bundling）：互斥特徵捆綁

優點：
✓ 訓練速度快
✓ 內存佔用低
✓ 準確度高
✓ 支持大規模數據
    """)

    # ========================================================================
    # 6.1 基礎 LightGBM
    # ========================================================================
    print("\n【6.1】基礎 LightGBM")

    lgb_basic = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    lgb_basic.fit(X_train, y_train)
    lgb_basic_results = evaluate_model(lgb_basic, X_train, X_test, y_train, y_test, "基礎 LightGBM")

else:
    print("\n⚠ LightGBM 未安裝，跳過 LightGBM 相關內容")
    lgb_basic_results = None

# ============================================================================
# 第七部分：綜合對比
# Part 7: Comprehensive Comparison
# ============================================================================
print("\n【第七部分】所有樹模型綜合對比")
print("-" * 80)

# 收集所有模型的結果
all_models = {
    '決策樹': (DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE), None),
    '隨機森林': (rf_basic, rf_basic_results),
    '梯度提升': (gb_basic, gb_basic_results)
}

if XGB_AVAILABLE:
    all_models['XGBoost'] = (xgb_basic, xgb_basic_results)

if LGB_AVAILABLE:
    all_models['LightGBM'] = (lgb_basic, lgb_basic_results)

# 重新評估所有模型並記錄訓練時間
comparison_data = []

for name, (model, cached_results) in all_models.items():
    if cached_results is None:
        # 訓練並計時
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 預測並計時
        start_time = time.time()
        y_test_pred = model.predict(X_test)
        predict_time = time.time() - start_time

        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
    else:
        # 使用緩存的結果
        test_r2 = cached_results['test_r2']
        test_rmse = cached_results['rmse']
        test_mae = cached_results['mae']

        # 重新計時
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        start_time = time.time()
        y_test_pred = model.predict(X_test)
        predict_time = time.time() - start_time

    comparison_data.append({
        '模型': name,
        'R² Score': test_r2,
        'RMSE': test_rmse,
        'MAE': test_mae,
        '訓練時間(s)': train_time,
        '預測時間(s)': predict_time
    })

    print(f"\n{name}:")
    print(f"  R²: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE: {test_mae:.2f}")
    print(f"  訓練時間: {train_time:.4f}s")
    print(f"  預測時間: {predict_time:.4f}s")

# 創建對比表格
comparison_df = pd.DataFrame(comparison_data)
print("\n" + "="*80)
print("模型性能對比表")
print("="*80)
print(comparison_df.to_string(index=False))

# ============================================================================
# 7.1 可視化：所有模型性能對比（柱狀圖）
# ============================================================================
print("\n【7.1】生成可視化：模型性能對比")

fig, axes = create_subplots(2, 2, figsize=(16, 12))

model_names = comparison_df['模型'].values
r2_scores = comparison_df['R² Score'].values
rmse_scores = comparison_df['RMSE'].values
mae_scores = comparison_df['MAE'].values
train_times = comparison_df['訓練時間(s)'].values

# 子圖1：R² Score
axes[0, 0].barh(model_names, r2_scores, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('R² Score', fontsize=12)
axes[0, 0].set_title('R² Score 對比\nR² Score Comparison', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(r2_scores):
    axes[0, 0].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')

# 子圖2：RMSE
axes[0, 1].barh(model_names, rmse_scores, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('RMSE', fontsize=12)
axes[0, 1].set_title('RMSE 對比（越低越好）\nRMSE Comparison (Lower is Better)',
                     fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(rmse_scores):
    axes[0, 1].text(v + 1, i, f'{v:.2f}', va='center', fontsize=10, fontweight='bold')

# 子圖3：MAE
axes[1, 0].barh(model_names, mae_scores, color='lightgreen', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('MAE', fontsize=12)
axes[1, 0].set_title('MAE 對比（越低越好）\nMAE Comparison (Lower is Better)',
                     fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(mae_scores):
    axes[1, 0].text(v + 1, i, f'{v:.2f}', va='center', fontsize=10, fontweight='bold')

# 子圖4：訓練時間
axes[1, 1].barh(model_names, train_times, color='gold', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('訓練時間 (秒)', fontsize=12)
axes[1, 1].set_title('訓練時間對比\nTraining Time Comparison', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(train_times):
    axes[1, 1].text(v + 0.001, i, f'{v:.4f}s', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
save_figure(fig, get_output_path('all_models_performance_comparison.png', 'Regression'))

# ============================================================================
# 7.2 可視化：預測vs真實值（多個模型）
# ============================================================================
print("\n【7.2】生成可視化：預測 vs 真實值")

n_models = len(all_models)
n_rows = (n_models + 2) // 3
n_cols = min(3, n_models)

fig, axes = create_subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
if n_models == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, (name, (model, _)) in enumerate(all_models.items()):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    axes[idx].scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   'r--', lw=2, label='完美預測線')
    axes[idx].set_xlabel('真實值 / True Values', fontsize=11)
    axes[idx].set_ylabel('預測值 / Predictions', fontsize=11)
    axes[idx].set_title(f'{name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.3)

# 隱藏多餘的子圖
for idx in range(n_models, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
save_figure(fig, get_output_path('predictions_vs_actual.png', 'Regression'))

# ============================================================================
# 7.3 可視化：殘差分析
# ============================================================================
print("\n【7.3】生成可視化：殘差分析")

fig, axes = create_subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
if n_models == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, (name, (model, _)) in enumerate(all_models.items()):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    axes[idx].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[idx].set_xlabel('預測值 / Predicted Values', fontsize=11)
    axes[idx].set_ylabel('殘差 / Residuals', fontsize=11)
    axes[idx].set_title(f'{name} - 殘差圖\nResidual Plot', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

# 隱藏多餘的子圖
for idx in range(n_models, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
save_figure(fig, get_output_path('residuals_analysis.png', 'Regression'))

# ============================================================================
# 第八部分：實際應用案例
# Part 8: Practical Application
# ============================================================================
print("\n【第八部分】實際應用案例 - 完整建模流程")
print("-" * 80)

# 創建一個更複雜的合成數據集用於演示
X_demo, y_demo = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    noise=10,
    random_state=RANDOM_STATE
)

X_train_demo, X_test_demo, y_train_demo, y_test_demo = train_test_split(
    X_demo, y_demo, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"演示數據集大小：{X_demo.shape}")
print(f"訓練集：{X_train_demo.shape}, 測試集：{X_test_demo.shape}")

# 使用最佳模型進行預測
print("\n使用隨機森林和 XGBoost 進行預測...")

# 隨機森林
rf_demo = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
rf_demo.fit(X_train_demo, y_train_demo)
y_pred_rf = rf_demo.predict(X_test_demo)
r2_rf = r2_score(y_test_demo, y_pred_rf)

print(f"\n隨機森林：")
print(f"  R²: {r2_rf:.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_demo, y_pred_rf)):.2f}")

if XGB_AVAILABLE:
    # XGBoost
    xgb_demo = xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1,
                               random_state=RANDOM_STATE, n_jobs=-1)
    xgb_demo.fit(X_train_demo, y_train_demo)
    y_pred_xgb = xgb_demo.predict(X_test_demo)
    r2_xgb = r2_score(y_test_demo, y_pred_xgb)

    print(f"\nXGBoost：")
    print(f"  R²: {r2_xgb:.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_demo, y_pred_xgb)):.2f}")

# ============================================================================
# 8.1 可視化：應用案例結果
# ============================================================================
print("\n【8.1】生成可視化：應用案例結果")

if XGB_AVAILABLE:
    fig, axes = create_subplots(1, 2, figsize=(16, 6))

    # 隨機森林
    axes[0].scatter(y_test_demo, y_pred_rf, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_test_demo.min(), y_test_demo.max()],
                [y_test_demo.min(), y_test_demo.max()], 'r--', lw=2)
    axes[0].set_xlabel('真實值 / True Values', fontsize=12)
    axes[0].set_ylabel('預測值 / Predictions', fontsize=12)
    axes[0].set_title(f'隨機森林 - 應用案例\nRandom Forest Application\nR² = {r2_rf:.4f}',
                     fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # XGBoost
    axes[1].scatter(y_test_demo, y_pred_xgb, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    axes[1].plot([y_test_demo.min(), y_test_demo.max()],
                [y_test_demo.min(), y_test_demo.max()], 'r--', lw=2)
    axes[1].set_xlabel('真實值 / True Values', fontsize=12)
    axes[1].set_ylabel('預測值 / Predictions', fontsize=12)
    axes[1].set_title(f'XGBoost - 應用案例\nXGBoost Application\nR² = {r2_xgb:.4f}',
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, get_output_path('application_case_comparison.png', 'Regression'))
else:
    fig, ax = create_subplots(1, 1, figsize=(10, 6))

    ax.scatter(y_test_demo, y_pred_rf, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    ax.plot([y_test_demo.min(), y_test_demo.max()],
           [y_test_demo.min(), y_test_demo.max()], 'r--', lw=2)
    ax.set_xlabel('真實值 / True Values', fontsize=12)
    ax.set_ylabel('預測值 / Predictions', fontsize=12)
    ax.set_title(f'隨機森林 - 應用案例\nRandom Forest Application\nR² = {r2_rf:.4f}',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, get_output_path('application_case_comparison.png', 'Regression'))

# ============================================================================
# 總結報告
# Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("基於樹的回歸教程總結".center(80))
print("=" * 80)

print("""
📊 本教程涵蓋的內容：

1. 決策樹回歸（Decision Tree Regressor）
   ✓ 樹的生長和剪枝
   ✓ max_depth 參數影響分析
   ✓ 樹結構可視化
   ✓ 過擬合問題識別

2. 隨機森林回歸（Random Forest Regressor）
   ✓ Bagging 集成原理
   ✓ n_estimators 和 max_depth 調優
   ✓ 特徵重要性（內置 + Permutation）
   ✓ OOB 評估

3. 梯度提升回歸（Gradient Boosting Regressor）
   ✓ Boosting 提升原理
   ✓ 殘差擬合機制
   ✓ learning_rate 和 n_estimators 權衡
   ✓ 學習曲線分析
""")

if XGB_AVAILABLE:
    print("""
4. XGBoost 回歸
   ✓ 正則化和二階導數優化
   ✓ 參數調優（max_depth, learning_rate）
   ✓ 3種特徵重要性（Weight, Gain, Cover）
   ✓ 與 Gradient Boosting 對比
""")

if LGB_AVAILABLE:
    print("""
5. LightGBM 回歸
   ✓ Histogram-based 算法
   ✓ Leaf-wise 生長策略
   ✓ 訓練速度優勢
""")

print("""
6. 綜合對比
   ✓ 所有模型性能評估（R², RMSE, MAE）
   ✓ 訓練時間和預測時間對比
   ✓ 預測 vs 真實值分析
   ✓ 殘差分析

7. 實際應用
   ✓ 完整建模流程演示
   ✓ 模型選擇建議

📈 生成的可視化圖表：
""")

print("• decision_tree_depth_visualization.png - 決策樹結構可視化")
print("• decision_tree_depth_analysis.png - 深度參數影響")
print("• random_forest_parameter_tuning.png - 隨機森林參數調優")
print("• random_forest_feature_importance.png - 隨機森林特徵重要性")
print("• gradient_boosting_learning_curve.png - 梯度提升學習曲線")
if XGB_AVAILABLE:
    print("• xgboost_feature_importance.png - XGBoost 特徵重要性")
print("• all_models_performance_comparison.png - 所有模型性能對比")
print("• predictions_vs_actual.png - 預測vs真實值")
print("• residuals_analysis.png - 殘差分析")
print("• application_case_comparison.png - 實際應用案例")

print("""
💡 關鍵要點：

模型選擇建議：
• 決策樹：適合快速原型、可解釋性要求高
• 隨機森林：穩定性好、易於使用、首選方案
• 梯度提升：準確度最高，但需要調參
• XGBoost：大規模數據、競賽首選
• LightGBM：超大規模數據、速度要求高

參數調優建議：
1. 決策樹：先調 max_depth（3-10），再調 min_samples_split
2. 隨機森林：n_estimators（100-500），max_depth（10-20）
3. 梯度提升：learning_rate 和 n_estimators 需要權衡
   - 小 learning_rate + 大 n_estimators = 更好性能但訓練慢
4. XGBoost/LightGBM：使用 early_stopping 自動確定最優迭代次數

防止過擬合：
• 限制樹的深度（max_depth）
• 增加最小樣本數（min_samples_split, min_samples_leaf）
• 使用正則化（XGBoost 的 reg_alpha, reg_lambda）
• 使用交叉驗證選擇參數

🎯 下一步：
• 學習模型集成和堆疊（Stacking, Blending）
• 探索深度學習回歸方法
• 實踐 Kaggle 回歸競賽
• 學習超參數優化（Optuna, Hyperopt）
""")

print("=" * 80)
print("教程結束！所有圖表已保存到 output/Regression/ 目錄".center(80))
print("=" * 80)
