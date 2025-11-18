"""
交叉驗證進階策略 | Advanced Cross-Validation Strategies

本教程涵蓋：
1. K-Fold vs Stratified K-Fold 深入對比
2. TimeSeriesSplit - 時間序列專用交叉驗證
3. Nested CV - 嵌套交叉驗證（模型選擇+評估）
4. GroupKFold - 處理分組數據
5. Leave-One-Out CV (LOOCV)
6. 自定義交叉驗證策略
7. 交叉驗證最佳實踐

作者：機器學習評估專家
版本：v2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from collections import defaultdict
from typing import List, Tuple, Dict, Any

# Scikit-learn imports
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    make_classification,
    make_regression
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    GroupKFold,
    LeaveOneOut,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    BaseCrossValidator
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    make_scorer
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print("交叉驗證進階策略教程 | Advanced Cross-Validation Strategies Tutorial".center(100))
print("=" * 100)

# ============================================================================
# 工具函數 | Utility Functions
# ============================================================================

def visualize_cv_splits(cv, X, y, groups=None, n_splits=None, title="Cross-Validation Strategy"):
    """
    可視化交叉驗證分割
    Visualize cross-validation splits

    Parameters:
    -----------
    cv : cross-validator
        交叉驗證器 | Cross-validation object
    X : array-like
        特徵矩陣 | Feature matrix
    y : array-like
        目標變量 | Target variable
    groups : array-like, optional
        分組標籤 | Group labels
    n_splits : int, optional
        要顯示的分割數量 | Number of splits to display
    title : str
        圖表標題 | Chart title
    """
    # 獲取分割 | Get splits
    splits = list(cv.split(X, y, groups))
    if n_splits is not None:
        splits = splits[:n_splits]

    n_splits_actual = len(splits)

    # 創建圖表 | Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, n_splits_actual * 0.5)))

    # 為每個分割創建可視化 | Visualize each split
    for idx, (train_idx, test_idx) in enumerate(splits):
        # 創建索引數組 | Create index array
        indices = np.zeros(len(X))
        indices[train_idx] = 1  # Training = 1
        indices[test_idx] = 2   # Testing = 2

        # 繪製 | Plot
        ax.scatter(range(len(indices)), [idx] * len(indices),
                  c=indices, cmap='RdYlBu', marker='s', s=50,
                  edgecolors='black', linewidth=0.5)

    # 設置標籤和標題 | Set labels and title
    ax.set_yticks(range(n_splits_actual))
    ax.set_yticklabels([f'Split {i+1}' for i in range(n_splits_actual)])
    ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('CV Split', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    # 添加圖例 | Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#313695', edgecolor='black', label='Training Set'),
        Patch(facecolor='#d73027', edgecolor='black', label='Testing Set')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_xlim(-5, len(X) + 5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig, ax


def evaluate_cv_strategy(cv, X, y, model, groups=None, cv_name="CV Strategy"):
    """
    評估交叉驗證策略的性能
    Evaluate cross-validation strategy performance

    Returns:
    --------
    dict : 包含評估結果的字典 | Dictionary containing evaluation results
    """
    start_time = time.time()

    # 執行交叉驗證 | Perform cross-validation
    if groups is not None:
        scores = cross_val_score(model, X, y, cv=cv, groups=groups)
    else:
        scores = cross_val_score(model, X, y, cv=cv)

    elapsed_time = time.time() - start_time

    return {
        'strategy': cv_name,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'min_score': scores.min(),
        'max_score': scores.max(),
        'scores': scores,
        'time': elapsed_time,
        'n_splits': len(scores)
    }


def print_cv_results(results: Dict[str, Any]):
    """
    打印交叉驗證結果
    Print cross-validation results
    """
    print(f"\n{results['strategy']}")
    print("-" * 70)
    print(f"  平均分數 | Mean Score:    {results['mean_score']:.4f}")
    print(f"  標準差 | Std Dev:        {results['std_score']:.4f}")
    print(f"  最小/最大 | Min/Max:      {results['min_score']:.4f} / {results['max_score']:.4f}")
    print(f"  分割數 | N Splits:       {results['n_splits']}")
    print(f"  耗時 | Time:             {results['time']:.4f} seconds")
    print(f"  各折分數 | Fold Scores:   {[f'{s:.4f}' for s in results['scores']]}")


def plot_class_distribution(y, train_idx, test_idx, fold_num, ax):
    """
    繪製訓練集和測試集的類別分佈
    Plot class distribution in training and testing sets
    """
    y_train = y[train_idx]
    y_test = y[test_idx]

    unique_classes = np.unique(y)
    train_dist = [np.sum(y_train == c) / len(y_train) for c in unique_classes]
    test_dist = [np.sum(y_test == c) / len(y_test) for c in unique_classes]

    x = np.arange(len(unique_classes))
    width = 0.35

    ax.bar(x - width/2, train_dist, width, label='Training', alpha=0.8)
    ax.bar(x + width/2, test_dist, width, label='Testing', alpha=0.8)

    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('Proportion', fontweight='bold')
    ax.set_title(f'Fold {fold_num} - Class Distribution', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {c}' for c in unique_classes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')


# ============================================================================
# Part 1: K-Fold vs Stratified K-Fold 深入對比
# Detailed Comparison of K-Fold vs Stratified K-Fold
# ============================================================================

print("\n" + "=" * 100)
print("Part 1: K-Fold vs Stratified K-Fold 深入對比".center(100))
print("=" * 100)

# 創建不平衡數據集 | Create imbalanced dataset
print("\n創建不平衡分類數據集...")
X_imb, y_imb = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    weights=[0.6, 0.3, 0.1],  # 不平衡類別 | Imbalanced classes
    random_state=42
)

print(f"數據集大小 | Dataset size: {X_imb.shape}")
print(f"類別分佈 | Class distribution:")
for cls in np.unique(y_imb):
    count = np.sum(y_imb == cls)
    print(f"  Class {cls}: {count} samples ({count/len(y_imb)*100:.1f}%)")

# 1.1 標準 K-Fold | Standard K-Fold
print("\n【1.1】標準 K-Fold 交叉驗證")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

kfold_results = evaluate_cv_strategy(
    kfold, X_imb, y_imb, rf_model, cv_name="Standard K-Fold (k=5)"
)
print_cv_results(kfold_results)

# 1.2 Stratified K-Fold
print("\n【1.2】Stratified K-Fold 交叉驗證")
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

skfold_results = evaluate_cv_strategy(
    skfold, X_imb, y_imb, rf_model, cv_name="Stratified K-Fold (k=5)"
)
print_cv_results(skfold_results)

# 1.3 可視化對比 | Visualization comparison
print("\n生成 K-Fold vs Stratified K-Fold 對比圖...")

fig1 = plt.figure(figsize=(18, 12))

# 1.3.1 K-Fold 分割可視化
ax1 = plt.subplot(3, 3, 1)
visualize_cv_splits(kfold, X_imb, y_imb, n_splits=5,
                   title="Standard K-Fold Splits")

# 1.3.2 Stratified K-Fold 分割可視化
ax2 = plt.subplot(3, 3, 2)
visualize_cv_splits(skfold, X_imb, y_imb, n_splits=5,
                   title="Stratified K-Fold Splits")

# 1.3.3 類別分佈對比（K-Fold）
ax3 = plt.subplot(3, 3, 4)
for idx, (train_idx, test_idx) in enumerate(kfold.split(X_imb, y_imb)):
    if idx >= 3:  # 只顯示前3個fold
        break
    plot_class_distribution(y_imb, train_idx, test_idx, idx + 1,
                          plt.subplot(3, 3, 4 + idx))

# 1.3.6 類別分佈對比（Stratified K-Fold）
for idx, (train_idx, test_idx) in enumerate(skfold.split(X_imb, y_imb)):
    if idx >= 3:  # 只顯示前3個fold
        break
    plot_class_distribution(y_imb, train_idx, test_idx, idx + 1,
                          plt.subplot(3, 3, 7 + idx))

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/05_ModelEvaluation/cv_part1_kfold_comparison.png',
            dpi=150, bbox_inches='tight')
print("✓ 已保存圖表: cv_part1_kfold_comparison.png")


# ============================================================================
# Part 2: TimeSeriesSplit - 時間序列交叉驗證
# Time Series Cross-Validation
# ============================================================================

print("\n" + "=" * 100)
print("Part 2: TimeSeriesSplit - 時間序列交叉驗證".center(100))
print("=" * 100)

# 2.1 生成時間序列數據 | Generate time series data
print("\n生成模擬股票價格數據...")
np.random.seed(42)
n_samples = 500

# 創建帶有趨勢和季節性的時間序列
time_steps = np.arange(n_samples)
trend = 0.05 * time_steps
seasonality = 10 * np.sin(2 * np.pi * time_steps / 50)
noise = np.random.randn(n_samples) * 5
price = 100 + trend + seasonality + noise

# 創建特徵：滑動窗口
window_size = 10
X_ts = []
y_ts = []

for i in range(window_size, len(price)):
    X_ts.append(price[i-window_size:i])
    y_ts.append(price[i])

X_ts = np.array(X_ts)
y_ts = np.array(y_ts)

print(f"時間序列數據集大小 | Time series dataset size: {X_ts.shape}")
print(f"時間範圍 | Time range: {len(y_ts)} time steps")

# 2.2 TimeSeriesSplit
print("\n【2.2】TimeSeriesSplit 交叉驗證")
tscv = TimeSeriesSplit(n_splits=5)
ridge_model = Ridge(alpha=1.0)

tscv_results = evaluate_cv_strategy(
    tscv, X_ts, y_ts, ridge_model, cv_name="TimeSeriesSplit (n_splits=5)"
)
print_cv_results(tscv_results)

# 2.3 與標準 K-Fold 對比（錯誤示範）
print("\n【2.3】錯誤示範：在時間序列上使用標準 K-Fold")
print("警告：這會導致數據洩漏！| Warning: This causes data leakage!")

kfold_ts = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_ts_results = evaluate_cv_strategy(
    kfold_ts, X_ts, y_ts, ridge_model, cv_name="K-Fold on Time Series (WRONG!)"
)
print_cv_results(kfold_ts_results)

print("\n⚠️  注意：K-Fold 分數虛高，因為使用了未來數據訓練！")
print("   TimeSeriesSplit 分數更真實地反映了模型的預測能力。")

# 2.4 可視化時間序列分割
print("\n生成時間序列分割可視化...")

fig2 = plt.figure(figsize=(18, 10))

# 2.4.1 原始時間序列
ax1 = plt.subplot(3, 2, 1)
ax1.plot(price[:200], linewidth=1.5)
ax1.set_xlabel('Time', fontweight='bold')
ax1.set_ylabel('Price', fontweight='bold')
ax1.set_title('Simulated Stock Price Time Series (First 200 points)',
             fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2.4.2 TimeSeriesSplit 可視化
ax2 = plt.subplot(3, 2, 2)
for idx, (train_idx, test_idx) in enumerate(tscv.split(X_ts)):
    ax2.scatter(train_idx, [idx] * len(train_idx),
               c='blue', marker='s', s=10, alpha=0.6, label='Train' if idx == 0 else '')
    ax2.scatter(test_idx, [idx] * len(test_idx),
               c='red', marker='s', s=10, alpha=0.6, label='Test' if idx == 0 else '')

ax2.set_yticks(range(5))
ax2.set_yticklabels([f'Split {i+1}' for i in range(5)])
ax2.set_xlabel('Time Index', fontweight='bold')
ax2.set_ylabel('Split', fontweight='bold')
ax2.set_title('TimeSeriesSplit Visualization', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')

# 2.4.3 K-Fold 在時間序列上的錯誤分割
ax3 = plt.subplot(3, 2, 3)
for idx, (train_idx, test_idx) in enumerate(kfold_ts.split(X_ts)):
    ax3.scatter(train_idx, [idx] * len(train_idx),
               c='blue', marker='s', s=10, alpha=0.6)
    ax3.scatter(test_idx, [idx] * len(test_idx),
               c='red', marker='s', s=10, alpha=0.6)

ax3.set_yticks(range(5))
ax3.set_yticklabels([f'Split {i+1}' for i in range(5)])
ax3.set_xlabel('Time Index', fontweight='bold')
ax3.set_ylabel('Split', fontweight='bold')
ax3.set_title('K-Fold on Time Series (WRONG - Data Leakage!)',
             fontsize=12, fontweight='bold', color='red')
ax3.grid(True, alpha=0.3, axis='x')

# 2.4.4 分數對比
ax4 = plt.subplot(3, 2, 4)
strategies = ['TimeSeriesSplit\n(Correct)', 'K-Fold\n(Wrong)']
scores = [tscv_results['mean_score'], kfold_ts_results['mean_score']]
errors = [tscv_results['std_score'], kfold_ts_results['std_score']]

bars = ax4.bar(strategies, scores, yerr=errors, capsize=10, alpha=0.7,
              color=['green', 'red'], edgecolor='black', linewidth=2)
ax4.set_ylabel('R² Score', fontweight='bold')
ax4.set_title('Performance Comparison: TimeSeriesSplit vs K-Fold',
             fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 添加數值標籤
for i, (bar, score, err) in enumerate(zip(bars, scores, errors)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.02,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

# 2.4.5 預測示例
ax5 = plt.subplot(3, 2, 5)
# 使用最後一個split進行預測
train_idx, test_idx = list(tscv.split(X_ts))[-1]
ridge_model.fit(X_ts[train_idx], y_ts[train_idx])
y_pred = ridge_model.predict(X_ts[test_idx])

ax5.plot(test_idx, y_ts[test_idx], 'o-', label='Actual', linewidth=2, markersize=4)
ax5.plot(test_idx, y_pred, 's-', label='Predicted', linewidth=2, markersize=4)
ax5.set_xlabel('Time Index', fontweight='bold')
ax5.set_ylabel('Price', fontweight='bold')
ax5.set_title('TimeSeriesSplit: Actual vs Predicted (Last Split)',
             fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 2.4.6 訓練集大小增長
ax6 = plt.subplot(3, 2, 6)
train_sizes = []
test_sizes = []
for train_idx, test_idx in tscv.split(X_ts):
    train_sizes.append(len(train_idx))
    test_sizes.append(len(test_idx))

x_pos = np.arange(len(train_sizes))
ax6.bar(x_pos - 0.2, train_sizes, 0.4, label='Training Size', alpha=0.7)
ax6.bar(x_pos + 0.2, test_sizes, 0.4, label='Testing Size', alpha=0.7)
ax6.set_xlabel('Split Number', fontweight='bold')
ax6.set_ylabel('Sample Count', fontweight='bold')
ax6.set_title('Training and Testing Set Sizes Across Splits',
             fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels([f'Split {i+1}' for i in range(len(train_sizes))])
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/05_ModelEvaluation/cv_part2_timeseries.png',
            dpi=150, bbox_inches='tight')
print("✓ 已保存圖表: cv_part2_timeseries.png")


# ============================================================================
# Part 3: Nested CV - 嵌套交叉驗證
# Nested Cross-Validation for Model Selection and Evaluation
# ============================================================================

print("\n" + "=" * 100)
print("Part 3: Nested CV - 嵌套交叉驗證（模型選擇 + 評估）".center(100))
print("=" * 100)

# 3.1 加載數據
print("\n加載 Breast Cancer 數據集...")
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

print(f"數據集大小 | Dataset size: {X_cancer.shape}")
print(f"類別分佈 | Class distribution: {np.bincount(y_cancer)}")

# 3.2 實現嵌套交叉驗證
print("\n【3.2】實現嵌套交叉驗證")
print("外層 CV：評估模型泛化性能 | Outer CV: Evaluate model generalization")
print("內層 CV：選擇最佳超參數 | Inner CV: Select best hyperparameters")

# 定義參數網格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
}

# 外層和內層 CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 存儲結果
nested_scores = []
non_nested_scores = []
best_params_list = []

print("\n執行嵌套交叉驗證...")
start_time = time.time()

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_cancer, y_cancer)):
    X_train_outer, X_test_outer = X_cancer[train_idx], X_cancer[test_idx]
    y_train_outer, y_test_outer = y_cancer[train_idx], y_cancer[test_idx]

    # 內層 CV：網格搜索找最佳參數
    grid_search = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train_outer, y_train_outer)

    # 使用最佳參數在外層測試集上評估
    nested_score = grid_search.score(X_test_outer, y_test_outer)
    nested_scores.append(nested_score)
    best_params_list.append(grid_search.best_params_)

    print(f"  Fold {fold_idx + 1}/5: Score = {nested_score:.4f}, "
          f"Best Params = {grid_search.best_params_}")

nested_time = time.time() - start_time

# 3.3 非嵌套 CV（錯誤做法）
print("\n【3.3】非嵌套 CV（錯誤做法 - 過於樂觀的估計）")
print("直接在整個數據集上進行網格搜索...")

start_time = time.time()
grid_search_non_nested = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=outer_cv,
    scoring='accuracy',
    n_jobs=-1
)
grid_search_non_nested.fit(X_cancer, y_cancer)
non_nested_score = grid_search_non_nested.best_score_
non_nested_time = time.time() - start_time

print(f"非嵌套 CV 最佳分數: {non_nested_score:.4f}")
print(f"最佳參數: {grid_search_non_nested.best_params_}")

# 3.4 結果對比
print("\n【3.4】嵌套 vs 非嵌套 CV 結果對比")
print("-" * 80)
print(f"嵌套 CV (Nested CV):")
print(f"  平均分數 | Mean:     {np.mean(nested_scores):.4f}")
print(f"  標準差 | Std:       {np.std(nested_scores):.4f}")
print(f"  耗時 | Time:        {nested_time:.2f} seconds")
print(f"\n非嵌套 CV (Non-Nested CV - BIASED!):")
print(f"  分數 | Score:       {non_nested_score:.4f}")
print(f"  耗時 | Time:        {non_nested_time:.2f} seconds")
print(f"\n差異 | Difference:   {non_nested_score - np.mean(nested_scores):.4f}")
print("⚠️  非嵌套 CV 分數通常過於樂觀！")

# 3.5 可視化嵌套 CV
print("\n生成嵌套 CV 可視化...")

fig3 = plt.figure(figsize=(18, 10))

# 3.5.1 嵌套 CV 結構圖
ax1 = plt.subplot(2, 3, 1)
# 繪制嵌套結構
outer_splits = 5
inner_splits = 3
y_pos = 0

for outer in range(outer_splits):
    # 外層
    ax1.barh(y_pos, 1, height=0.8, color='lightblue', edgecolor='black', linewidth=2)
    ax1.text(0.5, y_pos, f'Outer Fold {outer+1}', ha='center', va='center',
            fontweight='bold', fontsize=9)

    y_pos += 1

    # 內層
    for inner in range(inner_splits):
        ax1.barh(y_pos, 0.8, left=0.1, height=0.2, color='lightcoral',
                edgecolor='black', linewidth=1)
        ax1.text(0.5, y_pos, f'Inner {inner+1}', ha='center', va='center', fontsize=7)
        y_pos += 0.3

    y_pos += 0.5

ax1.set_xlim(0, 1.2)
ax1.set_ylim(-0.5, y_pos)
ax1.axis('off')
ax1.set_title('Nested CV Structure\n(5 Outer × 3 Inner Folds)',
             fontsize=12, fontweight='bold')

# 3.5.2 各折分數分佈
ax2 = plt.subplot(2, 3, 2)
ax2.bar(range(1, len(nested_scores) + 1), nested_scores, alpha=0.7,
       edgecolor='black', linewidth=2)
ax2.axhline(y=np.mean(nested_scores), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(nested_scores):.4f}')
ax2.axhline(y=non_nested_score, color='orange', linestyle='--', linewidth=2,
           label=f'Non-Nested: {non_nested_score:.4f}')
ax2.set_xlabel('Outer Fold', fontweight='bold')
ax2.set_ylabel('Accuracy', fontweight='bold')
ax2.set_title('Nested CV Scores per Outer Fold', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3.5.3 嵌套 vs 非嵌套對比
ax3 = plt.subplot(2, 3, 3)
comparison_data = [
    [np.mean(nested_scores), np.std(nested_scores)],
    [non_nested_score, 0]
]
labels = ['Nested CV\n(Unbiased)', 'Non-Nested CV\n(Biased)']
colors = ['green', 'orange']

bars = ax3.bar(labels, [d[0] for d in comparison_data],
              yerr=[d[1] for d in comparison_data],
              capsize=10, alpha=0.7, color=colors, edgecolor='black', linewidth=2)
ax3.set_ylabel('Accuracy', fontweight='bold')
ax3.set_title('Nested vs Non-Nested CV Comparison', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for bar, data in zip(bars, comparison_data):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{data[0]:.4f}', ha='center', va='bottom', fontweight='bold')

# 3.5.4 最佳參數分佈
ax4 = plt.subplot(2, 3, 4)
c_values = [params['C'] for params in best_params_list]
gamma_values = [params['gamma'] if isinstance(params['gamma'], str)
               else f"{params['gamma']:.3f}" for params in best_params_list]

param_text = "Best Parameters per Fold:\n\n"
for i, params in enumerate(best_params_list):
    param_text += f"Fold {i+1}: C={params['C']}, γ={params['gamma']}\n"

ax4.text(0.1, 0.5, param_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax4.axis('off')
ax4.set_title('Selected Hyperparameters\n(Inner CV Results)',
             fontsize=12, fontweight='bold')

# 3.5.5 計算時間對比
ax5 = plt.subplot(2, 3, 5)
time_comparison = [nested_time, non_nested_time]
time_labels = ['Nested CV', 'Non-Nested CV']

bars = ax5.bar(time_labels, time_comparison, alpha=0.7, color=['blue', 'orange'],
              edgecolor='black', linewidth=2)
ax5.set_ylabel('Time (seconds)', fontweight='bold')
ax5.set_title('Computation Time Comparison', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

for bar, t in zip(bars, time_comparison):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{t:.2f}s', ha='center', va='bottom', fontweight='bold')

# 3.5.6 偏差說明
ax6 = plt.subplot(2, 3, 6)
explanation = """
Nested CV 為何重要？
Why Nested CV is Important?

1. 無偏估計 | Unbiased Estimation
   - 外層CV獨立評估性能
   - Outer CV independently evaluates performance

2. 避免數據洩漏 | Avoid Data Leakage
   - 超參數選擇不影響測試集
   - Hyperparameter selection doesn't affect test set

3. 真實泛化能力 | True Generalization
   - 反映模型在新數據上的表現
   - Reflects performance on unseen data

⚠️ 非嵌套CV會過度樂觀！
   Non-nested CV is overly optimistic!
"""

ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes,
        fontsize=9, verticalalignment='top', family='sans-serif',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax6.axis('off')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/05_ModelEvaluation/cv_part3_nested_cv.png',
            dpi=150, bbox_inches='tight')
print("✓ 已保存圖表: cv_part3_nested_cv.png")


# ============================================================================
# Part 4: GroupKFold - 分組交叉驗證
# Group K-Fold for Grouped Data
# ============================================================================

print("\n" + "=" * 100)
print("Part 4: GroupKFold - 分組交叉驗證".center(100))
print("=" * 100)

# 4.1 創建分組數據（模擬醫療場景）
print("\n創建分組數據（模擬：50位患者，每人3-5次測量）...")
print("Creating grouped data (Simulated: 50 patients, 3-5 measurements each)...")

np.random.seed(42)
n_patients = 50
groups = []
X_grouped = []
y_grouped = []

for patient_id in range(n_patients):
    n_measurements = np.random.randint(3, 6)  # 每位患者3-5次測量

    # 每位患者有個體差異（基礎健康狀況）
    patient_bias = np.random.randn() * 2

    for _ in range(n_measurements):
        # 特徵：年齡、BMI、血壓、血糖等（帶有患者個體特徵）
        features = np.random.randn(10) + patient_bias * 0.5
        X_grouped.append(features)

        # 目標：疾病風險（0或1）
        risk_score = patient_bias + np.random.randn() * 0.5
        y_grouped.append(1 if risk_score > 0 else 0)

        groups.append(patient_id)

X_grouped = np.array(X_grouped)
y_grouped = np.array(y_grouped)
groups = np.array(groups)

print(f"總樣本數 | Total samples: {len(X_grouped)}")
print(f"患者數量 | Number of patients: {n_patients}")
print(f"平均每患者測量次數 | Avg measurements per patient: {len(X_grouped) / n_patients:.1f}")
print(f"類別分佈 | Class distribution: {np.bincount(y_grouped)}")

# 4.2 GroupKFold vs 標準 K-Fold
print("\n【4.2】GroupKFold vs 標準 K-Fold")

group_kfold = GroupKFold(n_splits=5)
standard_kfold = KFold(n_splits=5, shuffle=True, random_state=42)

rf_grouped = RandomForestClassifier(n_estimators=100, random_state=42)

# GroupKFold 評估
group_results = evaluate_cv_strategy(
    group_kfold, X_grouped, y_grouped, rf_grouped,
    groups=groups, cv_name="GroupKFold (n_splits=5)"
)
print_cv_results(group_results)

# 標準 K-Fold 評估（錯誤做法）
print("\n⚠️  錯誤做法：在分組數據上使用標準 K-Fold")
standard_results = evaluate_cv_strategy(
    standard_kfold, X_grouped, y_grouped, rf_grouped,
    cv_name="Standard K-Fold on Grouped Data (WRONG!)"
)
print_cv_results(standard_results)

print("\n說明：標準 K-Fold 會將同一患者的數據分散到訓練集和測試集，")
print("      導致數據洩漏和過於樂觀的評估結果！")

# 4.3 可視化分組交叉驗證
print("\n生成 GroupKFold 可視化...")

fig4 = plt.figure(figsize=(18, 12))

# 4.3.1 GroupKFold 分割可視化
ax1 = plt.subplot(3, 3, 1)
for idx, (train_idx, test_idx) in enumerate(group_kfold.split(X_grouped, y_grouped, groups)):
    # 按組著色
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]

    ax1.scatter(train_idx, [idx] * len(train_idx), c='blue',
               marker='s', s=20, alpha=0.6, label='Train' if idx == 0 else '')
    ax1.scatter(test_idx, [idx] * len(test_idx), c='red',
               marker='s', s=20, alpha=0.6, label='Test' if idx == 0 else '')

ax1.set_yticks(range(5))
ax1.set_yticklabels([f'Split {i+1}' for i in range(5)])
ax1.set_xlabel('Sample Index', fontweight='bold')
ax1.set_ylabel('Split', fontweight='bold')
ax1.set_title('GroupKFold Splits\n(Same patient never in both train & test)',
             fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='x')

# 4.3.2 標準 K-Fold 分割（顯示數據洩漏）
ax2 = plt.subplot(3, 3, 2)
for idx, (train_idx, test_idx) in enumerate(standard_kfold.split(X_grouped)):
    ax2.scatter(train_idx, [idx] * len(train_idx), c='blue',
               marker='s', s=20, alpha=0.6)
    ax2.scatter(test_idx, [idx] * len(test_idx), c='red',
               marker='s', s=20, alpha=0.6)

ax2.set_yticks(range(5))
ax2.set_yticklabels([f'Split {i+1}' for i in range(5)])
ax2.set_xlabel('Sample Index', fontweight='bold')
ax2.set_ylabel('Split', fontweight='bold')
ax2.set_title('Standard K-Fold on Grouped Data\n(WRONG - Data Leakage!)',
             fontsize=12, fontweight='bold', color='red')
ax2.grid(True, alpha=0.3, axis='x')

# 4.3.3 檢查數據洩漏
ax3 = plt.subplot(3, 3, 3)
leakage_counts = []
for train_idx, test_idx in standard_kfold.split(X_grouped):
    train_groups_set = set(groups[train_idx])
    test_groups_set = set(groups[test_idx])
    overlap = len(train_groups_set & test_groups_set)
    leakage_counts.append(overlap)

ax3.bar(range(1, 6), leakage_counts, alpha=0.7, color='red',
       edgecolor='black', linewidth=2)
ax3.set_xlabel('Fold', fontweight='bold')
ax3.set_ylabel('# Overlapping Groups', fontweight='bold')
ax3.set_title('Data Leakage in Standard K-Fold\n(Groups in both Train & Test)',
             fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for i, count in enumerate(leakage_counts):
    ax3.text(i + 1, count, str(count), ha='center', va='bottom', fontweight='bold')

# 4.3.4 分組統計
ax4 = plt.subplot(3, 3, 4)
measurements_per_patient = [np.sum(groups == g) for g in range(n_patients)]
ax4.hist(measurements_per_patient, bins=range(3, 7), alpha=0.7,
        edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Measurements per Patient', fontweight='bold')
ax4.set_ylabel('Frequency', fontweight='bold')
ax4.set_title('Distribution of Measurements per Patient',
             fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 4.3.5 性能對比
ax5 = plt.subplot(3, 3, 5)
strategies = ['GroupKFold\n(Correct)', 'K-Fold\n(Leakage)']
scores = [group_results['mean_score'], standard_results['mean_score']]
errors = [group_results['std_score'], standard_results['std_score']]

bars = ax5.bar(strategies, scores, yerr=errors, capsize=10, alpha=0.7,
              color=['green', 'red'], edgecolor='black', linewidth=2)
ax5.set_ylabel('Accuracy', fontweight='bold')
ax5.set_title('Performance: GroupKFold vs K-Fold on Grouped Data',
             fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, scores):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

# 4.3.6 每個fold的訓練/測試組數
ax6 = plt.subplot(3, 3, 6)
train_group_counts = []
test_group_counts = []

for train_idx, test_idx in group_kfold.split(X_grouped, y_grouped, groups):
    train_group_counts.append(len(np.unique(groups[train_idx])))
    test_group_counts.append(len(np.unique(groups[test_idx])))

x_pos = np.arange(5)
ax6.bar(x_pos - 0.2, train_group_counts, 0.4, label='Train Groups', alpha=0.7)
ax6.bar(x_pos + 0.2, test_group_counts, 0.4, label='Test Groups', alpha=0.7)
ax6.set_xlabel('Split', fontweight='bold')
ax6.set_ylabel('Number of Groups', fontweight='bold')
ax6.set_title('Number of Groups in Train/Test Sets', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels([f'Split {i+1}' for i in range(5)])
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# 4.3.7 使用場景說明
ax7 = plt.subplot(3, 3, 7)
use_cases = """
GroupKFold 使用場景
When to Use GroupKFold?

✓ 醫療數據：同一患者多次測量
  Medical: Multiple measurements per patient

✓ 時間序列：同一實體多個時間點
  Time series: Multiple time points per entity

✓ 圖像數據：同一對象多張照片
  Images: Multiple photos of same object

✓ 文本數據：同一作者多篇文章
  Text: Multiple articles by same author

⚠️ 關鍵：確保測試集完全獨立！
   Key: Ensure test set is truly independent!
"""

ax7.text(0.05, 0.95, use_cases, transform=ax7.transAxes,
        fontsize=9, verticalalignment='top', family='sans-serif',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax7.axis('off')

# 4.3.8 分組可視化（患者分組）
ax8 = plt.subplot(3, 3, 8)
# 選擇一個fold顯示分組情況
train_idx, test_idx = list(group_kfold.split(X_grouped, y_grouped, groups))[0]
train_groups_fold = np.unique(groups[train_idx])
test_groups_fold = np.unique(groups[test_idx])

# 創建分組可視化
all_groups_vis = np.zeros(n_patients)
all_groups_vis[train_groups_fold] = 1
all_groups_vis[test_groups_fold] = 2

ax8.imshow([all_groups_vis], aspect='auto', cmap='RdYlBu', interpolation='nearest')
ax8.set_yticks([])
ax8.set_xlabel('Patient ID', fontweight='bold')
ax8.set_title('Group Assignment in First Split\n(Blue=Train, Red=Test)',
             fontsize=12, fontweight='bold')

# 4.3.9 總結表格
ax9 = plt.subplot(3, 3, 9)
summary_data = [
    ['Metric', 'GroupKFold', 'K-Fold'],
    ['Mean Score', f"{group_results['mean_score']:.4f}", f"{standard_results['mean_score']:.4f}"],
    ['Std Dev', f"{group_results['std_score']:.4f}", f"{standard_results['std_score']:.4f}"],
    ['Data Leakage', 'No ✓', 'Yes ✗'],
    ['Realistic', 'Yes ✓', 'No ✗'],
]

table = ax9.table(cellText=summary_data, loc='center', cellLoc='center',
                 colWidths=[0.3, 0.35, 0.35],
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# 設置表頭樣式
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 設置交替行顏色
for i in range(1, len(summary_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax9.axis('off')
ax9.set_title('Comparison Summary', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/05_ModelEvaluation/cv_part4_groupkfold.png',
            dpi=150, bbox_inches='tight')
print("✓ 已保存圖表: cv_part4_groupkfold.png")


# ============================================================================
# Part 5: Leave-One-Out CV (LOOCV)
# ============================================================================

print("\n" + "=" * 100)
print("Part 5: Leave-One-Out CV (LOOCV)".center(100))
print("=" * 100)

# 5.1 創建小數據集
print("\n創建小型數據集（n=50）用於 LOOCV 演示...")
X_small, y_small = make_classification(
    n_samples=50,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

print(f"數據集大小 | Dataset size: {X_small.shape}")

# 5.2 LOOCV 評估
print("\n【5.2】Leave-One-Out 交叉驗證")
loo = LeaveOneOut()
lr_model = LogisticRegression(max_iter=1000, random_state=42)

loo_results = evaluate_cv_strategy(
    loo, X_small, y_small, lr_model, cv_name="Leave-One-Out CV (n=50)"
)
print_cv_results(loo_results)

# 5.3 與 K-Fold 對比
print("\n【5.3】LOOCV vs K-Fold 對比")
kfold_5 = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_10 = KFold(n_splits=10, shuffle=True, random_state=42)

kfold_5_results = evaluate_cv_strategy(
    kfold_5, X_small, y_small, lr_model, cv_name="5-Fold CV"
)
print_cv_results(kfold_5_results)

kfold_10_results = evaluate_cv_strategy(
    kfold_10, X_small, y_small, lr_model, cv_name="10-Fold CV"
)
print_cv_results(kfold_10_results)

# 5.4 計算成本分析
print("\n【5.4】計算成本分析")
print("-" * 80)

# 測試不同數據集大小
sample_sizes = [10, 20, 30, 50, 100, 200]
loo_times = []
kfold_times = []

for size in sample_sizes:
    X_temp, y_temp = make_classification(n_samples=size, n_features=10, random_state=42)

    # LOOCV
    loo_temp = LeaveOneOut()
    start = time.time()
    cross_val_score(lr_model, X_temp, y_temp, cv=loo_temp)
    loo_times.append(time.time() - start)

    # 10-Fold
    kfold_temp = KFold(n_splits=min(10, size), shuffle=True, random_state=42)
    start = time.time()
    cross_val_score(lr_model, X_temp, y_temp, cv=kfold_temp)
    kfold_times.append(time.time() - start)

    print(f"n={size:3d}: LOOCV={loo_times[-1]:.3f}s, 10-Fold={kfold_times[-1]:.3f}s, "
          f"Ratio={loo_times[-1]/kfold_times[-1]:.2f}x")

# 5.5 可視化 LOOCV
print("\n生成 LOOCV 可視化...")

fig5 = plt.figure(figsize=(18, 10))

# 5.5.1 LOOCV 分割示例（前20個樣本）
ax1 = plt.subplot(2, 3, 1)
n_show = 20
for idx, (train_idx, test_idx) in enumerate(loo.split(X_small)):
    if idx >= n_show:
        break
    ax1.scatter(train_idx, [idx] * len(train_idx), c='blue',
               marker='s', s=10, alpha=0.6, label='Train' if idx == 0 else '')
    ax1.scatter(test_idx, [idx] * len(test_idx), c='red',
               marker='s', s=30, alpha=0.8, label='Test' if idx == 0 else '')

ax1.set_yticks(range(0, n_show, 5))
ax1.set_yticklabels([f'Split {i+1}' for i in range(0, n_show, 5)])
ax1.set_xlabel('Sample Index', fontweight='bold')
ax1.set_ylabel('Split', fontweight='bold')
ax1.set_title(f'LOOCV Splits (First {n_show} of {len(X_small)} total)',
             fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='x')

# 5.5.2 性能對比
ax2 = plt.subplot(2, 3, 2)
strategies = ['LOOCV\n(n=50)', '10-Fold', '5-Fold']
scores = [loo_results['mean_score'], kfold_10_results['mean_score'],
         kfold_5_results['mean_score']]
errors = [loo_results['std_score'], kfold_10_results['std_score'],
         kfold_5_results['std_score']]

bars = ax2.bar(strategies, scores, yerr=errors, capsize=10, alpha=0.7,
              color=['purple', 'blue', 'green'], edgecolor='black', linewidth=2)
ax2.set_ylabel('Accuracy', fontweight='bold')
ax2.set_title('Performance Comparison', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, scores):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 5.5.3 計算時間對比
ax3 = plt.subplot(2, 3, 3)
time_comparison = [loo_results['time'], kfold_10_results['time'],
                  kfold_5_results['time']]

bars = ax3.bar(strategies, time_comparison, alpha=0.7,
              color=['purple', 'blue', 'green'], edgecolor='black', linewidth=2)
ax3.set_ylabel('Time (seconds)', fontweight='bold')
ax3.set_title('Computation Time Comparison', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for bar, t in zip(bars, time_comparison):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{t:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 5.5.4 時間複雜度隨樣本數增長
ax4 = plt.subplot(2, 3, 4)
ax4.plot(sample_sizes, loo_times, 'o-', linewidth=2, markersize=8, label='LOOCV')
ax4.plot(sample_sizes, kfold_times, 's-', linewidth=2, markersize=8, label='10-Fold')
ax4.set_xlabel('Sample Size', fontweight='bold')
ax4.set_ylabel('Time (seconds)', fontweight='bold')
ax4.set_title('Computation Time vs Sample Size', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5.5.5 方差對比
ax5 = plt.subplot(2, 3, 5)
variance_data = [loo_results['std_score'], kfold_10_results['std_score'],
                kfold_5_results['std_score']]

bars = ax5.bar(strategies, variance_data, alpha=0.7,
              color=['purple', 'blue', 'green'], edgecolor='black', linewidth=2)
ax5.set_ylabel('Standard Deviation', fontweight='bold')
ax5.set_title('Variance of CV Estimates', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

for bar, var in zip(bars, variance_data):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{var:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 5.5.6 LOOCV 優缺點總結
ax6 = plt.subplot(2, 3, 6)
summary = """
LOOCV 優缺點分析
Pros and Cons of LOOCV

優點 | Advantages:
✓ 最大化訓練數據利用
  Maximizes training data usage
✓ 確定性結果（無隨機性）
  Deterministic (no randomness)
✓ 適合小數據集
  Good for small datasets

缺點 | Disadvantages:
✗ 計算成本高（n次訓練）
  High computational cost (n trainings)
✗ 高方差估計
  High variance estimates
✗ 對異常值敏感
  Sensitive to outliers

建議 | Recommendation:
• n < 100: 考慮 LOOCV
  Consider LOOCV if n < 100
• n > 100: 使用 10-Fold
  Use 10-Fold if n > 100
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
        fontsize=8.5, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
ax6.axis('off')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/05_ModelEvaluation/cv_part5_loocv.png',
            dpi=150, bbox_inches='tight')
print("✓ 已保存圖表: cv_part5_loocv.png")


# ============================================================================
# Part 6: 自定義交叉驗證策略
# Custom Cross-Validation Strategies
# ============================================================================

print("\n" + "=" * 100)
print("Part 6: 自定義交叉驗證策略".center(100))
print("=" * 100)

# 6.1 實現自定義分割器
print("\n【6.1】實現自定義交叉驗證分割器")

class StratifiedGroupKFold(BaseCrossValidator):
    """
    結合 Stratified 和 Group 的交叉驗證
    Combines stratification with group-based splitting

    用於：既要保持類別平衡，又要避免組內數據洩漏
    Use case: Maintain class balance while avoiding group data leakage
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups):
        """生成索引以分割數據為訓練集和測試集"""
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        n_samples = len(X)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if n_groups < self.n_splits:
            raise ValueError(f"Cannot have {self.n_splits} splits with only "
                           f"{n_groups} groups.")

        # 計算每個組的主要類別
        group_to_class = {}
        for group in unique_groups:
            group_mask = groups == group
            group_labels = y[group_mask]
            # 使用多數類作為組的標籤
            group_to_class[group] = np.bincount(group_labels).argmax()

        # 按類別分組
        class_to_groups = defaultdict(list)
        for group, cls in group_to_class.items():
            class_to_groups[cls].append(group)

        # 為每個類別的組創建分割
        rng = np.random.RandomState(self.random_state)
        if self.shuffle:
            for cls in class_to_groups:
                rng.shuffle(class_to_groups[cls])

        # 創建 fold
        folds = [[] for _ in range(self.n_splits)]

        # 將每個類別的組均勻分配到各個 fold
        for cls, grps in class_to_groups.items():
            for i, grp in enumerate(grps):
                fold_idx = i % self.n_splits
                folds[fold_idx].append(grp)

        # 生成訓練/測試索引
        for i in range(self.n_splits):
            test_groups = folds[i]
            train_groups = [g for j, fold in enumerate(folds) if j != i for g in fold]

            test_indices = np.where(np.isin(groups, test_groups))[0]
            train_indices = np.where(np.isin(groups, train_groups))[0]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# 6.2 測試自定義分割器
print("\n【6.2】測試自定義 StratifiedGroupKFold")

# 創建測試數據（模擬：患者數據，有分組且類別不平衡）
np.random.seed(42)
n_patients_custom = 60
X_custom = []
y_custom = []
groups_custom = []

for patient_id in range(n_patients_custom):
    n_measurements = np.random.randint(2, 5)

    # 患者有不同的疾病風險（創建不平衡）
    if patient_id < 40:  # 大部分患者低風險
        patient_class = 0
        class_prob = 0.9
    else:  # 少部分患者高風險
        patient_class = 1
        class_prob = 0.9

    for _ in range(n_measurements):
        features = np.random.randn(10) + patient_class * 2
        X_custom.append(features)

        # 有一定概率是患者的主要類別
        y_custom.append(patient_class if np.random.rand() < class_prob else 1 - patient_class)
        groups_custom.append(patient_id)

X_custom = np.array(X_custom)
y_custom = np.array(y_custom)
groups_custom = np.array(groups_custom)

print(f"總樣本數 | Total samples: {len(X_custom)}")
print(f"患者數量 | Number of patients: {n_patients_custom}")
print(f"類別分佈 | Class distribution: {np.bincount(y_custom)}")
print(f"  Class 0: {np.sum(y_custom == 0)} ({np.sum(y_custom == 0)/len(y_custom)*100:.1f}%)")
print(f"  Class 1: {np.sum(y_custom == 1)} ({np.sum(y_custom == 1)/len(y_custom)*100:.1f}%)")

# 測試自定義分割器
stratified_group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
rf_custom = RandomForestClassifier(n_estimators=100, random_state=42)

custom_results = evaluate_cv_strategy(
    stratified_group_kfold, X_custom, y_custom, rf_custom,
    groups=groups_custom, cv_name="Custom StratifiedGroupKFold"
)
print_cv_results(custom_results)

# 與標準方法對比
print("\n【6.3】與標準方法對比")

# Standard GroupKFold（不考慮分層）
group_kfold_custom = GroupKFold(n_splits=5)
group_results_custom = evaluate_cv_strategy(
    group_kfold_custom, X_custom, y_custom, rf_custom,
    groups=groups_custom, cv_name="Standard GroupKFold"
)
print_cv_results(group_results_custom)

# Standard StratifiedKFold（不考慮分組）
stratified_kfold_custom = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_results_custom = evaluate_cv_strategy(
    stratified_kfold_custom, X_custom, y_custom, rf_custom,
    cv_name="Standard StratifiedKFold (ignores groups - WRONG!)"
)
print_cv_results(stratified_results_custom)

# 6.4 可視化自定義策略
print("\n生成自定義交叉驗證策略可視化...")

fig6 = plt.figure(figsize=(18, 10))

# 6.4.1 自定義分割器的分割可視化
ax1 = plt.subplot(2, 3, 1)
visualize_cv_splits(stratified_group_kfold, X_custom, y_custom,
                   groups=groups_custom, n_splits=5,
                   title="Custom StratifiedGroupKFold Splits")

# 6.4.2 檢查每個 fold 的類別平衡
ax2 = plt.subplot(2, 3, 2)
fold_class_distributions = {'fold': [], 'class_0_pct': [], 'class_1_pct': []}

for idx, (train_idx, test_idx) in enumerate(stratified_group_kfold.split(X_custom, y_custom, groups_custom)):
    y_test_fold = y_custom[test_idx]
    class_0_pct = np.sum(y_test_fold == 0) / len(y_test_fold) * 100
    class_1_pct = np.sum(y_test_fold == 1) / len(y_test_fold) * 100

    fold_class_distributions['fold'].append(f'Fold {idx+1}')
    fold_class_distributions['class_0_pct'].append(class_0_pct)
    fold_class_distributions['class_1_pct'].append(class_1_pct)

x_pos = np.arange(5)
width = 0.35

ax2.bar(x_pos - width/2, fold_class_distributions['class_0_pct'], width,
       label='Class 0', alpha=0.7)
ax2.bar(x_pos + width/2, fold_class_distributions['class_1_pct'], width,
       label='Class 1', alpha=0.7)

# 添加整體分佈線
overall_class_0_pct = np.sum(y_custom == 0) / len(y_custom) * 100
overall_class_1_pct = np.sum(y_custom == 1) / len(y_custom) * 100
ax2.axhline(y=overall_class_0_pct, color='C0', linestyle='--', linewidth=2, alpha=0.5)
ax2.axhline(y=overall_class_1_pct, color='C1', linestyle='--', linewidth=2, alpha=0.5)

ax2.set_xlabel('Fold', fontweight='bold')
ax2.set_ylabel('Class Percentage (%)', fontweight='bold')
ax2.set_title('Class Distribution per Fold\n(Custom StratifiedGroupKFold)',
             fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(fold_class_distributions['fold'])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 6.4.3 檢查分組完整性（確保無數據洩漏）
ax3 = plt.subplot(2, 3, 3)
leakage_check = []

for train_idx, test_idx in stratified_group_kfold.split(X_custom, y_custom, groups_custom):
    train_groups = set(groups_custom[train_idx])
    test_groups = set(groups_custom[test_idx])
    overlap = len(train_groups & test_groups)
    leakage_check.append(overlap)

ax3.bar(range(1, 6), leakage_check, alpha=0.7, color='green' if all(x == 0 for x in leakage_check) else 'red',
       edgecolor='black', linewidth=2)
ax3.set_xlabel('Fold', fontweight='bold')
ax3.set_ylabel('Overlapping Groups', fontweight='bold')
ax3.set_title('Data Leakage Check\n(Should be 0 for all folds)',
             fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(-0.5, max(leakage_check) + 1 if max(leakage_check) > 0 else 1)

for i, count in enumerate(leakage_check):
    ax3.text(i + 1, count + 0.1, '✓' if count == 0 else f'✗ {count}',
            ha='center', va='bottom', fontweight='bold', fontsize=12,
            color='green' if count == 0 else 'red')

# 6.4.4 性能對比
ax4 = plt.subplot(2, 3, 4)
strategies_custom = ['Custom\nStratifiedGroup', 'GroupKFold', 'StratifiedKFold\n(Wrong)']
scores_custom = [custom_results['mean_score'],
                group_results_custom['mean_score'],
                stratified_results_custom['mean_score']]
errors_custom = [custom_results['std_score'],
                group_results_custom['std_score'],
                stratified_results_custom['std_score']]

bars = ax4.bar(strategies_custom, scores_custom, yerr=errors_custom, capsize=10,
              alpha=0.7, color=['purple', 'blue', 'orange'],
              edgecolor='black', linewidth=2)
ax4.set_ylabel('Accuracy', fontweight='bold')
ax4.set_title('Performance Comparison', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, scores_custom):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 6.4.5 自定義策略的特性比較
ax5 = plt.subplot(2, 3, 5)
features_comparison = """
策略特性對比
Strategy Features Comparison

特性                Custom    Group    Stratified
                   SG-KFold  KFold    KFold
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
保持類別平衡          ✓         ✗         ✓
Class Balance

避免組數據洩漏        ✓         ✓         ✗
No Group Leakage

適用不平衡數據        ✓         ✗         ✓
Imbalanced Data

適用分組數據          ✓         ✓         ✗
Grouped Data

計算複雜度           中等      低        低
Complexity          Medium    Low       Low
"""

ax5.text(0.05, 0.95, features_comparison, transform=ax5.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax5.axis('off')

# 6.4.6 實現說明
ax6 = plt.subplot(2, 3, 6)
implementation_notes = """
自定義 CV 實現步驟
How to Implement Custom CV

1. 繼承 BaseCrossValidator
   Inherit from BaseCrossValidator

2. 實現 split(X, y, groups) 方法
   Implement split(X, y, groups) method

3. 實現 get_n_splits() 方法
   Implement get_n_splits() method

4. yield (train_indices, test_indices)
   Yield train/test index pairs

5. 確保：
   Ensure:
   • 無數據洩漏 | No data leakage
   • 完整覆蓋 | Complete coverage
   • 無重疊測試集 | No overlapping test sets

💡 提示：先在小數據上測試！
   Tip: Test on small data first!
"""

ax6.text(0.05, 0.95, implementation_notes, transform=ax6.transAxes,
        fontsize=9, verticalalignment='top', family='sans-serif',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
ax6.axis('off')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/05_ModelEvaluation/cv_part6_custom_cv.png',
            dpi=150, bbox_inches='tight')
print("✓ 已保存圖表: cv_part6_custom_cv.png")


# ============================================================================
# Part 7: 交叉驗證最佳實踐與決策指南
# Best Practices and Decision Guide
# ============================================================================

print("\n" + "=" * 100)
print("Part 7: 交叉驗證最佳實踐與決策指南".center(100))
print("=" * 100)

# 7.1 創建綜合性能對比表
print("\n【7.1】綜合性能對比表")

comparison_df = pd.DataFrame({
    'Strategy': [
        'K-Fold',
        'Stratified K-Fold',
        'TimeSeriesSplit',
        'GroupKFold',
        'LOOCV',
        'Nested CV',
        'Custom StratifiedGroupKFold'
    ],
    'Use Case': [
        'General purpose',
        'Imbalanced classes',
        'Time series data',
        'Grouped data',
        'Small datasets',
        'Model selection',
        'Grouped + imbalanced'
    ],
    'Data Leakage Risk': [
        'Low',
        'Low',
        'Low (if used correctly)',
        'Low',
        'Low',
        'Very Low',
        'Very Low'
    ],
    'Computational Cost': [
        'Medium',
        'Medium',
        'Medium',
        'Medium',
        'High',
        'Very High',
        'Medium-High'
    ],
    'Variance': [
        'Medium',
        'Medium',
        'Higher',
        'Medium',
        'Higher',
        'Lower',
        'Medium'
    ],
    'Min Sample Size': [
        '100+',
        '100+',
        '100+',
        '50+ groups',
        '< 100',
        '200+',
        '50+ groups'
    ]
})

print("\n" + "=" * 120)
print(comparison_df.to_string(index=False))
print("=" * 120)

# 7.2 決策樹函數
def suggest_cv_strategy(n_samples, has_groups=False, is_time_series=False,
                       is_imbalanced=False, need_model_selection=False):
    """
    根據數據特徵推薦交叉驗證策略
    Suggest CV strategy based on data characteristics

    Parameters:
    -----------
    n_samples : int
        樣本數量 | Number of samples
    has_groups : bool
        是否有分組結構 | Whether data has group structure
    is_time_series : bool
        是否為時間序列 | Whether data is time series
    is_imbalanced : bool
        類別是否不平衡 | Whether classes are imbalanced
    need_model_selection : bool
        是否需要模型選擇 | Whether model selection is needed

    Returns:
    --------
    dict : 推薦策略和原因 | Recommended strategy and reasoning
    """
    recommendations = []

    # 時間序列優先
    if is_time_series:
        recommendations.append({
            'strategy': 'TimeSeriesSplit',
            'reason': '時間序列數據必須使用 TimeSeriesSplit 避免使用未來數據',
            'priority': 1
        })

    # 分組數據
    elif has_groups:
        if is_imbalanced:
            recommendations.append({
                'strategy': 'Custom StratifiedGroupKFold',
                'reason': '分組數據 + 類別不平衡，需要自定義策略',
                'priority': 1
            })
        else:
            recommendations.append({
                'strategy': 'GroupKFold',
                'reason': '分組數據必須使用 GroupKFold 避免數據洩漏',
                'priority': 1
            })

    # 小數據集
    elif n_samples < 50:
        recommendations.append({
            'strategy': 'LOOCV',
            'reason': '小數據集（n<50）適合使用 LOOCV 最大化訓練數據',
            'priority': 1
        })

    # 類別不平衡
    elif is_imbalanced:
        recommendations.append({
            'strategy': 'Stratified K-Fold',
            'reason': '類別不平衡數據應使用 Stratified K-Fold 保持類別比例',
            'priority': 1
        })

    # 一般情況
    else:
        if n_samples < 100:
            recommendations.append({
                'strategy': '10-Fold CV',
                'reason': '中小數據集（50-100樣本）推薦 10-Fold',
                'priority': 1
            })
        else:
            recommendations.append({
                'strategy': '5-Fold CV',
                'reason': '大數據集（100+樣本）推薦 5-Fold 平衡性能和計算成本',
                'priority': 1
            })

    # 如果需要模型選擇，包裝為嵌套 CV
    if need_model_selection:
        base_strategy = recommendations[0]['strategy']
        recommendations.insert(0, {
            'strategy': f'Nested CV (outer: {base_strategy})',
            'reason': '需要模型選擇時必須使用嵌套 CV 避免過擬合',
            'priority': 0
        })

    return recommendations


print("\n【7.2】交叉驗證策略決策示例")
print("-" * 100)

# 測試案例
test_cases = [
    {
        'name': '大型平衡數據集',
        'params': {'n_samples': 1000, 'has_groups': False, 'is_time_series': False,
                  'is_imbalanced': False, 'need_model_selection': False}
    },
    {
        'name': '不平衡分類數據',
        'params': {'n_samples': 500, 'has_groups': False, 'is_time_series': False,
                  'is_imbalanced': True, 'need_model_selection': False}
    },
    {
        'name': '時間序列預測',
        'params': {'n_samples': 800, 'has_groups': False, 'is_time_series': True,
                  'is_imbalanced': False, 'need_model_selection': False}
    },
    {
        'name': '醫療分組數據（不平衡）',
        'params': {'n_samples': 300, 'has_groups': True, 'is_time_series': False,
                  'is_imbalanced': True, 'need_model_selection': False}
    },
    {
        'name': '小數據集',
        'params': {'n_samples': 30, 'has_groups': False, 'is_time_series': False,
                  'is_imbalanced': False, 'need_model_selection': False}
    },
    {
        'name': '需要超參數調優',
        'params': {'n_samples': 500, 'has_groups': False, 'is_time_series': False,
                  'is_imbalanced': True, 'need_model_selection': True}
    }
]

for case in test_cases:
    print(f"\n案例：{case['name']}")
    print(f"  參數：{case['params']}")
    recommendations = suggest_cv_strategy(**case['params'])
    print("  推薦策略：")
    for i, rec in enumerate(recommendations, 1):
        print(f"    {i}. {rec['strategy']}")
        print(f"       原因：{rec['reason']}")


# 7.3 最佳實踐總結可視化
print("\n生成最佳實踐總結可視化...")

fig7 = plt.figure(figsize=(20, 12))

# 7.3.1 決策樹可視化
ax1 = plt.subplot(3, 3, 1)
decision_tree = """
交叉驗證策略決策樹
CV Strategy Decision Tree

開始 START
  │
  ├─ 時間序列? → YES → TimeSeriesSplit
  │   Time Series?
  │
  ├─ 有分組? → YES ┬─ 不平衡? → YES → Custom
  │   Grouped?     │   Imbalanced?     StratifiedGroupKFold
  │                └─ NO → GroupKFold
  │
  ├─ n < 50? → YES → LOOCV
  │   Small?
  │
  ├─ 不平衡? → YES → Stratified K-Fold
  │   Imbalanced?
  │
  └─ 其他 → K-Fold (k=5 或 10)
      Other    K-Fold (k=5 or 10)
"""

ax1.text(0.05, 0.95, decision_tree, transform=ax1.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax1.axis('off')
ax1.set_title('Decision Tree for Choosing CV Strategy',
             fontsize=13, fontweight='bold', pad=20)

# 7.3.2 各策略的訓練/測試分割比例
ax2 = plt.subplot(3, 3, 2)
strategies_split = ['5-Fold', '10-Fold', 'LOOCV\n(n=50)']
train_ratios = [80, 90, 98]
test_ratios = [20, 10, 2]

x_pos = np.arange(len(strategies_split))
p1 = ax2.bar(x_pos, train_ratios, alpha=0.7, label='Training', color='skyblue')
p2 = ax2.bar(x_pos, test_ratios, bottom=train_ratios, alpha=0.7,
            label='Testing', color='salmon')

ax2.set_ylabel('Percentage (%)', fontweight='bold')
ax2.set_title('Train/Test Split Ratios', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(strategies_split)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 添加數值標籤
for i, (train, test) in enumerate(zip(train_ratios, test_ratios)):
    ax2.text(i, train/2, f'{train}%', ha='center', va='center',
            fontweight='bold', color='darkblue')
    ax2.text(i, train + test/2, f'{test}%', ha='center', va='center',
            fontweight='bold', color='darkred')

# 7.3.3 計算成本對比
ax3 = plt.subplot(3, 3, 3)
strategies_cost = ['5-Fold', '10-Fold', 'LOOCV\n(n=100)', 'Nested CV\n(5×3)']
n_trainings = [5, 10, 100, 15]  # 訓練次數

bars = ax3.bar(strategies_cost, n_trainings, alpha=0.7,
              color=['green', 'yellow', 'orange', 'red'],
              edgecolor='black', linewidth=2)
ax3.set_ylabel('Number of Model Trainings', fontweight='bold')
ax3.set_title('Computational Cost Comparison', fontsize=12, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, axis='y')

for bar, n in zip(bars, n_trainings):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            str(n), ha='center', va='bottom', fontweight='bold')

# 7.3.4 常見陷阱
ax4 = plt.subplot(3, 3, 4)
pitfalls = """
常見陷阱與錯誤
Common Pitfalls

❌ 1. 時間序列用標準 K-Fold
   Using K-Fold on time series
   → 使用未來數據訓練

❌ 2. 分組數據未使用 GroupKFold
   Not using GroupKFold for grouped data
   → 數據洩漏

❌ 3. 非嵌套 CV 做模型選擇
   Non-nested CV for model selection
   → 過於樂觀的評估

❌ 4. 不平衡數據用標準 K-Fold
   K-Fold on imbalanced data
   → 測試集類別分佈不一致

❌ 5. 數據預處理在 CV 外
   Preprocessing outside CV
   → 數據洩漏
"""

ax4.text(0.05, 0.95, pitfalls, transform=ax4.transAxes,
        fontsize=9, verticalalignment='top', family='sans-serif',
        bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
ax4.axis('off')

# 7.3.5 推薦的 k 值選擇
ax5 = plt.subplot(3, 3, 5)
sample_sizes_k = [50, 100, 200, 500, 1000, 5000, 10000]
recommended_k = [10, 10, 10, 5, 5, 5, 5]

ax5.plot(sample_sizes_k, recommended_k, 'o-', linewidth=3, markersize=10,
        color='purple')
ax5.set_xlabel('Sample Size', fontweight='bold')
ax5.set_ylabel('Recommended k', fontweight='bold')
ax5.set_title('Recommended k Value vs Sample Size', fontsize=12, fontweight='bold')
ax5.set_xscale('log')
ax5.grid(True, alpha=0.3)
ax5.set_yticks([5, 10])
ax5.set_ylim(4, 11)

# 添加區域標註
ax5.axhspan(9.5, 10.5, alpha=0.2, color='blue', label='k=10 for n<500')
ax5.axhspan(4.5, 5.5, alpha=0.2, color='green', label='k=5 for n≥500')
ax5.legend(loc='upper right')

# 7.3.6 方差-偏差權衡
ax6 = plt.subplot(3, 3, 6)
k_values = [2, 3, 5, 10, 20]
variance = [0.15, 0.12, 0.08, 0.06, 0.05]  # 模擬方差
bias = [0.05, 0.04, 0.03, 0.04, 0.06]      # 模擬偏差

ax6.plot(k_values, variance, 'o-', linewidth=2, markersize=8, label='Variance')
ax6.plot(k_values, bias, 's-', linewidth=2, markersize=8, label='Bias')
ax6.plot(k_values, np.array(variance) + np.array(bias), '^-',
        linewidth=2, markersize=8, label='Total Error')

ax6.set_xlabel('k (number of folds)', fontweight='bold')
ax6.set_ylabel('Error', fontweight='bold')
ax6.set_title('Bias-Variance Trade-off in k-Fold CV', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 標註最優點
optimal_idx = np.argmin(np.array(variance) + np.array(bias))
ax6.plot(k_values[optimal_idx], (np.array(variance) + np.array(bias))[optimal_idx],
        'r*', markersize=20, label='Optimal')

# 7.3.7 數據洩漏檢查清單
ax7 = plt.subplot(3, 3, 7)
checklist = """
數據洩漏檢查清單
Data Leakage Checklist

✓ 預處理在 CV 內部進行?
  Preprocessing inside CV?

✓ 特徵選擇在 CV 內部?
  Feature selection inside CV?

✓ 時間序列使用 TimeSeriesSplit?
  TimeSeriesSplit for time series?

✓ 分組數據使用 GroupKFold?
  GroupKFold for grouped data?

✓ 測試集完全獨立?
  Test set completely independent?

✓ 超參數調優使用嵌套 CV?
  Nested CV for hyperparameter tuning?
"""

ax7.text(0.05, 0.95, checklist, transform=ax7.transAxes,
        fontsize=9, verticalalignment='top', family='sans-serif',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax7.axis('off')

# 7.3.8 性能對比雷達圖
ax8 = plt.subplot(3, 3, 8, projection='polar')

categories = ['Accuracy', 'Robustness', 'Speed', 'Simplicity', 'Generality']
N = len(categories)

# 不同策略的評分（0-1）
strategies_radar = {
    'K-Fold': [0.7, 0.7, 0.8, 0.9, 0.9],
    'Stratified': [0.8, 0.8, 0.8, 0.8, 0.7],
    'LOOCV': [0.8, 0.6, 0.3, 0.7, 0.5],
    'Nested CV': [0.9, 0.9, 0.4, 0.5, 0.6]
}

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for strategy_name, values in strategies_radar.items():
    values += values[:1]
    ax8.plot(angles, values, 'o-', linewidth=2, label=strategy_name)
    ax8.fill(angles, values, alpha=0.1)

ax8.set_xticks(angles[:-1])
ax8.set_xticklabels(categories, fontsize=9)
ax8.set_ylim(0, 1)
ax8.set_title('Strategy Comparison Radar Chart', fontsize=12, fontweight='bold', pad=20)
ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax8.grid(True)

# 7.3.9 最終建議
ax9 = plt.subplot(3, 3, 9)
final_recommendations = """
最終建議 Final Recommendations

🎯 通用建議:
   • n < 50: LOOCV 或 10-Fold
   • 50 ≤ n < 500: 10-Fold
   • n ≥ 500: 5-Fold

⚡ 特殊情況:
   • 時間序列: TimeSeriesSplit (必須)
   • 分組數據: GroupKFold (必須)
   • 不平衡: Stratified K-Fold
   • 模型選擇: Nested CV (必須)

💡 最佳實踐:
   1. 始終在 CV 內做預處理
   2. 使用分層避免類別不平衡
   3. 檢查是否有數據洩漏
   4. 報告均值和標準差
   5. 考慮計算成本

⚠️ 記住：沒有完美的策略，
   選擇取決於具體問題！
"""

ax9.text(0.05, 0.95, final_recommendations, transform=ax9.transAxes,
        fontsize=9, verticalalignment='top', family='sans-serif',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax9.axis('off')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/05_ModelEvaluation/cv_part7_best_practices.png',
            dpi=150, bbox_inches='tight')
print("✓ 已保存圖表: cv_part7_best_practices.png")


# ============================================================================
# 總結報告
# Final Summary Report
# ============================================================================

print("\n" + "=" * 100)
print("教程總結報告 | Tutorial Summary Report".center(100))
print("=" * 100)

print("\n【已完成內容】")
print("-" * 100)
print("✓ Part 1: K-Fold vs Stratified K-Fold 深入對比")
print("✓ Part 2: TimeSeriesSplit - 時間序列專用交叉驗證")
print("✓ Part 3: Nested CV - 嵌套交叉驗證（模型選擇+評估）")
print("✓ Part 4: GroupKFold - 處理分組數據")
print("✓ Part 5: Leave-One-Out CV (LOOCV)")
print("✓ Part 6: 自定義交叉驗證策略")
print("✓ Part 7: 交叉驗證最佳實踐與決策指南")

print("\n【生成的可視化圖表】")
print("-" * 100)
visualizations = [
    'cv_part1_kfold_comparison.png',
    'cv_part2_timeseries.png',
    'cv_part3_nested_cv.png',
    'cv_part4_groupkfold.png',
    'cv_part5_loocv.png',
    'cv_part6_custom_cv.png',
    'cv_part7_best_practices.png'
]
for i, viz in enumerate(visualizations, 1):
    print(f"  {i}. {viz}")

print(f"\n總計圖表數量: {len(visualizations)} 張")

print("\n【關鍵學習要點】")
print("-" * 100)
print("""
1. K-Fold vs Stratified K-Fold:
   - 類別不平衡時必須使用 Stratified K-Fold
   - Stratified 確保每個 fold 的類別分佈一致

2. TimeSeriesSplit:
   - 時間序列數據唯一正確的交叉驗證方法
   - 避免使用未來數據訓練（數據洩漏）
   - 訓練集逐漸增大，模擬真實預測場景

3. Nested CV:
   - 模型選擇時必須使用嵌套 CV
   - 外層 CV：評估泛化性能（無偏估計）
   - 內層 CV：選擇最佳超參數
   - 非嵌套 CV 會導致過於樂觀的評估

4. GroupKFold:
   - 有分組結構的數據必須使用
   - 確保同一組的樣本不會同時出現在訓練集和測試集
   - 常見場景：醫療數據、多次測量、同一實體多個樣本

5. LOOCV:
   - 適合小數據集（n < 100）
   - 計算成本高（需要 n 次訓練）
   - 方差較大，對異常值敏感

6. 自定義策略:
   - 可以繼承 BaseCrossValidator 實現自定義分割
   - StratifiedGroupKFold 結合了分層和分組的優點
   - 適用於複雜場景

7. 最佳實踐:
   - 選擇策略取決於數據特徵和問題類型
   - 始終檢查數據洩漏
   - 預處理和特徵選擇必須在 CV 內部進行
   - 報告均值和標準差，而不只是單一分數
""")

print("\n【性能對比總結】")
print("-" * 100)
print("\n策略比較表:")
print(comparison_df.to_string(index=False))

print("\n" + "=" * 100)
print("教程完成！| Tutorial Completed!".center(100))
print("=" * 100)
print("\n所有圖表已保存至: /home/user/machineLearning-basics/05_ModelEvaluation/")
print("All visualizations saved to: /home/user/machineLearning-basics/05_ModelEvaluation/")
