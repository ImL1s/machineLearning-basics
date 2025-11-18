"""
超參數優化完整指南 | Complete Guide to Hyperparameter Optimization

本教程涵蓋：
1. GridSearchCV vs RandomizedSearchCV 深入對比
2. Bayesian Optimization（貝葉斯優化）
3. Optuna 自動調參
4. 超參數重要性分析
5. 多目標優化
6. 學習曲線與驗證曲線
7. 最佳實踐和常見陷阱

This tutorial covers:
1. GridSearchCV vs RandomizedSearchCV in-depth comparison
2. Bayesian Optimization
3. Optuna automatic tuning
4. Hyperparameter importance analysis
5. Multi-objective optimization
6. Learning curves and validation curves
7. Best practices and common pitfalls
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 機器學習庫 / Machine Learning Libraries
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    cross_val_score, learning_curve, validation_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    make_scorer, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler

# Utils 模塊 / Utils Module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    RANDOM_STATE, TEST_SIZE, setup_chinese_fonts,
    save_figure, get_output_path, create_subplots
)

setup_chinese_fonts()

print("="*80)
print("超參數優化完整指南 | Hyperparameter Optimization Guide")
print("="*80)

# ============================================================================
# Part 1: 數據準備 / Data Preparation
# ============================================================================
print("\n" + "="*80)
print("Part 1: 數據準備 / Data Preparation")
print("="*80)

# 加載真實數據集 / Load real dataset
X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

print(f"數據集大小 / Dataset size: {X.shape}")
print(f"特徵數量 / Number of features: {X.shape[1]}")
print(f"樣本數量 / Number of samples: {X.shape[0]}")
print(f"類別分佈 / Class distribution: {np.bincount(y)}")

# 數據分割 / Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 標準化 / Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n訓練集大小 / Training set size: {X_train.shape}")
print(f"測試集大小 / Test set size: {X_test.shape}")

# ============================================================================
# Part 2: GridSearchCV vs RandomizedSearchCV 深入對比
# ============================================================================
print("\n" + "="*80)
print("Part 2: GridSearchCV vs RandomizedSearchCV 深入對比")
print("Detailed Comparison of GridSearchCV vs RandomizedSearchCV")
print("="*80)

# 2.1 GridSearchCV - 網格搜索 / Grid Search
print("\n2.1 GridSearchCV - 網格搜索 / Grid Search")
print("-" * 80)

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

total_combinations = np.prod([len(v) for v in param_grid_rf.values()])
print(f"網格搜索參數組合數 / Grid search combinations: {total_combinations}")
print(f"預計評估次數（5折交叉驗證）/ Estimated evaluations (5-fold CV): {total_combinations * 5}")

import time
start_time = time.time()
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid_rf,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0,
    return_train_score=True
)
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print(f"\n✓ 網格搜索完成 / Grid search completed")
print(f"  耗時 / Time elapsed: {grid_time:.2f} 秒")
print(f"  最佳參數 / Best parameters: {grid_search.best_params_}")
print(f"  最佳交叉驗證分數 / Best CV score: {grid_search.best_score_:.4f}")

# 測試集評估 / Test set evaluation
grid_test_score = roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])
print(f"  測試集分數 / Test score: {grid_test_score:.4f}")

# 2.2 RandomizedSearchCV - 隨機搜索 / Randomized Search
print("\n2.2 RandomizedSearchCV - 隨機搜索 / Randomized Search")
print("-" * 80)

from scipy.stats import randint, uniform

param_dist_rf = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.3, 0.7)
}

n_iter_random = 50
print(f"隨機搜索參數組合數 / Random search combinations: {n_iter_random}")
print(f"預計評估次數（5折交叉驗證）/ Estimated evaluations (5-fold CV): {n_iter_random * 5}")

start_time = time.time()
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_dist_rf,
    n_iter=n_iter_random,
    cv=5,
    scoring='roc_auc',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0,
    return_train_score=True
)
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

print(f"\n✓ 隨機搜索完成 / Random search completed")
print(f"  耗時 / Time elapsed: {random_time:.2f} 秒")
print(f"  最佳參數 / Best parameters: {random_search.best_params_}")
print(f"  最佳交叉驗證分數 / Best CV score: {random_search.best_score_:.4f}")

# 測試集評估 / Test set evaluation
random_test_score = roc_auc_score(y_test, random_search.predict_proba(X_test)[:, 1])
print(f"  測試集分數 / Test score: {random_test_score:.4f}")

# 2.3 可視化對比 / Visualization Comparison
print("\n2.3 生成對比圖表 / Generating comparison plots...")

fig, axes = create_subplots(2, 2, figsize=(16, 12))

# 圖1: 搜索時間對比 / Search Time Comparison
axes[0, 0].bar(['Grid Search\n網格搜索', 'Random Search\n隨機搜索'],
               [grid_time, random_time],
               color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 0].set_ylabel('時間 (秒) / Time (seconds)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('搜索時間對比 / Search Time Comparison', fontsize=14, fontweight='bold', pad=20)
axes[0, 0].grid(True, alpha=0.3, linestyle='--')
for i, (method, t) in enumerate(zip(['Grid Search', 'Random Search'], [grid_time, random_time])):
    axes[0, 0].text(i, t, f'{t:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 圖2: 最佳分數對比 / Best Score Comparison
scores_data = {
    'Grid Search\n網格搜索': [grid_search.best_score_, grid_test_score],
    'Random Search\n隨機搜索': [random_search.best_score_, random_test_score]
}
x_pos = np.arange(len(scores_data))
width = 0.35
cv_scores = [grid_search.best_score_, random_search.best_score_]
test_scores = [grid_test_score, random_test_score]

bars1 = axes[0, 1].bar(x_pos - width/2, cv_scores, width, label='CV Score',
                       color='#3498db', alpha=0.7, edgecolor='black', linewidth=2)
bars2 = axes[0, 1].bar(x_pos + width/2, test_scores, width, label='Test Score',
                       color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 1].set_ylabel('ROC AUC 分數 / ROC AUC Score', fontsize=12, fontweight='bold')
axes[0, 1].set_title('最佳分數對比 / Best Score Comparison', fontsize=14, fontweight='bold', pad=20)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(['Grid Search\n網格搜索', 'Random Search\n隨機搜索'])
axes[0, 1].legend(loc='lower right', fontsize=10)
axes[0, 1].grid(True, alpha=0.3, linestyle='--')
axes[0, 1].set_ylim([0.95, 1.0])

# 添加數值標籤 / Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# 圖3: 網格搜索結果熱圖 / Grid Search Results Heatmap
grid_results = pd.DataFrame(grid_search.cv_results_)
# 創建參數組合標籤 / Create parameter combination labels
param_labels = []
for params in grid_results['params']:
    label = f"d{params['max_depth']}_n{params['n_estimators']}"
    param_labels.append(label)

# 繪製前20個結果 / Plot top 20 results
top_n = min(20, len(grid_results))
sorted_indices = grid_results['mean_test_score'].sort_values(ascending=False).index[:top_n]
top_results = grid_results.loc[sorted_indices]

y_pos = np.arange(top_n)
colors = plt.cm.RdYlGn(top_results['mean_test_score'].values)
bars = axes[1, 0].barh(y_pos, top_results['mean_test_score'].values, color=colors,
                        edgecolor='black', linewidth=1)
axes[1, 0].set_yticks(y_pos)
axes[1, 0].set_yticklabels([f"Rank {i+1}" for i in range(top_n)], fontsize=8)
axes[1, 0].set_xlabel('Mean CV Score / 平均交叉驗證分數', fontsize=12, fontweight='bold')
axes[1, 0].set_title(f'網格搜索 Top {top_n} 結果 / Grid Search Top {top_n} Results',
                    fontsize=14, fontweight='bold', pad=20)
axes[1, 0].grid(True, alpha=0.3, linestyle='--', axis='x')
axes[1, 0].set_xlim([0.95, 1.0])

# 圖4: 隨機搜索參數分佈 / Random Search Parameter Distribution
random_results = pd.DataFrame(random_search.cv_results_)
# 提取參數值 / Extract parameter values
max_depths = [params['max_depth'] for params in random_results['params']]
n_estimators = [params['n_estimators'] for params in random_results['params']]
scores = random_results['mean_test_score'].values

scatter = axes[1, 1].scatter(max_depths, n_estimators, c=scores, s=100,
                            cmap='RdYlGn', alpha=0.6, edgecolor='black', linewidth=1)
axes[1, 1].set_xlabel('Max Depth / 最大深度', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('N Estimators / 樹的數量', fontsize=12, fontweight='bold')
axes[1, 1].set_title('隨機搜索參數空間探索 / Random Search Parameter Space Exploration',
                    fontsize=14, fontweight='bold', pad=20)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')
cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label('CV Score', fontsize=11, fontweight='bold')

# 標記最佳點 / Mark best point
best_idx = random_results['mean_test_score'].idxmax()
best_params = random_results['params'].iloc[best_idx]
axes[1, 1].scatter(best_params['max_depth'], best_params['n_estimators'],
                  s=300, c='red', marker='*', edgecolor='black', linewidth=2,
                  label='Best', zorder=5)
axes[1, 1].legend(loc='upper right', fontsize=10)

plt.tight_layout()
save_figure(fig, get_output_path('grid_vs_random_search.png', 'ModelEvaluation'))

# ============================================================================
# Part 3: Bayesian Optimization（貝葉斯優化）
# ============================================================================
print("\n" + "="*80)
print("Part 3: Bayesian Optimization（貝葉斯優化）")
print("="*80)

BAYES_AVAILABLE = False
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical

    print("✓ scikit-optimize 已安裝 / scikit-optimize installed")

    # 定義搜索空間 / Define search space
    bayes_space = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(5, 30),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Real(0.1, 1.0, prior='uniform')
    }

    print(f"\n貝葉斯優化參數空間 / Bayesian optimization search space:")
    for param, space in bayes_space.items():
        print(f"  {param}: {space}")

    n_iter_bayes = 50
    print(f"\n迭代次數 / Number of iterations: {n_iter_bayes}")

    start_time = time.time()
    bayes_search = BayesSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        bayes_space,
        n_iter=n_iter_bayes,
        cv=5,
        scoring='roc_auc',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    bayes_search.fit(X_train, y_train)
    bayes_time = time.time() - start_time

    print(f"\n✓ 貝葉斯優化完成 / Bayesian optimization completed")
    print(f"  耗時 / Time elapsed: {bayes_time:.2f} 秒")
    print(f"  最佳參數 / Best parameters: {bayes_search.best_params_}")
    print(f"  最佳交叉驗證分數 / Best CV score: {bayes_search.best_score_:.4f}")

    # 測試集評估 / Test set evaluation
    bayes_test_score = roc_auc_score(y_test, bayes_search.predict_proba(X_test)[:, 1])
    print(f"  測試集分數 / Test score: {bayes_test_score:.4f}")

    BAYES_AVAILABLE = True

    # 可視化貝葉斯優化過程 / Visualize Bayesian optimization process
    print("\n生成貝葉斯優化圖表 / Generating Bayesian optimization plots...")

    fig, axes = create_subplots(2, 2, figsize=(16, 12))

    bayes_results = pd.DataFrame(bayes_search.cv_results_)

    # 圖1: 優化歷史 / Optimization History
    iterations = range(1, len(bayes_results) + 1)
    scores = bayes_results['mean_test_score'].values
    cummax_scores = np.maximum.accumulate(scores)

    axes[0, 0].plot(iterations, scores, 'o-', label='Current Score',
                   color='#3498db', alpha=0.6, markersize=6)
    axes[0, 0].plot(iterations, cummax_scores, 's-', label='Best Score So Far',
                   color='#e74c3c', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('迭代次數 / Iteration', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('ROC AUC 分數 / ROC AUC Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('貝葉斯優化歷史 / Bayesian Optimization History',
                        fontsize=14, fontweight='bold', pad=20)
    axes[0, 0].legend(loc='lower right', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    # 圖2: 收斂分析 / Convergence Analysis
    improvement = np.diff(cummax_scores, prepend=cummax_scores[0])
    axes[0, 1].bar(iterations, improvement, color='#2ecc71', alpha=0.7,
                  edgecolor='black', linewidth=1)
    axes[0, 1].set_xlabel('迭代次數 / Iteration', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('分數改進 / Score Improvement', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('每次迭代的改進 / Improvement per Iteration',
                        fontsize=14, fontweight='bold', pad=20)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)

    # 圖3: 參數探索熱圖 / Parameter Exploration Heatmap
    max_depths = [params['max_depth'] for params in bayes_results['params']]
    n_estimators = [params['n_estimators'] for params in bayes_results['params']]

    scatter = axes[1, 0].scatter(max_depths, n_estimators, c=scores, s=100,
                                cmap='viridis', alpha=0.6, edgecolor='black', linewidth=1)
    axes[1, 0].set_xlabel('Max Depth / 最大深度', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('N Estimators / 樹的數量', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('參數空間探索（貝葉斯）/ Parameter Space (Bayesian)',
                        fontsize=14, fontweight='bold', pad=20)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label('CV Score', fontsize=11, fontweight='bold')

    # 標記最佳點 / Mark best point
    best_idx = bayes_results['mean_test_score'].idxmax()
    best_params = bayes_results['params'].iloc[best_idx]
    axes[1, 0].scatter(best_params['max_depth'], best_params['n_estimators'],
                      s=300, c='red', marker='*', edgecolor='black', linewidth=2,
                      label='Best', zorder=5)
    axes[1, 0].legend(loc='upper right', fontsize=10)

    # 圖4: 分數分佈 / Score Distribution
    axes[1, 1].hist(scores, bins=20, color='#9b59b6', alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    axes[1, 1].axvline(bayes_search.best_score_, color='red', linestyle='--',
                      linewidth=2, label=f'Best: {bayes_search.best_score_:.4f}')
    axes[1, 1].axvline(np.mean(scores), color='green', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(scores):.4f}')
    axes[1, 1].set_xlabel('ROC AUC 分數 / ROC AUC Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('頻率 / Frequency', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('分數分佈 / Score Distribution', fontsize=14, fontweight='bold', pad=20)
    axes[1, 1].legend(loc='upper left', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    save_figure(fig, get_output_path('bayesian_optimization.png', 'ModelEvaluation'))

except ImportError:
    print("✗ scikit-optimize 未安裝 / scikit-optimize not installed")
    print("  安裝方法 / Install with: pip install scikit-optimize")
    print("  跳過貝葉斯優化部分 / Skipping Bayesian optimization section")

# ============================================================================
# Part 4: Optuna - 現代自動調參框架
# ============================================================================
print("\n" + "="*80)
print("Part 4: Optuna - 現代自動調參框架")
print("Modern Hyperparameter Optimization Framework")
print("="*80)

OPTUNA_AVAILABLE = False
try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances

    print("✓ Optuna 已安裝 / Optuna installed")

    # 設置日誌級別 / Set logging level
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        """
        Optuna 目標函數 / Optuna objective function

        定義超參數搜索空間並返回優化目標
        Defines hyperparameter search space and returns optimization objective
        """
        # 定義超參數搜索空間 / Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }

        # 創建模型 / Create model
        model = RandomForestClassifier(**params, random_state=RANDOM_STATE)

        # 交叉驗證 / Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    # 創建研究對象 / Create study object
    study = optuna.create_study(
        direction='maximize',
        study_name='RF_optimization',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )

    # 執行優化 / Execute optimization
    n_trials = 50
    print(f"\n開始 Optuna 優化（{n_trials} 次試驗）...")
    print(f"Starting Optuna optimization ({n_trials} trials)...")

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    optuna_time = time.time() - start_time

    print(f"\n✓ Optuna 優化完成 / Optuna optimization completed")
    print(f"  耗時 / Time elapsed: {optuna_time:.2f} 秒")
    print(f"  最佳參數 / Best parameters:")
    for param, value in study.best_params.items():
        print(f"    {param}: {value}")
    print(f"  最佳分數 / Best score: {study.best_value:.4f}")

    # 使用最佳參數訓練最終模型 / Train final model with best parameters
    best_model = RandomForestClassifier(**study.best_params, random_state=RANDOM_STATE)
    best_model.fit(X_train, y_train)
    optuna_test_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    print(f"  測試集分數 / Test score: {optuna_test_score:.4f}")

    OPTUNA_AVAILABLE = True

    # 可視化 Optuna 結果 / Visualize Optuna results
    print("\n生成 Optuna 優化圖表 / Generating Optuna optimization plots...")

    fig, axes = create_subplots(2, 2, figsize=(16, 12))

    # 圖1: 優化歷史 / Optimization History
    trial_numbers = [t.number for t in study.trials]
    trial_values = [t.value for t in study.trials]
    best_values = [study.best_trials[0].value if study.best_trials else None] * len(trial_numbers)
    cummax_values = np.maximum.accumulate(trial_values)

    axes[0, 0].plot(trial_numbers, trial_values, 'o-', label='Trial Value',
                   color='#3498db', alpha=0.6, markersize=6)
    axes[0, 0].plot(trial_numbers, cummax_values, 's-', label='Best Value So Far',
                   color='#e74c3c', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('試驗編號 / Trial Number', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('ROC AUC 分數 / ROC AUC Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Optuna 優化歷史 / Optuna Optimization History',
                        fontsize=14, fontweight='bold', pad=20)
    axes[0, 0].legend(loc='lower right', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    # 圖2: 參數重要性 / Parameter Importance
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        values = list(importances.values())

        colors_map = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(params)))
        bars = axes[0, 1].barh(params, values, color=colors_map,
                              edgecolor='black', linewidth=1.5)
        axes[0, 1].set_xlabel('重要性 / Importance', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('參數重要性分析 / Parameter Importance Analysis',
                            fontsize=14, fontweight='bold', pad=20)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--', axis='x')

        # 添加數值標籤 / Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            axes[0, 1].text(value, i, f' {value:.3f}', va='center', fontsize=10)
    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f'參數重要性計算失敗\n{str(e)}',
                       ha='center', va='center', transform=axes[0, 1].transAxes)

    # 圖3: 參數空間探索（max_depth vs n_estimators）
    max_depths = [t.params['max_depth'] for t in study.trials]
    n_estimators = [t.params['n_estimators'] for t in study.trials]

    scatter = axes[1, 0].scatter(max_depths, n_estimators, c=trial_values, s=100,
                                cmap='plasma', alpha=0.6, edgecolor='black', linewidth=1)
    axes[1, 0].set_xlabel('Max Depth / 最大深度', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('N Estimators / 樹的數量', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('參數空間探索（Optuna）/ Parameter Space (Optuna)',
                        fontsize=14, fontweight='bold', pad=20)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label('Objective Value', fontsize=11, fontweight='bold')

    # 標記最佳點 / Mark best point
    best_trial = study.best_trial
    axes[1, 0].scatter(best_trial.params['max_depth'],
                      best_trial.params['n_estimators'],
                      s=300, c='red', marker='*', edgecolor='black',
                      linewidth=2, label='Best', zorder=5)
    axes[1, 0].legend(loc='upper right', fontsize=10)

    # 圖4: 試驗持續時間分析 / Trial Duration Analysis
    durations = [(t.datetime_complete - t.datetime_start).total_seconds()
                 for t in study.trials if t.datetime_complete]

    axes[1, 1].plot(trial_numbers[:len(durations)], durations, 'o-',
                   color='#9b59b6', alpha=0.7, markersize=6)
    axes[1, 1].axhline(np.mean(durations), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(durations):.2f}s')
    axes[1, 1].set_xlabel('試驗編號 / Trial Number', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('持續時間 (秒) / Duration (seconds)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('試驗持續時間 / Trial Duration', fontsize=14, fontweight='bold', pad=20)
    axes[1, 1].legend(loc='upper right', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    save_figure(fig, get_output_path('optuna_optimization.png', 'ModelEvaluation'))

except ImportError:
    print("✗ Optuna 未安裝 / Optuna not installed")
    print("  安裝方法 / Install with: pip install optuna")
    print("  跳過 Optuna 部分 / Skipping Optuna section")

# ============================================================================
# Part 5: 超參數重要性分析
# ============================================================================
print("\n" + "="*80)
print("Part 5: 超參數重要性分析")
print("Hyperparameter Importance Analysis")
print("="*80)

# 使用隨機森林分析超參數重要性 / Use Random Forest to analyze hyperparameter importance
print("\n分析超參數對模型性能的影響...")
print("Analyzing hyperparameter impact on model performance...")

# 收集所有搜索結果 / Collect all search results
all_params = []
all_scores = []

for result_df, name in [(pd.DataFrame(grid_search.cv_results_), 'Grid'),
                         (pd.DataFrame(random_search.cv_results_), 'Random')]:
    for params, score in zip(result_df['params'], result_df['mean_test_score']):
        param_dict = params.copy()
        param_dict['score'] = score
        param_dict['method'] = name
        all_params.append(param_dict)
        all_scores.append(score)

params_df = pd.DataFrame(all_params)
print(f"\n✓ 收集了 {len(params_df)} 組參數配置")
print(f"  Collected {len(params_df)} parameter configurations")

# 可視化超參數影響 / Visualize hyperparameter impact
fig, axes = create_subplots(2, 2, figsize=(16, 12))

# 圖1: 各參數值與分數的關係 - max_depth
param = 'max_depth'
if param in params_df.columns:
    param_values = params_df[param].values
    scores = params_df['score'].values

    # 箱線圖 / Box plot
    unique_values = sorted(params_df[param].unique())
    data_for_box = [scores[param_values == val] for val in unique_values]

    bp = axes[0, 0].boxplot(data_for_box, labels=unique_values, patch_artist=True,
                            boxprops=dict(facecolor='#3498db', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5))
    axes[0, 0].set_xlabel('Max Depth / 最大深度', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('ROC AUC 分數 / ROC AUC Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Max Depth 對性能的影響 / Impact of Max Depth',
                        fontsize=14, fontweight='bold', pad=20)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--', axis='y')

# 圖2: n_estimators 的影響
param = 'n_estimators'
if param in params_df.columns:
    # 散點圖加趨勢線 / Scatter plot with trend line
    param_values = params_df[param].values
    scores = params_df['score'].values

    axes[0, 1].scatter(param_values, scores, alpha=0.5, s=50,
                      c=scores, cmap='viridis', edgecolor='black', linewidth=0.5)

    # 添加移動平均線 / Add moving average
    from scipy.ndimage import uniform_filter1d
    sorted_indices = np.argsort(param_values)
    sorted_params = param_values[sorted_indices]
    sorted_scores = scores[sorted_indices]

    # 分組平均 / Group average
    bins = np.linspace(sorted_params.min(), sorted_params.max(), 10)
    bin_indices = np.digitize(sorted_params, bins)
    bin_means_x = [sorted_params[bin_indices == i].mean()
                   for i in range(1, len(bins)) if (bin_indices == i).any()]
    bin_means_y = [sorted_scores[bin_indices == i].mean()
                   for i in range(1, len(bins)) if (bin_indices == i).any()]

    axes[0, 1].plot(bin_means_x, bin_means_y, 'r-', linewidth=3,
                   label='Trend', alpha=0.8)
    axes[0, 1].set_xlabel('N Estimators / 樹的數量', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('ROC AUC 分數 / ROC AUC Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('N Estimators 對性能的影響 / Impact of N Estimators',
                        fontsize=14, fontweight='bold', pad=20)
    axes[0, 1].legend(loc='lower right', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')

# 圖3: min_samples_split 的影響
param = 'min_samples_split'
if param in params_df.columns:
    unique_values = sorted(params_df[param].unique())
    mean_scores = [params_df[params_df[param] == val]['score'].mean()
                   for val in unique_values]
    std_scores = [params_df[params_df[param] == val]['score'].std()
                  for val in unique_values]

    axes[1, 0].errorbar(unique_values, mean_scores, yerr=std_scores,
                       fmt='o-', linewidth=2, markersize=8, capsize=5,
                       color='#e74c3c', ecolor='#34495e', alpha=0.7)
    axes[1, 0].set_xlabel('Min Samples Split / 最小分割樣本數', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('平均 ROC AUC / Mean ROC AUC', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Min Samples Split 對性能的影響 / Impact of Min Samples Split',
                        fontsize=14, fontweight='bold', pad=20)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')

# 圖4: min_samples_leaf 的影響
param = 'min_samples_leaf'
if param in params_df.columns:
    unique_values = sorted(params_df[param].unique())
    mean_scores = [params_df[params_df[param] == val]['score'].mean()
                   for val in unique_values]

    colors_map = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(unique_values)))
    bars = axes[1, 1].bar(range(len(unique_values)), mean_scores,
                         color=colors_map, edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[1, 1].set_xticks(range(len(unique_values)))
    axes[1, 1].set_xticklabels(unique_values)
    axes[1, 1].set_xlabel('Min Samples Leaf / 最小葉子樣本數', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('平均 ROC AUC / Mean ROC AUC', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Min Samples Leaf 對性能的影響 / Impact of Min Samples Leaf',
                        fontsize=14, fontweight='bold', pad=20)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')

    # 添加數值標籤 / Add value labels
    for i, (bar, score) in enumerate(zip(bars, mean_scores)):
        axes[1, 1].text(i, score, f'{score:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
save_figure(fig, get_output_path('hyperparameter_importance.png', 'ModelEvaluation'))

# ============================================================================
# Part 6: 學習曲線與驗證曲線
# ============================================================================
print("\n" + "="*80)
print("Part 6: 學習曲線與驗證曲線")
print("Learning Curves and Validation Curves")
print("="*80)

# 使用最佳參數 / Use best parameters
best_params = random_search.best_params_

# 6.1 學習曲線 / Learning Curve
print("\n6.1 生成學習曲線 / Generating learning curve...")

train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores, val_scores = learning_curve(
    RandomForestClassifier(**best_params, random_state=RANDOM_STATE),
    X_train, y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=RANDOM_STATE
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

print("✓ 學習曲線生成完成 / Learning curve generated")

# 6.2 驗證曲線 / Validation Curve
print("\n6.2 生成驗證曲線 / Generating validation curves...")

# 為不同超參數生成驗證曲線 / Generate validation curves for different hyperparameters
param_range_depth = np.arange(5, 31, 5)
param_range_estimators = np.arange(50, 301, 50)

# max_depth 驗證曲線
train_scores_depth, val_scores_depth = validation_curve(
    RandomForestClassifier(n_estimators=best_params['n_estimators'],
                          random_state=RANDOM_STATE),
    X_train, y_train,
    param_name='max_depth',
    param_range=param_range_depth,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

# n_estimators 驗證曲線
train_scores_est, val_scores_est = validation_curve(
    RandomForestClassifier(max_depth=best_params['max_depth'],
                          random_state=RANDOM_STATE),
    X_train, y_train,
    param_name='n_estimators',
    param_range=param_range_estimators,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

print("✓ 驗證曲線生成完成 / Validation curves generated")

# 可視化 / Visualization
fig, axes = create_subplots(2, 2, figsize=(16, 12))

# 圖1: 學習曲線 / Learning Curve
axes[0, 0].fill_between(train_sizes_abs,
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std,
                        alpha=0.3, color='#3498db')
axes[0, 0].fill_between(train_sizes_abs,
                        val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std,
                        alpha=0.3, color='#e74c3c')
axes[0, 0].plot(train_sizes_abs, train_scores_mean, 'o-',
               color='#3498db', linewidth=2, markersize=8,
               label='Training Score')
axes[0, 0].plot(train_sizes_abs, val_scores_mean, 's-',
               color='#e74c3c', linewidth=2, markersize=8,
               label='Cross-validation Score')
axes[0, 0].set_xlabel('訓練樣本數 / Training Samples', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('ROC AUC 分數 / ROC AUC Score', fontsize=12, fontweight='bold')
axes[0, 0].set_title('學習曲線 / Learning Curve', fontsize=14, fontweight='bold', pad=20)
axes[0, 0].legend(loc='lower right', fontsize=10)
axes[0, 0].grid(True, alpha=0.3, linestyle='--')

# 圖2: Max Depth 驗證曲線
train_mean_depth = np.mean(train_scores_depth, axis=1)
train_std_depth = np.std(train_scores_depth, axis=1)
val_mean_depth = np.mean(val_scores_depth, axis=1)
val_std_depth = np.std(val_scores_depth, axis=1)

axes[0, 1].fill_between(param_range_depth,
                        train_mean_depth - train_std_depth,
                        train_mean_depth + train_std_depth,
                        alpha=0.3, color='#3498db')
axes[0, 1].fill_between(param_range_depth,
                        val_mean_depth - val_std_depth,
                        val_mean_depth + val_std_depth,
                        alpha=0.3, color='#e74c3c')
axes[0, 1].plot(param_range_depth, train_mean_depth, 'o-',
               color='#3498db', linewidth=2, markersize=8,
               label='Training Score')
axes[0, 1].plot(param_range_depth, val_mean_depth, 's-',
               color='#e74c3c', linewidth=2, markersize=8,
               label='Cross-validation Score')
axes[0, 1].axvline(best_params['max_depth'], color='green',
                  linestyle='--', linewidth=2, alpha=0.7,
                  label=f"Best: {best_params['max_depth']}")
axes[0, 1].set_xlabel('Max Depth / 最大深度', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('ROC AUC 分數 / ROC AUC Score', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Max Depth 驗證曲線 / Max Depth Validation Curve',
                    fontsize=14, fontweight='bold', pad=20)
axes[0, 1].legend(loc='lower right', fontsize=10)
axes[0, 1].grid(True, alpha=0.3, linestyle='--')

# 圖3: N Estimators 驗證曲線
train_mean_est = np.mean(train_scores_est, axis=1)
train_std_est = np.std(train_scores_est, axis=1)
val_mean_est = np.mean(val_scores_est, axis=1)
val_std_est = np.std(val_scores_est, axis=1)

axes[1, 0].fill_between(param_range_estimators,
                        train_mean_est - train_std_est,
                        train_mean_est + train_std_est,
                        alpha=0.3, color='#3498db')
axes[1, 0].fill_between(param_range_estimators,
                        val_mean_est - val_std_est,
                        val_mean_est + val_std_est,
                        alpha=0.3, color='#e74c3c')
axes[1, 0].plot(param_range_estimators, train_mean_est, 'o-',
               color='#3498db', linewidth=2, markersize=8,
               label='Training Score')
axes[1, 0].plot(param_range_estimators, val_mean_est, 's-',
               color='#e74c3c', linewidth=2, markersize=8,
               label='Cross-validation Score')
axes[1, 0].axvline(best_params['n_estimators'], color='green',
                  linestyle='--', linewidth=2, alpha=0.7,
                  label=f"Best: {best_params['n_estimators']}")
axes[1, 0].set_xlabel('N Estimators / 樹的數量', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('ROC AUC 分數 / ROC AUC Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('N Estimators 驗證曲線 / N Estimators Validation Curve',
                    fontsize=14, fontweight='bold', pad=20)
axes[1, 0].legend(loc='lower right', fontsize=10)
axes[1, 0].grid(True, alpha=0.3, linestyle='--')

# 圖4: 過擬合/欠擬合分析 / Overfitting/Underfitting Analysis
gap_depth = train_mean_depth - val_mean_depth
gap_est = train_mean_est - val_mean_est

axes[1, 1].plot(param_range_depth, gap_depth, 'o-',
               color='#9b59b6', linewidth=2, markersize=8,
               label='Max Depth Gap')
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
axes[1, 1].axhline(0.02, color='red', linestyle='--', linewidth=1.5,
                  alpha=0.5, label='Overfitting Threshold')
axes[1, 1].set_xlabel('Max Depth / 最大深度', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('訓練-驗證分數差 / Train-Val Gap', fontsize=12, fontweight='bold')
axes[1, 1].set_title('過擬合分析 / Overfitting Analysis',
                    fontsize=14, fontweight='bold', pad=20)
axes[1, 1].legend(loc='upper left', fontsize=10)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
save_figure(fig, get_output_path('learning_validation_curves.png', 'ModelEvaluation'))

# ============================================================================
# Part 7: 多目標優化與權衡分析
# ============================================================================
print("\n" + "="*80)
print("Part 7: 多目標優化與權衡分析")
print("Multi-objective Optimization and Trade-off Analysis")
print("="*80)

# 7.1 多指標評估 / Multi-metric evaluation
print("\n7.1 多指標評估 / Multi-metric evaluation...")

# 定義多個評分指標 / Define multiple scoring metrics
scoring = {
    'roc_auc': 'roc_auc',
    'accuracy': 'accuracy',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall'
}

# 使用隨機搜索進行多指標優化 / Multi-metric optimization with random search
multi_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_dist_rf,
    n_iter=30,
    cv=5,
    scoring=scoring,
    refit='roc_auc',  # 使用 ROC AUC 作為主要指標
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0,
    return_train_score=True
)

multi_search.fit(X_train, y_train)

print("✓ 多指標優化完成 / Multi-metric optimization completed")
print(f"  最佳參數 / Best parameters: {multi_search.best_params_}")

# 提取結果 / Extract results
multi_results = pd.DataFrame(multi_search.cv_results_)

# 可視化多指標結果 / Visualize multi-metric results
fig, axes = create_subplots(2, 2, figsize=(16, 12))

# 圖1: 多指標得分對比 / Multi-metric score comparison
metrics = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall']
metric_names = ['ROC AUC', 'Accuracy', 'F1', 'Precision', 'Recall']
best_scores = [multi_results[f'mean_test_{metric}'].max() for metric in metrics]
mean_scores = [multi_results[f'mean_test_{metric}'].mean() for metric in metrics]

x_pos = np.arange(len(metrics))
width = 0.35

bars1 = axes[0, 0].bar(x_pos - width/2, best_scores, width,
                       label='Best Score', color='#2ecc71', alpha=0.7,
                       edgecolor='black', linewidth=1.5)
bars2 = axes[0, 0].bar(x_pos + width/2, mean_scores, width,
                       label='Mean Score', color='#3498db', alpha=0.7,
                       edgecolor='black', linewidth=1.5)

axes[0, 0].set_ylabel('分數 / Score', fontsize=12, fontweight='bold')
axes[0, 0].set_title('多指標性能對比 / Multi-metric Performance Comparison',
                    fontsize=14, fontweight='bold', pad=20)
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(metric_names, rotation=45, ha='right')
axes[0, 0].legend(loc='lower left', fontsize=10)
axes[0, 0].grid(True, alpha=0.3, linestyle='--', axis='y')

# 添加數值標籤 / Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 圖2: Precision-Recall 權衡 / Precision-Recall Trade-off
precision_scores = multi_results['mean_test_precision'].values
recall_scores = multi_results['mean_test_recall'].values
f1_scores = multi_results['mean_test_f1'].values

scatter = axes[0, 1].scatter(recall_scores, precision_scores, c=f1_scores,
                            s=100, cmap='RdYlGn', alpha=0.6,
                            edgecolor='black', linewidth=1)
axes[0, 1].set_xlabel('Recall / 召回率', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Precision / 精確率', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Precision-Recall 權衡 / Precision-Recall Trade-off',
                    fontsize=14, fontweight='bold', pad=20)
axes[0, 1].grid(True, alpha=0.3, linestyle='--')
cbar = plt.colorbar(scatter, ax=axes[0, 1])
cbar.set_label('F1 Score', fontsize=11, fontweight='bold')

# 標記最佳 F1 點 / Mark best F1 point
best_f1_idx = f1_scores.argmax()
axes[0, 1].scatter(recall_scores[best_f1_idx], precision_scores[best_f1_idx],
                  s=300, c='red', marker='*', edgecolor='black',
                  linewidth=2, label='Best F1', zorder=5)
axes[0, 1].legend(loc='lower left', fontsize=10)

# 圖3: 性能-複雜度權衡 / Performance-Complexity Trade-off
n_estimators_list = [params['n_estimators'] for params in multi_results['params']]
max_depth_list = [params['max_depth'] for params in multi_results['params']]
complexity = np.array(n_estimators_list) * np.array(max_depth_list)  # 模型複雜度代理
roc_auc_scores = multi_results['mean_test_roc_auc'].values

scatter = axes[1, 0].scatter(complexity, roc_auc_scores, c=roc_auc_scores,
                            s=100, cmap='viridis', alpha=0.6,
                            edgecolor='black', linewidth=1)
axes[1, 0].set_xlabel('模型複雜度 (n_estimators × max_depth) / Model Complexity',
                     fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('ROC AUC 分數 / ROC AUC Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('性能-複雜度權衡 / Performance-Complexity Trade-off',
                    fontsize=14, fontweight='bold', pad=20)
axes[1, 0].grid(True, alpha=0.3, linestyle='--')
cbar = plt.colorbar(scatter, ax=axes[1, 0])
cbar.set_label('ROC AUC', fontsize=11, fontweight='bold')

# 標記帕累托前沿 / Mark Pareto frontier
# 找到非支配解 / Find non-dominated solutions
pareto_indices = []
for i in range(len(complexity)):
    dominated = False
    for j in range(len(complexity)):
        if i != j:
            # 如果j比i更好（複雜度更低且性能更高）
            if complexity[j] <= complexity[i] and roc_auc_scores[j] >= roc_auc_scores[i]:
                if complexity[j] < complexity[i] or roc_auc_scores[j] > roc_auc_scores[i]:
                    dominated = True
                    break
    if not dominated:
        pareto_indices.append(i)

pareto_complexity = complexity[pareto_indices]
pareto_scores = roc_auc_scores[pareto_indices]
sorted_indices = np.argsort(pareto_complexity)
axes[1, 0].plot(pareto_complexity[sorted_indices], pareto_scores[sorted_indices],
               'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')
axes[1, 0].legend(loc='lower right', fontsize=10)

# 圖4: 相關性矩陣 / Correlation Matrix
correlation_data = {
    'ROC AUC': multi_results['mean_test_roc_auc'].values,
    'Accuracy': multi_results['mean_test_accuracy'].values,
    'F1': multi_results['mean_test_f1'].values,
    'Precision': multi_results['mean_test_precision'].values,
    'Recall': multi_results['mean_test_recall'].values
}
corr_df = pd.DataFrame(correlation_data)
corr_matrix = corr_df.corr()

im = axes[1, 1].imshow(corr_matrix, cmap='RdBu_r', aspect='auto',
                       vmin=-1, vmax=1, interpolation='nearest')
axes[1, 1].set_xticks(range(len(metric_names)))
axes[1, 1].set_yticks(range(len(metric_names)))
axes[1, 1].set_xticklabels(metric_names, rotation=45, ha='right')
axes[1, 1].set_yticklabels(metric_names)
axes[1, 1].set_title('指標相關性矩陣 / Metric Correlation Matrix',
                    fontsize=14, fontweight='bold', pad=20)

# 添加數值標籤 / Add value labels
for i in range(len(metric_names)):
    for j in range(len(metric_names)):
        text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha='center', va='center', color='black',
                              fontsize=10, fontweight='bold')

cbar = plt.colorbar(im, ax=axes[1, 1])
cbar.set_label('Correlation', fontsize=11, fontweight='bold')

plt.tight_layout()
save_figure(fig, get_output_path('multi_objective_optimization.png', 'ModelEvaluation'))

# ============================================================================
# Part 8: 方法比較與總結
# ============================================================================
print("\n" + "="*80)
print("Part 8: 方法比較與總結")
print("Method Comparison and Summary")
print("="*80)

# 8.1 創建比較表 / Create comparison table
comparison_data = {
    '方法 / Method': [],
    '耗時(秒) / Time(s)': [],
    'CV分數 / CV Score': [],
    '測試分數 / Test Score': [],
    '評估次數 / Evaluations': []
}

# Grid Search
comparison_data['方法 / Method'].append('Grid Search')
comparison_data['耗時(秒) / Time(s)'].append(f'{grid_time:.2f}')
comparison_data['CV分數 / CV Score'].append(f'{grid_search.best_score_:.4f}')
comparison_data['測試分數 / Test Score'].append(f'{grid_test_score:.4f}')
comparison_data['評估次數 / Evaluations'].append(total_combinations * 5)

# Random Search
comparison_data['方法 / Method'].append('Random Search')
comparison_data['耗時(秒) / Time(s)'].append(f'{random_time:.2f}')
comparison_data['CV分數 / CV Score'].append(f'{random_search.best_score_:.4f}')
comparison_data['測試分數 / Test Score'].append(f'{random_test_score:.4f}')
comparison_data['評估次數 / Evaluations'].append(n_iter_random * 5)

# Bayesian (if available)
if BAYES_AVAILABLE:
    comparison_data['方法 / Method'].append('Bayesian Opt')
    comparison_data['耗時(秒) / Time(s)'].append(f'{bayes_time:.2f}')
    comparison_data['CV分數 / CV Score'].append(f'{bayes_search.best_score_:.4f}')
    comparison_data['測試分數 / Test Score'].append(f'{bayes_test_score:.4f}')
    comparison_data['評估次數 / Evaluations'].append(n_iter_bayes * 5)

# Optuna (if available)
if OPTUNA_AVAILABLE:
    comparison_data['方法 / Method'].append('Optuna')
    comparison_data['耗時(秒) / Time(s)'].append(f'{optuna_time:.2f}')
    comparison_data['CV分數 / CV Score'].append(f'{study.best_value:.4f}')
    comparison_data['測試分數 / Test Score'].append(f'{optuna_test_score:.4f}')
    comparison_data['評估次數 / Evaluations'].append(n_trials * 5)

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "="*60)
print("方法比較 / Method Comparison:")
print("="*60)
print(comparison_df.to_string(index=False))
print("="*60)

# 8.2 生成綜合對比圖 / Generate comprehensive comparison plot
fig, axes = create_subplots(2, 2, figsize=(16, 12))

methods = comparison_df['方法 / Method'].tolist()
times = [float(t) for t in comparison_df['耗時(秒) / Time(s)'].tolist()]
cv_scores = [float(s) for s in comparison_df['CV分數 / CV Score'].tolist()]
test_scores = [float(s) for s in comparison_df['測試分數 / Test Score'].tolist()]
evaluations = comparison_df['評估次數 / Evaluations'].tolist()

# 圖1: 時間效率對比 / Time Efficiency Comparison
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:len(methods)]
bars = axes[0, 0].barh(methods, times, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=2)
axes[0, 0].set_xlabel('耗時 (秒) / Time (seconds)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('時間效率對比 / Time Efficiency Comparison',
                    fontsize=14, fontweight='bold', pad=20)
axes[0, 0].grid(True, alpha=0.3, linestyle='--', axis='x')

for i, (bar, time) in enumerate(zip(bars, times)):
    axes[0, 0].text(time, i, f' {time:.2f}s', va='center', fontsize=11, fontweight='bold')

# 圖2: 性能對比 / Performance Comparison
x_pos = np.arange(len(methods))
width = 0.35

bars1 = axes[0, 1].bar(x_pos - width/2, cv_scores, width,
                       label='CV Score', color='#3498db', alpha=0.7,
                       edgecolor='black', linewidth=1.5)
bars2 = axes[0, 1].bar(x_pos + width/2, test_scores, width,
                       label='Test Score', color='#2ecc71', alpha=0.7,
                       edgecolor='black', linewidth=1.5)

axes[0, 1].set_ylabel('ROC AUC 分數 / ROC AUC Score', fontsize=12, fontweight='bold')
axes[0, 1].set_title('性能對比 / Performance Comparison',
                    fontsize=14, fontweight='bold', pad=20)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
axes[0, 1].legend(loc='lower right', fontsize=10)
axes[0, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
axes[0, 1].set_ylim([min(min(cv_scores), min(test_scores)) - 0.01, 1.0])

# 添加數值標籤
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=8)

# 圖3: 效率-性能散點圖 / Efficiency-Performance Scatter
scatter = axes[1, 0].scatter(times, test_scores, s=300, c=colors[:len(methods)],
                            alpha=0.6, edgecolor='black', linewidth=2)

for i, method in enumerate(methods):
    axes[1, 0].annotate(method, (times[i], test_scores[i]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

axes[1, 0].set_xlabel('耗時 (秒) / Time (seconds)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('測試分數 / Test Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('效率-性能權衡 / Efficiency-Performance Trade-off',
                    fontsize=14, fontweight='bold', pad=20)
axes[1, 0].grid(True, alpha=0.3, linestyle='--')

# 圖4: 評估次數對比 / Evaluation Count Comparison
bars = axes[1, 1].bar(methods, evaluations, color=colors[:len(methods)],
                     alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('評估次數 / Number of Evaluations', fontsize=12, fontweight='bold')
axes[1, 1].set_title('評估次數對比 / Evaluation Count Comparison',
                    fontsize=14, fontweight='bold', pad=20)
axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')

for i, (bar, count) in enumerate(zip(bars, evaluations)):
    axes[1, 1].text(i, count, f' {count}', ha='center', va='bottom',
                   fontsize=11, fontweight='bold')

plt.tight_layout()
save_figure(fig, get_output_path('method_comparison.png', 'ModelEvaluation'))

# ============================================================================
# Part 9: 最佳實踐和常見陷阱
# ============================================================================
print("\n" + "="*80)
print("Part 9: 最佳實踐和常見陷阱")
print("Best Practices and Common Pitfalls")
print("="*80)

best_practices = """
✅ 最佳實踐 / Best Practices:

1. 搜索策略 / Search Strategy:
   • 從粗到細的搜索（先粗網格，再細化）/ Coarse-to-fine search
   • 使用對數空間搜索某些參數（如學習率）/ Use log-scale for certain parameters
   • 優先調整最重要的超參數 / Prioritize important hyperparameters

2. 計算效率 / Computational Efficiency:
   • 小數據集用 Grid Search，大數據集用 Random/Bayesian / Grid for small, Random/Bayesian for large
   • 使用 n_jobs=-1 並行計算 / Use n_jobs=-1 for parallelization
   • 考慮使用 early stopping / Consider early stopping

3. 避免過擬合 / Avoid Overfitting:
   • 使用獨立的測試集評估 / Use separate test set for evaluation
   • 監控訓練-驗證分數差距 / Monitor train-validation gap
   • 使用嵌套交叉驗證獲得無偏估計 / Use nested CV for unbiased estimates

4. 可復現性 / Reproducibility:
   • 設置 random_state / Set random_state
   • 記錄搜索空間和最佳參數 / Log search space and best parameters
   • 保存完整的搜索結果 / Save complete search results

5. 實際應用 / Practical Application:
   • 考慮模型複雜度與性能的權衡 / Consider complexity-performance trade-off
   • 多指標評估，不只看單一指標 / Multi-metric evaluation
   • 在真實場景中驗證模型 / Validate in real scenarios
"""

common_pitfalls = """
❌ 常見陷阱 / Common Pitfalls:

1. 數據洩漏 / Data Leakage:
   • ✗ 在整個數據集上調參 / Tuning on entire dataset
   • ✓ 只在訓練集上調參 / Tune only on training set

2. 過度優化 / Over-optimization:
   • ✗ 過多迭代導致驗證集過擬合 / Too many iterations overfitting validation set
   • ✓ 使用獨立測試集檢驗 / Use independent test set

3. 搜索空間問題 / Search Space Issues:
   • ✗ 搜索範圍太窄錯過最優解 / Search range too narrow
   • ✗ 搜索範圍太寬浪費計算 / Search range too wide
   • ✓ 根據經驗和文獻設定合理範圍 / Set reasonable range based on experience

4. 忽視實際約束 / Ignoring Practical Constraints:
   • ✗ 只追求性能忽視訓練/推理時間 / Only pursuing performance
   • ✓ 考慮實際部署的計算資源限制 / Consider deployment constraints

5. 缺乏驗證 / Lack of Validation:
   • ✗ 只看交叉驗證分數 / Only look at CV scores
   • ✓ 測試集、驗證曲線、學習曲線綜合分析 / Comprehensive analysis

6. 忘記標準化 / Forgetting Standardization:
   • ✗ 某些算法需要標準化特徵 / Some algorithms need standardization
   • ✓ 在 pipeline 中包含預處理 / Include preprocessing in pipeline
"""

print(best_practices)
print(common_pitfalls)

# ============================================================================
# Part 10: 實戰建議總結
# ============================================================================
print("\n" + "="*80)
print("Part 10: 實戰建議總結")
print("Practical Recommendations Summary")
print("="*80)

recommendations = """
📋 實戰建議 / Practical Recommendations:

何時使用哪種方法？/ When to use which method?

1. Grid Search (網格搜索):
   ✓ 適用場景: 參數空間小、計算資源充足
   ✓ 優點: 全面搜索、結果確定
   ✗ 缺點: 計算成本高、維度詛咒

2. Random Search (隨機搜索):
   ✓ 適用場景: 參數空間大、快速探索
   ✓ 優點: 效率高、易於並行
   ✗ 缺點: 可能錯過最優解

3. Bayesian Optimization (貝葉斯優化):
   ✓ 適用場景: 評估成本高、需要智能搜索
   ✓ 優點: 樣本效率高、利用歷史信息
   ✗ 缺點: 實現複雜、不易並行

4. Optuna (現代框架):
   ✓ 適用場景: 複雜優化任務、需要靈活性
   ✓ 優點: 功能豐富、易用性好、可視化佳
   ✗ 缺點: 額外依賴

推薦流程 / Recommended Workflow:

Step 1: 粗調 (Coarse Tuning)
   → 使用 Random Search 快速探索參數空間
   → 確定大致的參數範圍

Step 2: 精調 (Fine Tuning)
   → 在粗調結果附近使用 Grid Search 或 Bayesian Opt
   → 找到最優參數組合

Step 3: 驗證 (Validation)
   → 使用學習曲線檢查過擬合
   → 在測試集上評估最終性能
   → 多指標綜合評估

Step 4: 生產部署 (Production)
   → 考慮模型複雜度與性能權衡
   → 監控線上性能
   → 定期重新調參
"""

print(recommendations)

# ============================================================================
# 最終總結
# ============================================================================
print("\n" + "="*80)
print("教程完成總結 / Tutorial Summary")
print("="*80)

print(f"""
✓ 超參數優化完整教程已完成！

📊 生成的圖表 / Generated Plots:
   1. grid_vs_random_search.png - 網格搜索 vs 隨機搜索對比
   2. bayesian_optimization.png - 貝葉斯優化過程分析 (如果可用)
   3. optuna_optimization.png - Optuna 優化結果 (如果可用)
   4. hyperparameter_importance.png - 超參數重要性分析
   5. learning_validation_curves.png - 學習曲線與驗證曲線
   6. multi_objective_optimization.png - 多目標優化分析
   7. method_comparison.png - 方法綜合對比

📈 性能比較 / Performance Comparison:
   最佳方法: {comparison_df.loc[comparison_df['CV分數 / CV Score'].astype(float).idxmax(), '方法 / Method']}
   最快方法: {comparison_df.loc[comparison_df['耗時(秒) / Time(s)'].astype(float).idxmin(), '方法 / Method']}

💡 關鍵要點 / Key Takeaways:
   • 沒有"最好"的方法，只有最適合的方法
   • 平衡性能、時間、複雜度是關鍵
   • 多指標評估比單一指標更可靠
   • 避免過度優化驗證集
   • 在實際應用中驗證是必須的

📚 所有圖表已保存至 / All plots saved to:
   {get_output_path('', 'ModelEvaluation')}

""")

print("="*80)
print("Happy Hyperparameter Tuning! 🚀")
print("="*80)
