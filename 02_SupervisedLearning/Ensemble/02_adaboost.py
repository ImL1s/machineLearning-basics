"""
AdaBoost（Adaptive Boosting）
自適應提升算法 - Boosting 集成學習的經典方法

原理：
- 順序訓練多個弱分類器
- 每個分類器關注前一個分類器的錯誤
- 通過調整樣本權重，聚焦於難分類的樣本
- 加權組合所有弱分類器得到強分類器
- "三個臭皮匠，勝過一個諸葛亮"的進階版
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("AdaBoost（Adaptive Boosting）教程".center(80))
print("=" * 80)

# ============================================================================
# 1. AdaBoost 基本概念
# ============================================================================
print("\n【1】AdaBoost 基本概念")
print("-" * 80)
print("""
AdaBoost 核心思想：
• Boosting：提升算法，順序組合弱學習器
• Adaptive：自適應調整樣本權重和分類器權重

算法流程：
1. 初始化：所有樣本權重相等 w_i = 1/N
2. 對每個弱分類器 t = 1, 2, ..., T：
   a) 使用當前權重訓練弱分類器 h_t
   b) 計算加權錯誤率 ε_t = Σ w_i * I(y_i ≠ h_t(x_i))
   c) 計算分類器權重 α_t = 0.5 * ln((1-ε_t)/ε_t)
   d) 更新樣本權重：
      - 正確分類：w_i = w_i * exp(-α_t)
      - 錯誤分類：w_i = w_i * exp(α_t)
   e) 歸一化權重
3. 最終分類器：H(x) = sign(Σ α_t * h_t(x))

關鍵參數：
1. base_estimator：基礎分類器
   - 默認：DecisionTreeClassifier(max_depth=1)（決策樹樁）
   - 應該選擇弱分類器（略優於隨機猜測）

2. n_estimators：弱分類器數量
   - 越多越好，但收益遞減
   - 通常 50-500

3. learning_rate：學習率
   - 縮放每個分類器的貢獻
   - 較小的值需要更多的 n_estimators
   - 默認 1.0

4. algorithm：SAMME 或 SAMME.R
   - SAMME.R：使用概率估計，通常更好
   - SAMME：使用類別預測

優點：
✓ 簡單易用，效果好
✓ 不易過擬合（相比單個複雜模型）
✓ 可以使用任何分類器作為基礎
✓ 自動特徵選擇（通過權重）
✓ 不需要特徵縮放

缺點：
✗ 對噪聲和異常值敏感（會過度關注）
✗ 訓練時間較長（順序訓練）
✗ 難以並行化
✗ 弱分類器不能太複雜
""")

# ============================================================================
# 2. 數據準備：Breast Cancer 數據集
# ============================================================================
print("\n【2】數據準備：Breast Cancer 數據集")
print("-" * 80)

# 加載數據
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"數據集大小：{X.shape}")
print(f"特徵數量：{len(cancer.feature_names)}")
print(f"類別：{cancer.target_names}")
print(f"各類別樣本數：")
print(f"  Malignant (惡性): {np.sum(y == 0)}")
print(f"  Benign (良性): {np.sum(y == 1)}")
print(f"類別比例：{np.sum(y == 1) / len(y):.2%} 良性")

# 數據分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n訓練集大小：{X_train.shape}")
print(f"測試集大小：{X_test.shape}")

# 標準化（對某些基礎分類器有用）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 3. 基礎分類器性能對比
# ============================================================================
print("\n【3】基礎分類器性能對比")
print("-" * 80)

# 不同深度的決策樹作為基礎分類器
base_classifiers = []
for depth in [1, 2, 3, 5, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    score = dt.score(X_test, y_test)

    depth_str = str(depth) if depth else "∞"
    base_classifiers.append({
        'Base Estimator': f'Decision Tree (depth={depth_str})',
        'Accuracy': score,
        'Type': 'Single Tree'
    })
    print(f"決策樹深度 {depth_str}: {score:.4f}")

# ============================================================================
# 4. 訓練 AdaBoost 模型
# ============================================================================
print("\n【4】訓練 AdaBoost 模型")
print("-" * 80)

# 使用默認參數（決策樹樁，max_depth=1）
ada_default = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

print("訓練 AdaBoost (default)...")
ada_default.fit(X_train, y_train)
y_pred_default = ada_default.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

print(f"測試準確率：{accuracy_default:.4f}")
print(f"弱分類器數量：{len(ada_default.estimators_)}")

# 不同深度的 AdaBoost
ada_results = []
for depth in [1, 2, 3]:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators=50,
        random_state=42
    )
    ada.fit(X_train, y_train)
    score = ada.score(X_test, y_test)
    ada_results.append({
        'Base Estimator': f'AdaBoost (depth={depth})',
        'Accuracy': score,
        'Type': 'AdaBoost'
    })
    print(f"AdaBoost (深度={depth}): {score:.4f}")

# ============================================================================
# 5. 可視化結果
# ============================================================================
print("\n【5】可視化結果")
print("-" * 80)

fig1 = plt.figure(figsize=(20, 16))

# 5.1 基礎分類器 vs AdaBoost 性能對比
ax1 = plt.subplot(3, 3, 1)
all_results = base_classifiers + ada_results
results_df = pd.DataFrame(all_results)

colors = ['skyblue'] * 5 + ['orange'] * 3
bars = ax1.bar(range(len(results_df)), results_df['Accuracy'], color=colors)
ax1.set_xticks(range(len(results_df)))
ax1.set_xticklabels(results_df['Base Estimator'], rotation=45, ha='right')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Base Classifier vs AdaBoost\n基礎分類器 vs AdaBoost',
              fontsize=14, fontweight='bold')
ax1.axhline(y=results_df[results_df['Type'] == 'Single Tree']['Accuracy'].max(),
            color='gray', linestyle='--', alpha=0.7, label='Best Single Tree')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 添加數值標籤
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=8, rotation=0)

# ============================================================================
# 6. n_estimators 參數影響
# ============================================================================
print("\n【6】n_estimators 參數影響分析")
print("-" * 80)

n_estimators_range = range(1, 101, 2)
train_scores = []
test_scores = []

for n in n_estimators_range:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        random_state=42
    )
    ada.fit(X_train, y_train)
    train_scores.append(ada.score(X_train, y_train))
    test_scores.append(ada.score(X_test, y_test))

best_n = list(n_estimators_range)[np.argmax(test_scores)]
best_score = max(test_scores)
print(f"最優 n_estimators: {best_n}")
print(f"最佳測試準確率: {best_score:.4f}")

# 5.2 n_estimators vs Accuracy
ax2 = plt.subplot(3, 3, 2)
ax2.plot(n_estimators_range, train_scores, label='Training Accuracy',
         linewidth=2, alpha=0.8)
ax2.plot(n_estimators_range, test_scores, label='Testing Accuracy',
         linewidth=2, alpha=0.8)
ax2.axvline(x=best_n, color='red', linestyle='--', alpha=0.5,
            label=f'Best n={best_n}')
ax2.set_xlabel('Number of Estimators', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('n_estimators vs Accuracy\n弱分類器數量 vs 準確率',
              fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============================================================================
# 7. Learning Rate 影響
# ============================================================================
print("\n【7】Learning Rate 影響分析")
print("-" * 80)

learning_rates = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]
lr_scores = []

for lr in learning_rates:
    ada_lr = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=lr,
        random_state=42
    )
    ada_lr.fit(X_train, y_train)
    score = ada_lr.score(X_test, y_test)
    lr_scores.append(score)
    print(f"Learning Rate {lr:.2f}: {score:.4f}")

# 5.3 Learning Rate vs Accuracy
ax3 = plt.subplot(3, 3, 3)
ax3.plot(learning_rates, lr_scores, marker='o', linewidth=2, markersize=8)
ax3.set_xlabel('Learning Rate', fontsize=12)
ax3.set_ylabel('Test Accuracy', fontsize=12)
ax3.set_title('Learning Rate vs Accuracy\n學習率 vs 準確率',
              fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')

# ============================================================================
# 8. 特徵重要性
# ============================================================================
print("\n【8】特徵重要性分析")
print("-" * 80)

feature_importance = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance': ada_default.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 重要特徵：")
print(feature_importance.head(10).to_string(index=False))

# 5.4 特徵重要性
ax4 = plt.subplot(3, 3, 4)
top_features = feature_importance.head(15)
ax4.barh(range(len(top_features)), top_features['Importance'], color='coral')
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['Feature'], fontsize=9)
ax4.set_xlabel('Importance', fontsize=12)
ax4.set_title('Top 15 Feature Importances\n特徵重要性 Top 15',
              fontsize=14, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis='x')

# ============================================================================
# 9. 混淆矩陣
# ============================================================================
print("\n【9】混淆矩陣")
print("-" * 80)

cm = confusion_matrix(y_test, y_pred_default)
print("\n混淆矩陣：")
print(cm)

print("\n分類報告：")
print(classification_report(y_test, y_pred_default, target_names=cancer.target_names))

# 5.5 混淆矩陣
ax5 = plt.subplot(3, 3, 5)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names,
            cbar_kws={'label': 'Count'})
ax5.set_title('Confusion Matrix\n混淆矩陣', fontsize=14, fontweight='bold')
ax5.set_ylabel('True Label', fontsize=12)
ax5.set_xlabel('Predicted Label', fontsize=12)

# ============================================================================
# 10. ROC 曲線
# ============================================================================
print("\n【10】ROC 曲線分析")
print("-" * 80)

# 計算 ROC 曲線
y_pred_proba = ada_default.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"AUC 分數: {roc_auc:.4f}")

# 5.6 ROC 曲線
ax6 = plt.subplot(3, 3, 6)
ax6.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.4f})')
ax6.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax6.set_xlabel('False Positive Rate', fontsize=12)
ax6.set_ylabel('True Positive Rate', fontsize=12)
ax6.set_title('ROC Curve\nROC 曲線', fontsize=14, fontweight='bold')
ax6.legend(loc='lower right')
ax6.grid(True, alpha=0.3)

# ============================================================================
# 11. 學習曲線
# ============================================================================
print("\n【11】學習曲線分析")
print("-" * 80)

train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                      n_estimators=50, random_state=42),
    X_train, y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores_lc, axis=1)
train_std = np.std(train_scores_lc, axis=1)
val_mean = np.mean(val_scores_lc, axis=1)
val_std = np.std(val_scores_lc, axis=1)

print(f"最終訓練分數: {train_mean[-1]:.4f} (+/- {train_std[-1]:.4f})")
print(f"最終驗證分數: {val_mean[-1]:.4f} (+/- {val_std[-1]:.4f})")

# 5.7 學習曲線
ax7 = plt.subplot(3, 3, 7)
ax7.plot(train_sizes, train_mean, label='Training Score', marker='o', linewidth=2)
ax7.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
ax7.plot(train_sizes, val_mean, label='Validation Score', marker='s', linewidth=2)
ax7.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
ax7.set_xlabel('Training Size', fontsize=12)
ax7.set_ylabel('Accuracy', fontsize=12)
ax7.set_title('Learning Curve\n學習曲線', fontsize=14, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# ============================================================================
# 12. 弱分類器權重分析
# ============================================================================
print("\n【12】弱分類器權重分析")
print("-" * 80)

estimator_weights = ada_default.estimator_weights_
estimator_errors = ada_default.estimator_errors_

print(f"平均分類器權重: {np.mean(estimator_weights):.4f}")
print(f"權重標準差: {np.std(estimator_weights):.4f}")
print(f"平均錯誤率: {np.mean(estimator_errors):.4f}")

# 5.8 分類器權重分布
ax8 = plt.subplot(3, 3, 8)
ax8.plot(estimator_weights, label='Estimator Weights', linewidth=2)
ax8.set_xlabel('Estimator Index', fontsize=12)
ax8.set_ylabel('Weight', fontsize=12)
ax8.set_title('Weak Classifier Weights\n弱分類器權重',
              fontsize=14, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.legend()

# 5.9 錯誤率分布
ax9 = plt.subplot(3, 3, 9)
ax9.plot(estimator_errors, color='red', label='Estimator Errors', linewidth=2)
ax9.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random (0.5)')
ax9.set_xlabel('Estimator Index', fontsize=12)
ax9.set_ylabel('Error Rate', fontsize=12)
ax9.set_title('Weak Classifier Error Rates\n弱分類器錯誤率',
              fontsize=14, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/output/adaboost_analysis.png',
            dpi=150, bbox_inches='tight')
print("✓ 圖表已保存：output/adaboost_analysis.png")

# ============================================================================
# 13. 與其他集成算法對比
# ============================================================================
print("\n【13】與其他集成算法對比")
print("-" * 80)

fig2 = plt.figure(figsize=(20, 10))

# 訓練不同的集成算法
ensemble_algorithms = [
    ('AdaBoost', AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=42
    )),
    ('Random Forest', RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )),
    ('Gradient Boosting', GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42
    )),
    ('Single Decision Tree', DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
]

ensemble_results = []
ensemble_cv_scores = {}

for name, clf in ensemble_algorithms:
    print(f"\n訓練 {name}...")
    clf.fit(X_train, y_train)

    # 測試分數
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    # 交叉驗證
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    ensemble_cv_scores[name] = cv_scores

    # 預測時間（簡單測試）
    import time
    start_time = time.time()
    clf.predict(X_test)
    pred_time = time.time() - start_time

    ensemble_results.append({
        'Algorithm': name,
        'Train Accuracy': train_score,
        'Test Accuracy': test_score,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Prediction Time': pred_time
    })

    print(f"  訓練準確率: {train_score:.4f}")
    print(f"  測試準確率: {test_score:.4f}")
    print(f"  交叉驗證: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

ensemble_df = pd.DataFrame(ensemble_results)
print("\n算法對比總結：")
print(ensemble_df.to_string(index=False))

# 13.1 測試準確率對比
ax1 = plt.subplot(2, 4, 1)
colors_ensemble = ['orange', 'green', 'blue', 'gray']
bars = ax1.bar(ensemble_df['Algorithm'], ensemble_df['Test Accuracy'],
               color=colors_ensemble, alpha=0.7)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title('Algorithm Comparison\n算法對比', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=10)

# 13.2 訓練 vs 測試準確率
ax2 = plt.subplot(2, 4, 2)
x_pos = np.arange(len(ensemble_df))
width = 0.35
ax2.bar(x_pos - width/2, ensemble_df['Train Accuracy'], width,
        label='Train', alpha=0.7)
ax2.bar(x_pos + width/2, ensemble_df['Test Accuracy'], width,
        label='Test', alpha=0.7)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(ensemble_df['Algorithm'], rotation=45, ha='right')
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Train vs Test Accuracy\n訓練 vs 測試準確率',
              fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 13.3 交叉驗證分數箱線圖
ax3 = plt.subplot(2, 4, 3)
cv_data = [ensemble_cv_scores[name] for name in ensemble_df['Algorithm']]
bp = ax3.boxplot(cv_data, labels=ensemble_df['Algorithm'], patch_artist=True)
for patch, color in zip(bp['boxes'], colors_ensemble):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('CV Accuracy', fontsize=12)
ax3.set_title('Cross-Validation Scores\n交叉驗證分數',
              fontsize=14, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# 13.4 預測時間對比
ax4 = plt.subplot(2, 4, 4)
ax4.bar(ensemble_df['Algorithm'], ensemble_df['Prediction Time'],
        color=colors_ensemble, alpha=0.7)
ax4.set_ylabel('Time (seconds)', fontsize=12)
ax4.set_title('Prediction Time Comparison\n預測時間對比',
              fontsize=14, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3, axis='y')

# ============================================================================
# 14. AdaBoost 訓練過程可視化
# ============================================================================
print("\n【14】AdaBoost 訓練過程可視化")
print("-" * 80)

# 創建一個簡單的二維數據集用於可視化
X_simple, y_simple = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    flip_y=0.1,
    random_state=42
)

X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.3, random_state=42
)

# 訓練 AdaBoost
ada_simple = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=5,
    random_state=42
)
ada_simple.fit(X_train_simple, y_train_simple)

# 14.1-14.4 顯示前4個弱分類器的決策邊界
for i in range(min(4, len(ada_simple.estimators_))):
    ax = plt.subplot(2, 4, 5 + i)

    # 獲取第 i 個弱分類器
    estimator = ada_simple.estimators_[i]
    weight = ada_simple.estimator_weights_[i]
    error = ada_simple.estimator_errors_[i]

    # 繪製決策邊界
    try:
        from mlxtend.plotting import plot_decision_regions
        plot_decision_regions(X_train_simple, y_train_simple,
                            clf=estimator, legend=0, ax=ax)
    except:
        # 如果 mlxtend 不可用，使用簡單散點圖
        scatter = ax.scatter(X_train_simple[:, 0], X_train_simple[:, 1],
                           c=y_train_simple, cmap='viridis', alpha=0.6)

    ax.set_title(f'Weak Classifier {i+1}\nWeight={weight:.3f}, Error={error:.3f}',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)

print("AdaBoost 訓練過程：")
for i, (weight, error) in enumerate(zip(ada_simple.estimator_weights_,
                                        ada_simple.estimator_errors_)):
    print(f"  弱分類器 {i+1}: 權重={weight:.4f}, 錯誤率={error:.4f}")

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/output/adaboost_comparison.png',
            dpi=150, bbox_inches='tight')
print("✓ 圖表已保存：output/adaboost_comparison.png")

# ============================================================================
# 15. 驗證曲線：max_depth 參數調優
# ============================================================================
print("\n【15】驗證曲線：max_depth 參數調優")
print("-" * 80)

fig3 = plt.figure(figsize=(18, 6))

param_range = [1, 2, 3, 4, 5]
train_scores_vc, test_scores_vc = validation_curve(
    AdaBoostClassifier(n_estimators=50, random_state=42),
    X_train, y_train,
    param_name="estimator__max_depth",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

train_mean_vc = np.mean(train_scores_vc, axis=1)
train_std_vc = np.std(train_scores_vc, axis=1)
test_mean_vc = np.mean(test_scores_vc, axis=1)
test_std_vc = np.std(test_scores_vc, axis=1)

# 15.1 驗證曲線
ax1 = plt.subplot(1, 3, 1)
ax1.plot(param_range, train_mean_vc, label='Training Score', marker='o', linewidth=2)
ax1.fill_between(param_range, train_mean_vc - train_std_vc,
                train_mean_vc + train_std_vc, alpha=0.15)
ax1.plot(param_range, test_mean_vc, label='Validation Score', marker='s', linewidth=2)
ax1.fill_between(param_range, test_mean_vc - test_std_vc,
                test_mean_vc + test_std_vc, alpha=0.15)
ax1.set_xlabel('Max Depth of Base Estimator', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Validation Curve: max_depth\n驗證曲線：最大深度',
              fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

print(f"最優 max_depth: {param_range[np.argmax(test_mean_vc)]}")
print(f"最佳驗證分數: {max(test_mean_vc):.4f}")

# ============================================================================
# 16. 累積性能分析
# ============================================================================
print("\n【16】累積性能分析")
print("-" * 80)

# 計算累積性能
cumulative_train = []
cumulative_test = []

for i in range(1, 51):
    ada_temp = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=i,
        random_state=42
    )
    ada_temp.fit(X_train, y_train)
    cumulative_train.append(ada_temp.score(X_train, y_train))
    cumulative_test.append(ada_temp.score(X_test, y_test))

# 15.2 累積性能
ax2 = plt.subplot(1, 3, 2)
ax2.plot(range(1, 51), cumulative_train, label='Training', linewidth=2, alpha=0.8)
ax2.plot(range(1, 51), cumulative_test, label='Testing', linewidth=2, alpha=0.8)
ax2.set_xlabel('Number of Estimators', fontsize=12)
ax2.set_ylabel('Cumulative Accuracy', fontsize=12)
ax2.set_title('Cumulative Performance\n累積性能', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============================================================================
# 17. 總結對比表
# ============================================================================

# 15.3 總結表格
ax3 = plt.subplot(1, 3, 3)
ax3.axis('off')

summary_data = [
    ['算法', '測試準確率', 'CV 均值', '特點'],
    ['AdaBoost', f"{ensemble_df.iloc[0]['Test Accuracy']:.4f}",
     f"{ensemble_df.iloc[0]['CV Mean']:.4f}", '順序提升'],
    ['Random Forest', f"{ensemble_df.iloc[1]['Test Accuracy']:.4f}",
     f"{ensemble_df.iloc[1]['CV Mean']:.4f}", '並行 Bagging'],
    ['Gradient Boosting', f"{ensemble_df.iloc[2]['Test Accuracy']:.4f}",
     f"{ensemble_df.iloc[2]['CV Mean']:.4f}", '梯度提升'],
    ['Decision Tree', f"{ensemble_df.iloc[3]['Test Accuracy']:.4f}",
     f"{ensemble_df.iloc[3]['CV Mean']:.4f}", '單個樹']
]

table = ax3.table(cellText=summary_data, cellLoc='center', loc='center',
                 colWidths=[0.35, 0.25, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 設置表頭樣式
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 設置交替行顏色
for i in range(1, len(summary_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

# 突出顯示最佳結果
best_idx = ensemble_df['Test Accuracy'].idxmax() + 1
table[(best_idx, 1)].set_facecolor('#90EE90')
table[(best_idx, 1)].set_text_props(weight='bold')

ax3.set_title('Algorithm Summary\n算法總結', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/output/adaboost_detailed.png',
            dpi=150, bbox_inches='tight')
print("✓ 圖表已保存：output/adaboost_detailed.png")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("AdaBoost 要點總結")
print("=" * 80)
print(f"""
1. 模型性能：
   • AdaBoost 測試準確率: {accuracy_default:.4f}
   • 最佳單棵決策樹: {max([r['Accuracy'] for r in base_classifiers]):.4f}
   • 性能提升: {(accuracy_default - max([r['Accuracy'] for r in base_classifiers])) * 100:.2f}%

2. 最優參數：
   • n_estimators: {best_n}
   • max_depth（基礎分類器）: 1（決策樹樁效果最好）
   • learning_rate: 1.0（默認值表現良好）

3. 關鍵發現：
   • AdaBoost 顯著優於單個弱分類器
   • 使用決策樹樁（max_depth=1）作為基礎分類器效果最好
   • 增加 n_estimators 可以提升性能，但收益遞減
   • 對噪聲敏感，需要注意數據質量

4. 與其他算法比較：
   • AdaBoost: {ensemble_df.iloc[0]['Test Accuracy']:.4f}
   • Random Forest: {ensemble_df.iloc[1]['Test Accuracy']:.4f}
   • Gradient Boosting: {ensemble_df.iloc[2]['Test Accuracy']:.4f}

5. 適用場景：
   ✓ 二分類問題
   ✓ 數據質量較好（少噪聲）
   ✓ 需要高準確率
   ✓ 不需要實時預測

6. 最佳實踐：
   ✓ 使用弱分類器（決策樹樁或淺層樹）
   ✓ 從較小的 n_estimators 開始，逐步增加
   ✓ 調整 learning_rate 和 n_estimators 的組合
   ✓ 注意處理噪聲和異常值
   ✓ 使用交叉驗證評估穩定性

7. 注意事項：
   ✗ 對噪聲和異常值敏感
   ✗ 訓練時間較長（順序訓練）
   ✗ 難以並行化
   ✗ 不適合大規模數據集
""")

plt.show()
print("\n程序執行完成！")
