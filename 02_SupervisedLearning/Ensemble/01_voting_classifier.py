"""
投票分類器（Voting Classifier）
Ensemble Learning - 集成學習的基礎方法

原理：
- 結合多個不同的基礎分類器
- 通過"投票"機制做出最終決策
- 硬投票：多數投票（majority voting）
- 軟投票：平均概率（average probabilities）
- 利用模型多樣性提升整體性能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mlxtend.plotting import plot_decision_regions
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("投票分類器（Voting Classifier）教程".center(80))
print("=" * 80)

# ============================================================================
# 1. Voting Classifier 基本概念
# ============================================================================
print("\n【1】Voting Classifier 基本概念")
print("-" * 80)
print("""
投票分類器核心思想：
• 集成學習（Ensemble Learning）：組合多個模型的預測
• 三個臭皮匠勝過一個諸葛亮

兩種投票方式：

1. 硬投票（Hard Voting）：
   • 每個模型投一票給某個類別
   • 最終結果：得票最多的類別
   • 公式：y_pred = mode{c1(x), c2(x), ..., cn(x)}
   • 適用：所有分類器

2. 軟投票（Soft Voting）：
   • 每個模型給出類別概率
   • 最終結果：平均概率最高的類別
   • 公式：y_pred = argmax Σ p_i(y|x)
   • 適用：能輸出概率的分類器
   • 通常性能更好

關鍵要點：
✓ 基礎分類器應該多樣化（不同算法）
✓ 基礎分類器不能太差（至少優於隨機猜測）
✓ 基礎分類器之間相關性越低越好
✓ 軟投票通常優於硬投票

優點：
✓ 提升準確率和穩定性
✓ 減少過擬合風險
✓ 簡單易用，易於理解
✓ 可以結合不同類型的模型

缺點：
✗ 訓練時間增加
✗ 模型複雜度提高
✗ 不如單個模型易於解釋
""")

# ============================================================================
# 2. 數據準備：Iris 數據集
# ============================================================================
print("\n【2】數據準備：Iris 數據集")
print("-" * 80)

# 加載數據
iris = load_iris()
X, y = iris.data, iris.target

print(f"數據集大小：{X.shape}")
print(f"特徵名稱：{iris.feature_names}")
print(f"類別：{iris.target_names}")
print(f"各類別樣本數：{np.bincount(y)}")

# 為了可視化決策邊界，我們只使用前兩個特徵
X_2d = X[:, :2]
feature_names = iris.feature_names[:2]

# 數據分割
X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n訓練集大小：{X_train.shape}")
print(f"測試集大小：{X_test.shape}")
print(f"使用特徵：{feature_names}")

# 標準化（對某些算法很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 3. 創建基礎分類器
# ============================================================================
print("\n【3】創建基礎分類器")
print("-" * 80)

# 定義多個不同的基礎分類器
lr = LogisticRegression(max_iter=1000, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)  # probability=True for soft voting
rf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)

# 基礎分類器列表
base_classifiers = [
    ('Logistic Regression', lr),
    ('K-Nearest Neighbors', knn),
    ('Decision Tree', dt),
    ('SVM', svm),
    ('Random Forest', rf)
]

print("基礎分類器：")
for name, clf in base_classifiers:
    print(f"  • {name}: {clf.__class__.__name__}")

# ============================================================================
# 4. 訓練各個基礎分類器
# ============================================================================
print("\n【4】訓練各個基礎分類器")
print("-" * 80)

results = []

for name, clf in base_classifiers:
    # 訓練
    clf.fit(X_train_scaled, y_train)

    # 預測
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # 交叉驗證
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)

    results.append({
        'Classifier': name,
        'Test Accuracy': accuracy,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })

    print(f"{name}:")
    print(f"  測試準確率: {accuracy:.4f}")
    print(f"  交叉驗證: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

results_df = pd.DataFrame(results)

# ============================================================================
# 5. 創建投票分類器
# ============================================================================
print("\n【5】創建投票分類器")
print("-" * 80)

# 重新創建分類器（因為已經訓練過了）
estimators = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(max_depth=4, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42))
]

# 硬投票
voting_hard = VotingClassifier(
    estimators=estimators,
    voting='hard'
)

# 軟投票
voting_soft = VotingClassifier(
    estimators=estimators,
    voting='soft'
)

print("訓練硬投票分類器...")
voting_hard.fit(X_train_scaled, y_train)
y_pred_hard = voting_hard.predict(X_test_scaled)
accuracy_hard = accuracy_score(y_test, y_pred_hard)

print("訓練軟投票分類器...")
voting_soft.fit(X_train_scaled, y_train)
y_pred_soft = voting_soft.predict(X_test_scaled)
accuracy_soft = accuracy_score(y_test, y_pred_soft)

print(f"\n硬投票準確率: {accuracy_hard:.4f}")
print(f"軟投票準確率: {accuracy_soft:.4f}")

# 添加到結果
results.append({
    'Classifier': 'Voting (Hard)',
    'Test Accuracy': accuracy_hard,
    'CV Mean': cross_val_score(voting_hard, X_train_scaled, y_train, cv=5).mean(),
    'CV Std': cross_val_score(voting_hard, X_train_scaled, y_train, cv=5).std()
})

results.append({
    'Classifier': 'Voting (Soft)',
    'Test Accuracy': accuracy_soft,
    'CV Mean': cross_val_score(voting_soft, X_train_scaled, y_train, cv=5).mean(),
    'CV Std': cross_val_score(voting_soft, X_train_scaled, y_train, cv=5).std()
})

results_df = pd.DataFrame(results)
print("\n所有分類器性能對比：")
print(results_df.to_string(index=False))

# ============================================================================
# 6. 可視化：性能對比
# ============================================================================
print("\n【6】可視化結果")
print("-" * 80)

fig = plt.figure(figsize=(20, 16))

# 6.1 測試準確率對比
ax1 = plt.subplot(3, 3, 1)
colors = ['skyblue'] * 5 + ['orange', 'red']
bars = ax1.bar(range(len(results_df)), results_df['Test Accuracy'], color=colors)
ax1.set_xticks(range(len(results_df)))
ax1.set_xticklabels(results_df['Classifier'], rotation=45, ha='right')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Test Accuracy Comparison\n測試準確率對比', fontsize=14, fontweight='bold')
ax1.axhline(y=results_df['Test Accuracy'].iloc[:5].mean(), color='gray',
            linestyle='--', label='Base Avg', alpha=0.7)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 添加數值標籤
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=9)

# 6.2 交叉驗證對比
ax2 = plt.subplot(3, 3, 2)
ax2.bar(range(len(results_df)), results_df['CV Mean'],
        yerr=results_df['CV Std'], color=colors, capsize=5)
ax2.set_xticks(range(len(results_df)))
ax2.set_xticklabels(results_df['Classifier'], rotation=45, ha='right')
ax2.set_ylabel('CV Accuracy', fontsize=12)
ax2.set_title('Cross-Validation Scores\n交叉驗證分數', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 6.3 準確率提升
ax3 = plt.subplot(3, 3, 3)
base_mean = results_df['Test Accuracy'].iloc[:5].mean()
improvements = (results_df['Test Accuracy'] - base_mean) * 100
colors_imp = ['green' if x > 0 else 'red' for x in improvements]
ax3.barh(range(len(results_df)), improvements, color=colors_imp, alpha=0.7)
ax3.set_yticks(range(len(results_df)))
ax3.set_yticklabels(results_df['Classifier'])
ax3.set_xlabel('Improvement (%)', fontsize=12)
ax3.set_title('Improvement over Base Average\n相對基準的改進', fontsize=14, fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax3.grid(True, alpha=0.3, axis='x')

# ============================================================================
# 7. 混淆矩陣對比
# ============================================================================

# 7.1 最佳基礎分類器的混淆矩陣
best_base_idx = results_df['Test Accuracy'].iloc[:5].idxmax()
best_base_name = results_df['Classifier'].iloc[best_base_idx]
best_base_clf = base_classifiers[best_base_idx][1]
y_pred_best = best_base_clf.predict(X_test_scaled)

ax4 = plt.subplot(3, 3, 4)
cm_best = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
ax4.set_title(f'Confusion Matrix: {best_base_name}\n最佳基礎分類器',
              fontsize=12, fontweight='bold')
ax4.set_ylabel('True Label')
ax4.set_xlabel('Predicted Label')

# 7.2 硬投票混淆矩陣
ax5 = plt.subplot(3, 3, 5)
cm_hard = confusion_matrix(y_test, y_pred_hard)
sns.heatmap(cm_hard, annot=True, fmt='d', cmap='Oranges', ax=ax5,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
ax5.set_title('Confusion Matrix: Hard Voting\n硬投票混淆矩陣',
              fontsize=12, fontweight='bold')
ax5.set_ylabel('True Label')
ax5.set_xlabel('Predicted Label')

# 7.3 軟投票混淆矩陣
ax6 = plt.subplot(3, 3, 6)
cm_soft = confusion_matrix(y_test, y_pred_soft)
sns.heatmap(cm_soft, annot=True, fmt='d', cmap='Reds', ax=ax6,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
ax6.set_title('Confusion Matrix: Soft Voting\n軟投票混淆矩陣',
              fontsize=12, fontweight='bold')
ax6.set_ylabel('True Label')
ax6.set_xlabel('Predicted Label')

# ============================================================================
# 8. 決策邊界可視化
# ============================================================================

# 8.1 最佳基礎分類器決策邊界
ax7 = plt.subplot(3, 3, 7)
plot_decision_regions(X_train_scaled, y_train, clf=best_base_clf,
                     legend=2, ax=ax7)
ax7.set_xlabel(feature_names[0], fontsize=10)
ax7.set_ylabel(feature_names[1], fontsize=10)
ax7.set_title(f'Decision Boundary: {best_base_name}\n決策邊界',
              fontsize=12, fontweight='bold')

# 8.2 硬投票決策邊界
ax8 = plt.subplot(3, 3, 8)
plot_decision_regions(X_train_scaled, y_train, clf=voting_hard,
                     legend=2, ax=ax8)
ax8.set_xlabel(feature_names[0], fontsize=10)
ax8.set_ylabel(feature_names[1], fontsize=10)
ax8.set_title('Decision Boundary: Hard Voting\n硬投票決策邊界',
              fontsize=12, fontweight='bold')

# 8.3 軟投票決策邊界
ax9 = plt.subplot(3, 3, 9)
plot_decision_regions(X_train_scaled, y_train, clf=voting_soft,
                     legend=2, ax=ax9)
ax9.set_xlabel(feature_names[0], fontsize=10)
ax9.set_ylabel(feature_names[1], fontsize=10)
ax9.set_title('Decision Boundary: Soft Voting\n軟投票決策邊界',
              fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/output/voting_classifier_results.png',
            dpi=150, bbox_inches='tight')
print("✓ 圖表已保存：output/voting_classifier_results.png")

# ============================================================================
# 9. 詳細分類報告
# ============================================================================
print("\n【7】詳細分類報告")
print("-" * 80)

print(f"\n最佳基礎分類器 ({best_base_name})：")
print(classification_report(y_test, y_pred_best, target_names=iris.target_names))

print("\n硬投票分類器：")
print(classification_report(y_test, y_pred_hard, target_names=iris.target_names))

print("\n軟投票分類器：")
print(classification_report(y_test, y_pred_soft, target_names=iris.target_names))

# ============================================================================
# 10. 實驗2：不同數量的基礎分類器
# ============================================================================
print("\n【8】實驗：不同數量基礎分類器的影響")
print("-" * 80)

fig2 = plt.figure(figsize=(20, 10))

# 測試不同組合
estimators_list = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(max_depth=4, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42))
]

num_estimators = []
hard_scores = []
soft_scores = []

for i in range(2, len(estimators_list) + 1):
    current_estimators = estimators_list[:i]

    # 硬投票
    vc_hard = VotingClassifier(estimators=current_estimators, voting='hard')
    vc_hard.fit(X_train_scaled, y_train)
    hard_scores.append(vc_hard.score(X_test_scaled, y_test))

    # 軟投票
    vc_soft = VotingClassifier(estimators=current_estimators, voting='soft')
    vc_soft.fit(X_train_scaled, y_train)
    soft_scores.append(vc_soft.score(X_test_scaled, y_test))

    num_estimators.append(i)
    print(f"{i} 個分類器 - 硬投票: {hard_scores[-1]:.4f}, 軟投票: {soft_scores[-1]:.4f}")

# 10.1 分類器數量 vs 準確率
ax1 = plt.subplot(2, 3, 1)
ax1.plot(num_estimators, hard_scores, marker='o', label='Hard Voting', linewidth=2)
ax1.plot(num_estimators, soft_scores, marker='s', label='Soft Voting', linewidth=2)
ax1.set_xlabel('Number of Base Classifiers', fontsize=12)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title('Accuracy vs Number of Classifiers\n準確率 vs 分類器數量',
              fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ============================================================================
# 11. 實驗3：使用 Wine 數據集（完整特徵）
# ============================================================================
print("\n【9】實驗：Wine 數據集（完整特徵）")
print("-" * 80)

# 加載 Wine 數據集
wine = load_wine()
X_wine, y_wine = wine.data, wine.target

# 分割數據
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
)

# 標準化
scaler_wine = StandardScaler()
X_train_wine_scaled = scaler_wine.fit_transform(X_train_wine)
X_test_wine_scaled = scaler_wine.transform(X_test_wine)

print(f"Wine 數據集大小：{X_wine.shape}")
print(f"特徵數量：{X_wine.shape[1]}")
print(f"類別：{wine.target_names}")

# 訓練分類器
wine_results = []

for name, clf in base_classifiers:
    clf_wine = clf.__class__(**clf.get_params())
    clf_wine.fit(X_train_wine_scaled, y_train_wine)
    score = clf_wine.score(X_test_wine_scaled, y_test_wine)
    wine_results.append({'Classifier': name, 'Accuracy': score})
    print(f"{name}: {score:.4f}")

# 投票分類器
estimators_wine = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(max_depth=4, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42))
]

voting_hard_wine = VotingClassifier(estimators=estimators_wine, voting='hard')
voting_hard_wine.fit(X_train_wine_scaled, y_train_wine)
score_hard_wine = voting_hard_wine.score(X_test_wine_scaled, y_test_wine)

voting_soft_wine = VotingClassifier(estimators=estimators_wine, voting='soft')
voting_soft_wine.fit(X_train_wine_scaled, y_train_wine)
score_soft_wine = voting_soft_wine.score(X_test_wine_scaled, y_test_wine)

wine_results.append({'Classifier': 'Voting (Hard)', 'Accuracy': score_hard_wine})
wine_results.append({'Classifier': 'Voting (Soft)', 'Accuracy': score_soft_wine})

wine_results_df = pd.DataFrame(wine_results)
print(f"\nVoting (Hard): {score_hard_wine:.4f}")
print(f"Voting (Soft): {score_soft_wine:.4f}")

# 10.2 Wine 數據集結果
ax2 = plt.subplot(2, 3, 2)
colors_wine = ['skyblue'] * 5 + ['orange', 'red']
ax2.bar(range(len(wine_results_df)), wine_results_df['Accuracy'], color=colors_wine)
ax2.set_xticks(range(len(wine_results_df)))
ax2.set_xticklabels(wine_results_df['Classifier'], rotation=45, ha='right')
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Wine Dataset Results\nWine 數據集結果', fontsize=14, fontweight='bold')
ax2.axhline(y=wine_results_df['Accuracy'].iloc[:5].mean(), color='gray',
            linestyle='--', alpha=0.7)
ax2.grid(True, alpha=0.3, axis='y')

# ============================================================================
# 12. 權重投票實驗
# ============================================================================
print("\n【10】實驗：權重投票")
print("-" * 80)

# 根據性能設置權重
weights = [2, 1, 1, 2, 2]  # 給表現更好的模型更高權重

voting_weighted = VotingClassifier(
    estimators=estimators,
    voting='soft',
    weights=weights
)

voting_weighted.fit(X_train_scaled, y_train)
score_weighted = voting_weighted.score(X_test_scaled, y_test)

print(f"權重：{weights}")
print(f"均等權重軟投票: {accuracy_soft:.4f}")
print(f"加權軟投票: {score_weighted:.4f}")
print(f"改進: {(score_weighted - accuracy_soft) * 100:.2f}%")

# 10.3 權重實驗
ax3 = plt.subplot(2, 3, 3)
weight_comparison = pd.DataFrame({
    'Method': ['Equal Weights', 'Weighted'],
    'Accuracy': [accuracy_soft, score_weighted]
})
bars = ax3.bar(weight_comparison['Method'], weight_comparison['Accuracy'],
               color=['orange', 'green'], alpha=0.7)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('Equal vs Weighted Voting\n均等 vs 加權投票',
              fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=11)

# ============================================================================
# 13. 交叉驗證穩定性分析
# ============================================================================
print("\n【11】交叉驗證穩定性分析")
print("-" * 80)

cv_folds = 10
all_cv_scores = {}

classifiers_cv = [
    ('Best Base', base_classifiers[best_base_idx][1]),
    ('Hard Voting', voting_hard),
    ('Soft Voting', voting_soft),
    ('Weighted Voting', voting_weighted)
]

for name, clf in classifiers_cv:
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv_folds)
    all_cv_scores[name] = scores
    print(f"{name}:")
    print(f"  CV 分數: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"  最小值: {scores.min():.4f}, 最大值: {scores.max():.4f}")

# 10.4 CV 分數箱線圖
ax4 = plt.subplot(2, 3, 4)
cv_data = [all_cv_scores[name] for name in all_cv_scores.keys()]
bp = ax4.boxplot(cv_data, labels=all_cv_scores.keys(), patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'orange', 'red', 'lightgreen']):
    patch.set_facecolor(color)
ax4.set_ylabel('CV Accuracy', fontsize=12)
ax4.set_title('Cross-Validation Stability\n交叉驗證穩定性',
              fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ============================================================================
# 14. 預測概率分析（軟投票）
# ============================================================================
print("\n【12】預測概率分析")
print("-" * 80)

# 獲取預測概率
proba_soft = voting_soft.predict_proba(X_test_scaled)

# 計算預測置信度
confidence = np.max(proba_soft, axis=1)
predictions_correct = (voting_soft.predict(X_test_scaled) == y_test)

print(f"平均預測置信度: {confidence.mean():.4f}")
print(f"正確預測的平均置信度: {confidence[predictions_correct].mean():.4f}")
print(f"錯誤預測的平均置信度: {confidence[~predictions_correct].mean():.4f}")

# 10.5 預測置信度分布
ax5 = plt.subplot(2, 3, 5)
ax5.hist(confidence[predictions_correct], bins=20, alpha=0.6,
         label='Correct', color='green')
ax5.hist(confidence[~predictions_correct], bins=20, alpha=0.6,
         label='Incorrect', color='red')
ax5.set_xlabel('Prediction Confidence', fontsize=12)
ax5.set_ylabel('Frequency', fontsize=12)
ax5.set_title('Prediction Confidence Distribution\n預測置信度分布',
              fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# ============================================================================
# 15. 總結表格
# ============================================================================

# 10.6 總結對比表
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_data = [
    ['方法', '準確率', 'CV 均值', 'CV 標準差'],
    [best_base_name, f"{results_df['Test Accuracy'].iloc[best_base_idx]:.4f}",
     f"{all_cv_scores['Best Base'].mean():.4f}",
     f"{all_cv_scores['Best Base'].std():.4f}"],
    ['硬投票', f"{accuracy_hard:.4f}",
     f"{all_cv_scores['Hard Voting'].mean():.4f}",
     f"{all_cv_scores['Hard Voting'].std():.4f}"],
    ['軟投票', f"{accuracy_soft:.4f}",
     f"{all_cv_scores['Soft Voting'].mean():.4f}",
     f"{all_cv_scores['Soft Voting'].std():.4f}"],
    ['加權投票', f"{score_weighted:.4f}",
     f"{all_cv_scores['Weighted Voting'].mean():.4f}",
     f"{all_cv_scores['Weighted Voting'].std():.4f}"]
]

table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                 colWidths=[0.3, 0.2, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 設置表頭樣式
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 設置交替行顏色
for i in range(1, len(summary_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax6.set_title('Performance Summary\n性能總結', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/output/voting_classifier_experiments.png',
            dpi=150, bbox_inches='tight')
print("✓ 圖表已保存：output/voting_classifier_experiments.png")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("Voting Classifier 要點總結")
print("=" * 80)
print(f"""
1. 基礎分類器性能：
   • 最佳單個分類器: {best_base_name} ({results_df['Test Accuracy'].iloc[best_base_idx]:.4f})
   • 平均性能: {results_df['Test Accuracy'].iloc[:5].mean():.4f}

2. 投票分類器性能：
   • 硬投票: {accuracy_hard:.4f}
   • 軟投票: {accuracy_soft:.4f}
   • 加權投票: {score_weighted:.4f}
   • 相比最佳單個分類器提升: {(accuracy_soft - results_df['Test Accuracy'].iloc[best_base_idx]) * 100:.2f}%

3. 關鍵發現：
   • 軟投票通常優於硬投票（利用概率信息）
   • 加權投票可以進一步提升性能（給好模型更高權重）
   • 投票分類器更穩定（CV 標準差較小）
   • 基礎分類器越多樣化，效果越好

4. 最佳實踐：
   ✓ 使用不同類型的基礎分類器（增加多樣性）
   ✓ 優先使用軟投票（如果分類器支持概率輸出）
   ✓ 根據單個模型性能設置權重
   ✓ 確保基礎分類器至少優於隨機猜測
   ✓ 使用交叉驗證評估穩定性

5. 適用場景：
   • 追求最高準確率
   • 需要模型穩定性
   • 有多個候選模型難以選擇
   • 數據集不太大（訓練成本可接受）

6. 注意事項：
   ✗ 訓練時間是單個模型的 N 倍
   ✗ 預測速度較慢
   ✗ 模型可解釋性降低
   ✗ 內存佔用增加
""")

plt.show()
print("\n程序執行完成！")
