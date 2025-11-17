"""
堆疊集成（Stacking / Stacked Generalization）
高級集成學習方法 - 模型的模型

原理：
- 多層學習架構
- 第一層：多個不同的基礎模型（Base Learners）
- 第二層：元學習器（Meta-Learner）學習如何組合基礎模型
- 使用交叉驗證避免信息洩露
- 比簡單投票更靈活、更強大
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier, VotingClassifier, AdaBoostClassifier)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("堆疊集成（Stacking）教程".center(80))
print("=" * 80)

# ============================================================================
# 1. Stacking 基本概念
# ============================================================================
print("\n【1】Stacking 基本概念")
print("-" * 80)
print("""
Stacking（堆疊泛化）核心思想：
• 多層學習架構，類似深度學習的概念
• 第一層（Level 0）：多個不同的基礎模型
• 第二層（Level 1）：元學習器（Meta-Learner）

算法流程：
1. 數據準備：
   - 訓練集：用於訓練所有模型
   - 測試集：最終評估

2. 訓練第一層（基礎模型）：
   - 使用交叉驗證訓練多個不同的基礎模型
   - 對訓練集：獲取每個樣本的預測（通過 CV）
   - 對測試集：獲取每個基礎模型的預測
   - 這些預測成為新的特徵

3. 訓練第二層（元學習器）：
   - 輸入：基礎模型的預測（作為特徵）
   - 輸出：最終預測
   - 學習如何最優組合基礎模型

4. 預測：
   - 新數據 → 所有基礎模型 → 預測集合 → 元學習器 → 最終預測

關鍵要點：
✓ 基礎模型應該多樣化（不同算法、不同參數）
✓ 必須使用交叉驗證避免數據洩露
✓ 元學習器通常選擇簡單模型（如 Logistic Regression）
✓ 可以使用原始特徵 + 預測（passthrough=True）

Stacking vs Voting vs Boosting：
• Voting：簡單平均或多數投票（固定組合方式）
• Boosting：順序訓練，後面的模型關注前面的錯誤
• Stacking：學習如何組合（更靈活，性能通常更好）

優點：
✓ 通常比單個模型和簡單集成性能更好
✓ 可以結合不同類型模型的優勢
✓ 靈活性高，可以自定義各層模型
✓ 理論上可以堆疊多層（但通常 2 層足夠）

缺點：
✗ 訓練時間長（CV + 多個模型）
✗ 複雜度高，難以調試
✗ 容易過擬合（如果不正確使用 CV）
✗ 可解釋性差
""")

# ============================================================================
# 2. 數據準備：Digits 數據集
# ============================================================================
print("\n【2】數據準備：Digits 數據集")
print("-" * 80)

# 加載數據
digits = load_digits()
X, y = digits.data, digits.target

print(f"數據集大小：{X.shape}")
print(f"特徵數量：{X.shape[1]}")
print(f"類別數量：{len(np.unique(y))}")
print(f"類別：{np.unique(y)}")
print(f"各類別樣本數：")
for i in range(10):
    print(f"  數字 {i}: {np.sum(y == i)} 個樣本")

# 數據分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n訓練集大小：{X_train.shape}")
print(f"測試集大小：{X_test.shape}")

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 3. 定義基礎模型（第一層）
# ============================================================================
print("\n【3】定義基礎模型（Level 0）")
print("-" * 80)

# 創建多樣化的基礎模型
base_learners = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('nb', GaussianNB()),
]

print("基礎模型（Level 0）：")
for name, model in base_learners:
    print(f"  • {name}: {model.__class__.__name__}")

# ============================================================================
# 4. 訓練並評估各個基礎模型
# ============================================================================
print("\n【4】訓練並評估各個基礎模型")
print("-" * 80)

base_results = []

for name, model in base_learners:
    # 訓練
    model.fit(X_train_scaled, y_train)

    # 預測
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # 交叉驗證
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    base_results.append({
        'Model': name,
        'Full Name': model.__class__.__name__,
        'Test Accuracy': accuracy,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })

    print(f"{name:4s} ({model.__class__.__name__:25s}): "
          f"Test={accuracy:.4f}, CV={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

base_results_df = pd.DataFrame(base_results)

# ============================================================================
# 5. 定義元學習器（第二層）
# ============================================================================
print("\n【5】定義元學習器（Level 1）")
print("-" * 80)

# 元學習器通常使用簡單模型
meta_learner = LogisticRegression(max_iter=1000, random_state=42)

print(f"元學習器（Level 1）：{meta_learner.__class__.__name__}")
print("""
為什麼選擇 Logistic Regression 作為元學習器？
• 簡單，不易過擬合
• 訓練快速
• 可以輸出概率
• 易於解釋（可以看各個基礎模型的權重）
""")

# ============================================================================
# 6. 創建並訓練 Stacking 模型
# ============================================================================
print("\n【6】創建並訓練 Stacking 模型")
print("-" * 80)

# 創建 Stacking 分類器
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,  # 使用 5 折交叉驗證
    stack_method='auto',  # 'auto', 'predict_proba', 'decision_function', or 'predict'
    n_jobs=-1
)

print("訓練 Stacking 分類器...")
print("  • 使用 5 折交叉驗證")
print("  • 基礎模型數量：", len(base_learners))

stacking_clf.fit(X_train_scaled, y_train)

# 預測
y_pred_stacking = stacking_clf.predict(X_test_scaled)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)

print(f"\nStacking 測試準確率：{accuracy_stacking:.4f}")

# ============================================================================
# 7. 創建對比模型
# ============================================================================
print("\n【7】創建對比模型")
print("-" * 80)

# 7.1 Voting Classifier（硬投票）
voting_hard = VotingClassifier(
    estimators=base_learners,
    voting='hard',
    n_jobs=-1
)
voting_hard.fit(X_train_scaled, y_train)
accuracy_voting_hard = voting_hard.score(X_test_scaled, y_test)
print(f"Voting (Hard) 準確率：{accuracy_voting_hard:.4f}")

# 7.2 Voting Classifier（軟投票）
voting_soft = VotingClassifier(
    estimators=base_learners,
    voting='soft',
    n_jobs=-1
)
voting_soft.fit(X_train_scaled, y_train)
accuracy_voting_soft = voting_soft.score(X_test_scaled, y_test)
print(f"Voting (Soft) 準確率：{accuracy_voting_soft:.4f}")

# 7.3 最佳單個模型
best_base_idx = base_results_df['Test Accuracy'].idxmax()
best_base_name = base_results_df.iloc[best_base_idx]['Model']
best_base_accuracy = base_results_df.iloc[best_base_idx]['Test Accuracy']
print(f"最佳單個模型 ({best_base_name}): {best_base_accuracy:.4f}")

# ============================================================================
# 8. 可視化結果
# ============================================================================
print("\n【8】可視化結果")
print("-" * 80)

fig1 = plt.figure(figsize=(20, 16))

# 8.1 所有模型性能對比
ax1 = plt.subplot(3, 3, 1)

# 準備數據
all_models = base_results_df['Model'].tolist() + ['Vote(H)', 'Vote(S)', 'Stacking']
all_accuracies = (base_results_df['Test Accuracy'].tolist() +
                  [accuracy_voting_hard, accuracy_voting_soft, accuracy_stacking])

colors = ['skyblue'] * len(base_learners) + ['orange', 'orange', 'red']
bars = ax1.bar(range(len(all_models)), all_accuracies, color=colors, alpha=0.7)

ax1.set_xticks(range(len(all_models)))
ax1.set_xticklabels(all_models, rotation=45, ha='right')
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title('Model Comparison\n模型性能對比', fontsize=14, fontweight='bold')
ax1.axhline(y=best_base_accuracy, color='gray', linestyle='--',
            alpha=0.7, label=f'Best Base ({best_base_name})')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 添加數值標籤
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=8)

# 8.2 性能提升對比
ax2 = plt.subplot(3, 3, 2)
improvements = [(acc - best_base_accuracy) * 100
                for acc in [accuracy_voting_hard, accuracy_voting_soft, accuracy_stacking]]
methods = ['Voting\n(Hard)', 'Voting\n(Soft)', 'Stacking']
colors_imp = ['orange', 'orange', 'red']

bars = ax2.bar(methods, improvements, color=colors_imp, alpha=0.7)
ax2.set_ylabel('Improvement over Best Base (%)', fontsize=12)
ax2.set_title('Performance Improvement\n性能改進', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom' if height >= 0 else 'top', fontsize=11)

# 8.3 交叉驗證對比
ax3 = plt.subplot(3, 3, 3)
ensemble_methods = ['Best Base', 'Voting (Hard)', 'Voting (Soft)', 'Stacking']
ensemble_cv_scores = {}

# 獲取最佳基礎模型
best_base_model = base_learners[best_base_idx][1]
ensemble_cv_scores['Best Base'] = cross_val_score(
    best_base_model, X_train_scaled, y_train, cv=5
)
ensemble_cv_scores['Voting (Hard)'] = cross_val_score(
    voting_hard, X_train_scaled, y_train, cv=5
)
ensemble_cv_scores['Voting (Soft)'] = cross_val_score(
    voting_soft, X_train_scaled, y_train, cv=5
)
ensemble_cv_scores['Stacking'] = cross_val_score(
    stacking_clf, X_train_scaled, y_train, cv=5
)

cv_data = [ensemble_cv_scores[method] for method in ensemble_methods]
bp = ax3.boxplot(cv_data, labels=ensemble_methods, patch_artist=True)
colors_box = ['lightblue', 'orange', 'orange', 'red']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_ylabel('CV Accuracy', fontsize=12)
ax3.set_title('Cross-Validation Scores\n交叉驗證分數', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

print("\n交叉驗證結果：")
for method, scores in ensemble_cv_scores.items():
    print(f"{method:15s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ============================================================================
# 9. 混淆矩陣對比
# ============================================================================

# 9.1 最佳基礎模型混淆矩陣
ax4 = plt.subplot(3, 3, 4)
y_pred_best = best_base_model.predict(X_test_scaled)
cm_best = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=False)
ax4.set_title(f'Confusion Matrix: Best Base ({best_base_name})\n最佳基礎模型',
              fontsize=12, fontweight='bold')
ax4.set_ylabel('True Label')
ax4.set_xlabel('Predicted Label')

# 9.2 Voting 混淆矩陣
ax5 = plt.subplot(3, 3, 5)
y_pred_voting = voting_soft.predict(X_test_scaled)
cm_voting = confusion_matrix(y_test, y_pred_voting)
sns.heatmap(cm_voting, annot=True, fmt='d', cmap='Oranges', ax=ax5, cbar=False)
ax5.set_title('Confusion Matrix: Voting (Soft)\n軟投票',
              fontsize=12, fontweight='bold')
ax5.set_ylabel('True Label')
ax5.set_xlabel('Predicted Label')

# 9.3 Stacking 混淆矩陣
ax6 = plt.subplot(3, 3, 6)
cm_stacking = confusion_matrix(y_test, y_pred_stacking)
sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Reds', ax=ax6, cbar=False)
ax6.set_title('Confusion Matrix: Stacking\n堆疊集成',
              fontsize=12, fontweight='bold')
ax6.set_ylabel('True Label')
ax6.set_xlabel('Predicted Label')

# ============================================================================
# 10. 詳細分類報告
# ============================================================================
print("\n【9】詳細分類報告")
print("-" * 80)

print(f"\n最佳基礎模型 ({best_base_name})：")
print(classification_report(y_test, y_pred_best, target_names=[str(i) for i in range(10)]))

print("\nVoting (Soft)：")
print(classification_report(y_test, y_pred_voting, target_names=[str(i) for i in range(10)]))

print("\nStacking：")
print(classification_report(y_test, y_pred_stacking, target_names=[str(i) for i in range(10)]))

# ============================================================================
# 11. 元學習器分析
# ============================================================================
print("\n【10】元學習器分析")
print("-" * 80)

# 獲取元學習器的係數（如果是線性模型）
if hasattr(stacking_clf.final_estimator_, 'coef_'):
    meta_coef = stacking_clf.final_estimator_.coef_
    print(f"元學習器係數形狀：{meta_coef.shape}")

    # 計算每個基礎模型的平均重要性
    base_importance = np.abs(meta_coef).mean(axis=0)

    # 10.1 基礎模型在元學習器中的重要性
    ax7 = plt.subplot(3, 3, 7)
    base_names = [name for name, _ in base_learners]
    ax7.barh(base_names, base_importance, color='teal', alpha=0.7)
    ax7.set_xlabel('Importance (abs coef)', fontsize=12)
    ax7.set_title('Base Model Importance in Meta-Learner\n基礎模型在元學習器中的重要性',
                  fontsize=12, fontweight='bold')
    ax7.invert_yaxis()
    ax7.grid(True, alpha=0.3, axis='x')

    print("\n基礎模型重要性（元學習器視角）：")
    for name, imp in zip(base_names, base_importance):
        print(f"  {name:4s}: {imp:.4f}")
else:
    ax7 = plt.subplot(3, 3, 7)
    ax7.text(0.5, 0.5, 'Meta-learner coefficients\nnot available',
             ha='center', va='center', fontsize=14)
    ax7.axis('off')

# ============================================================================
# 12. 預測置信度分析
# ============================================================================
print("\n【11】預測置信度分析")
print("-" * 80)

# 獲取預測概率
proba_stacking = stacking_clf.predict_proba(X_test_scaled)
proba_voting = voting_soft.predict_proba(X_test_scaled)

# 計算置信度（最大概率）
confidence_stacking = np.max(proba_stacking, axis=1)
confidence_voting = np.max(proba_voting, axis=1)

# 判斷預測是否正確
correct_stacking = (y_pred_stacking == y_test)
correct_voting = (y_pred_voting == y_test)

print(f"Stacking 平均置信度：{confidence_stacking.mean():.4f}")
print(f"  正確預測: {confidence_stacking[correct_stacking].mean():.4f}")
print(f"  錯誤預測: {confidence_stacking[~correct_stacking].mean():.4f}")

print(f"\nVoting 平均置信度：{confidence_voting.mean():.4f}")
print(f"  正確預測: {confidence_voting[correct_voting].mean():.4f}")
print(f"  錯誤預測: {confidence_voting[~correct_voting].mean():.4f}")

# 10.2 預測置信度分布
ax8 = plt.subplot(3, 3, 8)
ax8.hist(confidence_stacking[correct_stacking], bins=20, alpha=0.6,
         label='Stacking (Correct)', color='green')
ax8.hist(confidence_stacking[~correct_stacking], bins=20, alpha=0.6,
         label='Stacking (Incorrect)', color='red')
ax8.set_xlabel('Prediction Confidence', fontsize=12)
ax8.set_ylabel('Frequency', fontsize=12)
ax8.set_title('Prediction Confidence: Stacking\n預測置信度分布',
              fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# ============================================================================
# 13. 各類別性能對比
# ============================================================================

# 計算每個類別的 F1 分數
from sklearn.metrics import f1_score

f1_best = f1_score(y_test, y_pred_best, average=None)
f1_voting = f1_score(y_test, y_pred_voting, average=None)
f1_stacking = f1_score(y_test, y_pred_stacking, average=None)

# 10.3 各類別 F1 分數對比
ax9 = plt.subplot(3, 3, 9)
x = np.arange(10)
width = 0.25

ax9.bar(x - width, f1_best, width, label=f'Best Base ({best_base_name})', alpha=0.7)
ax9.bar(x, f1_voting, width, label='Voting (Soft)', alpha=0.7)
ax9.bar(x + width, f1_stacking, width, label='Stacking', alpha=0.7)

ax9.set_xlabel('Digit Class', fontsize=12)
ax9.set_ylabel('F1 Score', fontsize=12)
ax9.set_title('F1 Score by Class\n各類別 F1 分數', fontsize=14, fontweight='bold')
ax9.set_xticks(x)
ax9.set_xticklabels([str(i) for i in range(10)])
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/output/stacking_analysis.png',
            dpi=150, bbox_inches='tight')
print("✓ 圖表已保存：output/stacking_analysis.png")

# ============================================================================
# 14. 高級實驗：多種元學習器對比
# ============================================================================
print("\n【12】高級實驗：多種元學習器對比")
print("-" * 80)

fig2 = plt.figure(figsize=(20, 10))

# 嘗試不同的元學習器
meta_learners = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Ridge Classifier', RidgeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)),
]

meta_results = []

for meta_name, meta_model in meta_learners:
    print(f"\n測試元學習器: {meta_name}")

    stacking_temp = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    stacking_temp.fit(X_train_scaled, y_train)

    # 測試分數
    test_score = stacking_temp.score(X_test_scaled, y_test)

    # 交叉驗證
    cv_scores = cross_val_score(stacking_temp, X_train_scaled, y_train, cv=5)

    meta_results.append({
        'Meta-Learner': meta_name,
        'Test Accuracy': test_score,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })

    print(f"  測試準確率: {test_score:.4f}")
    print(f"  交叉驗證: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

meta_results_df = pd.DataFrame(meta_results)

# 14.1 元學習器性能對比
ax1 = plt.subplot(2, 3, 1)
bars = ax1.bar(meta_results_df['Meta-Learner'], meta_results_df['Test Accuracy'],
               color=['red', 'blue', 'green', 'orange'], alpha=0.7)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title('Meta-Learner Comparison\n元學習器對比', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10)

# ============================================================================
# 15. passthrough 參數實驗
# ============================================================================
print("\n【13】passthrough 參數實驗")
print("-" * 80)
print("""
passthrough 參數說明：
• passthrough=False（默認）：元學習器只使用基礎模型的預測
• passthrough=True：元學習器使用基礎模型的預測 + 原始特徵

理論上，使用原始特徵可能提升性能，但也可能導致過擬合。
""")

# 不使用 passthrough
stacking_no_pass = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    passthrough=False,
    n_jobs=-1
)
stacking_no_pass.fit(X_train_scaled, y_train)
score_no_pass = stacking_no_pass.score(X_test_scaled, y_test)

# 使用 passthrough
stacking_pass = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    passthrough=True,
    n_jobs=-1
)
stacking_pass.fit(X_train_scaled, y_train)
score_pass = stacking_pass.score(X_test_scaled, y_test)

print(f"passthrough=False: {score_no_pass:.4f}")
print(f"passthrough=True:  {score_pass:.4f}")
print(f"差異: {(score_pass - score_no_pass) * 100:.2f}%")

# 14.2 passthrough 對比
ax2 = plt.subplot(2, 3, 2)
pass_comparison = pd.DataFrame({
    'Setting': ['passthrough=False', 'passthrough=True'],
    'Accuracy': [score_no_pass, score_pass]
})
bars = ax2.bar(pass_comparison['Setting'], pass_comparison['Accuracy'],
               color=['red', 'purple'], alpha=0.7)
ax2.set_ylabel('Test Accuracy', fontsize=12)
ax2.set_title('Passthrough Parameter Effect\npassthrough 參數影響',
              fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=11)

# ============================================================================
# 16. 不同 CV 折數的影響
# ============================================================================
print("\n【14】不同 CV 折數的影響")
print("-" * 80)

cv_folds = [3, 5, 7, 10]
cv_results = []

for cv in cv_folds:
    print(f"\n測試 CV={cv}")

    stacking_cv = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=cv,
        n_jobs=-1
    )

    # 訓練時間測試
    import time
    start_time = time.time()
    stacking_cv.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    score = stacking_cv.score(X_test_scaled, y_test)

    cv_results.append({
        'CV Folds': cv,
        'Test Accuracy': score,
        'Training Time': train_time
    })

    print(f"  準確率: {score:.4f}")
    print(f"  訓練時間: {train_time:.2f} 秒")

cv_results_df = pd.DataFrame(cv_results)

# 14.3 CV 折數 vs 準確率
ax3 = plt.subplot(2, 3, 3)
ax3.plot(cv_results_df['CV Folds'], cv_results_df['Test Accuracy'],
         marker='o', linewidth=2, markersize=10, color='red')
ax3.set_xlabel('Number of CV Folds', fontsize=12)
ax3.set_ylabel('Test Accuracy', fontsize=12)
ax3.set_title('CV Folds vs Accuracy\nCV 折數 vs 準確率',
              fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 14.4 CV 折數 vs 訓練時間
ax4 = plt.subplot(2, 3, 4)
ax4.plot(cv_results_df['CV Folds'], cv_results_df['Training Time'],
         marker='s', linewidth=2, markersize=10, color='blue')
ax4.set_xlabel('Number of CV Folds', fontsize=12)
ax4.set_ylabel('Training Time (seconds)', fontsize=12)
ax4.set_title('CV Folds vs Training Time\nCV 折數 vs 訓練時間',
              fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# ============================================================================
# 17. Stacking 架構圖
# ============================================================================

# 14.5 Stacking 架構示意圖
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

# 繪製簡化的架構圖
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 輸入層
input_box = FancyBboxPatch((0.1, 0.8), 0.8, 0.1, boxstyle="round,pad=0.01",
                          edgecolor='black', facecolor='lightblue', linewidth=2)
ax5.add_patch(input_box)
ax5.text(0.5, 0.85, 'Input Data\n(原始特徵)', ha='center', va='center',
         fontsize=11, fontweight='bold')

# Level 0 (基礎模型)
y_base = 0.5
base_width = 0.13
for i, (name, _) in enumerate(base_learners):
    x = 0.05 + i * 0.15
    box = FancyBboxPatch((x, y_base), base_width, 0.12,
                        boxstyle="round,pad=0.01",
                        edgecolor='darkblue', facecolor='skyblue', linewidth=1.5)
    ax5.add_patch(box)
    ax5.text(x + base_width/2, y_base + 0.06, name.upper(),
            ha='center', va='center', fontsize=9)

    # 箭頭從輸入到基礎模型
    arrow = FancyArrowPatch((0.5, 0.8), (x + base_width/2, 0.62),
                          arrowstyle='->', mutation_scale=15, linewidth=1,
                          color='gray', alpha=0.5)
    ax5.add_patch(arrow)

# Level 1 標籤
ax5.text(0.5, 0.36, 'Level 1: Meta-Learner', ha='center', va='center',
         fontsize=11, fontweight='bold', style='italic')

# 元學習器
meta_box = FancyBboxPatch((0.3, 0.15), 0.4, 0.15,
                         boxstyle="round,pad=0.01",
                         edgecolor='darkred', facecolor='lightcoral', linewidth=2)
ax5.add_patch(meta_box)
ax5.text(0.5, 0.225, 'Meta-Learner\n(Logistic Regression)',
         ha='center', va='center', fontsize=10, fontweight='bold')

# 箭頭從基礎模型到元學習器
for i in range(len(base_learners)):
    x = 0.05 + i * 0.15 + base_width/2
    arrow = FancyArrowPatch((x, 0.5), (0.5, 0.3),
                          arrowstyle='->', mutation_scale=15, linewidth=1,
                          color='gray', alpha=0.5)
    ax5.add_patch(arrow)

# 輸出
output_box = FancyBboxPatch((0.35, 0.01), 0.3, 0.08,
                           boxstyle="round,pad=0.01",
                           edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
ax5.add_patch(output_box)
ax5.text(0.5, 0.05, 'Final Prediction', ha='center', va='center',
         fontsize=10, fontweight='bold')

# 箭頭從元學習器到輸出
arrow = FancyArrowPatch((0.5, 0.15), (0.5, 0.09),
                       arrowstyle='->', mutation_scale=20, linewidth=2,
                       color='darkgreen')
ax5.add_patch(arrow)

# 標籤
ax5.text(0.02, 0.55, 'Level 0:\nBase Models', ha='left', va='center',
         fontsize=10, fontweight='bold', style='italic')

ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.set_title('Stacking Architecture\n堆疊架構', fontsize=14, fontweight='bold', pad=10)

# ============================================================================
# 18. 總結表格
# ============================================================================

# 14.6 總結對比表
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_data = [
    ['方法', '測試準確率', 'CV 均值', '改進 (%)'],
    [f'最佳基礎 ({best_base_name})',
     f"{best_base_accuracy:.4f}",
     f"{ensemble_cv_scores['Best Base'].mean():.4f}",
     '0.00%'],
    ['硬投票',
     f"{accuracy_voting_hard:.4f}",
     f"{ensemble_cv_scores['Voting (Hard)'].mean():.4f}",
     f"{(accuracy_voting_hard - best_base_accuracy) * 100:.2f}%"],
    ['軟投票',
     f"{accuracy_voting_soft:.4f}",
     f"{ensemble_cv_scores['Voting (Soft)'].mean():.4f}",
     f"{(accuracy_voting_soft - best_base_accuracy) * 100:.2f}%"],
    ['Stacking',
     f"{accuracy_stacking:.4f}",
     f"{ensemble_cv_scores['Stacking'].mean():.4f}",
     f"{(accuracy_stacking - best_base_accuracy) * 100:.2f}%"]
]

table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                 colWidths=[0.3, 0.25, 0.25, 0.2])
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
best_row = 4  # Stacking 行
table[(best_row, 1)].set_facecolor('#90EE90')
table[(best_row, 1)].set_text_props(weight='bold')
table[(best_row, 2)].set_facecolor('#90EE90')
table[(best_row, 2)].set_text_props(weight='bold')

ax6.set_title('Performance Summary\n性能總結', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/output/stacking_advanced.png',
            dpi=150, bbox_inches='tight')
print("✓ 圖表已保存：output/stacking_advanced.png")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("Stacking 要點總結")
print("=" * 80)
print(f"""
1. 模型性能對比：
   • 最佳基礎模型 ({best_base_name}): {best_base_accuracy:.4f}
   • Voting (Hard): {accuracy_voting_hard:.4f}
   • Voting (Soft): {accuracy_voting_soft:.4f}
   • Stacking: {accuracy_stacking:.4f}

2. 性能提升：
   • Stacking vs 最佳基礎: {(accuracy_stacking - best_base_accuracy) * 100:.2f}%
   • Stacking vs Voting (Soft): {(accuracy_stacking - accuracy_voting_soft) * 100:.2f}%

3. 元學習器對比：
   • 最佳元學習器: {meta_results_df.iloc[meta_results_df['Test Accuracy'].idxmax()]['Meta-Learner']}
   • 準確率: {meta_results_df['Test Accuracy'].max():.4f}

4. 參數影響：
   • passthrough=True vs False: {(score_pass - score_no_pass) * 100:.2f}%
   • 推薦 CV 折數: 5 (性能與時間的平衡)

5. 關鍵發現：
   ✓ Stacking 通常優於簡單的 Voting
   ✓ 基礎模型多樣性很重要
   ✓ 簡單的元學習器（如 Logistic Regression）通常效果最好
   ✓ 必須使用 CV 避免數據洩露
   ✓ passthrough 參數可能有幫助，但需要測試

6. 適用場景：
   ✓ 追求最高性能（如競賽）
   ✓ 數據量充足
   ✓ 訓練時間不是主要考慮
   ✓ 已有多個表現不錯的基礎模型

7. 最佳實踐：
   ✓ 選擇多樣化的基礎模型
   ✓ 使用簡單的元學習器（避免過擬合）
   ✓ 使用 5-10 折 CV
   ✓ 先在小數據集上驗證
   ✓ 監控訓練/驗證性能差距

8. 注意事項：
   ✗ 訓練時間長（CV × 基礎模型數量）
   ✗ 模型複雜，難以調試
   ✗ 容易過擬合（如果 CV 使用不當）
   ✗ 預測時間增加
   ✗ 可解釋性差

9. Voting vs Stacking 選擇：
   • Voting：簡單、快速、穩定
   • Stacking：性能更好、更靈活、更複雜
   • 建議：先嘗試 Voting，如需進一步提升再用 Stacking

10. 三種集成方法總結：
    • Voting：並行訓練，固定組合（平均/投票）
    • Boosting：順序訓練，關注錯誤樣本
    • Stacking：兩層架構，學習如何組合
""")

plt.show()
print("\n程序執行完成！")
