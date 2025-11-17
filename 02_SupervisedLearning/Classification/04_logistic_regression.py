"""
邏輯回歸（Logistic Regression）
最常用的二分類算法之一

雖然名字叫"回歸"，但實際上是分類算法
原理：使用 sigmoid 函數將線性回歸的輸出映射到 [0,1] 區間
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("邏輯回歸（Logistic Regression）教程".center(80))
print("=" * 80)

# ============================================================================
# 1. Logistic Regression 基本概念
# ============================================================================
print("\n【1】Logistic Regression 基本概念")
print("-" * 80)
print("""
邏輯回歸核心思想：
• 線性模型 + Sigmoid 函數
• 輸出概率值 P(y=1|x)
• 決策邊界為線性

數學公式：
P(y=1|x) = 1 / (1 + e^(-wx-b))

關鍵參數：
1. C（正則化強度的倒數）：
   - C 越大：正則化越弱，可能過擬合
   - C 越小：正則化越強，可能欠擬合
   - 默認 C=1.0

2. penalty（正則化類型）：
   - 'l1'：L1 正則化（特徵選擇）
   - 'l2'：L2 正則化（默認）
   - 'elasticnet'：L1 + L2
   - 'none'：無正則化

3. solver（優化算法）：
   - 'lbfgs'：適合小數據集（默認）
   - 'liblinear'：適合小數據集，支持 L1
   - 'sag'/'saga'：適合大數據集
   - 'newton-cg'：適合多分類

優點：
✓ 訓練速度快
✓ 可解釋性強
✓ 直接輸出概率
✓ 對線性可分問題效果好
✓ 不需要特徵縮放（但建議做）

缺點：
✗ 只能處理線性問題
✗ 對特徵工程要求高
✗ 容易欠擬合
""")

# ============================================================================
# 2. 實例：乳腺癌診斷
# ============================================================================
print("\n【2】實例：乳腺癌診斷")
print("-" * 80)

# 加載數據
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"數據集大小：{X.shape}")
print(f"類別：{cancer.target_names}")
print(f"各類別樣本數：良性 {np.sum(y==1)}, 惡性 {np.sum(y==0)}")

# 數據分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 特徵縮放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 3. 訓練基本模型
# ============================================================================
print("\n【3】訓練邏輯回歸模型")
print("-" * 80)

lr = LogisticRegression(random_state=42, max_iter=10000)
lr.fit(X_train_scaled, y_train)

# 預測
y_pred = lr.predict(X_test_scaled)
y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]

# 評估
accuracy = lr.score(X_test_scaled, y_test)
print(f"測試準確率：{accuracy:.4f}")
print(f"訓練準確率：{lr.score(X_train_scaled, y_train):.4f}")

print("\n分類報告：")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# ============================================================================
# 4. 不同 C 值的影響
# ============================================================================
print("\n【4】正則化參數 C 的影響")
print("-" * 80)

C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_scores = []
test_scores = []

for C in C_values:
    lr_c = LogisticRegression(C=C, random_state=42, max_iter=10000)
    lr_c.fit(X_train_scaled, y_train)
    train_scores.append(lr_c.score(X_train_scaled, y_train))
    test_scores.append(lr_c.score(X_test_scaled, y_test))

print("C 值對性能的影響：")
for C, train_score, test_score in zip(C_values, train_scores, test_scores):
    print(f"C={C:7.3f}: 訓練={train_score:.4f}, 測試={test_score:.4f}")

# ============================================================================
# 5. L1 vs L2 正則化
# ============================================================================
print("\n【5】L1 vs L2 正則化比較")
print("-" * 80)

lr_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)
lr_l2 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42, max_iter=10000)

lr_l1.fit(X_train_scaled, y_train)
lr_l2.fit(X_train_scaled, y_train)

print(f"L1 正則化測試準確率：{lr_l1.score(X_test_scaled, y_test):.4f}")
print(f"L2 正則化測試準確率：{lr_l2.score(X_test_scaled, y_test):.4f}")

print(f"\nL1 非零係數數量：{np.sum(lr_l1.coef_ != 0)}")
print(f"L2 非零係數數量：{np.sum(lr_l2.coef_ != 0)}")

# ============================================================================
# 6. 特徵重要性
# ============================================================================
print("\n【6】特徵重要性分析")
print("-" * 80)

feature_importance = pd.DataFrame({
    'feature': cancer.feature_names,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\n前10個最重要特徵：")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 可視化
# ============================================================================

fig = plt.figure(figsize=(16, 10))

# 1. 混淆矩陣
ax1 = plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names, ax=ax1)
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

# 2. ROC 曲線
ax2 = plt.subplot(2, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. C 值影響
ax3 = plt.subplot(2, 3, 3)
ax3.plot(C_values, train_scores, marker='o', label='Training Accuracy')
ax3.plot(C_values, test_scores, marker='s', label='Testing Accuracy')
ax3.set_xscale('log')
ax3.set_xlabel('C (log scale)')
ax3.set_ylabel('Accuracy')
ax3.set_title('C Parameter vs Accuracy', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 特徵係數
ax4 = plt.subplot(2, 3, 4)
top_features = feature_importance.head(15)
colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]
ax4.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['feature'], fontsize=8)
ax4.set_xlabel('Coefficient Value')
ax4.set_title('Top 15 Feature Coefficients', fontsize=12, fontweight='bold')
ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis='x')

# 5. 預測概率分布
ax5 = plt.subplot(2, 3, 5)
ax5.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Malignant (0)', color='red')
ax5.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Benign (1)', color='green')
ax5.set_xlabel('Predicted Probability')
ax5.set_ylabel('Frequency')
ax5.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. L1 vs L2 係數比較
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(range(len(lr_l1.coef_[0])), lr_l1.coef_[0],
           alpha=0.6, label='L1', s=50, marker='o')
ax6.scatter(range(len(lr_l2.coef_[0])), lr_l2.coef_[0],
           alpha=0.6, label='L2', s=50, marker='s')
ax6.set_xlabel('Feature Index')
ax6.set_ylabel('Coefficient Value')
ax6.set_title('L1 vs L2 Regularization Coefficients', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('02_SupervisedLearning/Classification/04_logistic_regression_results.png',
            dpi=150, bbox_inches='tight')
print("\n已保存結果圖表")

# ============================================================================
# 7. 交叉驗證
# ============================================================================
print("\n【7】交叉驗證")
print("-" * 80)

cv_scores = cross_val_score(
    LogisticRegression(random_state=42, max_iter=10000),
    X, y, cv=5, scoring='accuracy'
)

print(f"5折交叉驗證分數：{cv_scores}")
print(f"平均準確率：{cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("Logistic Regression 要點總結")
print("=" * 80)
print("""
1. 適用場景：
   • 二分類問題（也可擴展到多分類）
   • 需要概率輸出
   • 需要模型可解釋性
   • 線性可分或接近線性可分的數據

2. 參數調優：
   • C：通常從 [0.001, 0.01, 0.1, 1, 10, 100] 中選擇
   • penalty：L2（默認），L1（特徵選擇）
   • solver：小數據用 lbfgs，大數據用 saga

3. 特徵工程重要性：
   • 需要特徵縮放（StandardScaler）
   • 特徵交互可能提升性能
   • 多項式特徵可處理非線性

4. 與其他算法比較：
   • vs 線性回歸：LR 用於分類，輸出概率
   • vs SVM：LR 更快，SVM 更強大
   • vs Random Forest：LR 更可解釋，RF 更準確
   • vs 神經網絡：LR 更簡單，NN 更靈活

5. 最佳實踐：
   • 始終進行特徵縮放
   • 使用交叉驗證選擇 C
   • 檢查特徵係數的合理性
   • 考慮特徵多項式轉換
   • 處理類別不平衡（class_weight='balanced'）
""")
