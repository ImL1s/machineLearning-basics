"""
支持向量機（Support Vector Machine, SVM）
強大的分類算法，特別適合中小規模數據集

原理：
- 尋找最優超平面將不同類別分開
- 最大化類別之間的間隔（margin）
- 使用核技巧處理非線性問題
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_circles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("支持向量機（SVM）教程".center(80))
print("=" * 80)

# ============================================================================
# 1. SVM 基本概念
# ============================================================================
print("\n【1】SVM 基本概念")
print("-" * 80)
print("""
SVM 核心思想：
• 尋找最優分離超平面
• 最大化類別間的間隔（margin）
• 支持向量：最接近超平面的樣本點

關鍵參數：
1. C（正則化參數）：
   - C 越大：間隔越小，允許誤分類樣本越少（可能過擬合）
   - C 越小：間隔越大，允許誤分類樣本越多（可能欠擬合）

2. kernel（核函數）：
   - linear：線性核（線性可分問題）
   - rbf：徑向基核（最常用，處理非線性問題）
   - poly：多項式核
   - sigmoid：Sigmoid 核

3. gamma（rbf/poly/sigmoid 核的參數）：
   - gamma 越大：決策邊界越複雜（可能過擬合）
   - gamma 越小：決策邊界越平滑（可能欠擬合）

優點：
✓ 在高維空間表現優異
✓ 對於小樣本數據效果好
✓ 泛化能力強
✓ 可處理非線性問題

缺點：
✗ 大規模數據訓練慢
✗ 對參數和核函數選擇敏感
✗ 不直接提供概率估計
""")

# ============================================================================
# 2. 實例1：乳腺癌診斷（二分類問題）
# ============================================================================
print("\n【2】實例1：乳腺癌診斷（Breast Cancer Dataset）")
print("-" * 80)

# 加載數據
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"數據集大小：{X.shape}")
print(f"特徵數量：{X.shape[1]}")
print(f"類別：{cancer.target_names}")
print(f"良性樣本：{np.sum(y == 1)}, 惡性樣本：{np.sum(y == 0)}")

# 數據分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 特徵縮放（SVM 對特徵尺度非常敏感！）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n訓練集大小：{X_train.shape}")
print(f"測試集大小：{X_test.shape}")

# ============================================================================
# 3. 線性 SVM
# ============================================================================
print("\n【3】線性 SVM")
print("-" * 80)

svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

y_pred_linear = svm_linear.predict(X_test_scaled)
accuracy_linear = accuracy_score(y_test, y_pred_linear)

print(f"線性 SVM 準確率：{accuracy_linear:.4f}")
print(f"支持向量數量：{len(svm_linear.support_vectors_)}")

# ============================================================================
# 4. RBF 核 SVM
# ============================================================================
print("\n【4】RBF 核 SVM")
print("-" * 80)

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

y_pred_rbf = svm_rbf.predict(X_test_scaled)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

print(f"RBF SVM 準確率：{accuracy_rbf:.4f}")
print(f"支持向量數量：{len(svm_rbf.support_vectors_)}")

# ============================================================================
# 5. 參數調優（Grid Search）
# ============================================================================
print("\n【5】參數調優（Grid Search）")
print("-" * 80)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

print("正在進行網格搜索...")
grid_search.fit(X_train_scaled, y_train)

print(f"最佳參數：{grid_search.best_params_}")
print(f"最佳交叉驗證分數：{grid_search.best_score_:.4f}")

# 使用最佳參數訓練模型
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)

print(f"最佳模型測試準確率：{accuracy_best:.4f}")

# ============================================================================
# 6. 詳細評估
# ============================================================================
print("\n【6】模型評估")
print("-" * 80)

print("\n分類報告：")
print(classification_report(y_test, y_pred_best, target_names=cancer.target_names))

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.title('Confusion Matrix (Best SVM)', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# ============================================================================
# 7. 不同核函數比較
# ============================================================================
print("\n【7】不同核函數比較")
print("-" * 80)

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernel_scores = []

for kernel in kernels:
    svm_temp = SVC(kernel=kernel, C=1.0, random_state=42)
    svm_temp.fit(X_train_scaled, y_train)
    score = svm_temp.score(X_test_scaled, y_test)
    kernel_scores.append(score)
    print(f"{kernel:10s} 核準確率：{score:.4f}")

# 可視化核函數比較
plt.subplot(1, 3, 2)
bars = plt.bar(kernels, kernel_scores, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
plt.xlabel('Kernel Type', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Different Kernels Comparison', fontsize=14, fontweight='bold')
plt.ylim([0.9, 1.0])
for bar, score in zip(bars, kernel_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3, axis='y')

# ============================================================================
# 8. C 參數的影響
# ============================================================================
print("\n【8】C 參數的影響")
print("-" * 80)

C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_scores_C = []
test_scores_C = []

for C in C_values:
    svm_c = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
    svm_c.fit(X_train_scaled, y_train)
    train_scores_C.append(svm_c.score(X_train_scaled, y_train))
    test_scores_C.append(svm_c.score(X_test_scaled, y_test))

plt.subplot(1, 3, 3)
plt.plot(C_values, train_scores_C, label='Training Accuracy', marker='o')
plt.plot(C_values, test_scores_C, label='Testing Accuracy', marker='s')
plt.xscale('log')
plt.xlabel('C (log scale)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('C Parameter vs Accuracy', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_SupervisedLearning/Classification/svm_results.png', dpi=150)
print("\n已保存結果圖表")

# ============================================================================
# 9. 實例2：非線性分類（圓形數據）
# ============================================================================
print("\n【9】實例2：非線性分類問題")
print("-" * 80)

# 生成圓形數據
X_circle, y_circle = make_circles(n_samples=500, noise=0.1, factor=0.3, random_state=42)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_circle, y_circle, test_size=0.3, random_state=42
)

def plot_svm_decision_boundary(clf, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k', alpha=0.7)

    # 標記支持向量
    if hasattr(clf, 'support_vectors_'):
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                   s=200, facecolors='none', edgecolors='k', linewidths=2,
                   label='Support Vectors')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

plt.figure(figsize=(15, 5))

# 線性核（預期效果不好）
plt.subplot(1, 3, 1)
svm_circle_linear = SVC(kernel='linear', C=1.0)
svm_circle_linear.fit(X_train_c, y_train_c)
score_linear = svm_circle_linear.score(X_test_c, y_test_c)
plot_svm_decision_boundary(svm_circle_linear, X_circle, y_circle,
                           f'Linear Kernel\nAccuracy={score_linear:.3f}')

# RBF 核（預期效果好）
plt.subplot(1, 3, 2)
svm_circle_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_circle_rbf.fit(X_train_c, y_train_c)
score_rbf = svm_circle_rbf.score(X_test_c, y_test_c)
plot_svm_decision_boundary(svm_circle_rbf, X_circle, y_circle,
                          f'RBF Kernel\nAccuracy={score_rbf:.3f}')

# 多項式核
plt.subplot(1, 3, 3)
svm_circle_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_circle_poly.fit(X_train_c, y_train_c)
score_poly = svm_circle_poly.score(X_test_c, y_test_c)
plot_svm_decision_boundary(svm_circle_poly, X_circle, y_circle,
                          f'Polynomial Kernel (degree=3)\nAccuracy={score_poly:.3f}')

plt.tight_layout()
plt.savefig('02_SupervisedLearning/Classification/svm_kernels_comparison.png', dpi=150)
print("已保存核函數比較圖")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("SVM 算法要點總結")
print("=" * 80)
print("""
1. 核函數選擇：
   • 線性可分 → 線性核
   • 非線性問題 → RBF 核（首選）
   • 不確定 → 先嘗試 RBF 核

2. 參數調優：
   • C：控制錯誤分類的懲罰
   • gamma：控制單個訓練樣本的影響範圍
   • 使用 GridSearchCV 或 RandomizedSearchCV

3. 數據預處理：
   • 特徵縮放是必須的！
   • 標準化通常優於歸一化

4. 適用場景：
   • 中小規模數據集（< 10萬樣本）
   • 高維數據
   • 需要高準確率的二分類問題

5. 注意事項：
   • 大規模數據考慮使用 LinearSVC
   • 不平衡數據使用 class_weight='balanced'
   • 需要概率輸出設置 probability=True（會增加訓練時間）
""")
