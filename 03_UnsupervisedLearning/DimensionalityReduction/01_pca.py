"""
主成分分析（PCA - Principal Component Analysis）
最常用的降維技術

原理：找到數據變化最大的方向，投影到低維空間
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("=" * 80)
print("主成分分析（PCA）教程".center(80))
print("=" * 80)

# 實例1：鳶尾花數據降維可視化
print("\n【實例1】鳶尾花數據降維可視化")
print("-" * 80)

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# 標準化
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# PCA 降維到 2D
pca_2d = PCA(n_components=2)
X_iris_2d = pca_2d.fit_transform(X_iris_scaled)

print(f"原始維度：{X_iris.shape[1]}")
print(f"降維後維度：{X_iris_2d.shape[1]}")
print(f"解釋方差比：{pca_2d.explained_variance_ratio_}")
print(f"累積解釋方差：{np.sum(pca_2d.explained_variance_ratio_):.4f}")

# 實例2：手寫數字降維
print("\n【實例2】手寫數字數據降維")
print("-" * 80)

digits = load_digits()
X_digits, y_digits = digits.data, digits.target

print(f"原始數據形狀：{X_digits.shape}")
print(f"原始維度：{X_digits.shape[1]} (8x8 像素)")

# 確定最佳成分數量
pca_full = PCA()
pca_full.fit(X_digits)

cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)

# 找到解釋 95% 方差所需的成分數
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
print(f"保留 95% 方差需要 {n_components_95} 個主成分")

# 降維
pca_reduced = PCA(n_components=n_components_95)
X_digits_reduced = pca_reduced.fit_transform(X_digits)

print(f"降維後形狀：{X_digits_reduced.shape}")
print(f"維度壓縮率：{X_digits_reduced.shape[1] / X_digits.shape[1] * 100:.1f}%")

# 比較分類性能
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=42
)

X_train_pca, X_test_pca, _, _ = train_test_split(
    X_digits_reduced, y_digits, test_size=0.2, random_state=42
)

# 原始數據訓練
rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
rf_original.fit(X_train, y_train)
score_original = rf_original.score(X_test, y_test)

# PCA 數據訓練
rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)
score_pca = rf_pca.score(X_test_pca, y_test)

print(f"\n原始數據準確率：{score_original:.4f}")
print(f"PCA 數據準確率：{score_pca:.4f}")
print(f"性能差異：{abs(score_original - score_pca):.4f}")

# 可視化
fig = plt.figure(figsize=(18, 10))

# 1. 鳶尾花 2D 投影
ax1 = plt.subplot(2, 3, 1)
for i, target_name in enumerate(iris.target_names):
    ax1.scatter(X_iris_2d[y_iris == i, 0], X_iris_2d[y_iris == i, 1],
               label=target_name, alpha=0.7, edgecolors='k')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')
ax1.set_title('Iris Dataset - PCA 2D Projection', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 累積解釋方差
ax2 = plt.subplot(2, 3, 2)
ax2.plot(range(1, len(cumsum_var) + 1), cumsum_var, marker='o')
ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
ax2.axvline(x=n_components_95, color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 各成分解釋方差
ax3 = plt.subplot(2, 3, 3)
ax3.bar(range(1, 11), pca_full.explained_variance_ratio_[:10], alpha=0.7)
ax3.set_xlabel('Principal Component')
ax3.set_ylabel('Explained Variance Ratio')
ax3.set_title('Variance Explained by Each Component (Top 10)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. 手寫數字 2D 投影
pca_digits_2d = PCA(n_components=2)
X_digits_2d = pca_digits_2d.fit_transform(X_digits)

ax4 = plt.subplot(2, 3, 4)
scatter = ax4.scatter(X_digits_2d[:, 0], X_digits_2d[:, 1],
                     c=y_digits, cmap='tab10', alpha=0.6, edgecolors='k', s=20)
ax4.set_xlabel('First Principal Component')
ax4.set_ylabel('Second Principal Component')
ax4.set_title('Digits Dataset - PCA 2D Projection', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax4, label='Digit')
ax4.grid(True, alpha=0.3)

# 5. 原始 vs 重建圖像
ax5 = plt.subplot(2, 3, 5)
sample_idx = 0
original_image = X_digits[sample_idx].reshape(8, 8)
ax5.imshow(original_image, cmap='gray')
ax5.set_title('Original Image (64 features)', fontsize=12, fontweight='bold')
ax5.axis('off')

# 6. PCA 重建圖像
ax6 = plt.subplot(2, 3, 6)
reconstructed = pca_reduced.inverse_transform(X_digits_reduced[sample_idx:sample_idx+1])
reconstructed_image = reconstructed.reshape(8, 8)
ax6.imshow(reconstructed_image, cmap='gray')
ax6.set_title(f'Reconstructed ({n_components_95} components)', fontsize=12, fontweight='bold')
ax6.axis('off')

plt.tight_layout()
plt.savefig('03_UnsupervisedLearning/DimensionalityReduction/pca_results.png', dpi=150)
print("\n已保存結果圖表")

print("\n" + "=" * 80)
print("PCA 要點總結：")
print("=" * 80)
print("""
優點：
✓ 去除相關特徵，保留主要信息
✓ 降低計算複雜度
✓ 可視化高維數據
✓ 去除噪聲

缺點：
✗ 結果難以解釋（主成分是原特徵的線性組合）
✗ 假設數據線性相關
✗ 對特徵尺度敏感（需要標準化）

使用建議：
• 特徵數量很多時考慮使用
• 保留 95%-99% 的方差
• 數據必須先標準化
• 考慮使用 t-SNE 進行可視化（非線性降維）
""")
