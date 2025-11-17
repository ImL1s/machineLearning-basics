"""
K-Means 聚類算法
最常用的聚類算法

原理：將數據分成 K 個簇，使簇內距離最小化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("K-Means 聚類教程".center(80))
print("=" * 80)

# 生成數據
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# K-Means 聚類
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

print(f"聚類中心：\n{kmeans.cluster_centers_}")
print(f"慣性（Inertia）：{kmeans.inertia_:.2f}")
print(f"輪廓係數（Silhouette Score）：{silhouette_score(X, y_pred):.4f}")

# 手肘法找最佳 K 值
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans_temp.labels_))

# 可視化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 聚類結果
axes[0].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6, edgecolors='k')
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', marker='X', s=200, edgecolors='k', linewidths=2, label='Centroids')
axes[0].set_title('K-Means Clustering Results (K=4)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 手肘法
axes[1].plot(K_range, inertias, marker='o', linewidth=2)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Inertia', fontsize=12)
axes[1].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# 輪廓係數
axes[2].plot(K_range, silhouette_scores, marker='s', color='green', linewidth=2)
axes[2].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[2].set_ylabel('Silhouette Score', fontsize=12)
axes[2].set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_UnsupervisedLearning/Clustering/kmeans_results.png', dpi=150)
print("\n已保存結果圖表")

print("\n" + "=" * 80)
print("K-Means 要點：")
print("• 需要事先指定簇的數量 K")
print("• 使用手肘法或輪廓係數選擇最佳 K")
print("• 對初始值敏感，建議多次初始化（n_init）")
print("• 假設簇是凸形的、大小相似的")
print("• 適合大規模數據集")
print("=" * 80)
