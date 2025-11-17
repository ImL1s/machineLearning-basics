"""
層次聚類算法
Hierarchical Clustering

一種基於樹狀結構的聚類算法，通過逐步合併或分裂來形成層次化的簇結構
A tree-based clustering algorithm that builds hierarchical cluster structure through successive merges or splits

兩種方法：
- 凝聚式（Agglomerative）：自底向上，從單個樣本開始逐步合併
- 分裂式（Divisive）：自頂向下，從整體開始逐步分裂
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_iris, make_blobs
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import seaborn as sns
from utils import RANDOM_STATE, setup_chinese_fonts, save_figure, get_output_path

# 設置中文字體和繪圖風格
setup_chinese_fonts()
sns.set_style("whitegrid")

print("=" * 80)
print("層次聚類教程".center(80))
print("=" * 80)

# ============================================================================
# Part 1: 層次聚類原理介紹
# ============================================================================
print("\n【Part 1】層次聚類基礎原理")
print("-" * 80)

print("""
層次聚類（Hierarchical Clustering）

一、聚類方法：
┌─────────────────────────────────────────────────────────┐
│ 1. 凝聚式（Agglomerative）- 自底向上                   │
│    • 初始：每個樣本為一個簇                             │
│    • 過程：逐步合併最相似的簇                           │
│    • 結束：所有樣本合併為一個大簇                       │
│                                                         │
│ 2. 分裂式（Divisive）- 自頂向下                        │
│    • 初始：所有樣本為一個簇                             │
│    • 過程：逐步分裂最異質的簇                           │
│    • 結束：每個樣本為單獨的簇                           │
└─────────────────────────────────────────────────────────┘

二、鏈接方法（Linkage Methods）：
┌──────────────┬────────────────────────────────────┬────────────┐
│ 方法         │ 簇間距離定義                       │ 特點       │
├──────────────┼────────────────────────────────────┼────────────┤
│ Single       │ 兩簇最近點的距離                   │ 易形成鏈狀 │
│ Complete     │ 兩簇最遠點的距離                   │ 偏好緊湊簇 │
│ Average      │ 兩簇所有點對的平均距離             │ 折中方案   │
│ Ward         │ 合併後方差增加量                   │ 最常用     │
│ Centroid     │ 兩簇中心點的距離                   │ 可能倒置   │
│ Median       │ 兩簇中位數的距離                   │ 可能倒置   │
└──────────────┴────────────────────────────────────┴────────────┘

三、優點：
• 不需要預先指定簇的數量
• 提供完整的層次結構信息（樹狀圖）
• 可以在任意層次切割得到不同數量的簇
• 對於小數據集效果很好

四、缺點：
• 時間複雜度高：O(n² log n) 或 O(n³)
• 空間複雜度高：需要存儲距離矩陣
• 一旦合併/分裂無法撤銷
• 對噪聲和異常值敏感

五、應用場景：
• 生物學中的系統分類
• 文檔聚類和主題分析
• 社交網絡分析
• 圖像分割
""")

# ============================================================================
# Part 2: 加載和準備數據
# ============================================================================
print("\n【Part 2】數據準備")
print("-" * 80)

# 使用 Iris 數據集
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"✓ 加載 Iris 數據集：{X_iris.shape[0]} 樣本, {X_iris.shape[1]} 特徵")
print(f"  特徵名稱：{feature_names}")
print(f"  類別：{target_names}")

# 標準化數據
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# 生成人工數據集用於某些示例
X_synthetic, y_synthetic = make_blobs(n_samples=150, centers=3, n_features=2,
                                      cluster_std=0.7, random_state=RANDOM_STATE)
X_synthetic_scaled = StandardScaler().fit_transform(X_synthetic)

print(f"✓ 生成人工數據集：{X_synthetic.shape[0]} 樣本, {X_synthetic.shape[1]} 特徵")

# ============================================================================
# Part 3: 樹狀圖（Dendrogram）基礎
# ============================================================================
print("\n【Part 3】樹狀圖（Dendrogram）介紹")
print("-" * 80)

# 使用人工數據集繪製基礎樹狀圖
linkage_matrix = linkage(X_synthetic_scaled, method='ward')

fig1, axes = plt.subplots(2, 2, figsize=(16, 14))

# 完整樹狀圖
dendrogram(linkage_matrix, ax=axes[0, 0], color_threshold=0)
axes[0, 0].set_title('完整樹狀圖（Ward 鏈接）\nComplete Dendrogram',
                     fontsize=13, fontweight='bold', pad=10)
axes[0, 0].set_xlabel('樣本索引 (Sample Index)', fontsize=11)
axes[0, 0].set_ylabel('距離 (Distance)', fontsize=11)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 截斷樹狀圖（只顯示最後 30 次合併）
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, ax=axes[0, 1],
          color_threshold=0, show_leaf_counts=True)
axes[0, 1].set_title('截斷樹狀圖（最後 30 次合併）\nTruncated Dendrogram',
                     fontsize=13, fontweight='bold', pad=10)
axes[0, 1].set_xlabel('簇大小 (Cluster Size)', fontsize=11)
axes[0, 1].set_ylabel('距離 (Distance)', fontsize=11)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 帶顏色閾值的樹狀圖
threshold = 7
dendrogram(linkage_matrix, ax=axes[1, 0], color_threshold=threshold)
axes[1, 0].axhline(y=threshold, c='red', linestyle='--', linewidth=2, label=f'閾值 = {threshold}')
axes[1, 0].set_title(f'帶顏色閾值的樹狀圖\nDendrogram with Color Threshold',
                     fontsize=13, fontweight='bold', pad=10)
axes[1, 0].set_xlabel('樣本索引 (Sample Index)', fontsize=11)
axes[1, 0].set_ylabel('距離 (Distance)', fontsize=11)
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 原始數據分布
scatter = axes[1, 1].scatter(X_synthetic[:, 0], X_synthetic[:, 1],
                            c=y_synthetic, cmap='viridis',
                            s=80, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[1, 1].set_title('原始數據分布\nOriginal Data Distribution',
                    fontsize=13, fontweight='bold', pad=10)
axes[1, 1].set_xlabel('Feature 1', fontsize=11)
axes[1, 1].set_ylabel('Feature 2', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='True Label')

plt.tight_layout()
save_figure(fig1, get_output_path('hierarchical_01_dendrogram_basics.png',
                                 '03_UnsupervisedLearning/Clustering'))
plt.close()

print("✓ 樹狀圖解讀：")
print("  • Y軸：簇間距離（高度越高，簇間差異越大）")
print("  • X軸：樣本或簇的索引")
print("  • 水平線：表示簇的合併")
print("  • 顏色：不同顏色表示在指定閾值下的不同簇")

# ============================================================================
# Part 4: 不同鏈接方法對比
# ============================================================================
print("\n【Part 4】不同鏈接方法對比")
print("-" * 80)

linkage_methods = ['single', 'complete', 'average', 'ward']
method_names = {
    'single': 'Single Linkage\n（最近鄰）',
    'complete': 'Complete Linkage\n（最遠鄰）',
    'average': 'Average Linkage\n（平均）',
    'ward': 'Ward Linkage\n（最小方差）'
}

fig2, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, method in enumerate(linkage_methods):
    # 計算鏈接矩陣
    Z = linkage(X_synthetic_scaled, method=method)

    # 繪製樹狀圖
    dendrogram(Z, ax=axes[0, idx], no_labels=True, color_threshold=0)
    axes[0, idx].set_title(f'{method_names[method]}',
                          fontsize=12, fontweight='bold', pad=10)
    axes[0, idx].set_ylabel('距離 (Distance)', fontsize=10)
    axes[0, idx].grid(True, alpha=0.3, axis='y')

    # 使用 AgglomerativeClustering 得到聚類結果
    n_clusters = 3
    agg_clust = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    y_pred = agg_clust.fit_predict(X_synthetic_scaled)

    # 繪製聚類結果
    scatter = axes[1, idx].scatter(X_synthetic[:, 0], X_synthetic[:, 1],
                                  c=y_pred, cmap='viridis',
                                  s=60, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[1, idx].set_title(f'聚類結果 (K={n_clusters})',
                          fontsize=12, fontweight='bold', pad=10)
    axes[1, idx].set_xlabel('Feature 1', fontsize=10)
    axes[1, idx].set_ylabel('Feature 2', fontsize=10)
    axes[1, idx].grid(True, alpha=0.3)

    # 計算評估指標
    silhouette = silhouette_score(X_synthetic_scaled, y_pred)
    db_score = davies_bouldin_score(X_synthetic_scaled, y_pred)

    print(f"\n{method.upper()} Linkage:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Davies-Bouldin Score: {db_score:.4f}")

plt.tight_layout()
save_figure(fig2, get_output_path('hierarchical_02_linkage_comparison.png',
                                 '03_UnsupervisedLearning/Clustering'))
plt.close()

# ============================================================================
# Part 5: Iris 數據集的層次聚類分析
# ============================================================================
print("\n【Part 5】Iris 數據集層次聚類分析")
print("-" * 80)

# 只使用前兩個特徵以便可視化
X_iris_2d = X_iris[:, :2]
X_iris_2d_scaled = StandardScaler().fit_transform(X_iris_2d)

# 計算鏈接矩陣（使用 Ward 方法）
Z_iris = linkage(X_iris_2d_scaled, method='ward')

fig3, axes = plt.subplots(2, 2, figsize=(16, 14))

# 完整樹狀圖
dendrogram(Z_iris, ax=axes[0, 0], labels=y_iris, no_labels=True)
axes[0, 0].set_title('Iris 數據集樹狀圖（Ward）\nIris Dataset Dendrogram',
                    fontsize=13, fontweight='bold', pad=10)
axes[0, 0].set_xlabel('樣本索引 (Sample Index)', fontsize=11)
axes[0, 0].set_ylabel('距離 (Distance)', fontsize=11)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 帶標籤的樹狀圖（只顯示部分樣本）
sample_indices = np.random.choice(len(X_iris_2d), 30, replace=False)
sample_indices.sort()
X_sample = X_iris_2d_scaled[sample_indices]
y_sample = y_iris[sample_indices]
Z_sample = linkage(X_sample, method='ward')

labels = [f"{idx}({target_names[y_sample[i]]})" for i, idx in enumerate(sample_indices)]
dendrogram(Z_sample, ax=axes[0, 1], labels=labels, leaf_rotation=90, leaf_font_size=8)
axes[0, 1].set_title('樣本子集樹狀圖（帶標籤）\nSample Subset with Labels',
                    fontsize=13, fontweight='bold', pad=10)
axes[0, 1].set_ylabel('距離 (Distance)', fontsize=11)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 真實標籤分布
axes[1, 0].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=y_iris,
                  cmap='viridis', s=60, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[1, 0].set_title('真實標籤分布\nTrue Labels',
                    fontsize=13, fontweight='bold', pad=10)
axes[1, 0].set_xlabel(feature_names[0], fontsize=11)
axes[1, 0].set_ylabel(feature_names[1], fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# 使用不同簇數的聚類結果
for i, k in enumerate([2, 3, 4]):
    clusters = fcluster(Z_iris, k, criterion='maxclust')
    if i == 0:
        ax = axes[1, 1]
        ax.scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=clusters,
                  cmap='viridis', s=60, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.set_title(f'層次聚類結果 (K={k})\nHierarchical Clustering',
                    fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel(feature_names[0], fontsize=11)
        ax.set_ylabel(feature_names[1], fontsize=11)
        ax.grid(True, alpha=0.3)

        # 計算評估指標
        silhouette = silhouette_score(X_iris_2d_scaled, clusters)
        ari = adjusted_rand_score(y_iris, clusters)
        print(f"\nK={k}: Silhouette={silhouette:.4f}, ARI={ari:.4f}")

plt.tight_layout()
save_figure(fig3, get_output_path('hierarchical_03_iris_analysis.png',
                                 '03_UnsupervisedLearning/Clustering'))
plt.close()

# ============================================================================
# Part 6: 確定最佳簇數
# ============================================================================
print("\n【Part 6】確定最佳簇數")
print("-" * 80)

# 使用多種方法評估不同簇數
k_range = range(2, 11)
silhouette_scores = []
davies_bouldin_scores = []
inertias = []

print("\n評估不同簇數的性能：")
for k in k_range:
    # 層次聚類
    agg_clust = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg_clust.fit_predict(X_iris_scaled)

    # 計算評估指標
    silhouette = silhouette_score(X_iris_scaled, labels)
    db_score = davies_bouldin_score(X_iris_scaled, labels)
    silhouette_scores.append(silhouette)
    davies_bouldin_scores.append(db_score)

    # K-Means 的 inertia（用於對比）
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_iris_scaled)
    inertias.append(kmeans.inertia_)

    print(f"  K={k}: Silhouette={silhouette:.4f}, Davies-Bouldin={db_score:.4f}")

fig4, axes = plt.subplots(2, 2, figsize=(16, 12))

# Silhouette Score
axes[0, 0].plot(k_range, silhouette_scores, marker='o', linewidth=2,
               markersize=8, color='blue', label='Hierarchical')
axes[0, 0].set_xlabel('簇數量 (Number of Clusters)', fontsize=11)
axes[0, 0].set_ylabel('Silhouette Score', fontsize=11)
axes[0, 0].set_title('Silhouette Score vs 簇數量\n(越高越好)',
                    fontsize=13, fontweight='bold', pad=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
axes[0, 0].axvline(x=best_k_silhouette, color='red', linestyle='--',
                  linewidth=1.5, label=f'Best K={best_k_silhouette}')

# Davies-Bouldin Score
axes[0, 1].plot(k_range, davies_bouldin_scores, marker='s', linewidth=2,
               markersize=8, color='green', label='Hierarchical')
axes[0, 1].set_xlabel('簇數量 (Number of Clusters)', fontsize=11)
axes[0, 1].set_ylabel('Davies-Bouldin Score', fontsize=11)
axes[0, 1].set_title('Davies-Bouldin Score vs 簇數量\n(越低越好)',
                    fontsize=13, fontweight='bold', pad=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=10)
best_k_db = k_range[np.argmin(davies_bouldin_scores)]
axes[0, 1].axvline(x=best_k_db, color='red', linestyle='--',
                  linewidth=1.5, label=f'Best K={best_k_db}')

# Elbow Method (using K-Means inertia)
axes[1, 0].plot(k_range, inertias, marker='D', linewidth=2,
               markersize=8, color='orange', label='K-Means Inertia')
axes[1, 0].set_xlabel('簇數量 (Number of Clusters)', fontsize=11)
axes[1, 0].set_ylabel('Inertia', fontsize=11)
axes[1, 0].set_title('肘部法則（Elbow Method）\nK-Means Inertia',
                    fontsize=13, fontweight='bold', pad=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)

# 樹狀圖（標記建議的切割點）
dendrogram(Z_iris, ax=axes[1, 1], no_labels=True)
# 根據最佳簇數計算切割高度
if best_k_silhouette < len(Z_iris):
    cut_height = (Z_iris[-best_k_silhouette, 2] + Z_iris[-best_k_silhouette + 1, 2]) / 2
    axes[1, 1].axhline(y=cut_height, color='red', linestyle='--',
                      linewidth=2, label=f'建議切割（K={best_k_silhouette}）')
axes[1, 1].set_title('樹狀圖（帶建議切割點）\nDendrogram with Suggested Cut',
                    fontsize=13, fontweight='bold', pad=10)
axes[1, 1].set_xlabel('樣本索引 (Sample Index)', fontsize=11)
axes[1, 1].set_ylabel('距離 (Distance)', fontsize=11)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_figure(fig4, get_output_path('hierarchical_04_optimal_k.png',
                                 '03_UnsupervisedLearning/Clustering'))
plt.close()

print(f"\n最佳簇數建議：")
print(f"  基於 Silhouette Score: K = {best_k_silhouette}")
print(f"  基於 Davies-Bouldin Score: K = {best_k_db}")
print(f"  真實類別數: K = {len(np.unique(y_iris))}")

# ============================================================================
# Part 7: 層次聚類 vs K-Means 性能對比
# ============================================================================
print("\n【Part 7】層次聚類 vs K-Means 性能對比")
print("-" * 80)

n_clusters = 3

# 層次聚類
agg_clust = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
y_hierarchical = agg_clust.fit_predict(X_iris_2d_scaled)

# K-Means 聚類
kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
y_kmeans = kmeans.fit_predict(X_iris_2d_scaled)

fig5, axes = plt.subplots(2, 3, figsize=(18, 12))

# 真實標籤
axes[0, 0].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=y_iris,
                  cmap='viridis', s=80, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[0, 0].set_title('真實標籤\nTrue Labels', fontsize=13, fontweight='bold', pad=10)
axes[0, 0].set_xlabel(feature_names[0], fontsize=11)
axes[0, 0].set_ylabel(feature_names[1], fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# 層次聚類結果
axes[0, 1].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=y_hierarchical,
                  cmap='viridis', s=80, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[0, 1].set_title('層次聚類（Ward）\nHierarchical Clustering',
                    fontsize=13, fontweight='bold', pad=10)
axes[0, 1].set_xlabel(feature_names[0], fontsize=11)
axes[0, 1].set_ylabel(feature_names[1], fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# K-Means 結果
axes[0, 2].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=y_kmeans,
                  cmap='viridis', s=80, alpha=0.7, edgecolors='k', linewidth=0.5)
centers = StandardScaler().fit(X_iris_2d).inverse_transform(kmeans.cluster_centers_)
axes[0, 2].scatter(centers[:, 0], centers[:, 1], c='red', marker='X',
                  s=300, edgecolors='k', linewidth=2, label='Centroids', zorder=5)
axes[0, 2].set_title('K-Means 聚類\nK-Means Clustering',
                    fontsize=13, fontweight='bold', pad=10)
axes[0, 2].set_xlabel(feature_names[0], fontsize=11)
axes[0, 2].set_ylabel(feature_names[1], fontsize=11)
axes[0, 2].legend(fontsize=10)
axes[0, 2].grid(True, alpha=0.3)

# 使用全部特徵進行更準確的評估
y_hierarchical_full = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X_iris_scaled)
y_kmeans_full = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10).fit_predict(X_iris_scaled)

# 評估指標對比（全特徵）
metrics = {
    '層次聚類': {
        'Silhouette': silhouette_score(X_iris_scaled, y_hierarchical_full),
        'Davies-Bouldin': davies_bouldin_score(X_iris_scaled, y_hierarchical_full),
        'ARI': adjusted_rand_score(y_iris, y_hierarchical_full)
    },
    'K-Means': {
        'Silhouette': silhouette_score(X_iris_scaled, y_kmeans_full),
        'Davies-Bouldin': davies_bouldin_score(X_iris_scaled, y_kmeans_full),
        'ARI': adjusted_rand_score(y_iris, y_kmeans_full)
    }
}

# 繪製評估指標對比
metric_names = ['Silhouette', 'Davies-Bouldin', 'ARI']
x_pos = np.arange(len(metric_names))
hierarchical_values = [metrics['層次聚類'][m] for m in metric_names]
kmeans_values = [metrics['K-Means'][m] for m in metric_names]

width = 0.35
axes[1, 0].bar(x_pos - width/2, hierarchical_values, width, label='層次聚類',
              color='steelblue', alpha=0.8, edgecolor='k')
axes[1, 0].bar(x_pos + width/2, kmeans_values, width, label='K-Means',
              color='coral', alpha=0.8, edgecolor='k')
axes[1, 0].set_xlabel('評估指標 (Metrics)', fontsize=11)
axes[1, 0].set_ylabel('分數 (Score)', fontsize=11)
axes[1, 0].set_title('評估指標對比\nMetrics Comparison',
                    fontsize=13, fontweight='bold', pad=10)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(metric_names)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 混淆矩陣 - 層次聚類 vs 真實標籤
from sklearn.metrics import confusion_matrix
cm_hierarchical = confusion_matrix(y_iris, y_hierarchical_full)
sns.heatmap(cm_hierarchical, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
           xticklabels=target_names, yticklabels=target_names, cbar_kws={'label': 'Count'})
axes[1, 1].set_title('層次聚類混淆矩陣\nHierarchical Confusion Matrix',
                    fontsize=13, fontweight='bold', pad=10)
axes[1, 1].set_xlabel('預測簇 (Predicted Cluster)', fontsize=11)
axes[1, 1].set_ylabel('真實類別 (True Label)', fontsize=11)

# 混淆矩陣 - K-Means vs 真實標籤
cm_kmeans = confusion_matrix(y_iris, y_kmeans_full)
sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 2],
           xticklabels=target_names, yticklabels=target_names, cbar_kws={'label': 'Count'})
axes[1, 2].set_title('K-Means 混淆矩陣\nK-Means Confusion Matrix',
                    fontsize=13, fontweight='bold', pad=10)
axes[1, 2].set_xlabel('預測簇 (Predicted Cluster)', fontsize=11)
axes[1, 2].set_ylabel('真實類別 (True Label)', fontsize=11)

plt.tight_layout()
save_figure(fig5, get_output_path('hierarchical_05_vs_kmeans.png',
                                 '03_UnsupervisedLearning/Clustering'))
plt.close()

print("\n性能對比（使用全部 4 個特徵）：")
print(f"{'算法':<12} {'Silhouette':<15} {'Davies-Bouldin':<18} {'ARI':<10}")
print("-" * 60)
for algo, vals in metrics.items():
    print(f"{algo:<12} {vals['Silhouette']:>10.4f}     {vals['Davies-Bouldin']:>10.4f}     {vals['ARI']:>10.4f}")

# ============================================================================
# Part 8: 距離矩陣熱力圖
# ============================================================================
print("\n【Part 8】距離矩陣分析")
print("-" * 80)

# 計算樣本間的距離矩陣（使用子集以便可視化）
sample_size = 50
sample_indices = np.random.RandomState(RANDOM_STATE).choice(len(X_iris_scaled), sample_size, replace=False)
sample_indices.sort()

X_sample = X_iris_scaled[sample_indices]
y_sample = y_iris[sample_indices]

# 計算距離矩陣
distances = squareform(pdist(X_sample, metric='euclidean'))

# 根據真實標籤排序
sorted_indices = np.argsort(y_sample)
distances_sorted = distances[sorted_indices][:, sorted_indices]
y_sorted = y_sample[sorted_indices]

fig6, axes = plt.subplots(1, 2, figsize=(16, 7))

# 原始順序的距離矩陣
im1 = axes[0].imshow(distances, cmap='YlOrRd', aspect='auto')
axes[0].set_title('距離矩陣（原始順序）\nDistance Matrix (Original Order)',
                 fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('樣本索引 (Sample Index)', fontsize=11)
axes[0].set_ylabel('樣本索引 (Sample Index)', fontsize=11)
plt.colorbar(im1, ax=axes[0], label='歐氏距離 (Euclidean Distance)')

# 按真實標籤排序的距離矩陣
im2 = axes[1].imshow(distances_sorted, cmap='YlOrRd', aspect='auto')
axes[1].set_title('距離矩陣（按類別排序）\nDistance Matrix (Sorted by Class)',
                 fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('樣本索引 (Sample Index)', fontsize=11)
axes[1].set_ylabel('樣本索引 (Sample Index)', fontsize=11)

# 添加類別分隔線
class_boundaries = []
current_class = y_sorted[0]
for i, cls in enumerate(y_sorted):
    if cls != current_class:
        class_boundaries.append(i)
        current_class = cls

for boundary in class_boundaries:
    axes[1].axhline(y=boundary, color='blue', linewidth=2, linestyle='--')
    axes[1].axvline(x=boundary, color='blue', linewidth=2, linestyle='--')

plt.colorbar(im2, ax=axes[1], label='歐氏距離 (Euclidean Distance)')

plt.tight_layout()
save_figure(fig6, get_output_path('hierarchical_06_distance_matrix.png',
                                 '03_UnsupervisedLearning/Clustering'))
plt.close()

print("✓ 距離矩陣分析：")
print("  • 深色區域：樣本間距離小（相似度高）")
print("  • 淺色區域：樣本間距離大（相似度低）")
print("  • 排序後可以看到明顯的塊狀結構，對應不同類別")

# ============================================================================
# Part 9: 不同鏈接方法的詳細對比（使用 Iris）
# ============================================================================
print("\n【Part 9】Iris 數據集上的鏈接方法對比")
print("-" * 80)

fig7, axes = plt.subplots(2, 4, figsize=(20, 10))

linkage_methods = ['single', 'complete', 'average', 'ward']
method_descriptions = {
    'single': 'Single\n（最近鄰）',
    'complete': 'Complete\n（最遠鄰）',
    'average': 'Average\n（平均）',
    'ward': 'Ward\n（最小方差）'
}

for idx, method in enumerate(linkage_methods):
    # 計算鏈接矩陣
    Z = linkage(X_iris_scaled, method=method)

    # 繪製樹狀圖
    dendrogram(Z, ax=axes[0, idx], no_labels=True, color_threshold=0)
    axes[0, idx].set_title(f'{method_descriptions[method]}\n樹狀圖',
                          fontsize=12, fontweight='bold', pad=10)
    axes[0, idx].set_ylabel('距離 (Distance)', fontsize=10)
    axes[0, idx].grid(True, alpha=0.3, axis='y')

    # 聚類結果（使用前兩個特徵可視化）
    agg_clust = AgglomerativeClustering(n_clusters=3, linkage=method)
    y_pred = agg_clust.fit_predict(X_iris_scaled)

    axes[1, idx].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=y_pred,
                        cmap='viridis', s=60, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[1, idx].set_title(f'聚類結果 (K=3)', fontsize=12, fontweight='bold', pad=10)
    axes[1, idx].set_xlabel(feature_names[0], fontsize=10)
    axes[1, idx].set_ylabel(feature_names[1], fontsize=10)
    axes[1, idx].grid(True, alpha=0.3)

    # 計算評估指標（使用全部特徵）
    silhouette = silhouette_score(X_iris_scaled, y_pred)
    db_score = davies_bouldin_score(X_iris_scaled, y_pred)
    ari = adjusted_rand_score(y_iris, y_pred)

    print(f"\n{method.upper()}:")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  Davies-Bouldin: {db_score:.4f}")
    print(f"  ARI (vs true labels): {ari:.4f}")

plt.tight_layout()
save_figure(fig7, get_output_path('hierarchical_07_linkage_iris.png',
                                 '03_UnsupervisedLearning/Clustering'))
plt.close()

# ============================================================================
# Part 10: Silhouette 分析
# ============================================================================
print("\n【Part 10】Silhouette 詳細分析")
print("-" * 80)

from sklearn.metrics import silhouette_samples

# 使用 Ward 鏈接，K=3
agg_clust = AgglomerativeClustering(n_clusters=3, linkage='ward')
cluster_labels = agg_clust.fit_predict(X_iris_scaled)

# 計算每個樣本的 silhouette 值
silhouette_vals = silhouette_samples(X_iris_scaled, cluster_labels)

fig8, axes = plt.subplots(1, 2, figsize=(16, 6))

# Silhouette 圖
y_lower = 10
colors = plt.cm.Spectral(np.linspace(0, 1, 3))

for i in range(3):
    cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    axes[0].fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

    axes[0].text(-0.05, y_lower + 0.5 * size_cluster_i, f'簇 {i}',
                fontsize=11, fontweight='bold')

    y_lower = y_upper + 10

avg_silhouette = np.mean(silhouette_vals)
axes[0].axvline(x=avg_silhouette, color='red', linestyle='--', linewidth=2,
               label=f'平均值 = {avg_silhouette:.3f}')
axes[0].set_xlabel('Silhouette Coefficient', fontsize=11)
axes[0].set_ylabel('簇 (Cluster)', fontsize=11)
axes[0].set_title('Silhouette 分析圖\nSilhouette Plot',
                 fontsize=13, fontweight='bold', pad=10)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='x')

# 聚類結果（著色顯示 silhouette 值）
scatter = axes[1].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1],
                         c=silhouette_vals, cmap='RdYlGn',
                         s=80, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[1].set_title('樣本 Silhouette 值分布\nSilhouette Values Distribution',
                 fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel(feature_names[0], fontsize=11)
axes[1].set_ylabel(feature_names[1], fontsize=11)
axes[1].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[1])
cbar.set_label('Silhouette Coefficient', fontsize=10)

plt.tight_layout()
save_figure(fig8, get_output_path('hierarchical_08_silhouette_analysis.png',
                                 '03_UnsupervisedLearning/Clustering'))
plt.close()

print(f"\nSilhouette 分析結果：")
print(f"  平均 Silhouette 係數: {avg_silhouette:.4f}")
print(f"  最小值: {silhouette_vals.min():.4f}")
print(f"  最大值: {silhouette_vals.max():.4f}")

for i in range(3):
    cluster_silhouette = silhouette_vals[cluster_labels == i]
    print(f"  簇 {i}: 平均 = {cluster_silhouette.mean():.4f}, "
          f"樣本數 = {len(cluster_silhouette)}")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("層次聚類關鍵要點總結".center(80))
print("=" * 80)
print("""
1. 鏈接方法選擇建議：
   • Ward: 最常用，適合大多數情況，偏好形成大小相近的簇
   • Complete: 偏好緊湊的球形簇，對異常值較敏感
   • Average: 折中方案，較為穩健
   • Single: 容易形成鏈狀簇，適合細長形狀的數據

2. 確定最佳簇數：
   • 觀察樹狀圖的垂直線高度差
   • 使用 Silhouette Score（越高越好）
   • 使用 Davies-Bouldin Index（越低越好）
   • 結合領域知識

3. 優勢場景：
   • 不需要預先指定簇數
   • 想要了解數據的層次結構
   • 數據量不太大（< 10000 樣本）
   • 需要可解釋的結果（樹狀圖）

4. 劣勢場景：
   • 大規模數據集（計算複雜度高）
   • 高維數據（距離度量失效）
   • 需要在線學習或增量更新
   • 簇的密度差異很大

5. 與其他算法對比：
   • vs K-Means:
     - 層次聚類不需預設K值，但計算更慢
     - K-Means 適合大數據，層次聚類適合小數據
   • vs DBSCAN:
     - 層次聚類提供完整層次結構
     - DBSCAN 更適合發現任意形狀的簇和噪聲

6. 實踐技巧：
   • 始終標準化數據（StandardScaler）
   • 對於大數據集，考慮先抽樣
   • 使用多種評估指標綜合判斷
   • 可視化樹狀圖輔助決策
   • Ward 鏈接通常是最佳起點

7. Python 實現要點：
   • scipy.cluster.hierarchy: 繪製樹狀圖
   • sklearn.cluster.AgglomerativeClustering: 執行聚類
   • 使用 linkage() 計算層次結構
   • 使用 fcluster() 在指定層次切割

8. 常見陷阱：
   • 未標準化數據導致某些特徵主導
   • 選錯鏈接方法導致結果不理想
   • 忽略計算複雜度處理大數據
   • 過度依賴單一評估指標
""")
print("=" * 80)
print("✓ 層次聚類教程完成！已生成 8 張可視化圖表。")
print("=" * 80)
