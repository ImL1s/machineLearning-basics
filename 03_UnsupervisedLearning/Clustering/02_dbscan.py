"""
DBSCAN 密度聚類算法
Density-Based Spatial Clustering of Applications with Noise

一種基於密度的聚類算法，能夠發現任意形狀的簇並識別噪聲點
A density-based clustering algorithm that can discover clusters of arbitrary shape and identify noise points

核心概念：
- 核心點（Core Point）：eps 半徑內至少有 min_samples 個點
- 邊界點（Border Point）：不是核心點但在核心點的 eps 半徑內
- 噪聲點（Noise Point）：既不是核心點也不是邊界點
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from utils import RANDOM_STATE, setup_chinese_fonts, save_figure, get_output_path

# 設置中文字體
setup_chinese_fonts()

print("=" * 80)
print("DBSCAN 密度聚類教程".center(80))
print("=" * 80)

# ============================================================================
# Part 1: DBSCAN 原理介紹與基礎應用
# ============================================================================
print("\n【Part 1】DBSCAN 基礎原理")
print("-" * 80)

print("""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

核心參數：
1. eps (epsilon): 鄰域半徑，定義"鄰近"的距離
2. min_samples: 成為核心點所需的最小鄰居數量

優點：
• 不需要預先指定簇的數量
• 可以發現任意形狀的簇（非凸形狀）
• 能夠識別並標記噪聲點
• 對異常值魯棒

缺點：
• 對參數 eps 和 min_samples 敏感
• 難以處理密度差異大的數據集
• 高維數據效果較差（維度詛咒）

與 K-Means 的對比：
┌─────────────┬──────────────────┬──────────────────┐
│   特性      │    K-Means       │     DBSCAN       │
├─────────────┼──────────────────┼──────────────────┤
│ 簇的形狀    │ 凸形（球形）     │ 任意形狀         │
│ 簇的數量    │ 需要預先指定     │ 自動確定         │
│ 噪聲處理    │ 無法識別         │ 可以識別         │
│ 參數設置    │ 較簡單（K值）    │ 較複雜（eps, ms）│
│ 計算複雜度  │ O(n*k*i)         │ O(n log n)       │
└─────────────┴──────────────────┴──────────────────┘
""")

# ============================================================================
# Part 2: 生成多種形狀的測試數據
# ============================================================================
print("\n【Part 2】生成測試數據集")
print("-" * 80)

# 生成四種不同類型的數據集
datasets = []

# 1. 月牙形數據（非凸形）
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=RANDOM_STATE)
datasets.append(('Moons (月牙形)', X_moons, y_moons))

# 2. 同心圓數據（非凸形）
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=RANDOM_STATE)
datasets.append(('Circles (同心圓)', X_circles, y_circles))

# 3. 各向異性的 Blobs
random_state = np.random.RandomState(RANDOM_STATE)
X_aniso = np.dot(make_blobs(n_samples=300, random_state=RANDOM_STATE)[0],
                 [[0.6, -0.6], [-0.4, 0.8]])
y_aniso = make_blobs(n_samples=300, random_state=RANDOM_STATE)[1]
datasets.append(('Anisotropic Blobs (各向異性)', X_aniso, y_aniso))

# 4. 不同密度的簇
X_varied, y_varied = make_blobs(n_samples=[100, 200, 100],
                                cluster_std=[1.0, 2.5, 0.5],
                                centers=[[-5, -5], [0, 0], [5, 5]],
                                random_state=RANDOM_STATE)
datasets.append(('Varied Density (不同密度)', X_varied, y_varied))

print(f"✓ 生成了 {len(datasets)} 個測試數據集")
for name, X, y in datasets:
    print(f"  • {name}: {X.shape[0]} 樣本, {len(np.unique(y))} 個真實簇")

# ============================================================================
# Part 3: 可視化原始數據分布
# ============================================================================
print("\n【Part 3】原始數據分布可視化")
print("-" * 80)

fig1, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, (name, X, y) in enumerate(datasets):
    axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                     alpha=0.7, edgecolors='k', linewidth=0.5, s=50)
    axes[idx].set_title(f'{name}\n真實標籤分布', fontsize=13, fontweight='bold', pad=10)
    axes[idx].set_xlabel('Feature 1', fontsize=11)
    axes[idx].set_ylabel('Feature 2', fontsize=11)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig1, get_output_path('dbscan_01_original_data.png', '03_UnsupervisedLearning/Clustering'))
plt.close()

# ============================================================================
# Part 4: DBSCAN vs K-Means 對比實驗
# ============================================================================
print("\n【Part 4】DBSCAN vs K-Means 對比")
print("-" * 80)

fig2, axes = plt.subplots(4, 3, figsize=(16, 18))

for idx, (name, X, y) in enumerate(datasets):
    # 標準化數據
    X_scaled = StandardScaler().fit_transform(X)

    # 真實標籤
    axes[idx, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                        alpha=0.7, edgecolors='k', linewidth=0.5, s=40)
    axes[idx, 0].set_title(f'{name}\n真實標籤', fontsize=11, fontweight='bold')
    axes[idx, 0].grid(True, alpha=0.3)

    # DBSCAN 聚類
    if idx == 0:  # Moons
        dbscan = DBSCAN(eps=0.3, min_samples=5)
    elif idx == 1:  # Circles
        dbscan = DBSCAN(eps=0.3, min_samples=5)
    elif idx == 2:  # Anisotropic
        dbscan = DBSCAN(eps=0.5, min_samples=5)
    else:  # Varied
        dbscan = DBSCAN(eps=0.5, min_samples=5)

    y_dbscan = dbscan.fit_predict(X_scaled)

    # 計算噪聲點數量
    n_noise = np.sum(y_dbscan == -1)
    n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)

    # 繪製 DBSCAN 結果
    unique_labels = set(y_dbscan)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # 噪聲點用黑色

        class_member_mask = (y_dbscan == k)
        xy = X[class_member_mask]
        axes[idx, 1].scatter(xy[:, 0], xy[:, 1], c=[col],
                           alpha=0.7, edgecolors='k', linewidth=0.5, s=40,
                           marker='x' if k == -1 else 'o')

    axes[idx, 1].set_title(f'DBSCAN\n簇數: {n_clusters}, 噪聲點: {n_noise}',
                          fontsize=11, fontweight='bold')
    axes[idx, 1].grid(True, alpha=0.3)

    # K-Means 聚類（使用真實簇數）
    n_true_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_true_clusters, random_state=RANDOM_STATE, n_init=10)
    y_kmeans = kmeans.fit_predict(X_scaled)

    axes[idx, 2].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis',
                        alpha=0.7, edgecolors='k', linewidth=0.5, s=40)

    # 繪製 K-Means 中心點
    centers_original = StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_)
    axes[idx, 2].scatter(centers_original[:, 0], centers_original[:, 1],
                        c='red', marker='X', s=200, edgecolors='k', linewidth=2,
                        label='Centroids')
    axes[idx, 2].set_title(f'K-Means\nK = {n_true_clusters}',
                          fontsize=11, fontweight='bold')
    axes[idx, 2].grid(True, alpha=0.3)
    axes[idx, 2].legend(loc='upper right', fontsize=9)

    # 計算評估指標（忽略 DBSCAN 的噪聲點）
    if n_clusters > 0 and n_noise < len(y_dbscan):
        mask = y_dbscan != -1
        if len(np.unique(y_dbscan[mask])) > 1:
            dbscan_silhouette = silhouette_score(X_scaled[mask], y_dbscan[mask])
        else:
            dbscan_silhouette = -1
    else:
        dbscan_silhouette = -1

    kmeans_silhouette = silhouette_score(X_scaled, y_kmeans)

    print(f"\n{name}:")
    print(f"  DBSCAN: {n_clusters} 簇, {n_noise} 噪聲點, Silhouette = {dbscan_silhouette:.4f}")
    print(f"  K-Means: {n_true_clusters} 簇, Silhouette = {kmeans_silhouette:.4f}")

plt.tight_layout()
save_figure(fig2, get_output_path('dbscan_02_comparison.png', '03_UnsupervisedLearning/Clustering'))
plt.close()

# ============================================================================
# Part 5: eps 參數影響分析
# ============================================================================
print("\n【Part 5】eps 參數影響分析")
print("-" * 80)

# 使用月牙形數據進行詳細分析
X_test, y_test = make_moons(n_samples=300, noise=0.05, random_state=RANDOM_STATE)
X_test_scaled = StandardScaler().fit_transform(X_test)

eps_values = [0.15, 0.25, 0.35, 0.45]
min_samples = 5

fig3, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

print("\neps 參數實驗（min_samples=5）:")
for idx, eps in enumerate(eps_values):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = dbscan.fit_predict(X_test_scaled)

    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = np.sum(y_pred == -1)

    # 計算評估指標
    if n_clusters > 0 and n_noise < len(y_pred):
        mask = y_pred != -1
        if len(np.unique(y_pred[mask])) > 1:
            silhouette = silhouette_score(X_test_scaled[mask], y_pred[mask])
        else:
            silhouette = -1
    else:
        silhouette = -1

    # 繪製結果
    unique_labels = set(y_pred)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (y_pred == k)
        xy = X_test[class_member_mask]

        # 區分核心點和邊界點
        core_samples_mask = np.zeros_like(y_pred, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True

        # 核心點
        xy_core = X_test[class_member_mask & core_samples_mask]
        axes[idx].scatter(xy_core[:, 0], xy_core[:, 1], c=[col],
                         s=80, alpha=0.8, edgecolors='k', linewidth=0.5,
                         marker='o' if k != -1 else 'x')

        # 邊界點
        xy_border = X_test[class_member_mask & ~core_samples_mask]
        if k != -1:
            axes[idx].scatter(xy_border[:, 0], xy_border[:, 1], c=[col],
                             s=30, alpha=0.5, edgecolors='k', linewidth=0.5, marker='o')

    title = f'eps={eps}\n簇數: {n_clusters}, 噪聲: {n_noise}, Silhouette: {silhouette:.3f}'
    axes[idx].set_title(title, fontsize=12, fontweight='bold', pad=10)
    axes[idx].set_xlabel('Feature 1', fontsize=10)
    axes[idx].set_ylabel('Feature 2', fontsize=10)
    axes[idx].grid(True, alpha=0.3)

    print(f"  eps={eps}: {n_clusters} 簇, {n_noise} 噪聲點, "
          f"核心點: {len(dbscan.core_sample_indices_)}, Silhouette: {silhouette:.4f}")

plt.tight_layout()
save_figure(fig3, get_output_path('dbscan_03_eps_analysis.png', '03_UnsupervisedLearning/Clustering'))
plt.close()

# ============================================================================
# Part 6: min_samples 參數影響分析
# ============================================================================
print("\n【Part 6】min_samples 參數影響分析")
print("-" * 80)

min_samples_values = [3, 5, 10, 15]
eps = 0.3

fig4, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

print("\nmin_samples 參數實驗（eps=0.3）:")
for idx, min_samp in enumerate(min_samples_values):
    dbscan = DBSCAN(eps=eps, min_samples=min_samp)
    y_pred = dbscan.fit_predict(X_test_scaled)

    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = np.sum(y_pred == -1)

    # 計算評估指標
    if n_clusters > 0 and n_noise < len(y_pred):
        mask = y_pred != -1
        if len(np.unique(y_pred[mask])) > 1:
            silhouette = silhouette_score(X_test_scaled[mask], y_pred[mask])
        else:
            silhouette = -1
    else:
        silhouette = -1

    # 繪製結果
    unique_labels = set(y_pred)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (y_pred == k)
        xy = X_test[class_member_mask]
        axes[idx].scatter(xy[:, 0], xy[:, 1], c=[col],
                         alpha=0.7, edgecolors='k', linewidth=0.5, s=50,
                         marker='x' if k == -1 else 'o')

    title = f'min_samples={min_samp}\n簇數: {n_clusters}, 噪聲: {n_noise}, Silhouette: {silhouette:.3f}'
    axes[idx].set_title(title, fontsize=12, fontweight='bold', pad=10)
    axes[idx].set_xlabel('Feature 1', fontsize=10)
    axes[idx].set_ylabel('Feature 2', fontsize=10)
    axes[idx].grid(True, alpha=0.3)

    print(f"  min_samples={min_samp}: {n_clusters} 簇, {n_noise} 噪聲點, "
          f"核心點: {len(dbscan.core_sample_indices_)}, Silhouette: {silhouette:.4f}")

plt.tight_layout()
save_figure(fig4, get_output_path('dbscan_04_min_samples_analysis.png', '03_UnsupervisedLearning/Clustering'))
plt.close()

# ============================================================================
# Part 7: 核心點、邊界點、噪聲點詳細分析
# ============================================================================
print("\n【Part 7】核心點、邊界點、噪聲點分析")
print("-" * 80)

# 使用最佳參數
dbscan_best = DBSCAN(eps=0.3, min_samples=5)
y_pred_best = dbscan_best.fit_predict(X_test_scaled)

# 分類點的類型
core_samples_mask = np.zeros_like(y_pred_best, dtype=bool)
core_samples_mask[dbscan_best.core_sample_indices_] = True

core_points = core_samples_mask & (y_pred_best != -1)
border_points = ~core_samples_mask & (y_pred_best != -1)
noise_points = y_pred_best == -1

fig5, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左圖：點類型標記
axes[0].scatter(X_test[core_points, 0], X_test[core_points, 1],
               c='blue', s=80, alpha=0.8, edgecolors='k', linewidth=0.5,
               marker='o', label=f'核心點 (Core: {np.sum(core_points)})')
axes[0].scatter(X_test[border_points, 0], X_test[border_points, 1],
               c='green', s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
               marker='s', label=f'邊界點 (Border: {np.sum(border_points)})')
axes[0].scatter(X_test[noise_points, 0], X_test[noise_points, 1],
               c='red', s=40, alpha=0.8, edgecolors='k', linewidth=0.5,
               marker='x', label=f'噪聲點 (Noise: {np.sum(noise_points)})')
axes[0].set_title('DBSCAN 點類型分析\n(eps=0.3, min_samples=5)',
                 fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Feature 1', fontsize=11)
axes[0].set_ylabel('Feature 2', fontsize=11)
axes[0].legend(loc='upper right', fontsize=11, framealpha=0.9)
axes[0].grid(True, alpha=0.3)

# 右圖：聚類結果（帶點類型標記）
unique_labels = set(y_pred_best)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [1, 0, 0, 1]  # 噪聲點紅色

    class_member_mask = (y_pred_best == k)

    # 核心點（大圓）
    xy_core = X_test[class_member_mask & core_samples_mask]
    axes[1].scatter(xy_core[:, 0], xy_core[:, 1], c=[col],
                   s=100, alpha=0.9, edgecolors='k', linewidth=1.5,
                   marker='o' if k != -1 else 'x')

    # 邊界點（小圓）
    xy_border = X_test[class_member_mask & ~core_samples_mask]
    if k != -1:
        axes[1].scatter(xy_border[:, 0], xy_border[:, 1], c=[col],
                       s=30, alpha=0.5, edgecolors='k', linewidth=0.5, marker='o')

n_clusters = len(set(y_pred_best)) - (1 if -1 in y_pred_best else 0)
axes[1].set_title(f'DBSCAN 聚類結果\n發現 {n_clusters} 個簇',
                 fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Feature 1', fontsize=11)
axes[1].set_ylabel('Feature 2', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig5, get_output_path('dbscan_05_point_types.png', '03_UnsupervisedLearning/Clustering'))
plt.close()

print(f"\n點類型統計:")
print(f"  核心點: {np.sum(core_points)} ({np.sum(core_points)/len(X_test)*100:.1f}%)")
print(f"  邊界點: {np.sum(border_points)} ({np.sum(border_points)/len(X_test)*100:.1f}%)")
print(f"  噪聲點: {np.sum(noise_points)} ({np.sum(noise_points)/len(X_test)*100:.1f}%)")

# ============================================================================
# Part 8: 不同形狀數據集的最佳參數探索
# ============================================================================
print("\n【Part 8】最佳參數探索")
print("-" * 80)

# 生成包含噪聲的複雜數據集
X_complex, y_complex = make_moons(n_samples=300, noise=0.08, random_state=RANDOM_STATE)
# 添加噪聲點
rng = np.random.RandomState(RANDOM_STATE)
noise_points_to_add = rng.uniform(low=-2, high=2, size=(30, 2))
X_complex = np.vstack([X_complex, noise_points_to_add])
X_complex_scaled = StandardScaler().fit_transform(X_complex)

# 網格搜索最佳參數
eps_range = np.arange(0.1, 0.6, 0.05)
min_samples_range = [3, 5, 7, 10]

best_score = -1
best_params = {}
results = []

print("\n正在進行參數網格搜索...")
for eps in eps_range:
    for min_samp in min_samples_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samp)
        labels = dbscan.fit_predict(X_complex_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        if n_clusters > 1 and n_noise < len(labels):
            mask = labels != -1
            if len(np.unique(labels[mask])) > 1:
                score = silhouette_score(X_complex_scaled[mask], labels[mask])
                results.append({
                    'eps': eps,
                    'min_samples': min_samp,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette': score
                })

                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samp,
                                  'n_clusters': n_clusters, 'n_noise': n_noise}

print(f"\n最佳參數:")
print(f"  eps = {best_params['eps']:.2f}")
print(f"  min_samples = {best_params['min_samples']}")
print(f"  簇數 = {best_params['n_clusters']}")
print(f"  噪聲點 = {best_params['n_noise']}")
print(f"  Silhouette Score = {best_score:.4f}")

# 可視化參數搜索熱力圖
fig6, axes = plt.subplots(1, 2, figsize=(16, 6))

# 創建熱力圖數據
import pandas as pd
if results:
    results_df = pd.DataFrame(results)
    pivot_silhouette = results_df.pivot_table(values='silhouette',
                                              index='min_samples',
                                              columns='eps')
    pivot_clusters = results_df.pivot_table(values='n_clusters',
                                            index='min_samples',
                                            columns='eps')

    # Silhouette 分數熱力圖
    im1 = axes[0].imshow(pivot_silhouette.values, cmap='RdYlGn', aspect='auto')
    axes[0].set_xticks(np.arange(len(pivot_silhouette.columns)))
    axes[0].set_yticks(np.arange(len(pivot_silhouette.index)))
    axes[0].set_xticklabels([f'{x:.2f}' for x in pivot_silhouette.columns])
    axes[0].set_yticklabels(pivot_silhouette.index)
    axes[0].set_xlabel('eps', fontsize=11)
    axes[0].set_ylabel('min_samples', fontsize=11)
    axes[0].set_title('Silhouette Score 熱力圖', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im1, ax=axes[0])

    # 簇數量熱力圖
    im2 = axes[1].imshow(pivot_clusters.values, cmap='viridis', aspect='auto')
    axes[1].set_xticks(np.arange(len(pivot_clusters.columns)))
    axes[1].set_yticks(np.arange(len(pivot_clusters.index)))
    axes[1].set_xticklabels([f'{x:.2f}' for x in pivot_clusters.columns])
    axes[1].set_yticklabels(pivot_clusters.index)
    axes[1].set_xlabel('eps', fontsize=11)
    axes[1].set_ylabel('min_samples', fontsize=11)
    axes[1].set_title('簇數量熱力圖', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
save_figure(fig6, get_output_path('dbscan_06_parameter_search.png', '03_UnsupervisedLearning/Clustering'))
plt.close()

# ============================================================================
# Part 9: 最終評估與總結
# ============================================================================
print("\n【Part 9】最終評估")
print("-" * 80)

# 使用最佳參數
dbscan_final = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
y_final = dbscan_final.fit_predict(X_complex_scaled)

# 與 K-Means 比較
kmeans_final = KMeans(n_clusters=best_params['n_clusters'], random_state=RANDOM_STATE, n_init=10)
y_kmeans_final = kmeans_final.fit_predict(X_complex_scaled)

fig7, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原始數據（帶噪聲）
axes[0].scatter(X_complex[:300, 0], X_complex[:300, 1],
               c='blue', alpha=0.6, edgecolors='k', linewidth=0.5, s=50,
               label='Original data')
axes[0].scatter(X_complex[300:, 0], X_complex[300:, 1],
               c='red', alpha=0.8, edgecolors='k', linewidth=0.5, s=50,
               marker='x', label='Added noise')
axes[0].set_title('原始數據（含人工噪聲）', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Feature 1', fontsize=11)
axes[0].set_ylabel('Feature 2', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# DBSCAN 結果
unique_labels = set(y_final)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]

    class_member_mask = (y_final == k)
    xy = X_complex[class_member_mask]
    axes[1].scatter(xy[:, 0], xy[:, 1], c=[col],
                   alpha=0.7, edgecolors='k', linewidth=0.5, s=50,
                   marker='x' if k == -1 else 'o')

axes[1].set_title(f'DBSCAN 結果\n(eps={best_params["eps"]:.2f}, '
                 f'min_samples={best_params["min_samples"]})\n'
                 f'噪聲點: {np.sum(y_final == -1)}',
                 fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Feature 1', fontsize=11)
axes[1].set_ylabel('Feature 2', fontsize=11)
axes[1].grid(True, alpha=0.3)

# K-Means 結果
axes[2].scatter(X_complex[:, 0], X_complex[:, 1], c=y_kmeans_final,
               cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5, s=50)
centers_original = StandardScaler().fit(X_complex).inverse_transform(kmeans_final.cluster_centers_)
axes[2].scatter(centers_original[:, 0], centers_original[:, 1],
               c='red', marker='X', s=200, edgecolors='k', linewidth=2,
               label='Centroids')
axes[2].set_title(f'K-Means 結果\n(K={best_params["n_clusters"]})',
                 fontsize=13, fontweight='bold', pad=10)
axes[2].set_xlabel('Feature 1', fontsize=11)
axes[2].set_ylabel('Feature 2', fontsize=11)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig7, get_output_path('dbscan_07_final_comparison.png', '03_UnsupervisedLearning/Clustering'))
plt.close()

# 計算評估指標
mask_dbscan = y_final != -1
if len(np.unique(y_final[mask_dbscan])) > 1:
    dbscan_silhouette = silhouette_score(X_complex_scaled[mask_dbscan], y_final[mask_dbscan])
    dbscan_db = davies_bouldin_score(X_complex_scaled[mask_dbscan], y_final[mask_dbscan])
else:
    dbscan_silhouette = -1
    dbscan_db = -1

kmeans_silhouette = silhouette_score(X_complex_scaled, y_kmeans_final)
kmeans_db = davies_bouldin_score(X_complex_scaled, y_kmeans_final)

print("\n評估指標對比:")
print(f"{'算法':<10} {'Silhouette':<15} {'Davies-Bouldin':<15} {'噪聲識別':<10}")
print("-" * 60)
print(f"{'DBSCAN':<10} {dbscan_silhouette:>10.4f}     {dbscan_db:>10.4f}         "
      f"{np.sum(y_final == -1)} 點")
print(f"{'K-Means':<10} {kmeans_silhouette:>10.4f}     {kmeans_db:>10.4f}         N/A")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("DBSCAN 關鍵要點總結".center(80))
print("=" * 80)
print("""
1. 參數選擇建議：
   • eps: 使用 K-距離圖（K-distance graph）輔助選擇
   • min_samples: 通常設為數據維度 + 1，或根據數據密度調整
   • 對於 2D 數據，min_samples=5 是常見起點

2. 適用場景：
   • 數據有明顯的密度差異
   • 需要識別異常值/噪聲點
   • 簇的形狀不規則（非凸形）
   • 不需要預先知道簇的數量

3. 不適用場景：
   • 高維數據（維度詛咒）
   • 簇的密度差異很大
   • 數據量特別大（計算複雜度較高）

4. 與其他算法比較：
   • vs K-Means: DBSCAN 可處理任意形狀，K-Means 只適合凸形簇
   • vs 層次聚類: DBSCAN 計算效率更高，但參數選擇較困難
   • vs Mean Shift: DBSCAN 速度更快，但需要手動設置參數

5. 實踐建議：
   • 數據標準化（StandardScaler）通常能改善結果
   • 使用網格搜索找最佳參數組合
   • 結合 Silhouette Score 評估聚類質量
   • 可視化結果以驗證參數選擇的合理性
""")
print("=" * 80)
print("✓ DBSCAN 教程完成！已生成 7 張可視化圖表。")
print("=" * 80)
