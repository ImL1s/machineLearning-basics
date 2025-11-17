"""
t-SNE 和 UMAP 降維可視化
t-SNE and UMAP Dimensionality Reduction Visualization

t-SNE (t-Distributed Stochastic Neighbor Embedding):
    - 非線性降維技術，特別擅長可視化
    - 保持數據局部結構
    - 適合探索性數據分析

UMAP (Uniform Manifold Approximation and Projection):
    - 更快、更具可擴展性的非線性降維
    - 保持全局和局部結構
    - 支持監督和半監督學習
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_digits, load_iris, fetch_openml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 導入工具模塊 / Import utility modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import RANDOM_STATE, setup_chinese_fonts, create_subplots, save_figure

# 嘗試導入 UMAP / Try to import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
    print("✓ UMAP 已安裝並可用")
except ImportError:
    UMAP_AVAILABLE = False
    print("✗ UMAP 未安裝，部分功能將跳過")
    print("  安裝方法: pip install umap-learn")

# 設置中文字體 / Setup Chinese fonts
setup_chinese_fonts()

print("=" * 100)
print("t-SNE 和 UMAP 降維可視化教程".center(100))
print("t-SNE and UMAP Dimensionality Reduction Visualization Tutorial".center(100))
print("=" * 100)


# ============================================================================
# 第一部分：算法原理對比
# Part 1: Algorithm Comparison
# ============================================================================
print("\n" + "=" * 100)
print("第一部分：算法原理對比")
print("Part 1: Algorithm Comparison")
print("=" * 100)

print("""
【算法原理對比 / Algorithm Comparison】

1. PCA (Principal Component Analysis)
   原理 / Principle:
   • 線性降維，找到數據方差最大的方向
   • 保持全局結構，快速高效

   優點 / Advantages:
   • 計算速度快 O(min(n²p, np²))
   • 結果可重複
   • 適合預處理

   缺點 / Disadvantages:
   • 只能捕捉線性關係
   • 可能丟失重要的非線性結構

   適用場景 / Use Cases:
   • 特徵降維和數據壓縮
   • 快速可視化預覽
   • 噪聲過濾

2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
   原理 / Principle:
   • 非線性降維，保持局部鄰域結構
   • 將高維點間的距離轉換為概率分佈
   • 最小化高維和低維分佈的 KL 散度

   優點 / Advantages:
   • 優秀的可視化效果
   • 保持局部結構，聚類清晰
   • 適合探索性分析

   缺點 / Disadvantages:
   • 計算慢 O(n² log n)，不適合大數據
   • 結果不穩定（隨機初始化）
   • 距離不可解釋
   • 不保持全局結構

   適用場景 / Use Cases:
   • 高維數據可視化（<10000 樣本）
   • 聚類分析和模式發現
   • 異常值檢測

3. UMAP (Uniform Manifold Approximation and Projection)
   原理 / Principle:
   • 基於流形學習和拓撲數據分析
   • 構建高維模糊拓撲表示
   • 優化低維表示以匹配高維結構

   優點 / Advantages:
   • 速度快 O(n log n)，適合大數據
   • 保持全局和局部結構
   • 結果較穩定
   • 支持新數據投影

   缺點 / Disadvantages:
   • 參數較多，需要調優
   • 理論較複雜

   適用場景 / Use Cases:
   • 大規模數據可視化（>10000 樣本）
   • 需要保持全局結構時
   • 實時數據分析

【計算複雜度對比 / Computational Complexity】
• PCA:     O(min(n²p, np²)) - 最快
• UMAP:    O(n log n)       - 快
• t-SNE:   O(n² log n)      - 慢

其中 n 是樣本數，p 是特徵數
""")


# ============================================================================
# 第二部分：基礎應用（Digits 數據集）
# Part 2: Basic Application (Digits Dataset)
# ============================================================================
print("\n" + "=" * 100)
print("第二部分：基礎應用 - 手寫數字數據集")
print("Part 2: Basic Application - Handwritten Digits Dataset")
print("=" * 100)

# 加載 Digits 數據集 / Load Digits dataset
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

print(f"\n數據集信息 / Dataset Information:")
print(f"樣本數 / Samples: {X_digits.shape[0]}")
print(f"特徵數 / Features: {X_digits.shape[1]} (8x8 像素)")
print(f"類別數 / Classes: {len(np.unique(y_digits))} (數字 0-9)")

# 標準化 / Standardization
scaler = StandardScaler()
X_digits_scaled = scaler.fit_transform(X_digits)

# PCA 降維 / PCA dimensionality reduction
print("\n1. PCA 降維...")
start_time = time.time()
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_digits_scaled)
pca_time = time.time() - start_time
print(f"   PCA 完成，耗時: {pca_time:.3f} 秒")
print(f"   解釋方差比: {pca.explained_variance_ratio_}")
print(f"   累積解釋方差: {np.sum(pca.explained_variance_ratio_):.4f}")

# t-SNE 降維 / t-SNE dimensionality reduction
print("\n2. t-SNE 降維...")
start_time = time.time()
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_digits_scaled)
tsne_time = time.time() - start_time
print(f"   t-SNE 完成，耗時: {tsne_time:.3f} 秒")
print(f"   KL 散度: {tsne.kl_divergence_:.4f}")

# UMAP 降維 / UMAP dimensionality reduction
if UMAP_AVAILABLE:
    print("\n3. UMAP 降維...")
    start_time = time.time()
    umap_reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15)
    X_umap = umap_reducer.fit_transform(X_digits_scaled)
    umap_time = time.time() - start_time
    print(f"   UMAP 完成，耗時: {umap_time:.3f} 秒")
else:
    X_umap = None
    umap_time = 0

# 可視化對比 / Visualization comparison
fig, axes = create_subplots(1, 3 if UMAP_AVAILABLE else 2, figsize=(18, 5))

# PCA 可視化
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10',
                          alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
axes[0].set_title(f'PCA (耗時: {pca_time:.2f}s)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('第一主成分 / First PC')
axes[0].set_ylabel('第二主成分 / Second PC')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='數字 / Digit')

# t-SNE 可視化
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10',
                          alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
axes[1].set_title(f't-SNE (耗時: {tsne_time:.2f}s)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('t-SNE 維度 1 / t-SNE Dim 1')
axes[1].set_ylabel('t-SNE 維度 2 / t-SNE Dim 2')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='數字 / Digit')

# UMAP 可視化
if UMAP_AVAILABLE:
    scatter3 = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y_digits, cmap='tab10',
                              alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
    axes[2].set_title(f'UMAP (耗時: {umap_time:.2f}s)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('UMAP 維度 1 / UMAP Dim 1')
    axes[2].set_ylabel('UMAP 維度 2 / UMAP Dim 2')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[2], label='數字 / Digit')

plt.tight_layout()
save_figure(fig, '03_UnsupervisedLearning/DimensionalityReduction/01_algorithm_comparison.png')
plt.close()

# 評估降維質量 / Evaluate dimensionality reduction quality
print("\n【降維質量評估 / Quality Evaluation】")

# 使用輪廓係數評估聚類質量 / Use silhouette score to evaluate clustering quality
if X_umap is not None:
    scores = {
        'PCA': silhouette_score(X_pca, y_digits),
        't-SNE': silhouette_score(X_tsne, y_digits),
        'UMAP': silhouette_score(X_umap, y_digits)
    }
else:
    scores = {
        'PCA': silhouette_score(X_pca, y_digits),
        't-SNE': silhouette_score(X_tsne, y_digits)
    }

for method, score in scores.items():
    print(f"{method:8s} 輪廓係數: {score:.4f}")


# ============================================================================
# 第三部分：t-SNE 參數分析
# Part 3: t-SNE Parameter Analysis
# ============================================================================
print("\n" + "=" * 100)
print("第三部分：t-SNE 參數分析")
print("Part 3: t-SNE Parameter Analysis")
print("=" * 100)

print("""
【t-SNE 關鍵參數 / Key Parameters】

1. perplexity (困惑度):
   • 範圍: 5-50（通常使用 30）
   • 含義: 每個點考慮的鄰居數量
   • 效果:
     - 小值: 關注局部結構
     - 大值: 關注全局結構
   • 建議: 數據量大時可用較大值

2. learning_rate (學習率):
   • 範圍: 10-1000（通常使用 200）
   • 效果:
     - 太小: 收斂慢，可能陷入局部最優
     - 太大: 不穩定
   • 建議: 如果點聚成一團，增加學習率

3. n_iter (迭代次數):
   • 範圍: 250-1000（至少 250）
   • 效果: 更多迭代通常更好，但更慢
   • 建議: 觀察 KL 散度是否收斂
""")

# perplexity 參數影響 / Effect of perplexity parameter
print("\n測試不同的 perplexity 值...")
perplexities = [5, 30, 50, 100]
tsne_results_perp = {}

fig, axes = create_subplots(2, 2, figsize=(16, 14))
axes = axes.ravel()

for idx, perp in enumerate(perplexities):
    print(f"  perplexity = {perp}...")
    start_time = time.time()
    tsne_temp = TSNE(n_components=2, random_state=RANDOM_STATE,
                     perplexity=perp, n_iter=1000)
    X_tsne_temp = tsne_temp.fit_transform(X_digits_scaled)
    elapsed = time.time() - start_time

    tsne_results_perp[perp] = X_tsne_temp

    scatter = axes[idx].scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1],
                               c=y_digits, cmap='tab10', alpha=0.6, s=30,
                               edgecolors='k', linewidths=0.5)
    axes[idx].set_title(f'perplexity={perp} (KL={tsne_temp.kl_divergence_:.2f}, {elapsed:.2f}s)',
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('t-SNE 維度 1')
    axes[idx].set_ylabel('t-SNE 維度 2')
    axes[idx].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[idx], label='數字')

plt.suptitle('t-SNE: perplexity 參數影響 / Effect of Perplexity Parameter',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
save_figure(fig, '03_UnsupervisedLearning/DimensionalityReduction/02_tsne_perplexity.png')
plt.close()

# learning_rate 參數影響 / Effect of learning_rate parameter
print("\n測試不同的 learning_rate 值...")
learning_rates = [10, 100, 200, 1000]
tsne_results_lr = {}

fig, axes = create_subplots(2, 2, figsize=(16, 14))
axes = axes.ravel()

for idx, lr in enumerate(learning_rates):
    print(f"  learning_rate = {lr}...")
    start_time = time.time()
    tsne_temp = TSNE(n_components=2, random_state=RANDOM_STATE,
                     perplexity=30, learning_rate=lr, n_iter=1000)
    X_tsne_temp = tsne_temp.fit_transform(X_digits_scaled)
    elapsed = time.time() - start_time

    tsne_results_lr[lr] = X_tsne_temp

    scatter = axes[idx].scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1],
                               c=y_digits, cmap='tab10', alpha=0.6, s=30,
                               edgecolors='k', linewidths=0.5)
    axes[idx].set_title(f'learning_rate={lr} (KL={tsne_temp.kl_divergence_:.2f}, {elapsed:.2f}s)',
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('t-SNE 維度 1')
    axes[idx].set_ylabel('t-SNE 維度 2')
    axes[idx].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[idx], label='數字')

plt.suptitle('t-SNE: learning_rate 參數影響 / Effect of Learning Rate Parameter',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
save_figure(fig, '03_UnsupervisedLearning/DimensionalityReduction/03_tsne_learning_rate.png')
plt.close()


# ============================================================================
# 第四部分：UMAP 參數分析
# Part 4: UMAP Parameter Analysis
# ============================================================================
if UMAP_AVAILABLE:
    print("\n" + "=" * 100)
    print("第四部分：UMAP 參數分析")
    print("Part 4: UMAP Parameter Analysis")
    print("=" * 100)

    print("""
【UMAP 關鍵參數 / Key Parameters】

1. n_neighbors (鄰居數量):
   • 範圍: 2-100（通常使用 15）
   • 含義: 局部鄰域大小
   • 效果:
     - 小值: 關注局部結構，細節更多
     - 大值: 關注全局結構，更緊湊
   • 建議: 數據量大時可用較大值

2. min_dist (最小距離):
   • 範圍: 0.0-0.99（通常使用 0.1）
   • 含義: 低維空間中點之間的最小距離
   • 效果:
     - 小值: 點更緊密，聚類更明顯
     - 大值: 點更分散，保持更多結構
   • 建議: 可視化用小值，保持拓撲用大值

3. metric (距離度量):
   • 選項: 'euclidean', 'manhattan', 'cosine' 等
   • 含義: 計算距離的方式
   • 建議: 根據數據類型選擇
    """)

    # n_neighbors 參數影響 / Effect of n_neighbors parameter
    print("\n測試不同的 n_neighbors 值...")
    n_neighbors_list = [5, 15, 50, 100]
    umap_results_nn = {}

    fig, axes = create_subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    for idx, nn in enumerate(n_neighbors_list):
        print(f"  n_neighbors = {nn}...")
        start_time = time.time()
        umap_temp = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                             n_neighbors=nn, min_dist=0.1)
        X_umap_temp = umap_temp.fit_transform(X_digits_scaled)
        elapsed = time.time() - start_time

        umap_results_nn[nn] = X_umap_temp

        scatter = axes[idx].scatter(X_umap_temp[:, 0], X_umap_temp[:, 1],
                                   c=y_digits, cmap='tab10', alpha=0.6, s=30,
                                   edgecolors='k', linewidths=0.5)
        axes[idx].set_title(f'n_neighbors={nn} ({elapsed:.2f}s)',
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('UMAP 維度 1')
        axes[idx].set_ylabel('UMAP 維度 2')
        axes[idx].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[idx], label='數字')

    plt.suptitle('UMAP: n_neighbors 參數影響 / Effect of n_neighbors Parameter',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    save_figure(fig, '03_UnsupervisedLearning/DimensionalityReduction/04_umap_n_neighbors.png')
    plt.close()

    # min_dist 參數影響 / Effect of min_dist parameter
    print("\n測試不同的 min_dist 值...")
    min_dists = [0.0, 0.1, 0.5, 0.9]
    umap_results_md = {}

    fig, axes = create_subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    for idx, md in enumerate(min_dists):
        print(f"  min_dist = {md}...")
        start_time = time.time()
        umap_temp = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                             n_neighbors=15, min_dist=md)
        X_umap_temp = umap_temp.fit_transform(X_digits_scaled)
        elapsed = time.time() - start_time

        umap_results_md[md] = X_umap_temp

        scatter = axes[idx].scatter(X_umap_temp[:, 0], X_umap_temp[:, 1],
                                   c=y_digits, cmap='tab10', alpha=0.6, s=30,
                                   edgecolors='k', linewidths=0.5)
        axes[idx].set_title(f'min_dist={md} ({elapsed:.2f}s)',
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('UMAP 維度 1')
        axes[idx].set_ylabel('UMAP 維度 2')
        axes[idx].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[idx], label='數字')

    plt.suptitle('UMAP: min_dist 參數影響 / Effect of min_dist Parameter',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    save_figure(fig, '03_UnsupervisedLearning/DimensionalityReduction/05_umap_min_dist.png')
    plt.close()


# ============================================================================
# 第五部分：高維數據應用（MNIST 子集）
# Part 5: High-Dimensional Data Application (MNIST Subset)
# ============================================================================
print("\n" + "=" * 100)
print("第五部分：高維數據應用 - MNIST 子集")
print("Part 5: High-Dimensional Data Application - MNIST Subset")
print("=" * 100)

# 加載 MNIST 子集 / Load MNIST subset
print("\n加載 MNIST 數據集（這可能需要一些時間）...")
try:
    # 使用較小的子集以加快演示速度
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X_mnist = mnist.data[:5000].values if hasattr(mnist.data, 'values') else mnist.data[:5000]
    y_mnist = mnist.target[:5000].astype(int).values if hasattr(mnist.target, 'values') else mnist.target[:5000].astype(int)

    print(f"✓ MNIST 數據集加載成功")
    print(f"  樣本數: {X_mnist.shape[0]}")
    print(f"  特徵數: {X_mnist.shape[1]} (28x28 像素)")
    print(f"  類別數: {len(np.unique(y_mnist))}")

    # 標準化
    X_mnist_scaled = StandardScaler().fit_transform(X_mnist)

    # 運行時間對比 / Runtime comparison
    print("\n運行時間對比...")
    runtimes = {}

    # PCA
    print("  1. PCA...")
    start_time = time.time()
    pca_mnist = PCA(n_components=2, random_state=RANDOM_STATE)
    X_mnist_pca = pca_mnist.fit_transform(X_mnist_scaled)
    runtimes['PCA'] = time.time() - start_time
    print(f"     完成，耗時: {runtimes['PCA']:.2f}s")

    # t-SNE
    print("  2. t-SNE...")
    start_time = time.time()
    tsne_mnist = TSNE(n_components=2, random_state=RANDOM_STATE,
                      perplexity=30, n_iter=1000, verbose=0)
    X_mnist_tsne = tsne_mnist.fit_transform(X_mnist_scaled)
    runtimes['t-SNE'] = time.time() - start_time
    print(f"     完成，耗時: {runtimes['t-SNE']:.2f}s")

    # UMAP
    if UMAP_AVAILABLE:
        print("  3. UMAP...")
        start_time = time.time()
        umap_mnist = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                              n_neighbors=15, min_dist=0.1)
        X_mnist_umap = umap_mnist.fit_transform(X_mnist_scaled)
        runtimes['UMAP'] = time.time() - start_time
        print(f"     完成，耗時: {runtimes['UMAP']:.2f}s")

    # 可視化運行時間對比 / Visualize runtime comparison
    fig, axes = create_subplots(1, 2, figsize=(16, 6))

    # 運行時間柱狀圖
    methods = list(runtimes.keys())
    times = list(runtimes.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(methods)]

    bars = axes[0].bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('運行時間 (秒) / Runtime (seconds)', fontsize=12)
    axes[0].set_title('MNIST 數據集降維運行時間對比\nRuntime Comparison on MNIST',
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # 在柱狀圖上添加數值
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.2f}s',
                    ha='center', va='bottom', fontweight='bold')

    # 速度提升對比
    if len(runtimes) > 1:
        baseline = runtimes['t-SNE']
        speedups = {k: baseline/v for k, v in runtimes.items()}

        bars2 = axes[1].bar(methods, list(speedups.values()),
                           color=colors, alpha=0.7, edgecolor='black')
        axes[1].axhline(y=1, color='red', linestyle='--', label='基準線 (t-SNE)')
        axes[1].set_ylabel('相對速度 (相對於 t-SNE) / Relative Speed', fontsize=12)
        axes[1].set_title('速度提升對比 (以 t-SNE 為基準)\nSpeed Improvement (vs t-SNE)',
                         fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        # 添加倍數標籤
        for bar, speedup in zip(bars2, speedups.values()):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{speedup:.1f}x',
                        ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, '03_UnsupervisedLearning/DimensionalityReduction/06_mnist_runtime_comparison.png')
    plt.close()

    # 可視化降維結果 / Visualize dimensionality reduction results
    fig, axes = create_subplots(1, 3 if UMAP_AVAILABLE else 2, figsize=(18, 5))

    # PCA 可視化
    scatter1 = axes[0].scatter(X_mnist_pca[:, 0], X_mnist_pca[:, 1],
                              c=y_mnist, cmap='tab10', alpha=0.5, s=10,
                              edgecolors='none')
    axes[0].set_title(f'PCA on MNIST\n({runtimes["PCA"]:.2f}s)',
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('第一主成分 / PC1')
    axes[0].set_ylabel('第二主成分 / PC2')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='數字')

    # t-SNE 可視化
    scatter2 = axes[1].scatter(X_mnist_tsne[:, 0], X_mnist_tsne[:, 1],
                              c=y_mnist, cmap='tab10', alpha=0.5, s=10,
                              edgecolors='none')
    axes[1].set_title(f't-SNE on MNIST\n({runtimes["t-SNE"]:.2f}s)',
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE 維度 1')
    axes[1].set_ylabel('t-SNE 維度 2')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='數字')

    # UMAP 可視化
    if UMAP_AVAILABLE:
        scatter3 = axes[2].scatter(X_mnist_umap[:, 0], X_mnist_umap[:, 1],
                                  c=y_mnist, cmap='tab10', alpha=0.5, s=10,
                                  edgecolors='none')
        axes[2].set_title(f'UMAP on MNIST\n({runtimes["UMAP"]:.2f}s)',
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('UMAP 維度 1')
        axes[2].set_ylabel('UMAP 維度 2')
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[2], label='數字')

    plt.tight_layout()
    save_figure(fig, '03_UnsupervisedLearning/DimensionalityReduction/07_mnist_visualization.png')
    plt.close()

    print("\n✓ MNIST 數據集分析完成")

except Exception as e:
    print(f"✗ 加載 MNIST 數據集失敗: {e}")
    print("  跳過 MNIST 數據集分析")


# ============================================================================
# 第六部分：實際應用案例
# Part 6: Practical Application Cases
# ============================================================================
print("\n" + "=" * 100)
print("第六部分：實際應用案例")
print("Part 6: Practical Application Cases")
print("=" * 100)

# 案例1：特徵空間可視化（Iris 所有特徵）
print("\n【案例1】特徵空間可視化 - Iris 數據集")
print("Case 1: Feature Space Visualization - Iris Dataset")
print("-" * 100)

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# 標準化
X_iris_scaled = StandardScaler().fit_transform(X_iris)

# 應用降維
iris_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_iris_scaled)
iris_tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30).fit_transform(X_iris_scaled)

if UMAP_AVAILABLE:
    iris_umap = umap.UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(X_iris_scaled)

# 可視化
fig, axes = create_subplots(1, 3 if UMAP_AVAILABLE else 2, figsize=(18, 5))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, name in enumerate(iris.target_names):
    mask = y_iris == i
    axes[0].scatter(iris_pca[mask, 0], iris_pca[mask, 1],
                   label=name, c=colors[i], alpha=0.7, s=80,
                   edgecolors='black', linewidths=1)
    axes[1].scatter(iris_tsne[mask, 0], iris_tsne[mask, 1],
                   label=name, c=colors[i], alpha=0.7, s=80,
                   edgecolors='black', linewidths=1)
    if UMAP_AVAILABLE:
        axes[2].scatter(iris_umap[mask, 0], iris_umap[mask, 1],
                       label=name, c=colors[i], alpha=0.7, s=80,
                       edgecolors='black', linewidths=1)

axes[0].set_title('PCA - Iris Dataset', fontsize=14, fontweight='bold')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('t-SNE - Iris Dataset', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Dimension 1')
axes[1].set_ylabel('Dimension 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

if UMAP_AVAILABLE:
    axes[2].set_title('UMAP - Iris Dataset', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Dimension 1')
    axes[2].set_ylabel('Dimension 2')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, '03_UnsupervisedLearning/DimensionalityReduction/08_iris_feature_space.png')
plt.close()

print("✓ Iris 特徵空間可視化完成")

# 案例2：聚類結果可視化
print("\n【案例2】聚類結果可視化")
print("Case 2: Clustering Results Visualization")
print("-" * 100)

# 對 Digits 數據集進行 K-Means 聚類
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
clusters = kmeans.fit_predict(X_digits_scaled)

# 使用 t-SNE 可視化聚類結果
tsne_for_clustering = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
X_tsne_cluster = tsne_for_clustering.fit_transform(X_digits_scaled)

# 創建對比圖：真實標籤 vs 聚類結果
fig, axes = create_subplots(1, 2, figsize=(16, 6))

# 真實標籤
scatter1 = axes[0].scatter(X_tsne_cluster[:, 0], X_tsne_cluster[:, 1],
                          c=y_digits, cmap='tab10', alpha=0.6, s=30,
                          edgecolors='k', linewidths=0.5)
axes[0].set_title('真實標籤 / True Labels', fontsize=14, fontweight='bold')
axes[0].set_xlabel('t-SNE 維度 1')
axes[0].set_ylabel('t-SNE 維度 2')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='數字')

# 聚類結果
scatter2 = axes[1].scatter(X_tsne_cluster[:, 0], X_tsne_cluster[:, 1],
                          c=clusters, cmap='tab10', alpha=0.6, s=30,
                          edgecolors='k', linewidths=0.5)
axes[1].set_title(f'K-Means 聚類結果 (k={n_clusters})\nClustering Results',
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('t-SNE 維度 1')
axes[1].set_ylabel('t-SNE 維度 2')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='簇編號')

plt.tight_layout()
save_figure(fig, '03_UnsupervisedLearning/DimensionalityReduction/09_clustering_visualization.png')
plt.close()

print("✓ 聚類結果可視化完成")

# 案例3：不同數據集的對比
print("\n【案例3】多數據集降維效果對比")
print("Case 3: Multi-Dataset Dimensionality Reduction Comparison")
print("-" * 100)

# 使用 t-SNE 處理不同數據集
datasets = {
    'Iris (4D)': (X_iris_scaled, y_iris, 'Set3'),
    'Digits (64D)': (X_digits_scaled, y_digits, 'tab10')
}

fig, axes = create_subplots(2, 2, figsize=(16, 14))

idx = 0
for name, (X, y, cmap) in datasets.items():
    # PCA
    X_pca_temp = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    scatter = axes[idx, 0].scatter(X_pca_temp[:, 0], X_pca_temp[:, 1],
                                  c=y, cmap=cmap, alpha=0.6, s=40,
                                  edgecolors='k', linewidths=0.5)
    axes[idx, 0].set_title(f'{name} - PCA', fontsize=12, fontweight='bold')
    axes[idx, 0].set_xlabel('PC1')
    axes[idx, 0].set_ylabel('PC2')
    axes[idx, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[idx, 0])

    # t-SNE
    X_tsne_temp = TSNE(n_components=2, random_state=RANDOM_STATE,
                       perplexity=min(30, len(X)//4)).fit_transform(X)
    scatter = axes[idx, 1].scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1],
                                  c=y, cmap=cmap, alpha=0.6, s=40,
                                  edgecolors='k', linewidths=0.5)
    axes[idx, 1].set_title(f'{name} - t-SNE', fontsize=12, fontweight='bold')
    axes[idx, 1].set_xlabel('Dimension 1')
    axes[idx, 1].set_ylabel('Dimension 2')
    axes[idx, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[idx, 1])

    idx += 1

plt.suptitle('多數據集降維效果對比 / Multi-Dataset Comparison',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
save_figure(fig, '03_UnsupervisedLearning/DimensionalityReduction/10_multi_dataset_comparison.png')
plt.close()

print("✓ 多數據集對比完成")


# ============================================================================
# 總結與最佳實踐
# Summary and Best Practices
# ============================================================================
print("\n" + "=" * 100)
print("總結與最佳實踐指南")
print("Summary and Best Practices")
print("=" * 100)

print("""
【算法選擇指南 / Algorithm Selection Guide】

1. 選擇 PCA 當:
   ✓ 需要快速降維
   ✓ 數據呈線性結構
   ✓ 需要可解釋的結果
   ✓ 作為其他算法的預處理步驟
   ✓ 需要壓縮數據

2. 選擇 t-SNE 當:
   ✓ 主要目標是可視化 (2D/3D)
   ✓ 數據量適中 (< 10,000 樣本)
   ✓ 想要發現局部聚類結構
   ✓ 不需要投影新數據
   ✓ 願意花時間調參

3. 選擇 UMAP 當:
   ✓ 數據量較大 (> 10,000 樣本)
   ✓ 需要保持全局和局部結構
   ✓ 運行速度重要
   ✓ 需要投影新數據
   ✓ 要進行後續機器學習任務

【參數調優建議 / Parameter Tuning Recommendations】

t-SNE:
• perplexity:
  - 小數據集 (< 1000): 5-30
  - 中等數據集 (1000-10000): 30-50
  - 大數據集 (> 10000): 50-100

• learning_rate:
  - 開始值: 200
  - 如果點聚成一團: 增加到 500-1000
  - 如果太分散: 減少到 10-100

• n_iter:
  - 最少: 250
  - 推薦: 1000
  - 複雜數據: 2000-5000

UMAP:
• n_neighbors:
  - 局部結構: 5-15
  - 平衡: 15-30
  - 全局結構: 30-100

• min_dist:
  - 緊湊聚類: 0.0-0.1
  - 平衡: 0.1-0.3
  - 保持結構: 0.3-0.99

• metric:
  - 數值特徵: 'euclidean'
  - 文本/稀疏: 'cosine'
  - 類別特徵: 'hamming'

【最佳實踐 / Best Practices】

1. 數據預處理:
   ✓ 總是標準化數據 (StandardScaler)
   ✓ 處理缺失值和異常值
   ✓ 考慮先用 PCA 降到 50 維

2. 可視化技巧:
   ✓ 使用不同顏色區分類別
   ✓ 調整點的透明度 (alpha)
   ✓ 添加清晰的圖例和標題
   ✓ 保存高分辨率圖像 (DPI ≥ 150)

3. 結果驗證:
   ✓ 使用輪廓係數評估聚類質量
   ✓ 嘗試多次運行 (不同隨機種子)
   ✓ 對比不同方法的結果
   ✓ 結合領域知識解釋

4. 性能優化:
   ✓ 大數據集先用 PCA 降維
   ✓ 使用數據子集進行參數探索
   ✓ UMAP 優於 t-SNE 處理大數據
   ✓ 考慮使用 GPU 加速 (如 cuML)

【常見陷阱 / Common Pitfalls】

✗ 不要過度解釋點之間的距離
✗ 不要用於維度災難之外的特徵提取
✗ 不要忽略數據標準化
✗ 不要只依賴默認參數
✗ 不要用 t-SNE 的結果訓練模型

【進階應用 / Advanced Applications】

• 結合監督學習: UMAP 支持 supervised/semi-supervised 模式
• 異常檢測: 尋找孤立的點或離群簇
• 時間序列: 可視化序列數據的演化
• 高維數據探索: 發現隱藏的模式和結構
• 特徵工程: 創建新的低維表示特徵
""")

print("\n" + "=" * 100)
print("教程完成！")
print("Tutorial Completed!")
print("=" * 100)

print("""
已生成的可視化圖表 / Generated Visualizations:
1. 01_algorithm_comparison.png       - 三種算法基礎對比
2. 02_tsne_perplexity.png           - t-SNE perplexity 參數分析
3. 03_tsne_learning_rate.png        - t-SNE learning_rate 參數分析
4. 04_umap_n_neighbors.png          - UMAP n_neighbors 參數分析 (如果可用)
5. 05_umap_min_dist.png             - UMAP min_dist 參數分析 (如果可用)
6. 06_mnist_runtime_comparison.png  - MNIST 運行時間對比
7. 07_mnist_visualization.png       - MNIST 降維可視化
8. 08_iris_feature_space.png        - Iris 特徵空間可視化
9. 09_clustering_visualization.png  - 聚類結果可視化
10. 10_multi_dataset_comparison.png - 多數據集對比

關鍵要點 / Key Takeaways:
• PCA 最快但只能處理線性關係
• t-SNE 可視化效果最好但速度較慢
• UMAP 在速度和質量之間取得最佳平衡
• 參數選擇對結果影響巨大
• 標準化是必須的預處理步驟
• 不同算法適合不同應用場景
""")
