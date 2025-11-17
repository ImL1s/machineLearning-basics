"""
K-近鄰算法（K-Nearest Neighbors, KNN）
最簡單直觀的分類算法之一

原理：
- 找到距離測試樣本最近的 K 個訓練樣本
- 通過多數投票決定測試樣本的類別
- 不需要訓練過程，屬於懶惰學習（Lazy Learning）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("K-近鄰算法（KNN）教程".center(80))
print("=" * 80)

# ============================================================================
# 1. KNN 基本概念
# ============================================================================
print("\n【1】KNN 基本概念")
print("-" * 80)
print("""
KNN 算法的核心思想：
• 物以類聚：相似的樣本往往屬於同一類別
• K 值選擇：鄰居的數量
  - K 太小：容易過擬合，對噪聲敏感
  - K 太大：容易欠擬合，分類邊界過於平滑
• 距離度量：常用歐氏距離、曼哈頓距離

優點：
✓ 簡單易懂，容易實現
✓ 無需訓練過程
✓ 對異常值不敏感（當 K 較大時）
✓ 適合多分類問題

缺點：
✗ 預測速度慢（需要計算與所有訓練樣本的距離）
✗ 需要大量存儲空間
✗ 對不平衡數據集表現不佳
✗ 特徵縮放很重要
""")

# ============================================================================
# 2. 實例1：鳶尾花分類（經典三分類問題）
# ============================================================================
print("\n【2】實例1：鳶尾花分類（Iris Dataset）")
print("-" * 80)

# 加載數據
iris = load_iris()
X, y = iris.data, iris.target

print(f"數據集大小：{X.shape}")
print(f"特徵名稱：{iris.feature_names}")
print(f"類別名稱：{iris.target_names}")
print(f"各類別樣本數：{np.bincount(y)}")

# 數據分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 特徵縮放（KNN 對特徵尺度敏感！）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n訓練集大小：{X_train.shape}")
print(f"測試集大小：{X_test.shape}")

# ============================================================================
# 3. 尋找最佳 K 值
# ============================================================================
print("\n【3】尋找最佳 K 值")
print("-" * 80)

k_range = range(1, 31)
train_scores = []
test_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_scores.append(knn.score(X_train_scaled, y_train))
    test_scores.append(knn.score(X_test_scaled, y_test))

# 可視化 K 值的影響
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, train_scores, label='Training Accuracy', marker='o')
plt.plot(k_range, test_scores, label='Testing Accuracy', marker='s')
plt.xlabel('K Value', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('KNN: K Value vs Accuracy', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

best_k = k_range[np.argmax(test_scores)]
best_score = max(test_scores)
print(f"最佳 K 值：{best_k}")
print(f"最佳測試準確率：{best_score:.4f}")

# ============================================================================
# 4. 訓練最終模型
# ============================================================================
print("\n【4】訓練最終模型")
print("-" * 80)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

# 預測
y_pred = knn.predict(X_test_scaled)

# 評估
accuracy = accuracy_score(y_test, y_pred)
print(f"測試準確率：{accuracy:.4f}")

print("\n分類報告：")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('02_SupervisedLearning/Classification/knn_iris_results.png', dpi=150)
print("\n已保存結果圖表")

# ============================================================================
# 5. 實例2：非線性分類問題
# ============================================================================
print("\n【5】實例2：非線性分類問題（Moon Dataset）")
print("-" * 80)

# 生成月牙形數據
X_moon, y_moon = make_moons(n_samples=300, noise=0.2, random_state=42)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_moon, y_moon, test_size=0.3, random_state=42
)

# 訓練 KNN
knn_moon = KNeighborsClassifier(n_neighbors=15)
knn_moon.fit(X_train_m, y_train_m)

# 創建決策邊界
def plot_decision_boundary(clf, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k', alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.figure(figsize=(15, 5))

# 不同 K 值的決策邊界
for idx, k in enumerate([1, 5, 15], 1):
    plt.subplot(1, 3, idx)
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_m, y_train_m)
    score = knn_temp.score(X_test_m, y_test_m)
    plot_decision_boundary(knn_temp, X_moon, y_moon,
                          f'KNN (K={k}), Accuracy={score:.3f}')

plt.tight_layout()
plt.savefig('02_SupervisedLearning/Classification/knn_decision_boundary.png', dpi=150)
print("已保存決策邊界圖")

# ============================================================================
# 6. 交叉驗證
# ============================================================================
print("\n【6】交叉驗證")
print("-" * 80)

knn_cv = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn_cv, X, y, cv=5)

print(f"5折交叉驗證分數：{cv_scores}")
print(f"平均分數：{cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# 7. 距離度量的影響
# ============================================================================
print("\n【7】不同距離度量的比較")
print("-" * 80)

metrics = ['euclidean', 'manhattan', 'minkowski']
for metric in metrics:
    knn_metric = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn_metric.fit(X_train_scaled, y_train)
    score = knn_metric.score(X_test_scaled, y_test)
    print(f"{metric:12s} 距離準確率：{score:.4f}")

# ============================================================================
# 8. 實際應用示例
# ============================================================================
print("\n【8】實際應用：預測新樣本")
print("-" * 80)

# 創建新樣本（花萼長度=5.1, 花萼寬度=3.5, 花瓣長度=1.4, 花瓣寬度=0.2）
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample_scaled = scaler.transform(new_sample)

# 預測
prediction = knn.predict(new_sample_scaled)
probabilities = knn.predict_proba(new_sample_scaled)

print(f"新樣本特徵：{new_sample[0]}")
print(f"預測類別：{iris.target_names[prediction[0]]}")
print(f"預測概率：")
for i, prob in enumerate(probabilities[0]):
    print(f"  {iris.target_names[i]:12s}: {prob:.4f}")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("KNN 算法要點總結")
print("=" * 80)
print("""
1. K 值選擇：
   • 通過交叉驗證選擇最佳 K 值
   • 一般選擇奇數避免平票
   • 常用 K 值範圍：3-10

2. 特徵縮放：
   • 必須進行特徵標準化或歸一化
   • 否則大數值特徵會主導距離計算

3. 距離度量：
   • 歐氏距離（最常用）
   • 曼哈頓距離
   • 閔可夫斯基距離

4. 適用場景：
   • 樣本量不太大的分類問題
   • 特徵數量適中
   • 類別邊界不規則的問題

5. 優化建議：
   • 使用 KD-Tree 或 Ball-Tree 加速搜索
   • 特徵選擇去除無關特徵
   • 數據預處理去除噪聲
""")
