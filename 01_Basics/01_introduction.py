"""
機器學習基礎概念 - 入門介紹
Machine Learning Basics - Introduction

本教程涵蓋機器學習的核心概念和基本術語
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# 設置中文字體支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("機器學習基礎概念".center(80))
print("=" * 80)

# ============================================================================
# 1. 什麼是機器學習？
# ============================================================================
print("\n【1】什麼是機器學習？")
print("-" * 80)
print("""
機器學習是人工智能的一個分支，它讓計算機能夠從數據中學習規律，
而無需明確編程。機器學習的核心思想是：

• 從數據中學習模式和規律
• 基於學習到的模式進行預測或決策
• 隨著更多數據的輸入，模型性能會不斷改進

機器學習三要素：
1. 數據（Data）：訓練模型的原材料
2. 模型（Model）：學習數據規律的算法
3. 目標（Objective）：優化的目標函數
""")

# ============================================================================
# 2. 機器學習的主要類型
# ============================================================================
print("\n【2】機器學習的主要類型")
print("-" * 80)
print("""
1. 監督學習（Supervised Learning）
   - 有標籤的訓練數據
   - 分類（Classification）：預測離散標籤
     例如：郵件分類（垃圾郵件/正常郵件）
   - 回歸（Regression）：預測連續值
     例如：房價預測

2. 非監督學習（Unsupervised Learning）
   - 無標籤的訓練數據
   - 聚類（Clustering）：將相似數據分組
     例如：客戶分群
   - 降維（Dimensionality Reduction）：減少特徵數量
     例如：數據可視化

3. 強化學習（Reinforcement Learning）
   - 通過與環境互動學習
   - 通過獎勵和懲罰來優化決策
     例如：遊戲AI、機器人控制
""")

# ============================================================================
# 3. 基本術語
# ============================================================================
print("\n【3】機器學習基本術語")
print("-" * 80)
print("""
• 特徵（Features）：輸入變量，用於描述數據的屬性
• 標籤（Labels）：輸出變量，我們要預測的目標
• 樣本（Sample）：數據集中的一條記錄
• 訓練集（Training Set）：用於訓練模型的數據
• 測試集（Test Set）：用於評估模型性能的數據
• 驗證集（Validation Set）：用於調整超參數的數據
• 過擬合（Overfitting）：模型在訓練集上表現很好，但在測試集上表現差
• 欠擬合（Underfitting）：模型過於簡單，無法捕捉數據的規律
• 泛化能力（Generalization）：模型在新數據上的表現能力
""")

# ============================================================================
# 4. 實際示例：分類問題
# ============================================================================
print("\n【4】實際示例：二分類問題")
print("-" * 80)

# 生成分類數據集
X_class, y_class = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

print(f"數據形狀：{X_class.shape}")
print(f"特徵數量：{X_class.shape[1]}")
print(f"樣本數量：{X_class.shape[0]}")
print(f"類別分布：類別0有 {np.sum(y_class == 0)} 個樣本，類別1有 {np.sum(y_class == 1)} 個樣本")

# 可視化分類數據
plt.figure(figsize=(10, 6))
plt.scatter(X_class[y_class == 0, 0], X_class[y_class == 0, 1],
            c='blue', label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X_class[y_class == 1, 0], X_class[y_class == 1, 1],
            c='red', label='Class 1', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Classification Dataset Example', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_Basics/classification_example.png', dpi=150)
print("已保存分類數據可視化圖：classification_example.png")

# ============================================================================
# 5. 實際示例：回歸問題
# ============================================================================
print("\n【5】實際示例：回歸問題")
print("-" * 80)

# 生成回歸數據集
X_reg, y_reg = make_regression(
    n_samples=200,
    n_features=1,
    noise=20,
    random_state=42
)

print(f"數據形狀：{X_reg.shape}")
print(f"目標值範圍：[{y_reg.min():.2f}, {y_reg.max():.2f}]")

# 可視化回歸數據
plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, c='green', alpha=0.6, edgecolors='k')
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Target Value', fontsize=12)
plt.title('Regression Dataset Example', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_Basics/regression_example.png', dpi=150)
print("已保存回歸數據可視化圖：regression_example.png")

# ============================================================================
# 6. 數據分割：訓練集和測試集
# ============================================================================
print("\n【6】數據分割：訓練集和測試集")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class,
    test_size=0.2,  # 20% 用於測試
    random_state=42
)

print(f"原始數據集大小：{len(X_class)}")
print(f"訓練集大小：{len(X_train)} ({len(X_train)/len(X_class)*100:.1f}%)")
print(f"測試集大小：{len(X_test)} ({len(X_test)/len(X_class)*100:.1f}%)")
print("""
為什麼要分割數據？
• 訓練集用於訓練模型（讓模型學習）
• 測試集用於評估模型（測試模型在未見過的數據上的表現）
• 這樣可以檢驗模型的泛化能力，避免過擬合
""")

# ============================================================================
# 7. 機器學習工作流程
# ============================================================================
print("\n【7】機器學習典型工作流程")
print("-" * 80)
print("""
1. 問題定義
   └─ 明確要解決的問題（分類？回歸？聚類？）

2. 數據收集
   └─ 收集足夠的高質量數據

3. 數據探索與分析（EDA）
   └─ 了解數據分布、特徵關係、缺失值等

4. 數據預處理
   ├─ 處理缺失值
   ├─ 特徵縮放（標準化/歸一化）
   ├─ 特徵編碼（類別變量轉數值）
   └─ 特徵工程（創建新特徵）

5. 模型選擇
   └─ 選擇合適的算法（決策樹、SVM、神經網絡等）

6. 模型訓練
   └─ 用訓練數據訓練模型

7. 模型評估
   ├─ 在測試集上評估性能
   └─ 使用適當的評估指標（準確率、F1分數、RMSE等）

8. 模型調優
   └─ 調整超參數以提升性能

9. 模型部署
   └─ 將模型應用到實際場景

10. 監控與維護
    └─ 持續監控模型性能，必要時重新訓練
""")

# ============================================================================
# 8. 常見的評估指標
# ============================================================================
print("\n【8】常見的評估指標")
print("-" * 80)
print("""
分類問題：
• 準確率（Accuracy）：正確預測的比例
• 精確率（Precision）：預測為正的樣本中真正為正的比例
• 召回率（Recall）：真正為正的樣本中被正確預測的比例
• F1分數（F1-Score）：精確率和召回率的調和平均
• AUC-ROC：接收者操作特徵曲線下面積

回歸問題：
• MAE（Mean Absolute Error）：平均絕對誤差
• MSE（Mean Squared Error）：均方誤差
• RMSE（Root Mean Squared Error）：均方根誤差
• R²（R-squared）：決定係數，表示模型解釋變異的比例
""")

# ============================================================================
# 9. 偏差-方差權衡
# ============================================================================
print("\n【9】偏差-方差權衡（Bias-Variance Tradeoff）")
print("-" * 80)
print("""
• 偏差（Bias）：模型的預測值與真實值之間的差距
  - 高偏差 → 欠擬合（模型過於簡單）

• 方差（Variance）：模型對訓練數據的敏感程度
  - 高方差 → 過擬合（模型過於複雜）

理想情況：
  低偏差 + 低方差 = 良好的泛化能力

調整方法：
  - 增加模型複雜度 → 降低偏差，增加方差
  - 增加訓練數據 → 降低方差
  - 正則化 → 降低方差
  - 特徵選擇 → 降低方差
""")

print("\n" + "=" * 80)
print("基礎概念介紹完畢！接下來可以學習具體的算法實作。")
print("=" * 80)
