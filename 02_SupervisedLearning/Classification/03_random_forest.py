"""
隨機森林（Random Forest）
強大的集成學習算法，基於決策樹

原理：
- 建立多個決策樹（森林）
- 每棵樹使用隨機樣本和隨機特徵訓練
- 通過投票（分類）或平均（回歸）得到最終結果
- Bagging（Bootstrap Aggregating）思想
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("隨機森林（Random Forest）教程".center(80))
print("=" * 80)

# ============================================================================
# 1. Random Forest 基本概念
# ============================================================================
print("\n【1】Random Forest 基本概念")
print("-" * 80)
print("""
隨機森林核心思想：
• 集成學習：三個臭皮匠勝過一個諸葛亮
• Bagging：自助採樣聚合
• 隨機性：
  1. 數據隨機：每棵樹使用隨機抽取的樣本（有放回抽樣）
  2. 特徵隨機：每次分裂時隨機選擇特徵子集

關鍵參數：
1. n_estimators：樹的數量
   - 越多越好，但計算時間增加
   - 通常 100-500

2. max_depth：樹的最大深度
   - 控制過擬合
   - None 表示不限制

3. min_samples_split：分裂所需的最小樣本數
   - 增大可以防止過擬合

4. min_samples_leaf：葉節點最小樣本數
   - 增大可以平滑模型

5. max_features：分裂時考慮的最大特徵數
   - 'sqrt'：√n_features（分類默認）
   - 'log2'：log2(n_features)
   - None：n_features

優點：
✓ 準確率高，泛化能力強
✓ 可處理高維數據
✓ 可以評估特徵重要性
✓ 對缺失值和異常值不敏感
✓ 不需要特徵縮放
✓ 可以並行訓練

缺點：
✗ 模型較大，預測速度慢
✗ 不適合稀疏數據
✗ 難以解釋
""")

# ============================================================================
# 2. 實例1：葡萄酒分類
# ============================================================================
print("\n【2】實例1：葡萄酒分類（Wine Dataset）")
print("-" * 80)

# 加載數據
wine = load_wine()
X, y = wine.data, wine.target

print(f"數據集大小：{X.shape}")
print(f"特徵名稱：{wine.feature_names}")
print(f"類別：{wine.target_names}")
print(f"各類別樣本數：{np.bincount(y)}")

# 數據分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n訓練集大小：{X_train.shape}")
print(f"測試集大小：{X_test.shape}")

# ============================================================================
# 3. 訓練基本模型
# ============================================================================
print("\n【3】訓練隨機森林模型")
print("-" * 80)

# 創建隨機森林
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1  # 使用所有 CPU
)

rf.fit(X_train, y_train)

# 預測
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"測試準確率：{accuracy:.4f}")
print(f"森林中的樹數量：{len(rf.estimators_)}")

# ============================================================================
# 4. 特徵重要性
# ============================================================================
print("\n【4】特徵重要性分析")
print("-" * 80)

feature_importance = pd.DataFrame({
    'feature': wine.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特徵重要性排名：")
print(feature_importance.to_string(index=False))

# 可視化特徵重要性
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance', fontsize=12)
plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# ============================================================================
# 5. 模型評估
# ============================================================================
print("\n【5】模型評估")
print("-" * 80)

print("\n分類報告：")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)

plt.subplot(2, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=wine.target_names,
            yticklabels=wine.target_names)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# ============================================================================
# 6. 樹的數量的影響
# ============================================================================
print("\n【6】樹的數量對性能的影響")
print("-" * 80)

n_trees = range(10, 201, 10)
train_scores = []
test_scores = []

for n in n_trees:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    train_scores.append(rf_temp.score(X_train, y_train))
    test_scores.append(rf_temp.score(X_test, y_test))

plt.subplot(2, 3, 3)
plt.plot(n_trees, train_scores, label='Training Accuracy', marker='o', markersize=3)
plt.plot(n_trees, test_scores, label='Testing Accuracy', marker='s', markersize=3)
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Number of Trees vs Accuracy', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

print(f"最優樹數量：{n_trees[np.argmax(test_scores)]}")
print(f"最佳測試準確率：{max(test_scores):.4f}")

# ============================================================================
# 7. 最大深度的影響
# ============================================================================
print("\n【7】最大深度對性能的影響")
print("-" * 80)

max_depths = range(1, 21)
train_scores_depth = []
test_scores_depth = []

for depth in max_depths:
    rf_depth = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42,
        n_jobs=-1
    )
    rf_depth.fit(X_train, y_train)
    train_scores_depth.append(rf_depth.score(X_train, y_train))
    test_scores_depth.append(rf_depth.score(X_test, y_test))

plt.subplot(2, 3, 4)
plt.plot(max_depths, train_scores_depth, label='Training Accuracy', marker='o')
plt.plot(max_depths, test_scores_depth, label='Testing Accuracy', marker='s')
plt.xlabel('Max Depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Max Depth vs Accuracy', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# ============================================================================
# 8. Out-of-Bag (OOB) 評估
# ============================================================================
print("\n【8】Out-of-Bag (OOB) 評估")
print("-" * 80)

rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # 啟用 OOB 評估
    random_state=42,
    n_jobs=-1
)

rf_oob.fit(X_train, y_train)

print(f"OOB Score：{rf_oob.oob_score_:.4f}")
print(f"測試準確率：{rf_oob.score(X_test, y_test):.4f}")
print("""
OOB 評估說明：
• 每棵樹只使用約 2/3 的訓練數據
• 剩餘的 1/3 數據稱為 Out-of-Bag 樣本
• 可以用 OOB 樣本評估模型，無需額外的驗證集
""")

# ============================================================================
# 9. 學習曲線
# ============================================================================
print("\n【9】學習曲線分析")
print("-" * 80)

train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores_lc, axis=1)
train_std = np.std(train_scores_lc, axis=1)
val_mean = np.mean(val_scores_lc, axis=1)
val_std = np.std(val_scores_lc, axis=1)

plt.subplot(2, 3, 5)
plt.plot(train_sizes, train_mean, label='Training Score', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
plt.plot(train_sizes, val_mean, label='Validation Score', marker='s')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
plt.xlabel('Training Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Learning Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# ============================================================================
# 10. 單棵樹可視化
# ============================================================================
print("\n【10】隨機森林中的單棵樹可視化")
print("-" * 80)

plt.subplot(2, 3, 6)
# 可視化第一棵樹（限制深度以便查看）
rf_visual = RandomForestClassifier(
    n_estimators=1,
    max_depth=3,
    random_state=42
)
rf_visual.fit(X_train, y_train)

plot_tree(rf_visual.estimators_[0],
          feature_names=wine.feature_names,
          class_names=wine.target_names,
          filled=True,
          fontsize=6,
          rounded=True)
plt.title('Decision Tree Example\n(1st tree, max_depth=3)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('02_SupervisedLearning/Classification/random_forest_results.png', dpi=150, bbox_inches='tight')
print("已保存結果圖表")

# ============================================================================
# 11. 與單棵決策樹比較
# ============================================================================
print("\n【11】隨機森林 vs 單棵決策樹")
print("-" * 80)

from sklearn.tree import DecisionTreeClassifier

# 單棵決策樹
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)

# 隨機森林
rf_compare = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_compare.fit(X_train, y_train)
rf_score = rf_compare.score(X_test, y_test)

print(f"單棵決策樹準確率：{dt_score:.4f}")
print(f"隨機森林準確率：{rf_score:.4f}")
print(f"性能提升：{(rf_score - dt_score) * 100:.2f}%")

# ============================================================================
# 12. 交叉驗證
# ============================================================================
print("\n【12】交叉驗證")
print("-" * 80)

cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X, y,
    cv=5,
    scoring='accuracy'
)

print(f"5折交叉驗證分數：{cv_scores}")
print(f"平均準確率：{cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("Random Forest 算法要點總結")
print("=" * 80)
print("""
1. 參數調優建議：
   • n_estimators：從 100 開始，增加到穩定為止
   • max_depth：None（不限制）或根據數據調整
   • min_samples_split：2-10（防止過擬合）
   • max_features：'sqrt'（分類）或 'log2'

2. 特徵重要性：
   • Random Forest 可以自然地評估特徵重要性
   • 用於特徵選擇和理解數據

3. OOB 評估：
   • 無需額外的驗證集
   • 可以作為交叉驗證的替代

4. 適用場景：
   • 表格數據的分類和回歸
   • 需要特徵重要性分析
   • 數據量中等到大規模
   • 對準確率要求較高

5. 優化技巧：
   • 使用 n_jobs=-1 並行訓練
   • 合理設置 max_depth 防止過擬合
   • 不需要特徵縮放
   • 可以處理缺失值（部分實現）

6. 與其他算法比較：
   • vs 決策樹：更穩定，準確率更高
   • vs XGBoost：訓練速度快，但準確率可能稍低
   • vs SVM：處理大規模數據更快
   • vs 神經網絡：更易解釋，訓練簡單
""")
