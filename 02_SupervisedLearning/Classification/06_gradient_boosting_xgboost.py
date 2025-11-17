"""
梯度提升（Gradient Boosting）與 XGBoost
最強大的機器學習算法之一，Kaggle 競賽常勝軍

包括：
- Gradient Boosting (scikit-learn)
- XGBoost (極限梯度提升)
- LightGBM (輕量級梯度提升)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

# 嘗試導入 XGBoost 和 LightGBM
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("警告：XGBoost 未安裝，部分功能將跳過")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("警告：LightGBM 未安裝，部分功能將跳過")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("梯度提升算法完整指南".center(80))
print("=" * 80)

# ============================================================================
# 1. Gradient Boosting 基本概念
# ============================================================================
print("\n【1】Gradient Boosting 基本概念")
print("-" * 80)
print("""
梯度提升核心思想：
• Boosting：串行集成學習，後一個模型修正前一個模型的錯誤
• 梯度下降：在函數空間中進行梯度下降優化
• 弱學習器：通常使用決策樹（CART）

主要算法比較：

1. Gradient Boosting (scikit-learn)：
   • 經典實現，穩定可靠
   • 支持多種損失函數
   • 訓練較慢
   • 適合小到中等規模數據

2. XGBoost (eXtreme Gradient Boosting)：
   • 速度快，性能強
   • 支持並行計算
   • 內置正則化
   • Kaggle 最受歡迎
   • 支持缺失值處理

3. LightGBM (Light Gradient Boosting Machine)：
   • 微軟出品，速度最快
   • 適合大規模數據
   • 基於直方圖的算法
   • 葉子優先生長策略

關鍵參數：
• n_estimators：樹的數量（100-1000）
• learning_rate：學習率（0.01-0.3）
• max_depth：樹的深度（3-10）
• subsample：樣本採樣比例
• colsample_bytree：特徵採樣比例

優點：
✓ 預測性能極強
✓ 可處理各種類型數據
✓ 特徵重要性分析
✓ 魯棒性好
✓ 支持自定義損失函數

缺點：
✗ 訓練時間長
✗ 參數調優複雜
✗ 容易過擬合（需要調參）
✗ 模型解釋性較差
""")

# ============================================================================
# 2. 數據準備
# ============================================================================
print("\n【2】數據準備")
print("-" * 80)

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"訓練集大小：{X_train.shape}")
print(f"測試集大小：{X_test.shape}")
print(f"特徵數量：{X.shape[1]}")

# ============================================================================
# 3. Gradient Boosting (scikit-learn)
# ============================================================================
print("\n【3】Gradient Boosting Classifier")
print("-" * 80)

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]

accuracy_gb = gb.score(X_test, y_test)
roc_auc_gb = roc_auc_score(y_test, y_pred_proba_gb)

print(f"測試準確率：{accuracy_gb:.4f}")
print(f"ROC-AUC：{roc_auc_gb:.4f}")

print("\n分類報告：")
print(classification_report(y_test, y_pred_gb, target_names=cancer.target_names))

# ============================================================================
# 4. XGBoost
# ============================================================================
if XGB_AVAILABLE:
    print("\n【4】XGBoost Classifier")
    print("-" * 80)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric='logloss'
    )

    xgb_clf.fit(X_train, y_train)

    y_pred_xgb = xgb_clf.predict(X_test)
    y_pred_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]

    accuracy_xgb = xgb_clf.score(X_test, y_test)
    roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

    print(f"測試準確率：{accuracy_xgb:.4f}")
    print(f"ROC-AUC：{roc_auc_xgb:.4f}")

# ============================================================================
# 5. LightGBM
# ============================================================================
if LGB_AVAILABLE:
    print("\n【5】LightGBM Classifier")
    print("-" * 80)

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbosity=-1
    )

    lgb_clf.fit(X_train, y_train)

    y_pred_lgb = lgb_clf.predict(X_test)
    y_pred_proba_lgb = lgb_clf.predict_proba(X_test)[:, 1]

    accuracy_lgb = lgb_clf.score(X_test, y_test)
    roc_auc_lgb = roc_auc_score(y_test, y_pred_proba_lgb)

    print(f"測試準確率：{accuracy_lgb:.4f}")
    print(f"ROC-AUC：{roc_auc_lgb:.4f}")

# ============================================================================
# 6. 特徵重要性
# ============================================================================
print("\n【6】特徵重要性分析")
print("-" * 80)

feature_importance_gb = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)

print("\n前10個最重要特徵（Gradient Boosting）：")
print(feature_importance_gb.head(10).to_string(index=False))

# ============================================================================
# 7. 學習率影響
# ============================================================================
print("\n【7】學習率對性能的影響")
print("-" * 80)

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
scores_lr = []

for lr in learning_rates:
    gb_lr = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )
    gb_lr.fit(X_train, y_train)
    score = gb_lr.score(X_test, y_test)
    scores_lr.append(score)
    print(f"Learning Rate = {lr:.2f}: Accuracy = {score:.4f}")

# ============================================================================
# 8. 樹的數量影響
# ============================================================================
print("\n【8】樹的數量對性能的影響")
print("-" * 80)

n_estimators_range = range(10, 201, 10)
train_scores_ne = []
test_scores_ne = []

for n in n_estimators_range:
    gb_ne = GradientBoostingClassifier(
        n_estimators=n,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_ne.fit(X_train, y_train)
    train_scores_ne.append(gb_ne.score(X_train, y_train))
    test_scores_ne.append(gb_ne.score(X_test, y_test))

best_n = n_estimators_range[np.argmax(test_scores_ne)]
print(f"最佳樹數量：{best_n}")
print(f"最佳測試準確率：{max(test_scores_ne):.4f}")

# ============================================================================
# 可視化
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# 1. 混淆矩陣
ax1 = plt.subplot(3, 3, 1)
cm_gb = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names, ax=ax1)
ax1.set_title('Gradient Boosting\nConfusion Matrix', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

# 2. ROC 曲線
ax2 = plt.subplot(3, 3, 2)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_proba_gb)
ax2.plot(fpr_gb, tpr_gb, label=f'GB (AUC={roc_auc_gb:.3f})', lw=2)

if XGB_AVAILABLE:
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
    ax2.plot(fpr_xgb, tpr_xgb, label=f'XGB (AUC={roc_auc_xgb:.3f})', lw=2)

if LGB_AVAILABLE:
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_pred_proba_lgb)
    ax2.plot(fpr_lgb, tpr_lgb, label=f'LGB (AUC={roc_auc_lgb:.3f})', lw=2)

ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 特徵重要性
ax3 = plt.subplot(3, 3, 3)
top_features = feature_importance_gb.head(15)
ax3.barh(range(len(top_features)), top_features['importance'], alpha=0.7)
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features['feature'], fontsize=8)
ax3.set_xlabel('Importance')
ax3.set_title('Top 15 Feature Importances', fontsize=12, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# 4. 學習率影響
ax4 = plt.subplot(3, 3, 4)
ax4.plot(learning_rates, scores_lr, marker='o', linewidth=2, markersize=8)
ax4.set_xlabel('Learning Rate')
ax4.set_ylabel('Accuracy')
ax4.set_title('Learning Rate vs Accuracy', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. 樹數量影響
ax5 = plt.subplot(3, 3, 5)
ax5.plot(n_estimators_range, train_scores_ne, label='Training', marker='o', markersize=4)
ax5.plot(n_estimators_range, test_scores_ne, label='Testing', marker='s', markersize=4)
ax5.set_xlabel('Number of Trees')
ax5.set_ylabel('Accuracy')
ax5.set_title('Number of Trees vs Accuracy', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 模型性能比較
ax6 = plt.subplot(3, 3, 6)
models = ['GB']
accuracies = [accuracy_gb]

if XGB_AVAILABLE:
    models.append('XGBoost')
    accuracies.append(accuracy_xgb)

if LGB_AVAILABLE:
    models.append('LightGBM')
    accuracies.append(accuracy_lgb)

colors_models = ['blue', 'green', 'orange'][:len(models)]
bars = ax6.bar(models, accuracies, color=colors_models, alpha=0.7, edgecolor='black')
ax6.set_ylabel('Accuracy')
ax6.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax6.set_ylim([0.9, 1.0])
ax6.grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.4f}', ha='center', va='bottom')

# 7. 訓練過程（損失曲線）
ax7 = plt.subplot(3, 3, 7)
train_scores_cumsum = np.cumsum(gb.train_score_)
ax7.plot(range(1, len(train_scores_cumsum) + 1), train_scores_cumsum,
        label='Cumulative Training Score', linewidth=2)
ax7.set_xlabel('Boosting Iteration')
ax7.set_ylabel('Cumulative Score')
ax7.set_title('Training Process', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. 預測概率分布
ax8 = plt.subplot(3, 3, 8)
ax8.hist(y_pred_proba_gb[y_test == 0], bins=30, alpha=0.7,
        label='Malignant (0)', color='red', edgecolor='black')
ax8.hist(y_pred_proba_gb[y_test == 1], bins=30, alpha=0.7,
        label='Benign (1)', color='green', edgecolor='black')
ax8.set_xlabel('Predicted Probability')
ax8.set_ylabel('Frequency')
ax8.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# 9. XGBoost 特徵重要性（如果可用）
if XGB_AVAILABLE:
    ax9 = plt.subplot(3, 3, 9)
    xgb.plot_importance(xgb_clf, ax=ax9, max_num_features=15, importance_type='weight')
    ax9.set_title('XGBoost Feature Importance', fontsize=12, fontweight='bold')
else:
    ax9 = plt.subplot(3, 3, 9)
    ax9.text(0.5, 0.5, 'XGBoost not installed\npip install xgboost',
            ha='center', va='center', fontsize=12)
    ax9.axis('off')

plt.tight_layout()
plt.savefig('02_SupervisedLearning/Classification/06_gradient_boosting_results.png',
            dpi=150, bbox_inches='tight')
print("\n已保存結果圖表")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("Gradient Boosting 要點總結")
print("=" * 80)
print("""
1. 三種主要實現比較：
   • Gradient Boosting：穩定，適合小數據
   • XGBoost：速度快，性能強，最常用
   • LightGBM：最快，適合大數據

2. 關鍵參數調優順序：
   ① max_depth：從3開始，通常3-10
   ② n_estimators：從100開始，逐漸增加
   ③ learning_rate：0.01-0.3，與 n_estimators 權衡
   ④ subsample：0.5-1.0
   ⑤ colsample_bytree：0.5-1.0

3. 防止過擬合：
   • 降低 learning_rate，增加 n_estimators
   • 增加 min_samples_split
   • 增加正則化（XGBoost 的 reg_alpha, reg_lambda）
   • 使用 early_stopping（提前停止）
   • 減小 max_depth

4. 性能優化：
   • 並行訓練（XGBoost, LightGBM）
   • GPU 加速（XGBoost, LightGBM）
   • 特徵工程很重要
   • 缺失值處理（XGBoost 自動處理）

5. 適用場景：
   • Kaggle 競賽
   • 表格數據分類/回歸
   • 需要高準確率的場景
   • 中小規模結構化數據

6. 最佳實踐：
   • 使用交叉驗證
   • 設置 early_stopping_rounds
   • 保存最佳模型
   • 特徵重要性分析
   • 調參使用 GridSearchCV 或 Optuna

7. 常見配置：

   # 快速原型
   XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

   # 生產環境
   XGBClassifier(
       n_estimators=1000,
       learning_rate=0.01,
       max_depth=5,
       subsample=0.8,
       colsample_bytree=0.8,
       early_stopping_rounds=50
   )

   # 大數據
   LGBMClassifier(
       n_estimators=1000,
       learning_rate=0.05,
       num_leaves=31,
       n_jobs=-1
   )
""")

if not XGB_AVAILABLE:
    print("\n提示：安裝 XGBoost 以獲得更好的性能:")
    print("pip install xgboost")

if not LGB_AVAILABLE:
    print("\n提示：安裝 LightGBM 以處理大規模數據:")
    print("pip install lightgbm")
