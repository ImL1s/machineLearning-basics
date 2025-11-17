"""
模型評估與調參完整指南
Model Evaluation and Hyperparameter Tuning Guide

如何正確評估模型性能並優化參數
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV, learning_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                            roc_auc_score, precision_recall_curve, mean_squared_error,
                            mean_absolute_error, r2_score)
import seaborn as sns

print("=" * 80)
print("模型評估與調參完整指南".center(80))
print("=" * 80)

# ============================================================================
# Part 1: 分類模型評估
# ============================================================================
print("\n【Part 1】分類模型評估")
print("-" * 80)

# 加載數據
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 訓練基礎模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# 1. 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
print("\n1. 混淆矩陣：")
print(cm)

# 2. 分類報告
print("\n2. 分類報告：")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# 3. ROC-AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n3. ROC-AUC Score: {roc_auc:.4f}")

# 4. Precision-Recall 曲線
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)

# ============================================================================
# Part 2: 交叉驗證
# ============================================================================
print("\n【Part 2】交叉驗證")
print("-" * 80)

cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"5折交叉驗證分數：{cv_scores}")
print(f"平均準確率：{cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# Part 3: 網格搜索（Grid Search）
# ============================================================================
print("\n【Part 3】網格搜索調參")
print("-" * 80)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

print("正在進行網格搜索...")
grid_search.fit(X_train, y_train)

print(f"最佳參數：{grid_search.best_params_}")
print(f"最佳CV分數：{grid_search.best_score_:.4f}")
print(f"測試集分數：{grid_search.score(X_test, y_test):.4f}")

# ============================================================================
# Part 4: 隨機搜索（Random Search）
# ============================================================================
print("\n【Part 4】隨機搜索調參")
print("-" * 80)

from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # 嘗試20組隨機參數
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

print("正在進行隨機搜索...")
random_search.fit(X_train, y_train)

print(f"最佳參數：{random_search.best_params_}")
print(f"最佳CV分數：{random_search.best_score_:.4f}")

# ============================================================================
# Part 5: 學習曲線
# ============================================================================
print("\n【Part 5】學習曲線分析")
print("-" * 80)

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

print(f"訓練集分數：{train_mean[-1]:.4f} (+/- {train_std[-1]:.4f})")
print(f"驗證集分數：{val_mean[-1]:.4f} (+/- {val_std[-1]:.4f})")

# ============================================================================
# Part 6: 回歸模型評估
# ============================================================================
print("\n【Part 6】回歸模型評估")
print("-" * 80)

diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train_reg, y_train_reg)
y_pred_reg = gbr.predict(X_test_reg)

# 回歸評估指標
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"MSE (均方誤差): {mse:.2f}")
print(f"RMSE (均方根誤差): {rmse:.2f}")
print(f"MAE (平均絕對誤差): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# ============================================================================
# 可視化
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# 1. 混淆矩陣
ax1 = plt.subplot(3, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names, ax=ax1)
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

# 2. ROC 曲線
ax2 = plt.subplot(3, 3, 2)
ax2.plot(fpr, tpr, color='darkorange', lw=2,
        label=f'ROC curve (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Precision-Recall 曲線
ax3 = plt.subplot(3, 3, 3)
ax3.plot(recall, precision, color='green', lw=2)
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. 交叉驗證分數
ax4 = plt.subplot(3, 3, 4)
ax4.bar(range(1, len(cv_scores) + 1), cv_scores, alpha=0.7, edgecolor='black')
ax4.axhline(y=cv_scores.mean(), color='r', linestyle='--',
           label=f'Mean: {cv_scores.mean():.4f}')
ax4.set_xlabel('Fold')
ax4.set_ylabel('Accuracy')
ax4.set_title('Cross-Validation Scores', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. 網格搜索結果
ax5 = plt.subplot(3, 3, 5)
results_df = pd.DataFrame(grid_search.cv_results_)
top_results = results_df.nsmallest(10, 'rank_test_score')
ax5.barh(range(len(top_results)), top_results['mean_test_score'], alpha=0.7)
ax5.set_yticks(range(len(top_results)))
ax5.set_yticklabels([f"Config {i+1}" for i in range(len(top_results))])
ax5.set_xlabel('Mean CV Score')
ax5.set_title('Top 10 Grid Search Configurations', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')
ax5.invert_yaxis()

# 6. 學習曲線
ax6 = plt.subplot(3, 3, 6)
ax6.plot(train_sizes, train_mean, 'o-', label='Training Score')
ax6.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
ax6.plot(train_sizes, val_mean, 'o-', label='Validation Score')
ax6.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
ax6.set_xlabel('Training Size')
ax6.set_ylabel('Score')
ax6.set_title('Learning Curves', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. 回歸：預測 vs 真實值
ax7 = plt.subplot(3, 3, 7)
ax7.scatter(y_test_reg, y_pred_reg, alpha=0.6)
ax7.plot([y_test_reg.min(), y_test_reg.max()],
        [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
ax7.set_xlabel('True Values')
ax7.set_ylabel('Predictions')
ax7.set_title(f'Regression: Predictions vs True (R²={r2:.3f})',
             fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. 回歸：殘差圖
ax8 = plt.subplot(3, 3, 8)
residuals_reg = y_test_reg - y_pred_reg
ax8.scatter(y_pred_reg, residuals_reg, alpha=0.6)
ax8.axhline(y=0, color='r', linestyle='--', lw=2)
ax8.set_xlabel('Predicted Values')
ax8.set_ylabel('Residuals')
ax8.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# 9. 特徵重要性
ax9 = plt.subplot(3, 3, 9)
feature_importance = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(10)

ax9.barh(range(len(feature_importance)), feature_importance['importance'], alpha=0.7)
ax9.set_yticks(range(len(feature_importance)))
ax9.set_yticklabels(feature_importance['feature'], fontsize=8)
ax9.set_xlabel('Importance')
ax9.set_title('Top 10 Feature Importances', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='x')
ax9.invert_yaxis()

plt.tight_layout()
plt.savefig('05_ModelEvaluation/model_evaluation_results.png', dpi=150, bbox_inches='tight')
print("\n已保存結果圖表")

print("\n" + "=" * 80)
print("模型評估與調參要點總結")
print("=" * 80)
print("""
1. 分類評估指標：
   • 準確率（Accuracy）：整體正確率
   • 精確率（Precision）：預測為正中真正為正的比例
   • 召回率（Recall）：真正為正中被預測為正的比例
   • F1分數：精確率和召回率的調和平均
   • ROC-AUC：綜合評估分類性能
   • 混淆矩陣：詳細分析預測結果

2. 回歸評估指標：
   • MAE：平均絕對誤差（對異常值魯棒）
   • MSE：均方誤差（放大誤差）
   • RMSE：均方根誤差（與目標同單位）
   • R²：決定係數（解釋變異比例）

3. 交叉驗證：
   • K折交叉驗證：更可靠的性能估計
   • 分層交叉驗證：保持類別比例
   • 時間序列交叉驗證：時序數據專用

4. 超參數調優：
   • 網格搜索（Grid Search）：窮舉所有組合
   • 隨機搜索（Random Search）：隨機採樣，更快
   • 貝葉斯優化：更智能的搜索策略

5. 診斷工具：
   • 學習曲線：判斷過擬合/欠擬合
   • 驗證曲線：分析單個參數影響
   • 殘差圖：檢查回歸假設

6. 最佳實踐：
   • 始終使用獨立的測試集
   • 使用交叉驗證評估模型
   • 選擇合適的評估指標
   • 理解業務需求（精確率 vs 召回率）
   • 避免數據洩漏
   • 考慮模型的可解釋性
""")
