"""
處理不平衡數據完整指南
Handling Imbalanced Data

實際工作中經常遇到的問題：
- 欺詐檢測（欺詐樣本很少）
- 疾病診斷（患病樣本很少）
- 異常檢測（異常樣本很少）

學會如何處理這類問題！
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            precision_recall_curve, roc_auc_score, f1_score,
                            precision_score, recall_score)
from sklearn.utils import resample
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("處理不平衡數據完整指南".center(80))
print("=" * 80)

# ============================================================================
# 1. 什麼是不平衡數據？
# ============================================================================
print("\n【1】什麼是不平衡數據？")
print("-" * 80)
print("""
不平衡數據：
• 一個類別的樣本數遠多於另一個類別
• 例如：正樣本 95%，負樣本 5%

為什麼是問題：
• 模型傾向於預測多數類
• 準確率高但沒有意義
• 少數類（往往是重要類）被忽略

常見場景：
• 欺詐檢測（欺詐交易 < 1%）
• 疾病診斷（患病率 5-10%）
• 點擊率預測（點擊率 1-2%）
• 設備故障（故障率 < 5%）
• 客戶流失（流失率 10-20%）

解決方法分類：
1. 數據層面：重採樣（過採樣、欠採樣）
2. 算法層面：調整類別權重、使用不同算法
3. 評估層面：使用合適的評估指標
""")

# ============================================================================
# 2. 創建不平衡數據集
# ============================================================================
print("\n【2】創建不平衡數據集示例")
print("-" * 80)

# 創建高度不平衡的數據集
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.95, 0.05],  # 95% vs 5% 的不平衡
    random_state=42
)

print(f"數據集大小：{X.shape}")
print(f"類別分布：")
print(f"  類別 0（多數類）：{np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
print(f"  類別 1（少數類）：{np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
print(f"不平衡比例：{np.sum(y == 0) / np.sum(y == 1):.1f}:1")

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================================
# 3. 錯誤的做法：只看準確率
# ============================================================================
print("\n【3】錯誤示例：只關注準確率")
print("-" * 80)

# 訓練基礎模型
baseline_model = LogisticRegression(random_state=42, max_iter=1000)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

# 評估
accuracy = baseline_model.score(X_test, y_test)
print(f"準確率：{accuracy:.4f}  ← 看起來很高！")

# 但看看混淆矩陣
cm = confusion_matrix(y_test, y_pred_baseline)
print(f"\n混淆矩陣：")
print(cm)
print(f"\n問題：模型可能全預測為多數類！")

# 詳細報告
print("\n分類報告：")
print(classification_report(y_test, y_pred_baseline,
                          target_names=['Majority (0)', 'Minority (1)']))

# ============================================================================
# 4. 方法1：調整類別權重（算法層面）
# ============================================================================
print("\n【4】方法1：調整類別權重（class_weight='balanced'）")
print("-" * 80)

# 使用 class_weight='balanced'
weighted_model = LogisticRegression(
    class_weight='balanced',  # 自動計算權重
    random_state=42,
    max_iter=1000
)

weighted_model.fit(X_train, y_train)
y_pred_weighted = weighted_model.predict(X_test)

print(f"準確率：{weighted_model.score(X_test, y_test):.4f}")
print(f"F1 分數：{f1_score(y_test, y_pred_weighted):.4f}")
print(f"召回率（Recall）：{recall_score(y_test, y_pred_weighted):.4f}")
print(f"精確率（Precision）：{precision_score(y_test, y_pred_weighted):.4f}")

print("\n分類報告：")
print(classification_report(y_test, y_pred_weighted,
                          target_names=['Majority (0)', 'Minority (1)']))

# ============================================================================
# 5. 方法2：過採樣（Over-sampling）
# ============================================================================
print("\n【5】方法2：過採樣少數類")
print("-" * 80)

# 分離多數類和少數類
X_train_majority = X_train[y_train == 0]
X_train_minority = X_train[y_train == 1]
y_train_majority = y_train[y_train == 0]
y_train_minority = y_train[y_train == 1]

print(f"原始訓練集：")
print(f"  多數類：{len(X_train_majority)}")
print(f"  少數類：{len(X_train_minority)}")

# 過採樣少數類（有放回抽樣）
X_minority_upsampled, y_minority_upsampled = resample(
    X_train_minority,
    y_train_minority,
    n_samples=len(X_train_majority),  # 採樣到與多數類相同
    replace=True,  # 有放回抽樣
    random_state=42
)

# 合併
X_train_balanced = np.vstack([X_train_majority, X_minority_upsampled])
y_train_balanced = np.hstack([y_train_majority, y_minority_upsampled])

print(f"\n過採樣後：")
print(f"  多數類：{np.sum(y_train_balanced == 0)}")
print(f"  少數類：{np.sum(y_train_balanced == 1)}")

# 訓練模型
oversampled_model = LogisticRegression(random_state=42, max_iter=1000)
oversampled_model.fit(X_train_balanced, y_train_balanced)
y_pred_oversampled = oversampled_model.predict(X_test)

print(f"\n過採樣模型性能：")
print(f"F1 分數：{f1_score(y_test, y_pred_oversampled):.4f}")
print(f"召回率：{recall_score(y_test, y_pred_oversampled):.4f}")

# ============================================================================
# 6. 方法3：欠採樣（Under-sampling）
# ============================================================================
print("\n【6】方法3：欠採樣多數類")
print("-" * 80)

# 欠採樣多數類
X_majority_downsampled, y_majority_downsampled = resample(
    X_train_majority,
    y_train_majority,
    n_samples=len(X_train_minority),  # 採樣到與少數類相同
    replace=False,  # 無放回抽樣
    random_state=42
)

# 合併
X_train_undersampled = np.vstack([X_majority_downsampled, X_train_minority])
y_train_undersampled = np.hstack([y_majority_downsampled, y_train_minority])

print(f"欠採樣後：")
print(f"  多數類：{np.sum(y_train_undersampled == 0)}")
print(f"  少數類：{np.sum(y_train_undersampled == 1)}")

# 訓練模型
undersampled_model = LogisticRegression(random_state=42, max_iter=1000)
undersampled_model.fit(X_train_undersampled, y_train_undersampled)
y_pred_undersampled = undersampled_model.predict(X_test)

print(f"\n欠採樣模型性能：")
print(f"F1 分數：{f1_score(y_test, y_pred_undersampled):.4f}")
print(f"召回率：{recall_score(y_test, y_pred_undersampled):.4f}")

# ============================================================================
# 7. 方法4：SMOTE（Synthetic Minority Over-sampling）
# ============================================================================
print("\n【7】方法4：SMOTE（合成少數類過採樣）")
print("-" * 80)

try:
    from imblearn.over_sampling import SMOTE

    # 使用 SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"SMOTE 後：")
    print(f"  多數類：{np.sum(y_train_smote == 0)}")
    print(f"  少數類：{np.sum(y_train_smote == 1)}")

    # 訓練模型
    smote_model = LogisticRegression(random_state=42, max_iter=1000)
    smote_model.fit(X_train_smote, y_train_smote)
    y_pred_smote = smote_model.predict(X_test)

    print(f"\nSMOTE 模型性能：")
    print(f"F1 分數：{f1_score(y_test, y_pred_smote):.4f}")
    print(f"召回率：{recall_score(y_test, y_pred_smote):.4f}")

    SMOTE_AVAILABLE = True
except ImportError:
    print("SMOTE 需要 imbalanced-learn 套件")
    print("安裝：pip install imbalanced-learn")
    SMOTE_AVAILABLE = False
    y_pred_smote = y_pred_weighted  # 備用

# ============================================================================
# 8. 方法5：集成方法（Random Forest with balanced）
# ============================================================================
print("\n【8】方法5：使用 Random Forest + class_weight")
print("-" * 80)

rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

rf_balanced.fit(X_train, y_train)
y_pred_rf = rf_balanced.predict(X_test)

print(f"Random Forest 性能：")
print(f"F1 分數：{f1_score(y_test, y_pred_rf):.4f}")
print(f"召回率：{recall_score(y_test, y_pred_rf):.4f}")

# ============================================================================
# 9. 評估指標對比
# ============================================================================
print("\n【9】不同方法的評估指標對比")
print("-" * 80)

methods = {
    'Baseline': y_pred_baseline,
    'Class Weight': y_pred_weighted,
    'Over-sampling': y_pred_oversampled,
    'Under-sampling': y_pred_undersampled,
    'SMOTE': y_pred_smote if SMOTE_AVAILABLE else y_pred_weighted,
    'RF Balanced': y_pred_rf
}

results = []
for name, y_pred in methods.items():
    results.append({
        'Method': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print("\n評估指標比較：")
print(results_df.to_string(index=False))

# ============================================================================
# 可視化
# ============================================================================
fig = plt.figure(figsize=(18, 12))

# 1. 原始數據分布
ax1 = plt.subplot(3, 3, 1)
ax1.bar(['Majority\n(95%)', 'Minority\n(5%)'],
       [np.sum(y == 0), np.sum(y == 1)],
       color=['blue', 'red'], alpha=0.7)
ax1.set_ylabel('Count')
ax1.set_title('Original Imbalanced Data', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 2. 各方法性能比較（F1 Score）
ax2 = plt.subplot(3, 3, 2)
bars = ax2.bar(range(len(results_df)), results_df['F1'],
              color=['gray', 'blue', 'green', 'orange', 'purple', 'red'], alpha=0.7)
ax2.set_xticks(range(len(results_df)))
ax2.set_xticklabels(results_df['Method'], rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('F1 Score')
ax2.set_title('F1 Score Comparison', fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3, axis='y')
for bar, f1 in zip(bars, results_df['F1']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{f1:.3f}', ha='center', va='bottom', fontsize=8)

# 3. Precision vs Recall
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(results_df['Recall'], results_df['Precision'],
           s=200, alpha=0.6, c=range(len(results_df)), cmap='viridis')
for i, method in enumerate(results_df['Method']):
    ax3.annotate(method, (results_df['Recall'].iloc[i], results_df['Precision'].iloc[i]),
                fontsize=8, ha='right')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision vs Recall', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])

# 4-9. 各方法的混淆矩陣
for idx, (name, y_pred) in enumerate(methods.items(), 4):
    if idx > 9:
        break
    ax = plt.subplot(3, 3, idx)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=['Maj', 'Min'],
               yticklabels=['Maj', 'Min'])
    ax.set_title(f'{name}\nConfusion Matrix', fontweight='bold', fontsize=10)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('04_FeatureEngineering/imbalanced_data_results.png',
           dpi=150, bbox_inches='tight')
print("\n✓ 已保存結果圖表")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("處理不平衡數據要點總結")
print("=" * 80)
print("""
1. 識別不平衡：
   • 檢查類別分布
   • 不平衡比例 > 3:1 需要注意
   • 不平衡比例 > 10:1 必須處理

2. 評估指標選擇：
   ❌ 不要只看準確率！
   ✓ 使用 F1 分數（精確率和召回率的調和平均）
   ✓ 使用 Precision（精確率）- 預測為正的準確性
   ✓ 使用 Recall（召回率）- 找出所有正樣本的能力
   ✓ 使用 ROC-AUC
   ✓ 查看混淆矩陣

3. 處理方法：

   方法1：調整類別權重（推薦）
   • class_weight='balanced'
   • 簡單有效
   • 不改變數據

   方法2：過採樣（Over-sampling）
   • 複製少數類樣本
   • 可能過擬合
   • SMOTE 更好（生成合成樣本）

   方法3：欠採樣（Under-sampling）
   • 刪除多數類樣本
   • 可能丟失信息
   • 數據量大時可用

   方法4：SMOTE
   • 合成新的少數類樣本
   • 避免簡單複製
   • 推薦使用

   方法5：集成方法
   • Random Forest + balanced
   • XGBoost + scale_pos_weight
   • 通常效果最好

4. 選擇建議：
   • 數據量小：SMOTE 或 class_weight
   • 數據量大：class_weight 或 under-sampling
   • 追求性能：集成方法 + SMOTE
   • 生產環境：class_weight（簡單穩定）

5. 注意事項：
   • 分層抽樣（stratify=y）
   • 只在訓練集上重採樣
   • 測試集保持原始分布
   • 根據業務選擇 Precision vs Recall
   • 考慮誤分類的成本

6. 實際案例：
   • 欺詐檢測：高 Recall（不漏過欺詐）
   • 垃圾郵件：高 Precision（不誤殺正常郵件）
   • 疾病診斷：高 Recall（不漏診）
   • 推薦系統：平衡 Precision 和 Recall

7. 代碼模板：
   # 方法1：class_weight
   model = LogisticRegression(class_weight='balanced')

   # 方法2：SMOTE
   from imblearn.over_sampling import SMOTE
   smote = SMOTE()
   X_train, y_train = smote.fit_resample(X_train, y_train)

   # 方法3：Random Forest
   rf = RandomForestClassifier(class_weight='balanced')

   # 評估
   from sklearn.metrics import f1_score, recall_score
   print(f'F1: {f1_score(y_test, y_pred):.3f}')
   print(f'Recall: {recall_score(y_test, y_pred):.3f}')
""")

if not SMOTE_AVAILABLE:
    print("\n提示：安裝 imbalanced-learn 以使用 SMOTE:")
    print("pip install imbalanced-learn")
