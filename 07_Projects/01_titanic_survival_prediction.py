"""
實戰項目1：泰坦尼克號生存預測
Titanic Survival Prediction

這是一個完整的機器學習項目流程示例，包括：
1. 數據探索和分析（EDA）
2. 數據預處理
3. 特徵工程
4. 模型訓練和評估
5. 模型優化
6. 結果解釋
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import RANDOM_STATE, TEST_SIZE, DPI, setup_chinese_fonts, save_figure, get_output_path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

setup_chinese_fonts()

print("=" * 90)
print("實戰項目：泰坦尼克號生存預測".center(90))
print("=" * 90)

# ============================================================================
# 1. 創建示例數據（模擬泰坦尼克號數據集）
# ============================================================================
print("\n【第1步】創建/加載數據")
print("-" * 90)

# 創建模擬的泰坦尼克號數據
np.random.seed(RANDOM_STATE)
n_samples = 891

# 生成模擬數據
data = pd.DataFrame({
    'PassengerId': range(1, n_samples + 1),
    'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
    'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
    'Age': np.random.normal(30, 14, n_samples).clip(0.5, 80),
    'SibSp': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.01]),
    'Parch': np.random.choice([0, 1, 2, 3], n_samples, p=[0.76, 0.13, 0.08, 0.03]),
    'Fare': np.random.lognormal(2.5, 1.2, n_samples).clip(0, 512),
    'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
})

# 添加一些缺失值
data.loc[np.random.choice(data.index, 20, replace=False), 'Age'] = np.nan
data.loc[np.random.choice(data.index, 2, replace=False), 'Embarked'] = np.nan

# 創建目標變量（生存與否）- 基於一些規則
survived = []
for idx, row in data.iterrows():
    prob = 0.3  # 基礎生存率

    # 女性生存率更高
    if row['Sex'] == 'female':
        prob += 0.4

    # 1等艙生存率更高
    if row['Pclass'] == 1:
        prob += 0.2
    elif row['Pclass'] == 2:
        prob += 0.1

    # 兒童生存率更高
    if pd.notna(row['Age']) and row['Age'] < 16:
        prob += 0.15

    # 家庭成員數影響
    family_size = row['SibSp'] + row['Parch']
    if 1 <= family_size <= 3:
        prob += 0.1

    # 添加隨機性
    survived.append(1 if np.random.random() < prob else 0)

data['Survived'] = survived

print(f"數據集大小：{data.shape}")
print(f"\n前5行數據：")
print(data.head())

print(f"\n數據信息：")
print(data.info())

print(f"\n生存統計：")
print(f"生還人數：{data['Survived'].sum()} ({data['Survived'].mean()*100:.1f}%)")
print(f"遇難人數：{len(data) - data['Survived'].sum()} ({(1-data['Survived'].mean())*100:.1f}%)")

# ============================================================================
# 2. 探索性數據分析（EDA）
# ============================================================================
print("\n【第2步】探索性數據分析（EDA）")
print("-" * 90)

# 缺失值統計
print("\n缺失值統計：")
missing = data.isnull().sum()
missing_pct = 100 * missing / len(data)
missing_table = pd.DataFrame({
    'Missing': missing,
    'Percent': missing_pct
}).sort_values('Missing', ascending=False)
print(missing_table[missing_table['Missing'] > 0])

# 數值特徵統計
print("\n數值特徵描述性統計：")
print(data.describe())

# 類別特徵分析
print("\n類別特徵分布：")
for col in ['Pclass', 'Sex', 'Embarked']:
    print(f"\n{col}:")
    print(data[col].value_counts())

# ============================================================================
# 3. 數據可視化
# ============================================================================
print("\n【第3步】數據可視化")
print("-" * 90)

fig = plt.figure(figsize=(18, 12))

# 1. 生存率
ax1 = plt.subplot(3, 4, 1)
survival_counts = data['Survived'].value_counts()
ax1.bar(['Died', 'Survived'], survival_counts.values, color=['red', 'green'], alpha=0.7)
ax1.set_ylabel('Count')
ax1.set_title('Overall Survival', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 2. 性別 vs 生存
ax2 = plt.subplot(3, 4, 2)
gender_survival = data.groupby(['Sex', 'Survived']).size().unstack()
gender_survival.plot(kind='bar', ax=ax2, color=['red', 'green'], alpha=0.7)
ax2.set_ylabel('Count')
ax2.set_title('Survival by Gender', fontweight='bold')
ax2.legend(['Died', 'Survived'])
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.grid(True, alpha=0.3, axis='y')

# 3. 客艙等級 vs 生存
ax3 = plt.subplot(3, 4, 3)
class_survival = data.groupby(['Pclass', 'Survived']).size().unstack()
class_survival.plot(kind='bar', ax=ax3, color=['red', 'green'], alpha=0.7)
ax3.set_ylabel('Count')
ax3.set_title('Survival by Class', fontweight='bold')
ax3.legend(['Died', 'Survived'])
ax3.set_xlabel('Passenger Class')
ax3.grid(True, alpha=0.3, axis='y')

# 4. 年齡分布
ax4 = plt.subplot(3, 4, 4)
ax4.hist([data[data['Survived']==0]['Age'].dropna(),
         data[data['Survived']==1]['Age'].dropna()],
        bins=20, label=['Died', 'Survived'], color=['red', 'green'], alpha=0.6)
ax4.set_xlabel('Age')
ax4.set_ylabel('Count')
ax4.set_title('Age Distribution by Survival', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 票價分布
ax5 = plt.subplot(3, 4, 5)
ax5.hist([data[data['Survived']==0]['Fare'],
         data[data['Survived']==1]['Fare']],
        bins=30, label=['Died', 'Survived'], color=['red', 'green'], alpha=0.6)
ax5.set_xlabel('Fare')
ax5.set_ylabel('Count')
ax5.set_title('Fare Distribution by Survival', fontweight='bold')
ax5.set_xlim(0, 300)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 登船港口 vs 生存
ax6 = plt.subplot(3, 4, 6)
port_survival = data.groupby(['Embarked', 'Survived']).size().unstack()
port_survival.plot(kind='bar', ax=ax6, color=['red', 'green'], alpha=0.7)
ax6.set_ylabel('Count')
ax6.set_title('Survival by Embarkation Port', fontweight='bold')
ax6.legend(['Died', 'Survived'])
ax6.grid(True, alpha=0.3, axis='y')

print("生成數據可視化圖表...")

# ============================================================================
# 4. 數據預處理和特徵工程
# ============================================================================
print("\n【第4步】數據預處理和特徵工程")
print("-" * 90)

# 複製數據
df = data.copy()

# 4.1 處理缺失值
print("\n4.1 處理缺失值...")

# Age: 用中位數填充
df['Age'].fillna(df['Age'].median(), inplace=True)
print(f"  ✓ Age 缺失值已用中位數 {df['Age'].median():.1f} 填充")

# Embarked: 用眾數填充
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print(f"  ✓ Embarked 缺失值已用眾數 '{df['Embarked'].mode()[0]}' 填充")

# 4.2 特徵工程
print("\n4.2 創建新特徵...")

# 家庭大小
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print(f"  ✓ 創建特徵 FamilySize")

# 是否獨自一人
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print(f"  ✓ 創建特徵 IsAlone")

# 年齡分組
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Senior'])
print(f"  ✓ 創建特徵 AgeGroup")

# 票價分組
df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
print(f"  ✓ 創建特徵 FareGroup")

# 4.3 編碼類別特徵
print("\n4.3 編碼類別特徵...")

# 性別編碼
df['Sex_encoded'] = LabelEncoder().fit_transform(df['Sex'])
print(f"  ✓ Sex 已編碼")

# Embarked 獨熱編碼
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_dummies], axis=1)
print(f"  ✓ Embarked 已進行獨熱編碼")

# AgeGroup 獨熱編碼
agegroup_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
df = pd.concat([df, agegroup_dummies], axis=1)
print(f"  ✓ AgeGroup 已進行獨熱編碼")

# 選擇最終特徵
feature_columns = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare',
                  'FamilySize', 'IsAlone'] + list(embarked_dummies.columns)

X = df[feature_columns]
y = df['Survived']

print(f"\n最終特徵集：{list(X.columns)}")
print(f"特徵矩陣形狀：{X.shape}")

# ============================================================================
# 5. 分割數據
# ============================================================================
print("\n【第5步】分割數據")
print("-" * 90)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"訓練集大小：{X_train.shape}")
print(f"測試集大小：{X_test.shape}")

# 特徵縮放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 6. 訓練多個模型
# ============================================================================
print("\n【第6步】訓練多個模型")
print("-" * 90)

models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
}

results = {}

for name, model in models.items():
    print(f"\n訓練 {name}...")
    # 訓練
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

    # 評估
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results[name] = {
        'model': model,
        'train_score': train_score,
        'test_score': test_score,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

    print(f"  訓練準確率：{train_score:.4f}")
    print(f"  測試準確率：{test_score:.4f}")
    print(f"  ROC-AUC：{roc_auc:.4f}")

# ============================================================================
# 7. 模型比較
# ============================================================================
print("\n【第7步】模型比較")
print("-" * 90)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [r['train_score'] for r in results.values()],
    'Test Accuracy': [r['test_score'] for r in results.values()],
    'ROC-AUC': [r['roc_auc'] for r in results.values()]
}).sort_values('Test Accuracy', ascending=False)

print("\n模型性能比較：")
print(comparison_df.to_string(index=False))

# 選擇最佳模型
best_model_name = comparison_df.iloc[0]['Model']
best_model_results = results[best_model_name]

print(f"\n最佳模型：{best_model_name}")
print(f"測試準確率：{best_model_results['test_score']:.4f}")

# ============================================================================
# 8. 特徵重要性（使用 Random Forest）
# ============================================================================
print("\n【第8步】特徵重要性分析")
print("-" * 90)

rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n特徵重要性排名：")
print(feature_importance.head(10).to_string(index=False))

# 添加更多可視化到圖表
ax7 = plt.subplot(3, 4, 7)
top_features = feature_importance.head(10)
ax7.barh(range(len(top_features)), top_features['Importance'], alpha=0.7)
ax7.set_yticks(range(len(top_features)))
ax7.set_yticklabels(top_features['Feature'], fontsize=8)
ax7.set_xlabel('Importance')
ax7.set_title('Top 10 Feature Importances', fontweight='bold')
ax7.invert_yaxis()
ax7.grid(True, alpha=0.3, axis='x')

# 8. 模型性能比較
ax8 = plt.subplot(3, 4, 8)
model_names = list(results.keys())
test_scores = [r['test_score'] for r in results.values()]
ax8.bar(range(len(model_names)), test_scores, alpha=0.7, edgecolor='black')
ax8.set_xticks(range(len(model_names)))
ax8.set_xticklabels(['LR', 'RF', 'GB'], rotation=0)
ax8.set_ylabel('Accuracy')
ax8.set_title('Model Performance Comparison', fontweight='bold')
ax8.set_ylim([0.6, 1.0])
ax8.grid(True, alpha=0.3, axis='y')

for i, score in enumerate(test_scores):
    ax8.text(i, score, f'{score:.3f}', ha='center', va='bottom')

# 9. ROC 曲線
ax9 = plt.subplot(3, 4, 9)
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    ax9.plot(fpr, tpr, label=f"{name[:3]} (AUC={result['roc_auc']:.3f})", linewidth=2)

ax9.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
ax9.set_xlabel('False Positive Rate')
ax9.set_ylabel('True Positive Rate')
ax9.set_title('ROC Curves', fontweight='bold')
ax9.legend(loc='lower right', fontsize=8)
ax9.grid(True, alpha=0.3)

# 10. 混淆矩陣
ax10 = plt.subplot(3, 4, 10)
cm = confusion_matrix(y_test, best_model_results['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax10,
           xticklabels=['Died', 'Survived'],
           yticklabels=['Died', 'Survived'])
ax10.set_title(f'Confusion Matrix\n({best_model_name})', fontweight='bold')
ax10.set_ylabel('True Label')
ax10.set_xlabel('Predicted Label')

# 11. 年齡 vs 生存率
ax11 = plt.subplot(3, 4, 11)
age_bins = [0, 10, 20, 30, 40, 50, 60, 100]
age_groups = pd.cut(data['Age'], bins=age_bins)
survival_by_age = data.groupby(age_groups)['Survived'].mean()
ax11.plot(range(len(survival_by_age)), survival_by_age.values, marker='o', linewidth=2)
ax11.set_xticks(range(len(survival_by_age)))
ax11.set_xticklabels(['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+'],
                    rotation=45, fontsize=8)
ax11.set_ylabel('Survival Rate')
ax11.set_xlabel('Age Group')
ax11.set_title('Survival Rate by Age', fontweight='bold')
ax11.grid(True, alpha=0.3)

# 12. 家庭大小 vs 生存率
ax12 = plt.subplot(3, 4, 12)
family_survival = df.groupby('FamilySize')['Survived'].mean()
ax12.bar(family_survival.index, family_survival.values, alpha=0.7, edgecolor='black')
ax12.set_xlabel('Family Size')
ax12.set_ylabel('Survival Rate')
ax12.set_title('Survival Rate by Family Size', fontweight='bold')
ax12.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_figure(fig, get_output_path('titanic_analysis_results.png', 'Projects'))
print("\n✓ 分析結果圖表已保存")

# ============================================================================
# 9. 結論和建議
# ============================================================================
print("\n" + "=" * 90)
print("項目總結和建議".center(90))
print("=" * 90)

print(f"""
1. 關鍵發現：
   • 性別是最重要的特徵（女性生存率更高）
   • 客艙等級顯著影響生存率（1等艙更高）
   • 年齡影響生存率（兒童優先）
   • 適度的家庭規模有利於生存

2. 模型性能：
   • 最佳模型：{best_model_name}
   • 測試準確率：{best_model_results['test_score']:.2%}
   • ROC-AUC：{best_model_results['roc_auc']:.4f}

3. 改進建議：
   • 收集更多特徵（如姓名頭銜、客艙號等）
   • 嘗試更多特徵工程
   • 使用集成學習方法
   • 調整模型超參數
   • 處理類別不平衡問題

4. 業務洞察：
   • 在緊急情況下，優先救援女性和兒童
   • 高等級客艙的乘客獲救機會更大
   • 家庭團體比獨自旅行者更容易生還

5. 下一步：
   • 使用真實的泰坦尼克號數據集
   • 嘗試深度學習方法
   • 建立在線預測服務
   • 添加模型解釋性分析（SHAP）
""")

print("=" * 90)
print("項目完成！".center(90))
print("=" * 90)
