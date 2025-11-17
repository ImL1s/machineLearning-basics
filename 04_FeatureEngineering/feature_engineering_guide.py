"""
特徵工程完整指南
Feature Engineering Guide

特徵工程是機器學習中最重要的環節之一
好的特徵工程往往比複雜的模型更重要
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("=" * 80)
print("特徵工程完整指南".center(80))
print("=" * 80)

# 創建示例數據
data = pd.DataFrame({
    '年齡': [25, 35, np.nan, 28, 45, 32, 50, 29],
    '收入': [50000, 80000, 60000, 55000, 120000, 75000, 150000, 58000],
    '教育': ['本科', '碩士', '本科', '專科', '博士', '碩士', '博士', '本科'],
    '城市': ['北京', '上海', '廣州', '深圳', '北京', '上海', '北京', '廣州'],
    '是否購買': [0, 1, 0, 0, 1, 1, 1, 0]
})

print("\n【1】原始數據")
print("-" * 80)
print(data)

# ============================================================================
# 1. 處理缺失值
# ============================================================================
print("\n【2】處理缺失值")
print("-" * 80)

print(f"缺失值統計：\n{data.isnull().sum()}")

# 方法1：刪除缺失值
# data_dropped = data.dropna()

# 方法2：填充缺失值（均值）
data['年齡'].fillna(data['年齡'].mean(), inplace=True)
print(f"\n填充後：\n{data['年齡']}")

# ============================================================================
# 2. 特徵縮放
# ============================================================================
print("\n【3】特徵縮放")
print("-" * 80)

# 標準化（Z-score normalization）
scaler_standard = StandardScaler()
income_standard = scaler_standard.fit_transform(data[['收入']])

# 歸一化（Min-Max scaling）
scaler_minmax = MinMaxScaler()
income_minmax = scaler_minmax.fit_transform(data[['收入']])

print("原始收入：", data['收入'].values)
print("標準化後：", income_standard.flatten())
print("歸一化後：", income_minmax.flatten())

# ============================================================================
# 3. 類別特徵編碼
# ============================================================================
print("\n【4】類別特徵編碼")
print("-" * 80)

# 標籤編碼（Label Encoding）
le = LabelEncoder()
data['教育_編碼'] = le.fit_transform(data['教育'])
print(f"\n教育標籤編碼：\n{data[['教育', '教育_編碼']]}")

# 獨熱編碼（One-Hot Encoding）
city_onehot = pd.get_dummies(data['城市'], prefix='城市')
print(f"\n城市獨熱編碼：\n{city_onehot}")

# ============================================================================
# 4. 特徵創建
# ============================================================================
print("\n【5】特徵創建（Feature Creation）")
print("-" * 80)

# 多項式特徵
data['年齡_平方'] = data['年齡'] ** 2
data['收入_對數'] = np.log1p(data['收入'])

# 交互特徵
data['年齡_收入_交互'] = data['年齡'] * data['收入'] / 1000

# 分箱（Binning）
data['年齡_分組'] = pd.cut(data['年齡'], bins=[0, 30, 40, 100],
                        labels=['青年', '中年', '老年'])

print(data[['年齡', '年齡_平方', '年齡_分組']].head())

# ============================================================================
# 5. 特徵選擇示例
# ============================================================================
print("\n【6】特徵選擇")
print("-" * 80)

# 創建更大的數據集用於特徵選擇示例
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"原始特徵數：{X.shape[1]}")

# 方法1：基於統計的特徵選擇
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = cancer.feature_names[selector.get_support()]

print(f"\n選擇的前10個特徵：")
for feature in selected_features:
    print(f"  • {feature}")

# 方法2：基於模型的特徵重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n特徵重要性前10名：")
print(feature_importance.head(10).to_string(index=False))

# 可視化
plt.figure(figsize=(15, 10))

# 1. 特徵縮放比較
plt.subplot(2, 3, 1)
plt.scatter(range(len(data)), data['收入'], label='Original', alpha=0.7)
plt.scatter(range(len(data)), income_standard * 10000 + 80000,
           label='Standardized', alpha=0.7)
plt.scatter(range(len(data)), income_minmax * 100000 + 50000,
           label='Normalized', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Feature Scaling Comparison', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 特徵重要性
plt.subplot(2, 3, 2)
top_10_features = feature_importance.head(10)
plt.barh(top_10_features['feature'], top_10_features['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# 3. 年齡分布
plt.subplot(2, 3, 3)
plt.hist(data['年齡'], bins=10, edgecolor='black', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 4. 收入 vs 年齡
plt.subplot(2, 3, 4)
plt.scatter(data['年齡'], data['收入'], c=data['是否購買'],
           cmap='RdYlGn', s=100, alpha=0.7, edgecolors='k')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Age vs Income (colored by purchase)', fontsize=12, fontweight='bold')
plt.colorbar(label='Purchase')
plt.grid(True, alpha=0.3)

# 5. 類別特徵分布
plt.subplot(2, 3, 5)
education_counts = data['教育'].value_counts()
plt.bar(education_counts.index, education_counts.values, alpha=0.7, edgecolor='black')
plt.xlabel('Education')
plt.ylabel('Count')
plt.title('Education Distribution', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# 6. 相關性熱圖（使用數值特徵）
plt.subplot(2, 3, 6)
numeric_data = data[['年齡', '收入', '教育_編碼', '是否購買']].corr()
im = plt.imshow(numeric_data, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
plt.xticks(range(len(numeric_data.columns)), numeric_data.columns, rotation=45)
plt.yticks(range(len(numeric_data.columns)), numeric_data.columns)
plt.title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

# 添加相關係數數值
for i in range(len(numeric_data.columns)):
    for j in range(len(numeric_data.columns)):
        plt.text(j, i, f'{numeric_data.iloc[i, j]:.2f}',
                ha='center', va='center', color='white' if abs(numeric_data.iloc[i, j]) > 0.5 else 'black')

plt.tight_layout()
plt.savefig('04_FeatureEngineering/feature_engineering_results.png', dpi=150)
print("\n已保存結果圖表")

print("\n" + "=" * 80)
print("特徵工程要點總結")
print("=" * 80)
print("""
1. 數據清洗：
   • 處理缺失值（刪除、填充、插值）
   • 處理異常值（IQR、Z-score）
   • 處理重複值

2. 特徵縮放：
   • 標準化（StandardScaler）：均值0，標準差1
   • 歸一化（MinMaxScaler）：縮放到[0,1]
   • 魯棒縮放（RobustScaler）：對異常值不敏感

3. 類別特徵編碼：
   • 標籤編碼：有序類別
   • 獨熱編碼：無序類別
   • 目標編碼：用目標變量統計量編碼

4. 特徵創建：
   • 多項式特徵
   • 交互特徵
   • 數學變換（log、sqrt、exp）
   • 分箱/離散化
   • 時間特徵提取

5. 特徵選擇：
   • 過濾法（Filter）：統計測試
   • 包裝法（Wrapper）：RFE
   • 嵌入法（Embedded）：L1正則化、特徵重要性

6. 降維：
   • PCA：線性降維
   • t-SNE：非線性降維，可視化
   • UMAP：更快的非線性降維

重要原則：
• 特徵工程是迭代過程
• 理解業務和數據是關鍵
• 好的特徵 > 複雜的模型
• 避免數據洩漏
• 先在訓練集上fit，再transform測試集
""")
