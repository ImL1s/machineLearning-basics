"""
實戰項目2：房價預測（回歸任務）
House Price Prediction (Regression)

這是一個完整的回歸任務機器學習項目，包括：
1. 問題定義和業務目標
2. 數據探索和分析（EDA）
3. 特徵工程
4. 多種回歸模型訓練
5. 模型調優和評估
6. 業務洞察和建議
7. 模型保存和部署準備

業務場景：房地產公司需要準確預測房價，以便：
- 為客戶提供合理的定價建議
- 識別被低估/高估的房產
- 優化投資決策
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import RANDOM_STATE, TEST_SIZE, DPI, setup_chinese_fonts, save_figure, get_output_path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings('ignore')

# 設置隨機種子以確保可重現性
# Set random seed for reproducibility
np.random.seed(RANDOM_STATE)

# 設置中文字體
# Set Chinese font
setup_chinese_fonts()

# 設置繪圖風格
# Set plotting style
sns.set_style("whitegrid")

print("=" * 100)
print("實戰項目：房價預測（回歸任務）".center(100))
print("House Price Prediction (Regression Task)".center(100))
print("=" * 100)

# ============================================================================
# 第一部分：項目介紹和數據加載
# Part 1: Project Introduction and Data Loading
# ============================================================================
print("\n" + "=" * 100)
print("第一部分：項目介紹和數據加載 | Part 1: Project Introduction and Data Loading")
print("=" * 100)

print("""
【問題描述 Problem Description】
預測房屋的銷售價格，這是一個典型的回歸問題。
Predict the sales price of houses, a typical regression problem.

【業務目標 Business Objectives】
1. 為房地產中介提供準確的定價工具
   Provide accurate pricing tools for real estate agents
2. 幫助買家評估房產價值是否合理
   Help buyers evaluate if property prices are reasonable
3. 為投資者識別潛在的投資機會
   Identify potential investment opportunities for investors

【評估指標 Evaluation Metrics】
- R² Score: 解釋方差比例 (Proportion of variance explained)
- RMSE: 均方根誤差 (Root Mean Squared Error)
- MAE: 平均絕對誤差 (Mean Absolute Error)
- MAPE: 平均絕對百分比誤差 (Mean Absolute Percentage Error)
""")

print("\n【數據集創建 Dataset Creation】")
print("-" * 100)

# 創建模擬房價數據集
# Create simulated housing dataset
n_samples = 1000

# 設置隨機種子
np.random.seed(RANDOM_STATE)

# 基礎特徵 Basic features
data = pd.DataFrame({
    # 房屋面積 (平方英尺) - House area (square feet)
    'LotArea': np.random.normal(8000, 2000, n_samples).clip(1500, 20000),

    # 居住面積 (平方英尺) - Living area (square feet)
    'GrLivArea': np.random.normal(1500, 400, n_samples).clip(500, 5000),

    # 地下室面積 (平方英尺) - Basement area (square feet)
    'TotalBsmtSF': np.random.normal(1000, 300, n_samples).clip(0, 3000),

    # 車庫面積 (平方英尺) - Garage area (square feet)
    'GarageArea': np.random.normal(500, 150, n_samples).clip(0, 1500),

    # 房齡 (年) - House age (years)
    'YearBuilt': np.random.randint(1950, 2024, n_samples),

    # 整體質量評分 (1-10) - Overall quality score (1-10)
    'OverallQual': np.random.choice(range(1, 11), n_samples, p=[0.02, 0.03, 0.05, 0.08, 0.15, 0.2, 0.2, 0.15, 0.08, 0.04]),

    # 整體狀況評分 (1-10) - Overall condition score (1-10)
    'OverallCond': np.random.choice(range(1, 11), n_samples, p=[0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.2, 0.12, 0.07, 0.03]),

    # 臥室數量 - Number of bedrooms
    'BedroomAbvGr': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.03, 0.15, 0.4, 0.3, 0.1, 0.02]),

    # 浴室數量 - Number of bathrooms
    'FullBath': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.5, 0.15, 0.05]),

    # 半浴室數量 - Number of half bathrooms
    'HalfBath': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.35, 0.05]),

    # 樓層數 - Number of floors
    'TotRmsAbvGrd': np.random.choice(range(3, 13), n_samples),

    # 壁爐數量 - Number of fireplaces
    'Fireplaces': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.4, 0.15, 0.05]),
})

# 類別特徵 Categorical features
data['Neighborhood'] = np.random.choice(
    ['Downtown', 'Suburb', 'Rural', 'Waterfront', 'Industrial'],
    n_samples,
    p=[0.25, 0.4, 0.2, 0.1, 0.05]
)

data['HouseStyle'] = np.random.choice(
    ['1Story', '2Story', 'Split', 'SplitFoyer'],
    n_samples,
    p=[0.4, 0.35, 0.15, 0.1]
)

data['ExterQual'] = np.random.choice(
    ['Excellent', 'Good', 'Average', 'Fair', 'Poor'],
    n_samples,
    p=[0.15, 0.35, 0.35, 0.12, 0.03]
)

data['KitchenQual'] = np.random.choice(
    ['Excellent', 'Good', 'Average', 'Fair'],
    n_samples,
    p=[0.2, 0.4, 0.3, 0.1]
)

data['HeatingQC'] = np.random.choice(
    ['Excellent', 'Good', 'Average', 'Fair', 'Poor'],
    n_samples,
    p=[0.25, 0.35, 0.25, 0.1, 0.05]
)

# 添加一些缺失值（模擬真實情況）
# Add some missing values (simulate real scenarios)
missing_indices = np.random.choice(data.index, size=30, replace=False)
data.loc[missing_indices[:10], 'GarageArea'] = np.nan
data.loc[missing_indices[10:20], 'TotalBsmtSF'] = np.nan
data.loc[missing_indices[20:], 'LotArea'] = np.nan

# 生成目標變量：房價 (基於多個因素)
# Generate target variable: house price (based on multiple factors)
base_price = 100000

# 計算房價（使用複雜的公式模擬真實情況）
# Calculate house price (using complex formula to simulate reality)
price = base_price.copy()

# 面積影響 Area impact
price = data['GrLivArea'] * 80
price += data['TotalBsmtSF'].fillna(0) * 50
price += data['GarageArea'].fillna(0) * 100
price += data['LotArea'].fillna(8000) * 2

# 質量影響 Quality impact
price *= (data['OverallQual'] / 5)
price *= (data['OverallCond'] / 7)

# 房齡影響 Age impact
age = 2024 - data['YearBuilt']
price *= np.exp(-age * 0.01)

# 位置影響 Location impact
location_multiplier = {'Downtown': 1.3, 'Suburb': 1.0, 'Rural': 0.8, 'Waterfront': 1.5, 'Industrial': 0.7}
price *= data['Neighborhood'].map(location_multiplier)

# 房屋風格影響 House style impact
style_multiplier = {'1Story': 1.0, '2Story': 1.1, 'Split': 0.95, 'SplitFoyer': 0.9}
price *= data['HouseStyle'].map(style_multiplier)

# 質量等級影響 Quality grade impact
qual_multiplier = {'Excellent': 1.2, 'Good': 1.1, 'Average': 1.0, 'Fair': 0.9, 'Poor': 0.8}
price *= data['ExterQual'].map(qual_multiplier)
price *= data['KitchenQual'].map(qual_multiplier)

# 添加隨機噪聲 Add random noise
price *= np.random.normal(1, 0.15, n_samples).clip(0.7, 1.3)

data['SalePrice'] = price.round(-3)  # 四捨五入到千位

print(f"✓ 數據集大小 Dataset size: {data.shape[0]} samples, {data.shape[1]} features")
print(f"✓ 目標變量 Target variable: SalePrice (房價)")
print(f"\n前5行數據 First 5 rows:")
print(data.head())

print(f"\n數據類型 Data types:")
print(data.dtypes.value_counts())

print(f"\n目標變量統計 Target variable statistics:")
print(f"- 平均房價 Mean: ${data['SalePrice'].mean():,.0f}")
print(f"- 中位房價 Median: ${data['SalePrice'].median():,.0f}")
print(f"- 最低房價 Min: ${data['SalePrice'].min():,.0f}")
print(f"- 最高房價 Max: ${data['SalePrice'].max():,.0f}")
print(f"- 標準差 Std: ${data['SalePrice'].std():,.0f}")

# ============================================================================
# 第二部分：探索性數據分析（EDA）
# Part 2: Exploratory Data Analysis (EDA)
# ============================================================================
print("\n" + "=" * 100)
print("第二部分：探索性數據分析（EDA） | Part 2: Exploratory Data Analysis")
print("=" * 100)

print("\n【1. 數據基本信息 Basic Information】")
print("-" * 100)

# 數據形狀 Data shape
print(f"數據形狀 Shape: {data.shape[0]} rows × {data.shape[1]} columns")

# 缺失值統計 Missing values statistics
print("\n缺失值統計 Missing Values:")
missing = data.isnull().sum()
missing_pct = 100 * missing / len(data)
missing_table = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
}).sort_values('Missing_Count', ascending=False)
print(missing_table[missing_table['Missing_Count'] > 0])

# 數值特徵描述性統計 Descriptive statistics for numerical features
print("\n數值特徵描述性統計 Numerical Features Statistics:")
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
print(data[numeric_features].describe())

# 類別特徵統計 Categorical features statistics
print("\n類別特徵統計 Categorical Features Statistics:")
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
for col in categorical_features:
    print(f"\n{col}:")
    print(data[col].value_counts())

print("\n【2. 目標變量分析 Target Variable Analysis】")
print("-" * 100)

# 檢查偏度和峰度 Check skewness and kurtosis
skewness = data['SalePrice'].skew()
kurt = data['SalePrice'].kurtosis()
print(f"房價分佈偏度 Skewness: {skewness:.4f}")
print(f"房價分佈峰度 Kurtosis: {kurt:.4f}")

if abs(skewness) > 0.5:
    print("⚠ 目標變量呈偏態分佈，可能需要對數變換")
    print("  Target variable is skewed, may need log transformation")

# ============================================================================
# 第三部分：數據可視化
# Part 3: Data Visualization
# ============================================================================
print("\n" + "=" * 100)
print("第三部分：數據可視化 | Part 3: Data Visualization")
print("=" * 100)
print("正在生成可視化圖表... Generating visualization charts...")

# 創建大型圖表集合 Create large figure collection
fig = plt.figure(figsize=(24, 20))

# 圖表1: 房價分佈 - Chart 1: House Price Distribution
ax1 = plt.subplot(4, 4, 1)
ax1.hist(data['SalePrice'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(data['SalePrice'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${data["SalePrice"].mean():,.0f}')
ax1.axvline(data['SalePrice'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${data["SalePrice"].median():,.0f}')
ax1.set_xlabel('Sale Price ($)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax1.set_title('Chart 1: House Price Distribution\n房價分佈', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 圖表2: 房價箱線圖 - Chart 2: House Price Box Plot
ax2 = plt.subplot(4, 4, 2)
bp = ax2.boxplot(data['SalePrice'], vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_alpha(0.7)
ax2.set_ylabel('Sale Price ($)', fontsize=10, fontweight='bold')
ax2.set_title('Chart 2: House Price Box Plot\n房價箱線圖', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 圖表3: Log變換後的房價分佈 - Chart 3: Log-transformed Price Distribution
ax3 = plt.subplot(4, 4, 3)
log_prices = np.log1p(data['SalePrice'])
ax3.hist(log_prices, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Log(Sale Price)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax3.set_title('Chart 3: Log-transformed Price\nLog變換後房價分佈', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 圖表4: Q-Q圖 - Chart 4: Q-Q Plot
ax4 = plt.subplot(4, 4, 4)
stats.probplot(data['SalePrice'], dist="norm", plot=ax4)
ax4.set_title('Chart 4: Q-Q Plot\n正態性檢驗', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 圖表5: 居住面積 vs 房價 - Chart 5: Living Area vs Price
ax5 = plt.subplot(4, 4, 5)
ax5.scatter(data['GrLivArea'], data['SalePrice'], alpha=0.5, c='steelblue', s=20)
z = np.polyfit(data['GrLivArea'], data['SalePrice'], 1)
p = np.poly1d(z)
ax5.plot(data['GrLivArea'], p(data['GrLivArea']), "r--", linewidth=2, label='Trend line')
ax5.set_xlabel('Living Area (sqft)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Sale Price ($)', fontsize=10, fontweight='bold')
ax5.set_title('Chart 5: Living Area vs Price\n居住面積vs房價', fontsize=11, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 圖表6: 整體質量 vs 房價 - Chart 6: Overall Quality vs Price
ax6 = plt.subplot(4, 4, 6)
quality_price = data.groupby('OverallQual')['SalePrice'].mean().sort_index()
ax6.bar(quality_price.index, quality_price.values, color='green', alpha=0.7, edgecolor='black')
ax6.set_xlabel('Overall Quality', fontsize=10, fontweight='bold')
ax6.set_ylabel('Average Sale Price ($)', fontsize=10, fontweight='bold')
ax6.set_title('Chart 6: Quality vs Price\n質量vs房價', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# 圖表7: 房齡 vs 房價 - Chart 7: House Age vs Price
ax7 = plt.subplot(4, 4, 7)
data['Age'] = 2024 - data['YearBuilt']
ax7.scatter(data['Age'], data['SalePrice'], alpha=0.5, c='orange', s=20)
z = np.polyfit(data['Age'], data['SalePrice'], 1)
p = np.poly1d(z)
ax7.plot(data['Age'], p(data['Age']), "r--", linewidth=2)
ax7.set_xlabel('House Age (years)', fontsize=10, fontweight='bold')
ax7.set_ylabel('Sale Price ($)', fontsize=10, fontweight='bold')
ax7.set_title('Chart 7: House Age vs Price\n房齡vs房價', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 圖表8: 位置 vs 房價 - Chart 8: Neighborhood vs Price
ax8 = plt.subplot(4, 4, 8)
neighborhood_price = data.groupby('Neighborhood')['SalePrice'].mean().sort_values()
ax8.barh(neighborhood_price.index, neighborhood_price.values, color='purple', alpha=0.7, edgecolor='black')
ax8.set_xlabel('Average Sale Price ($)', fontsize=10, fontweight='bold')
ax8.set_ylabel('Neighborhood', fontsize=10, fontweight='bold')
ax8.set_title('Chart 8: Neighborhood vs Price\n位置vs房價', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='x')

# 圖表9: 相關性熱力圖 - Chart 9: Correlation Heatmap
ax9 = plt.subplot(4, 4, 9)
correlation_features = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'GarageArea',
                       'OverallQual', 'OverallCond', 'YearBuilt', 'BedroomAbvGr']
corr_matrix = data[correlation_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax9, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
ax9.set_title('Chart 9: Correlation Heatmap\n相關性熱力圖', fontsize=11, fontweight='bold')

# 圖表10: 臥室數量 vs 房價 - Chart 10: Bedrooms vs Price
ax10 = plt.subplot(4, 4, 10)
bedroom_price = data.groupby('BedroomAbvGr')['SalePrice'].mean()
ax10.plot(bedroom_price.index, bedroom_price.values, marker='o', linewidth=2, markersize=8, color='teal')
ax10.set_xlabel('Number of Bedrooms', fontsize=10, fontweight='bold')
ax10.set_ylabel('Average Sale Price ($)', fontsize=10, fontweight='bold')
ax10.set_title('Chart 10: Bedrooms vs Price\n臥室數量vs房價', fontsize=11, fontweight='bold')
ax10.grid(True, alpha=0.3)

# 圖表11: 車庫面積 vs 房價 - Chart 11: Garage Area vs Price
ax11 = plt.subplot(4, 4, 11)
garage_data = data.dropna(subset=['GarageArea'])
ax11.scatter(garage_data['GarageArea'], garage_data['SalePrice'], alpha=0.5, c='brown', s=20)
z = np.polyfit(garage_data['GarageArea'], garage_data['SalePrice'], 1)
p = np.poly1d(z)
ax11.plot(garage_data['GarageArea'], p(garage_data['GarageArea']), "r--", linewidth=2)
ax11.set_xlabel('Garage Area (sqft)', fontsize=10, fontweight='bold')
ax11.set_ylabel('Sale Price ($)', fontsize=10, fontweight='bold')
ax11.set_title('Chart 11: Garage Area vs Price\n車庫面積vs房價', fontsize=11, fontweight='bold')
ax11.grid(True, alpha=0.3)

# 圖表12: 數值特徵分佈 - Chart 12: Numerical Features Distribution
ax12 = plt.subplot(4, 4, 12)
important_features = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'LotArea']
for feat in important_features:
    ax12.hist(data[feat].dropna(), bins=30, alpha=0.4, label=feat)
ax12.set_xlabel('Feature Value', fontsize=10, fontweight='bold')
ax12.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax12.set_title('Chart 12: Feature Distributions\n特徵分佈', fontsize=11, fontweight='bold')
ax12.legend(fontsize=8)
ax12.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('house_price_eda.png', 'Projects'))
print("✓ 已生成 12 張可視化圖表 Generated 12 visualization charts")

# ============================================================================
# 第四部分：特徵工程
# Part 4: Feature Engineering
# ============================================================================
print("\n" + "=" * 100)
print("第四部分：特徵工程 | Part 4: Feature Engineering")
print("=" * 100)

# 創建數據副本 Create a copy of data
df = data.copy()

print("\n【1. 缺失值處理 Missing Value Handling】")
print("-" * 100)

# 數值特徵：使用中位數填充 Numerical features: fill with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"✓ {col}: 填充缺失值 {df[col].isnull().sum()} 個，使用中位數 {median_val:.0f}")

print(f"\n剩餘缺失值 Remaining missing values: {df.isnull().sum().sum()}")

print("\n【2. 特徵創建 Feature Creation】")
print("-" * 100)

# 創建新特徵 Create new features

# 總面積 Total area
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
print("✓ 創建特徵 TotalSF = GrLivArea + TotalBsmtSF")

# 總浴室數 Total bathrooms
df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']
print("✓ 創建特徵 TotalBath = FullBath + 0.5 * HalfBath")

# 房屋新舊程度 House newness
df['HouseAge'] = 2024 - df['YearBuilt']
df['IsNew'] = (df['HouseAge'] <= 5).astype(int)
print("✓ 創建特徵 HouseAge 和 IsNew")

# 每平方英尺價格 Price per square foot (will be removed before training)
df['PricePerSqft'] = df['SalePrice'] / df['TotalSF']

# 房間密度 Room density
df['RoomDensity'] = df['TotRmsAbvGrd'] / df['GrLivArea']
print("✓ 創建特徵 RoomDensity = TotRmsAbvGrd / GrLivArea")

# 車庫與房屋面積比 Garage to house area ratio
df['GarageRatio'] = df['GarageArea'] / df['GrLivArea']
print("✓ 創建特徵 GarageRatio = GarageArea / GrLivArea")

# 質量與狀況的綜合評分 Combined quality and condition score
df['QualCondScore'] = df['OverallQual'] * df['OverallCond']
print("✓ 創建特徵 QualCondScore = OverallQual × OverallCond")

# 是否有地下室 Has basement
df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
print("✓ 創建特徵 HasBsmt")

# 是否有車庫 Has garage
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
print("✓ 創建特徵 HasGarage")

# 是否有壁爐 Has fireplace
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
print("✓ 創建特徵 HasFireplace")

print(f"\n當前特徵數量 Current number of features: {df.shape[1]}")

print("\n【3. 特徵編碼 Feature Encoding】")
print("-" * 100)

# 目標編碼（Target Encoding）用於類別特徵
# Target encoding for categorical features
categorical_cols = ['Neighborhood', 'HouseStyle', 'ExterQual', 'KitchenQual', 'HeatingQC']

for col in categorical_cols:
    # 計算每個類別的平均房價 Calculate mean price for each category
    target_mean = df.groupby(col)['SalePrice'].mean()
    # 創建新的編碼列 Create new encoded column
    df[f'{col}_Encoded'] = df[col].map(target_mean)
    print(f"✓ {col}: 目標編碼完成 Target encoding completed")

# 為質量評級創建序數編碼 Create ordinal encoding for quality ratings
quality_map = {'Poor': 1, 'Fair': 2, 'Average': 3, 'Good': 4, 'Excellent': 5}

for col in ['ExterQual', 'KitchenQual', 'HeatingQC']:
    df[f'{col}_Ordinal'] = df[col].map(quality_map)
    print(f"✓ {col}: 序數編碼完成 Ordinal encoding completed")

print("\n【4. 異常值檢測和處理 Outlier Detection and Handling】")
print("-" * 100)

# 使用 IQR 方法檢測異常值 Detect outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

# 檢測主要特徵的異常值 Detect outliers in main features
outlier_features = ['SalePrice', 'GrLivArea', 'TotalSF', 'LotArea']
for col in outlier_features:
    n_outliers, lower, upper = detect_outliers_iqr(df, col)
    print(f"{col}: 檢測到 {n_outliers} 個異常值 (範圍: {lower:.0f} - {upper:.0f})")

# 對於極端異常值，使用蓋帽法處理 Cap extreme outliers
def cap_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    lower_cap = df[column].quantile(lower_percentile)
    upper_cap = df[column].quantile(upper_percentile)
    df[column] = df[column].clip(lower_cap, upper_cap)
    return df

# 對某些特徵應用蓋帽法 Apply capping to certain features
for col in ['LotArea', 'GrLivArea']:
    df = cap_outliers(df, col)
    print(f"✓ {col}: 異常值已處理 Outliers capped")

print("\n【5. 準備訓練數據 Prepare Training Data】")
print("-" * 100)

# 移除原始類別特徵（保留編碼後的特徵）
# Remove original categorical features (keep encoded features)
df_processed = df.drop(columns=categorical_cols + ['PricePerSqft'])

# 分離特徵和目標變量 Separate features and target
X = df_processed.drop('SalePrice', axis=1)
y = df_processed['SalePrice']

print(f"✓ 特徵矩陣形狀 Feature matrix shape: {X.shape}")
print(f"✓ 目標變量形狀 Target shape: {y.shape}")

# 劃分訓練集和測試集 Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"✓ 訓練集大小 Training set: {X_train.shape[0]} samples")
print(f"✓ 測試集大小 Test set: {X_test.shape[0]} samples")

print("\n【6. 特徵縮放 Feature Scaling】")
print("-" * 100)

# 使用 RobustScaler（對異常值更穩健）Use RobustScaler (more robust to outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ 使用 RobustScaler 完成特徵縮放 Feature scaling completed with RobustScaler")
print(f"✓ 縮放後訓練集形狀 Scaled training set shape: {X_train_scaled.shape}")

# ============================================================================
# 第五部分：模型訓練
# Part 5: Model Training
# ============================================================================
print("\n" + "=" * 100)
print("第五部分：模型訓練 | Part 5: Model Training")
print("=" * 100)

print("\n【1. 基線模型 Baseline Models】")
print("-" * 100)

# 基線1：使用平均值預測 Baseline 1: Mean prediction
y_pred_mean = np.full(len(y_test), y_train.mean())
baseline_mean_r2 = r2_score(y_test, y_pred_mean)
baseline_mean_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
print(f"平均值基線 Mean Baseline:")
print(f"  R² Score: {baseline_mean_r2:.4f}")
print(f"  RMSE: ${baseline_mean_rmse:,.0f}")

# 基線2：使用中位數預測 Baseline 2: Median prediction
y_pred_median = np.full(len(y_test), y_train.median())
baseline_median_r2 = r2_score(y_test, y_pred_median)
baseline_median_rmse = np.sqrt(mean_squared_error(y_test, y_pred_median))
print(f"\n中位數基線 Median Baseline:")
print(f"  R² Score: {baseline_median_r2:.4f}")
print(f"  RMSE: ${baseline_median_rmse:,.0f}")

print("\n【2. 訓練多個回歸模型 Train Multiple Regression Models】")
print("-" * 100)

# 定義模型字典 Define model dictionary
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10, random_state=RANDOM_STATE),
    'Lasso Regression': Lasso(alpha=100, random_state=RANDOM_STATE),
    'ElasticNet': ElasticNet(alpha=100, l1_ratio=0.5, random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
    'SVR': SVR(kernel='rbf', C=1000, gamma=0.001)
}

# 存儲結果 Store results
results = []

print("正在訓練模型... Training models...\n")

for name, model in models.items():
    # 訓練模型 Train model
    model.fit(X_train_scaled, y_train)

    # 預測 Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # 評估指標 Evaluation metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # 交叉驗證 Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5,
                                scoring='r2', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results.append({
        'Model': name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'CV_R2_Mean': cv_mean,
        'CV_R2_Std': cv_std,
        'RMSE': test_rmse,
        'MAE': test_mae
    })

    print(f"{name}:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  CV R²: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"  RMSE: ${test_rmse:,.0f}")
    print(f"  MAE: ${test_mae:,.0f}")
    print()

# 創建結果DataFrame Results DataFrame
results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False)
print("\n模型性能排名 Model Performance Ranking:")
print(results_df.to_string(index=False))

# ============================================================================
# 第六部分：模型調優
# Part 6: Model Tuning
# ============================================================================
print("\n" + "=" * 100)
print("第六部分：模型調優 | Part 6: Model Tuning")
print("=" * 100)

# 選擇表現最好的模型進行調優 Select best model for tuning
best_model_name = results_df.iloc[0]['Model']
print(f"\n最佳模型 Best Model: {best_model_name}")
print(f"Test R²: {results_df.iloc[0]['Test_R2']:.4f}")

print("\n【網格搜索超參數調優 Grid Search Hyperparameter Tuning】")
print("-" * 100)

# 為隨機森林或梯度提升進行網格搜索 Grid search for Random Forest or Gradient Boosting
if 'Random Forest' in best_model_name:
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    base_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

elif 'Gradient Boosting' in best_model_name:
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
    base_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
else:
    # 默認使用隨機森林 Default to Random Forest
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5]
    }
    base_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

print(f"正在進行網格搜索... Performing grid search...")
print(f"參數網格 Parameter grid: {param_grid}")

grid_search = GridSearchCV(
    base_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ 網格搜索完成 Grid search completed")
print(f"最佳參數 Best parameters: {grid_search.best_params_}")
print(f"最佳CV R²分數 Best CV R² score: {grid_search.best_score_:.4f}")

# 使用最佳模型 Use best model
best_model = grid_search.best_estimator_

# 在測試集上評估 Evaluate on test set
y_pred_best = best_model.predict(X_test_scaled)
best_r2 = r2_score(y_test, y_pred_best)
best_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
best_mae = mean_absolute_error(y_test, y_pred_best)
best_mape = mean_absolute_percentage_error(y_test, y_pred_best)

print(f"\n優化後模型性能 Tuned Model Performance:")
print(f"  R² Score: {best_r2:.4f}")
print(f"  RMSE: ${best_rmse:,.0f}")
print(f"  MAE: ${best_mae:,.0f}")
print(f"  MAPE: {best_mape:.2%}")

# ============================================================================
# 第七部分：模型評估和可視化
# Part 7: Model Evaluation and Visualization
# ============================================================================
print("\n" + "=" * 100)
print("第七部分：模型評估和可視化 | Part 7: Model Evaluation and Visualization")
print("=" * 100)

# 創建評估圖表 Create evaluation charts
fig2 = plt.figure(figsize=(20, 12))

# 圖表13: 模型性能對比 - Chart 13: Model Performance Comparison
ax13 = plt.subplot(3, 3, 1)
models_sorted = results_df.sort_values('Test_R2')
colors = ['lightcoral' if x < 0.7 else 'lightgreen' if x < 0.85 else 'darkgreen'
          for x in models_sorted['Test_R2']]
ax13.barh(models_sorted['Model'], models_sorted['Test_R2'], color=colors, edgecolor='black')
ax13.set_xlabel('R² Score', fontsize=10, fontweight='bold')
ax13.set_title('Chart 13: Model Performance Comparison\n模型性能對比', fontsize=11, fontweight='bold')
ax13.grid(True, alpha=0.3, axis='x')

# 圖表14: 預測值 vs 真實值 - Chart 14: Predicted vs Actual
ax14 = plt.subplot(3, 3, 2)
ax14.scatter(y_test, y_pred_best, alpha=0.6, c='steelblue', s=30)
ax14.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
ax14.set_xlabel('Actual Price ($)', fontsize=10, fontweight='bold')
ax14.set_ylabel('Predicted Price ($)', fontsize=10, fontweight='bold')
ax14.set_title('Chart 14: Predicted vs Actual\n預測值vs真實值', fontsize=11, fontweight='bold')
ax14.legend()
ax14.grid(True, alpha=0.3)

# 圖表15: 殘差分佈 - Chart 15: Residuals Distribution
ax15 = plt.subplot(3, 3, 3)
residuals = y_test - y_pred_best
ax15.hist(residuals, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax15.axvline(0, color='red', linestyle='--', linewidth=2)
ax15.set_xlabel('Residuals ($)', fontsize=10, fontweight='bold')
ax15.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax15.set_title('Chart 15: Residuals Distribution\n殘差分佈', fontsize=11, fontweight='bold')
ax15.grid(True, alpha=0.3)

# 圖表16: 殘差 vs 預測值 - Chart 16: Residuals vs Predicted
ax16 = plt.subplot(3, 3, 4)
ax16.scatter(y_pred_best, residuals, alpha=0.6, c='purple', s=30)
ax16.axhline(0, color='red', linestyle='--', linewidth=2)
ax16.set_xlabel('Predicted Price ($)', fontsize=10, fontweight='bold')
ax16.set_ylabel('Residuals ($)', fontsize=10, fontweight='bold')
ax16.set_title('Chart 16: Residuals vs Predicted\n殘差vs預測值', fontsize=11, fontweight='bold')
ax16.grid(True, alpha=0.3)

# 圖表17: 特徵重要性 - Chart 17: Feature Importance
ax17 = plt.subplot(3, 3, 5)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    ax17.barh(feature_importance['feature'], feature_importance['importance'],
             color='teal', edgecolor='black')
    ax17.set_xlabel('Importance', fontsize=10, fontweight='bold')
    ax17.set_title('Chart 17: Top 15 Feature Importance\nTop 15特徵重要性', fontsize=11, fontweight='bold')
    ax17.grid(True, alpha=0.3, axis='x')

# 圖表18: 學習曲線 - Chart 18: Learning Curve
ax18 = plt.subplot(3, 3, 6)
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_scaled, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
)
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

ax18.plot(train_sizes, train_mean, label='Training Score', color='blue', linewidth=2)
ax18.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
ax18.plot(train_sizes, val_mean, label='Cross-validation Score', color='red', linewidth=2)
ax18.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
ax18.set_xlabel('Training Set Size', fontsize=10, fontweight='bold')
ax18.set_ylabel('R² Score', fontsize=10, fontweight='bold')
ax18.set_title('Chart 18: Learning Curve\n學習曲線', fontsize=11, fontweight='bold')
ax18.legend()
ax18.grid(True, alpha=0.3)

# 圖表19: RMSE對比 - Chart 19: RMSE Comparison
ax19 = plt.subplot(3, 3, 7)
rmse_comparison = results_df.sort_values('RMSE', ascending=False)
ax19.barh(rmse_comparison['Model'], rmse_comparison['RMSE']/1000,
         color='orange', alpha=0.7, edgecolor='black')
ax19.set_xlabel('RMSE (thousands $)', fontsize=10, fontweight='bold')
ax19.set_title('Chart 19: RMSE Comparison\nRMSE對比', fontsize=11, fontweight='bold')
ax19.grid(True, alpha=0.3, axis='x')

# 圖表20: Q-Q圖（殘差） - Chart 20: Q-Q Plot (Residuals)
ax20 = plt.subplot(3, 3, 8)
stats.probplot(residuals, dist="norm", plot=ax20)
ax20.set_title('Chart 20: Residuals Q-Q Plot\n殘差正態性檢驗', fontsize=11, fontweight='bold')
ax20.grid(True, alpha=0.3)

# 圖表21: 預測誤差百分比分佈 - Chart 21: Prediction Error Percentage Distribution
ax21 = plt.subplot(3, 3, 9)
error_pct = ((y_pred_best - y_test) / y_test) * 100
ax21.hist(error_pct, bins=50, color='green', alpha=0.7, edgecolor='black')
ax21.axvline(0, color='red', linestyle='--', linewidth=2)
ax21.set_xlabel('Prediction Error (%)', fontsize=10, fontweight='bold')
ax21.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax21.set_title('Chart 21: Error Percentage Distribution\n預測誤差百分比分佈', fontsize=11, fontweight='bold')
ax21.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig2, get_output_path('house_price_evaluation.png', 'Projects'))
print("✓ 已生成 9 張評估可視化圖表 Generated 9 evaluation visualization charts")

# ============================================================================
# 第八部分：業務洞察和建議
# Part 8: Business Insights and Recommendations
# ============================================================================
print("\n" + "=" * 100)
print("第八部分：業務洞察和建議 | Part 8: Business Insights and Recommendations")
print("=" * 100)

print("\n【1. 特徵重要性分析 Feature Importance Analysis】")
print("-" * 100)

if hasattr(best_model, 'feature_importances_'):
    feature_importance_full = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 最重要特徵:")
    for idx, row in feature_importance_full.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

    print("\n特徵重要性洞察 Feature Importance Insights:")
    top_feature = feature_importance_full.iloc[0]['Feature']
    print(f"✓ {top_feature} 是最重要的特徵，對房價預測影響最大")
    print(f"  {top_feature} is the most important feature with the highest impact on price prediction")

print("\n【2. 模型性能總結 Model Performance Summary】")
print("-" * 100)

print(f"""
最佳模型 Best Model: {best_model_name}
--------------------------
✓ R² Score: {best_r2:.4f}
  - 模型可以解釋房價變異的 {best_r2*100:.1f}%
  - The model explains {best_r2*100:.1f}% of price variance

✓ RMSE: ${best_rmse:,.0f}
  - 平均預測誤差約為 ${best_rmse:,.0f}
  - Average prediction error is approximately ${best_rmse:,.0f}

✓ MAE: ${best_mae:,.0f}
  - 中位預測誤差約為 ${best_mae:,.0f}
  - Median prediction error is approximately ${best_mae:,.0f}

✓ MAPE: {best_mape:.2%}
  - 平均預測誤差百分比為 {best_mape:.2%}
  - Mean absolute percentage error is {best_mape:.2%}
""")

print("\n【3. 業務建議 Business Recommendations】")
print("-" * 100)

print("""
基於模型分析，以下是關鍵業務建議：
Based on model analysis, here are key business recommendations:

1. 定價策略 Pricing Strategy:
   ✓ 使用模型預測作為定價基準
     Use model predictions as pricing baseline
   ✓ 對於誤差較大的預測（>15%），需要人工審核
     Manual review needed for predictions with large errors (>15%)
   ✓ 重點關注居住面積、整體質量等關鍵特徵
     Focus on key features like living area and overall quality

2. 投資機會識別 Investment Opportunity Identification:
   ✓ 尋找實際價格低於預測價格15%以上的房產
     Look for properties with actual prices 15%+ below predictions
   ✓ 優先考慮質量評分高但價格合理的房產
     Prioritize high-quality properties with reasonable prices
   ✓ 關注新興社區的發展潛力
     Pay attention to development potential in emerging neighborhoods

3. 數據收集改進 Data Collection Improvement:
   ✓ 收集更多關於房屋狀況的詳細信息
     Collect more detailed information about property conditions
   ✓ 增加位置相關的特徵（如學區、交通便利性）
     Add location-related features (school districts, transportation)
   ✓ 定期更新數據以反映市場變化
     Regularly update data to reflect market changes

4. 模型應用場景 Model Application Scenarios:
   ✓ 自動化估價系統
     Automated valuation system
   ✓ 市場趨勢分析
     Market trend analysis
   ✓ 投資組合優化
     Portfolio optimization
""")

print("\n【4. 風險提示 Risk Warnings】")
print("-" * 100)

print(f"""
模型使用注意事項 Model Usage Precautions:

⚠ 模型局限性 Model Limitations:
  - 模型基於歷史數據訓練，可能不能準確預測市場突變
    Model trained on historical data may not predict market disruptions
  - 對於極端價格（過高或過低）的預測可能不準確
    Predictions for extreme prices may be inaccurate
  - 殘差分析顯示仍存在約 {best_mape:.1%} 的平均誤差
    Residual analysis shows average error of {best_mape:.1%}

⚠ 使用建議 Usage Recommendations:
  - 結合領域專家判斷，不要完全依賴模型
    Combine with domain expert judgment, don't rely solely on model
  - 定期重新訓練模型以適應市場變化
    Regularly retrain model to adapt to market changes
  - 對預測結果進行合理性檢查
    Perform sanity checks on predictions
""")

# ============================================================================
# 第九部分：模型保存和部署準備
# Part 9: Model Saving and Deployment Preparation
# ============================================================================
print("\n" + "=" * 100)
print("第九部分：模型保存和部署準備 | Part 9: Model Saving and Deployment")
print("=" * 100)

print("\n【創建預測函數 Create Prediction Function】")
print("-" * 100)

def predict_house_price(features_dict):
    """
    預測房價的函數
    Function to predict house price

    Parameters:
    -----------
    features_dict : dict
        包含所有必要特徵的字典
        Dictionary containing all necessary features

    Returns:
    --------
    float
        預測的房價
        Predicted house price
    """
    # 創建特徵DataFrame Create features DataFrame
    features_df = pd.DataFrame([features_dict])

    # 特徵工程（與訓練時相同的步驟）
    # Feature engineering (same steps as training)
    features_df['TotalSF'] = features_df['GrLivArea'] + features_df['TotalBsmtSF']
    features_df['TotalBath'] = features_df['FullBath'] + 0.5 * features_df['HalfBath']
    features_df['HouseAge'] = 2024 - features_df['YearBuilt']
    features_df['IsNew'] = (features_df['HouseAge'] <= 5).astype(int)
    features_df['RoomDensity'] = features_df['TotRmsAbvGrd'] / features_df['GrLivArea']
    features_df['GarageRatio'] = features_df['GarageArea'] / features_df['GrLivArea']
    features_df['QualCondScore'] = features_df['OverallQual'] * features_df['OverallCond']
    features_df['HasBsmt'] = (features_df['TotalBsmtSF'] > 0).astype(int)
    features_df['HasGarage'] = (features_df['GarageArea'] > 0).astype(int)
    features_df['HasFireplace'] = (features_df['Fireplaces'] > 0).astype(int)

    # 特徵縮放 Feature scaling
    features_scaled = scaler.transform(features_df[X.columns])

    # 預測 Predict
    prediction = best_model.predict(features_scaled)[0]

    return prediction

# 測試預測函數 Test prediction function
print("✓ 預測函數已創建 Prediction function created")

# 示例預測 Example prediction
example_house = {
    'LotArea': 8000,
    'GrLivArea': 1500,
    'TotalBsmtSF': 1000,
    'GarageArea': 500,
    'YearBuilt': 2010,
    'OverallQual': 7,
    'OverallCond': 7,
    'BedroomAbvGr': 3,
    'FullBath': 2,
    'HalfBath': 1,
    'TotRmsAbvGrd': 8,
    'Fireplaces': 1,
    'Neighborhood_Encoded': df['Neighborhood_Encoded'].median(),
    'HouseStyle_Encoded': df['HouseStyle_Encoded'].median(),
    'ExterQual_Encoded': df['ExterQual_Encoded'].median(),
    'KitchenQual_Encoded': df['KitchenQual_Encoded'].median(),
    'HeatingQC_Encoded': df['HeatingQC_Encoded'].median(),
    'ExterQual_Ordinal': 4,
    'KitchenQual_Ordinal': 4,
    'HeatingQC_Ordinal': 4
}

predicted_price = predict_house_price(example_house)
print(f"\n示例房屋預測價格 Example house predicted price: ${predicted_price:,.0f}")

print("\n" + "=" * 100)
print("項目完成 | Project Completed")
print("=" * 100)

print(f"""
項目總結 Project Summary:
-----------------------
✓ 數據樣本數 Sample size: {len(data)}
✓ 特徵數量 Number of features: {X.shape[1]}
✓ 最佳模型 Best model: {best_model_name}
✓ 模型R² Model R²: {best_r2:.4f}
✓ 模型RMSE Model RMSE: ${best_rmse:,.0f}
✓ 可視化圖表 Visualization charts: 21 張

關鍵成果 Key Achievements:
- 成功構建了高精度的房價預測模型
  Successfully built high-accuracy house price prediction model
- 識別了影響房價的關鍵因素
  Identified key factors affecting house prices
- 提供了可操作的業務建議
  Provided actionable business recommendations
- 創建了可部署的預測函數
  Created deployable prediction function

下一步 Next Steps:
1. 將模型部署到生產環境
   Deploy model to production environment
2. 設置模型監控和自動重訓練機制
   Set up model monitoring and automatic retraining
3. 收集實際應用反饋並持續改進
   Collect real-world feedback and continuously improve
4. 擴展模型功能（如價格趨勢預測、市場細分）
   Expand model capabilities (price trend prediction, market segmentation)
""")

plt.show()
print("\n程序執行完畢 Program execution completed")
