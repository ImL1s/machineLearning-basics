"""
NumPy 和 Pandas 基礎教程
NumPy and Pandas Basics

機器學習必備的數據處理工具
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 80)
print("NumPy 和 Pandas 基礎教程".center(80))
print("=" * 80)

# ============================================================================
# Part 1: NumPy 基礎
# ============================================================================
print("\n【Part 1】NumPy 基礎")
print("-" * 80)

# 1.1 創建數組
print("\n1.1 創建數組")
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
range_arr = np.arange(0, 10, 2)
linspace_arr = np.linspace(0, 1, 5)

print(f"一維數組：{arr1}")
print(f"二維數組：\n{arr2}")
print(f"全零數組：\n{zeros}")
print(f"全一數組：\n{ones}")
print(f"範圍數組：{range_arr}")
print(f"等間距數組：{linspace_arr}")

# 1.2 數組屬性
print("\n1.2 數組屬性")
print(f"形狀（shape）：{arr2.shape}")
print(f"維度（ndim）：{arr2.ndim}")
print(f"大小（size）：{arr2.size}")
print(f"數據類型（dtype）：{arr2.dtype}")

# 1.3 數組運算
print("\n1.3 數組運算")
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"加法：{a + b}")
print(f"減法：{a - b}")
print(f"乘法（元素級）：{a * b}")
print(f"除法：{a / b}")
print(f"平方：{a ** 2}")
print(f"點積：{np.dot(a, b)}")

# 1.4 數組索引和切片
print("\n1.4 數組索引和切片")
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print(f"原始數組：\n{arr}")
print(f"第一行：{arr[0]}")
print(f"第一列：{arr[:, 0]}")
print(f"前兩行前三列：\n{arr[:2, :3]}")
print(f"條件索引（大於5）：{arr[arr > 5]}")

# 1.5 統計函數
print("\n1.5 統計函數")
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"數據：{data}")
print(f"平均值：{np.mean(data):.2f}")
print(f"中位數：{np.median(data):.2f}")
print(f"標準差：{np.std(data):.2f}")
print(f"方差：{np.var(data):.2f}")
print(f"最小值：{np.min(data)}")
print(f"最大值：{np.max(data)}")
print(f"總和：{np.sum(data)}")

# 1.6 數組形狀操作
print("\n1.6 數組形狀操作")
original = np.arange(12)
print(f"原始數組：{original}")
print(f"重塑為3x4：\n{original.reshape(3, 4)}")
print(f"轉置：\n{original.reshape(3, 4).T}")
print(f"展平：{original.reshape(3, 4).flatten()}")

# 1.7 隨機數生成
print("\n1.7 隨機數生成")
np.random.seed(42)
print(f"均勻分布[0,1)：{np.random.random(5)}")
print(f"正態分布N(0,1)：{np.random.randn(5)}")
print(f"隨機整數[0,10)：{np.random.randint(0, 10, 5)}")

# ============================================================================
# Part 2: Pandas 基礎
# ============================================================================
print("\n" + "=" * 80)
print("【Part 2】Pandas 基礎")
print("-" * 80)

# 2.1 Series（一維數據）
print("\n2.1 Series（一維數據）")
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print("Series 示例：")
print(s)
print(f"\n索引：{s.index.tolist()}")
print(f"值：{s.values}")

# 自定義索引
s_custom = pd.Series([100, 200, 300], index=['a', 'b', 'c'])
print("\n自定義索引的 Series：")
print(s_custom)

# 2.2 DataFrame（二維數據）
print("\n2.2 DataFrame（二維數據）")

# 從字典創建
data_dict = {
    '姓名': ['張三', '李四', '王五', '趙六'],
    '年齡': [25, 30, 35, 28],
    '城市': ['北京', '上海', '廣州', '深圳'],
    '薪水': [8000, 12000, 15000, 10000]
}
df = pd.DataFrame(data_dict)
print("DataFrame 示例：")
print(df)

# 2.3 查看數據
print("\n2.3 查看數據")
print(f"前3行：\n{df.head(3)}")
print(f"\n數據信息：")
print(df.info())
print(f"\n描述性統計：")
print(df.describe())

# 2.4 選擇數據
print("\n2.4 選擇數據")
print(f"選擇單列：\n{df['姓名']}")
print(f"\n選擇多列：\n{df[['姓名', '薪水']]}")
print(f"\n按行索引選擇（iloc）：\n{df.iloc[0]}")
print(f"\n按標籤選擇（loc）：\n{df.loc[0, '姓名']}")

# 2.5 條件過濾
print("\n2.5 條件過濾")
high_salary = df[df['薪水'] > 10000]
print("薪水大於10000的員工：")
print(high_salary)

# 2.6 添加新列
print("\n2.6 添加新列")
df['年薪'] = df['薪水'] * 12
df['是否高薪'] = df['薪水'] > 10000
print("添加新列後：")
print(df)

# 2.7 排序
print("\n2.7 排序")
print("按薪水降序排列：")
print(df.sort_values('薪水', ascending=False))

# 2.8 分組統計
print("\n2.8 分組統計")
# 創建更多數據用於分組示例
data_group = {
    '部門': ['銷售', '技術', '銷售', '技術', '人事', '人事'],
    '姓名': ['張三', '李四', '王五', '趙六', '錢七', '孫八'],
    '薪水': [8000, 12000, 9000, 13000, 7000, 7500]
}
df_group = pd.DataFrame(data_group)
print("原始數據：")
print(df_group)
print("\n按部門分組計算平均薪水：")
print(df_group.groupby('部門')['薪水'].mean())

# 2.9 處理缺失值
print("\n2.9 處理缺失值")
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})
print("含缺失值的數據：")
print(df_missing)
print(f"\n檢查缺失值：\n{df_missing.isnull()}")
print(f"\n缺失值數量：\n{df_missing.isnull().sum()}")
print(f"\n刪除含缺失值的行：\n{df_missing.dropna()}")
print(f"\n填充缺失值（用0）：\n{df_missing.fillna(0)}")
print(f"\n填充缺失值（用平均值）：\n{df_missing.fillna(df_missing.mean())}")

# 2.10 讀取和保存數據
print("\n2.10 讀取和保存數據")

# 保存為 CSV
df.to_csv('01_Basics/sample_data.csv', index=False, encoding='utf-8-sig')
print("已保存數據到 sample_data.csv")

# 讀取 CSV
df_read = pd.read_csv('01_Basics/sample_data.csv')
print("從 CSV 讀取的數據：")
print(df_read)

# ============================================================================
# Part 3: 數據可視化基礎
# ============================================================================
print("\n" + "=" * 80)
print("【Part 3】數據可視化基礎")
print("-" * 80)

# 創建示例數據
np.random.seed(42)
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 創建多個子圖
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 3.1 折線圖
axes[0, 0].plot(x, y1, label='sin(x)', color='blue', linewidth=2)
axes[0, 0].plot(x, y2, label='cos(x)', color='red', linewidth=2)
axes[0, 0].set_title('Line Plot', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 3.2 散點圖
data_scatter = np.random.randn(100, 2)
axes[0, 1].scatter(data_scatter[:, 0], data_scatter[:, 1],
                   c=np.random.randn(100), cmap='viridis', alpha=0.6)
axes[0, 1].set_title('Scatter Plot', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Feature 1')
axes[0, 1].set_ylabel('Feature 2')
axes[0, 1].grid(True, alpha=0.3)

# 3.3 柱狀圖
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
axes[1, 0].bar(categories, values, color='green', alpha=0.7)
axes[1, 0].set_title('Bar Chart', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Value')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 3.4 直方圖
data_hist = np.random.normal(0, 1, 1000)
axes[1, 1].hist(data_hist, bins=30, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Histogram', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('01_Basics/visualization_basics.png', dpi=150)
print("已保存可視化圖表到 visualization_basics.png")

# ============================================================================
# Part 4: 實用技巧
# ============================================================================
print("\n" + "=" * 80)
print("【Part 4】實用技巧")
print("-" * 80)

# 4.1 鏈式操作
print("\n4.1 Pandas 鏈式操作")
result = (df_group
          .groupby('部門')['薪水']
          .mean()
          .sort_values(ascending=False)
          .reset_index())
print("鏈式操作結果：")
print(result)

# 4.2 Apply 函數
print("\n4.2 使用 Apply 函數")
df_group['薪水等級'] = df_group['薪水'].apply(
    lambda x: '高' if x > 10000 else ('中' if x > 8000 else '低')
)
print(df_group)

# 4.3 透視表
print("\n4.3 透視表")
pivot = df_group.pivot_table(
    values='薪水',
    index='部門',
    aggfunc=['mean', 'count']
)
print(pivot)

print("\n" + "=" * 80)
print("NumPy 和 Pandas 基礎教程完成！")
print("=" * 80)
print("""
重要提示：
1. NumPy 是數值計算的基礎，提供高效的數組運算
2. Pandas 是數據分析的利器，提供靈活的數據結構
3. 這兩個庫是機器學習的必備工具
4. 多練習才能熟練掌握！
""")
