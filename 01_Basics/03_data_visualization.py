"""
數據可視化完整教程
Data Visualization Masterclass

深入學習 Matplotlib 和 Seaborn
機器學習中最重要的技能之一：看懂數據！
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, make_classification

# 設置風格
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")

print("=" * 80)
print("數據可視化完整教程".center(80))
print("=" * 80)

# ============================================================================
# 1. 為什麼數據可視化很重要？
# ============================================================================
print("\n【1】為什麼數據可視化很重要？")
print("-" * 80)
print("""
數據可視化的重要性：
• 快速理解數據分布和模式
• 發現異常值和錯誤
• 探索特徵之間的關係
• 向他人展示分析結果
• 調試模型和解釋預測

機器學習中常用的圖表：
• 散點圖：特徵關係、聚類結果
• 直方圖：數據分布
• 箱線圖：檢測異常值
• 熱圖：相關性分析
• 折線圖：訓練過程、學習曲線
• 條形圖：特徵重要性、模型比較
""")

# 準備數據
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = iris.target
df_iris['species_name'] = df_iris['species'].map({
    0: 'setosa', 1: 'versicolor', 2: 'virginica'
})

# ============================================================================
# 2. Matplotlib 基礎
# ============================================================================
print("\n【2】Matplotlib 基礎")
print("-" * 80)

fig = plt.figure(figsize=(18, 12))

# 2.1 折線圖
ax1 = plt.subplot(3, 4, 1)
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), label='sin(x)', linewidth=2, color='blue')
ax1.plot(x, np.cos(x), label='cos(x)', linewidth=2, color='red', linestyle='--')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Line Plot', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2.2 散點圖
ax2 = plt.subplot(3, 4, 2)
colors = ['red', 'green', 'blue']
for i, species in enumerate(['setosa', 'versicolor', 'virginica']):
    mask = df_iris['species_name'] == species
    ax2.scatter(df_iris[mask]['sepal length (cm)'],
               df_iris[mask]['sepal width (cm)'],
               c=colors[i], label=species, alpha=0.6, edgecolors='k')
ax2.set_xlabel('Sepal Length (cm)')
ax2.set_ylabel('Sepal Width (cm)')
ax2.set_title('Scatter Plot', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 2.3 直方圖
ax3 = plt.subplot(3, 4, 3)
ax3.hist(df_iris['petal length (cm)'], bins=20, color='green',
        alpha=0.7, edgecolor='black')
ax3.set_xlabel('Petal Length (cm)')
ax3.set_ylabel('Frequency')
ax3.set_title('Histogram', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 2.4 箱線圖
ax4 = plt.subplot(3, 4, 4)
data_boxplot = [df_iris[df_iris['species']==i]['sepal length (cm)'].values
               for i in range(3)]
bp = ax4.boxplot(data_boxplot, labels=['setosa', 'versicolor', 'virginica'],
                patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax4.set_ylabel('Sepal Length (cm)')
ax4.set_title('Box Plot', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 2.5 條形圖
ax5 = plt.subplot(3, 4, 5)
species_counts = df_iris['species_name'].value_counts()
bars = ax5.bar(species_counts.index, species_counts.values,
              color=colors, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Count')
ax5.set_title('Bar Chart', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')

# 2.6 餅圖
ax6 = plt.subplot(3, 4, 6)
ax6.pie(species_counts.values, labels=species_counts.index,
       colors=colors, autopct='%1.1f%%', startangle=90)
ax6.set_title('Pie Chart', fontweight='bold')

# ============================================================================
# 3. Seaborn 進階
# ============================================================================
print("\n【3】Seaborn 進階可視化")
print("-" * 80)

# 3.1 Pairplot - 特徵關係矩陣
print("  生成 Pairplot...")
ax7 = plt.subplot(3, 4, 7)
# 由於 pairplot 會創建新圖，我們用散點矩陣代替
from pandas.plotting import scatter_matrix
scatter_matrix(df_iris[['sepal length (cm)', 'sepal width (cm)',
                        'petal length (cm)']].iloc[:50],
              alpha=0.6, figsize=(4, 4), diagonal='hist', ax=ax7)
ax7.set_title('Scatter Matrix', fontweight='bold', fontsize=8)

# 3.2 小提琴圖
ax8 = plt.subplot(3, 4, 8)
parts = ax8.violinplot([df_iris[df_iris['species']==i]['petal width (cm)'].values
                        for i in range(3)],
                       positions=[1, 2, 3],
                       showmeans=True,
                       showmedians=True)
ax8.set_xticks([1, 2, 3])
ax8.set_xticklabels(['setosa', 'versicolor', 'virginica'])
ax8.set_ylabel('Petal Width (cm)')
ax8.set_title('Violin Plot', fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')

# 3.3 KDE 圖（核密度估計）
ax9 = plt.subplot(3, 4, 9)
for species, color in zip(['setosa', 'versicolor', 'virginica'], colors):
    mask = df_iris['species_name'] == species
    data = df_iris[mask]['petal length (cm)']
    ax9.hist(data, bins=15, alpha=0.3, color=color, density=True, label=species)
    # 添加 KDE 曲線
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 100)
    ax9.plot(x_range, kde(x_range), color=color, linewidth=2)
ax9.set_xlabel('Petal Length (cm)')
ax9.set_ylabel('Density')
ax9.set_title('KDE Plot', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

# 3.4 熱圖（相關性矩陣）
ax10 = plt.subplot(3, 4, 10)
correlation_matrix = df_iris.iloc[:, :4].corr()
im = ax10.imshow(correlation_matrix, cmap='coolwarm', aspect='auto',
                vmin=-1, vmax=1)
ax10.set_xticks(range(4))
ax10.set_yticks(range(4))
ax10.set_xticklabels(['SL', 'SW', 'PL', 'PW'], fontsize=8)
ax10.set_yticklabels(['SL', 'SW', 'PL', 'PW'], fontsize=8)
ax10.set_title('Correlation Heatmap', fontweight='bold')

# 添加數值
for i in range(4):
    for j in range(4):
        text = ax10.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black",
                        fontsize=8)

# 添加顏色條
plt.colorbar(im, ax=ax10, fraction=0.046, pad=0.04)

# 3.5 聯合分布圖
ax11 = plt.subplot(3, 4, 11)
for species, color in zip(['setosa', 'versicolor', 'virginica'], colors):
    mask = df_iris['species_name'] == species
    ax11.scatter(df_iris[mask]['sepal length (cm)'],
                df_iris[mask]['petal length (cm)'],
                c=color, label=species, alpha=0.6, edgecolors='k')

# 添加回歸線
from scipy.stats import linregress
x_data = df_iris['sepal length (cm)']
y_data = df_iris['petal length (cm)']
slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
line_x = np.array([x_data.min(), x_data.max()])
line_y = slope * line_x + intercept
ax11.plot(line_x, line_y, 'k--', linewidth=2, label=f'R²={r_value**2:.3f}')

ax11.set_xlabel('Sepal Length (cm)')
ax11.set_ylabel('Petal Length (cm)')
ax11.set_title('Joint Distribution with Regression', fontweight='bold')
ax11.legend(fontsize=8)
ax11.grid(True, alpha=0.3)

# 3.6 多子圖組合
ax12 = plt.subplot(3, 4, 12)
# 分面條形圖
x_pos = np.arange(3)
means = [df_iris[df_iris['species']==i]['petal length (cm)'].mean()
        for i in range(3)]
stds = [df_iris[df_iris['species']==i]['petal length (cm)'].std()
       for i in range(3)]
ax12.bar(x_pos, means, yerr=stds, color=colors, alpha=0.7,
        edgecolor='black', capsize=5)
ax12.set_xticks(x_pos)
ax12.set_xticklabels(['setosa', 'versicolor', 'virginica'], rotation=15)
ax12.set_ylabel('Mean Petal Length (cm)')
ax12.set_title('Mean with Error Bars', fontweight='bold')
ax12.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('01_Basics/data_visualization_examples.png', dpi=150, bbox_inches='tight')
print("\n已保存可視化示例圖表")

# ============================================================================
# 4. 機器學習特定可視化
# ============================================================================
print("\n【4】機器學習特定可視化")
print("-" * 80)

fig2 = plt.figure(figsize=(16, 10))

# 4.1 決策邊界
print("  生成決策邊界圖...")
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 使用兩個特徵
X_2d = df_iris[['petal length (cm)', 'petal width (cm)']].values
y_2d = df_iris['species'].values

# 訓練簡單分類器
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_2d, y_2d)

# 創建網格
h = 0.02
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 預測
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax1 = plt.subplot(2, 3, 1)
ax1.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
for i, color in enumerate(colors):
    mask = y_2d == i
    ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=color, label=iris.target_names[i],
               edgecolors='k', alpha=0.7)
ax1.set_xlabel('Petal Length (cm)')
ax1.set_ylabel('Petal Width (cm)')
ax1.set_title('Decision Boundary', fontweight='bold')
ax1.legend()

# 4.2 學習曲線
print("  生成學習曲線...")
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(max_depth=3, random_state=42),
    X_2d, y_2d, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

ax2 = plt.subplot(2, 3, 2)
ax2.plot(train_sizes, train_mean, 'o-', label='Training Score', linewidth=2)
ax2.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
ax2.plot(train_sizes, val_mean, 's-', label='Validation Score', linewidth=2)
ax2.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
ax2.set_xlabel('Training Size')
ax2.set_ylabel('Accuracy')
ax2.set_title('Learning Curve', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 4.3 特徵重要性
print("  生成特徵重要性圖...")
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(iris.data, iris.target)

ax3 = plt.subplot(2, 3, 3)
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

ax3.barh(range(len(feature_importance)), feature_importance['importance'],
        color='steelblue', alpha=0.7)
ax3.set_yticks(range(len(feature_importance)))
ax3.set_yticklabels([f.replace(' (cm)', '').title()
                     for f in feature_importance['feature']], fontsize=9)
ax3.set_xlabel('Importance')
ax3.set_title('Feature Importance', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# 4.4 混淆矩陣
print("  生成混淆矩陣...")
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
clf_cm = RandomForestClassifier(n_estimators=100, random_state=42)
clf_cm.fit(X_train, y_train)
y_pred = clf_cm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

ax4 = plt.subplot(2, 3, 4)
im = ax4.imshow(cm, cmap='Blues', aspect='auto')
ax4.set_xticks(range(3))
ax4.set_yticks(range(3))
ax4.set_xticklabels(iris.target_names, rotation=45, ha='right')
ax4.set_yticklabels(iris.target_names)
ax4.set_xlabel('Predicted')
ax4.set_ylabel('True')
ax4.set_title('Confusion Matrix', fontweight='bold')

for i in range(3):
    for j in range(3):
        text = ax4.text(j, i, cm[i, j],
                       ha="center", va="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black")

plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

# 4.5 ROC 曲線（二分類）
print("  生成 ROC 曲線...")
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 二值化標籤（一對多）
y_bin = label_binarize(iris.target, classes=[0, 1, 2])
n_classes = 3

# 訓練分類器
from sklearn.multiclass import OneVsRestClassifier
clf_roc = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, y_bin, test_size=0.3, random_state=42
)
clf_roc.fit(X_train, y_train)
y_score = clf_roc.predict_proba(X_test)

ax5 = plt.subplot(2, 3, 5)

# 計算每個類別的 ROC 曲線
for i, color in enumerate(colors):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax5.plot(fpr, tpr, color=color, lw=2,
            label=f'{iris.target_names[i]} (AUC = {roc_auc:.2f})')

ax5.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
ax5.set_xlabel('False Positive Rate')
ax5.set_ylabel('True Positive Rate')
ax5.set_title('ROC Curves (One-vs-Rest)', fontweight='bold')
ax5.legend(loc='lower right', fontsize=8)
ax5.grid(True, alpha=0.3)

# 4.6 分布對比
ax6 = plt.subplot(2, 3, 6)
for i, (species, color) in enumerate(zip(['setosa', 'versicolor', 'virginica'], colors)):
    mask = df_iris['species_name'] == species
    data = df_iris[mask]['sepal length (cm)']
    ax6.hist(data, bins=15, alpha=0.5, color=color, label=species, density=True)

ax6.set_xlabel('Sepal Length (cm)')
ax6.set_ylabel('Density')
ax6.set_title('Distribution Comparison', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_Basics/ml_specific_visualizations.png', dpi=150, bbox_inches='tight')
print("已保存機器學習可視化圖表")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("數據可視化要點總結")
print("=" * 80)
print("""
1. 基礎圖表：
   • 折線圖：趨勢、時間序列
   • 散點圖：特徵關係、聚類
   • 直方圖：分布
   • 箱線圖：異常值檢測
   • 條形圖：類別比較

2. 進階圖表：
   • 熱圖：相關性矩陣
   • KDE圖：密度估計
   • 小提琴圖：分布+箱線圖
   • Pairplot：多特徵關係

3. 機器學習專用：
   • 決策邊界：模型如何分類
   • 學習曲線：訓練過程
   • 混淆矩陣：分類錯誤模式
   • ROC曲線：分類器性能
   • 特徵重要性：哪些特徵重要

4. 可視化最佳實踐：
   • 選擇合適的圖表類型
   • 使用清晰的標籤和標題
   • 選擇易於區分的顏色
   • 添加圖例
   • 保持圖表簡潔
   • 使用網格線輔助閱讀

5. Matplotlib vs Seaborn：
   • Matplotlib：底層，靈活，完全控制
   • Seaborn：高層，美觀，統計圖表

6. 常用代碼模板：
   # 創建圖表
   fig, ax = plt.subplots(figsize=(10, 6))

   # 繪圖
   ax.plot(x, y, label='Data', linewidth=2)

   # 設置
   ax.set_xlabel('X Label')
   ax.set_ylabel('Y Label')
   ax.set_title('Title')
   ax.legend()
   ax.grid(True, alpha=0.3)

   # 保存
   plt.savefig('output.png', dpi=150, bbox_inches='tight')
   plt.show()
""")
