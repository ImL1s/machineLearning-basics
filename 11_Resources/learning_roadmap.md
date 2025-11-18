# 机器学习完整学习路线图 (Machine Learning Roadmap)

> 从零基础到实战项目的完整学习路径

## 目录
- [学习路线概览](#学习路线概览)
- [前置知识](#前置知识)
- [阶段0：快速入门](#阶段0快速入门)
- [阶段1：基础知识](#阶段1基础知识)
- [阶段2：监督学习](#阶段2监督学习)
- [阶段3：非监督学习](#阶段3非监督学习)
- [阶段4：集成学习](#阶段4集成学习)
- [阶段5：深度学习基础](#阶段5深度学习基础)
- [阶段6：实战项目](#阶段6实战项目)
- [进阶方向](#进阶方向)
- [学习资源推荐](#学习资源推荐)

---

## 学习路线概览

```
时间轴（建议）：
│
├─ 前置知识（2-4周）
│   └─ Python、NumPy、Pandas、数学基础
│
├─ 阶段0：快速入门（1天）
│   └─ 第一个ML项目体验
│
├─ 阶段1：基础知识（2-3周）
│   ├─ ML基本概念
│   ├─ 数据处理
│   └─ 工具使用
│
├─ 阶段2：监督学习（4-6周）
│   ├─ 分类算法
│   ├─ 回归算法
│   └─ 模型评估
│
├─ 阶段3：非监督学习（2-3周）
│   ├─ 聚类
│   └─ 降维
│
├─ 阶段4：集成学习（2-3周）
│   ├─ Bagging
│   ├─ Boosting
│   └─ Stacking
│
├─ 阶段5：深度学习基础（4-6周）
│   ├─ 神经网络
│   ├─ CNN
│   └─ RNN
│
├─ 阶段6：实战项目（持续）
│   ├─ Kaggle竞赛
│   ├─ 个人项目
│   └─ 开源贡献
│
└─ 进阶方向（根据兴趣）
    ├─ NLP
    ├─ Computer Vision
    ├─ Reinforcement Learning
    └─ MLOps
```

---

## 前置知识

### 必备技能清单

#### 1. Python编程（2周）

**学习目标：**
- [ ] 掌握Python基础语法
- [ ] 熟悉函数和类
- [ ] 理解列表推导式
- [ ] 掌握异常处理
- [ ] 熟悉文件操作

**推荐资源：**
- [Python官方教程](https://docs.python.org/3/tutorial/)
- [廖雪峰Python教程](https://www.liaoxuefeng.com/wiki/1016959663602400)

**实践项目：**
```python
# 练习1：数据结构操作
students = [
    {'name': 'Alice', 'score': 85},
    {'name': 'Bob', 'score': 92},
    {'name': 'Charlie', 'score': 78}
]

# 找出分数大于80的学生
high_scorers = [s['name'] for s in students if s['score'] > 80]

# 练习2：函数定义
def calculate_average(scores):
    """计算平均分"""
    return sum(scores) / len(scores)

# 练习3：类定义
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def is_passing(self, threshold=60):
        return self.score >= threshold
```

#### 2. NumPy（1周）

**学习目标：**
- [ ] 理解数组操作
- [ ] 掌握数组索引和切片
- [ ] 熟悉数学运算
- [ ] 掌握广播机制
- [ ] 理解向量化操作

**核心知识点：**
```python
import numpy as np

# 1. 数组创建
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
random = np.random.rand(2, 2)

# 2. 数组操作
arr.shape           # 形状
arr.reshape(5, 1)   # 重塑
arr.T               # 转置

# 3. 索引和切片
arr[0]              # 第一个元素
arr[1:4]            # 切片
arr[arr > 2]        # 条件索引

# 4. 数学运算
arr + 10            # 加法
arr * 2             # 乘法
np.mean(arr)        # 均值
np.std(arr)         # 标准差

# 5. 广播
arr + np.array([[1], [2], [3]])
```

**练习题：**
```python
# 练习1：创建10x10的矩阵，边界为1，内部为0
matrix = np.ones((10, 10))
matrix[1:-1, 1:-1] = 0

# 练习2：生成0-1之间的随机矩阵并归一化
random_matrix = np.random.rand(5, 5)
normalized = (random_matrix - random_matrix.min()) / (random_matrix.max() - random_matrix.min())

# 练习3：计算两个向量的余弦相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

#### 3. Pandas（1周）

**学习目标：**
- [ ] 掌握DataFrame操作
- [ ] 熟悉数据读取和保存
- [ ] 掌握数据清洗
- [ ] 理解分组聚合
- [ ] 掌握数据合并

**核心知识点：**
```python
import pandas as pd

# 1. 数据读取
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')

# 2. 数据查看
df.head()
df.info()
df.describe()

# 3. 数据选择
df['column']                    # 选择列
df[['col1', 'col2']]           # 多列
df.loc[0:5, 'col1']            # 标签索引
df.iloc[0:5, 0:3]              # 位置索引

# 4. 数据清洗
df.dropna()                     # 删除缺失值
df.fillna(0)                    # 填充缺失值
df.drop_duplicates()            # 删除重复

# 5. 数据转换
df['new_col'] = df['col1'] * 2
df.apply(lambda x: x * 2)

# 6. 分组聚合
df.groupby('category')['value'].mean()
df.groupby('category').agg({'value': ['mean', 'sum', 'count']})

# 7. 数据合并
pd.concat([df1, df2])
pd.merge(df1, df2, on='key')
```

**练习题：**
```python
# 练习1：数据清洗流程
df = pd.read_csv('messy_data.csv')
df = df.dropna(subset=['important_column'])
df = df[df['age'] > 0]
df = df.drop_duplicates()

# 练习2：特征工程
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['Youth', 'Adult', 'Middle', 'Senior'])
df['total_purchases'] = df.groupby('customer_id')['purchase'].transform('sum')

# 练习3：数据分析
summary = df.groupby('category').agg({
    'sales': ['sum', 'mean'],
    'quantity': 'sum',
    'customer_id': 'nunique'
})
```

#### 4. Matplotlib / Seaborn（1周）

**学习目标：**
- [ ] 掌握基本绘图
- [ ] 理解子图布局
- [ ] 熟悉统计图表
- [ ] 掌握图表美化

**核心知识点：**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 基本绘图
plt.plot(x, y)
plt.scatter(x, y)
plt.bar(categories, values)
plt.hist(data, bins=30)

# 2. 子图
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y1)
axes[0, 1].scatter(x, y2)

# 3. Seaborn统计图
sns.boxplot(x='category', y='value', data=df)
sns.violinplot(x='category', y='value', data=df)
sns.heatmap(correlation_matrix, annot=True)
sns.pairplot(df)

# 4. 美化
plt.style.use('seaborn')
sns.set_palette('husl')
plt.title('Title', fontsize=14)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.legend()
plt.grid(True)
```

#### 5. 数学基础（2周）

**线性代数：**
- [ ] 向量和矩阵
- [ ] 矩阵乘法
- [ ] 特征值和特征向量
- [ ] 奇异值分解（SVD）

**微积分：**
- [ ] 导数
- [ ] 梯度
- [ ] 链式法则
- [ ] 偏导数

**概率统计：**
- [ ] 概率分布（正态、伯努利、泊松）
- [ ] 期望和方差
- [ ] 贝叶斯定理
- [ ] 最大似然估计

**推荐资源：**
- 3Blue1Brown（YouTube频道）
- Khan Academy
- MIT 18.06 Linear Algebra

---

## 阶段0：快速入门

**时间：1天**

### 目标
体验完整的机器学习流程，建立直观认识

### 学习清单
- [ ] 运行QuickStart教程
- [ ] 理解数据加载
- [ ] 理解训练/测试分割
- [ ] 运行第一个模型
- [ ] 评估模型性能

### 实践项目
```python
# 快速入门项目：鸢尾花分类

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 2. 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 训练模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. 预测
y_pred = clf.predict(X_test)

# 5. 评估
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### 里程碑
成功运行第一个机器学习模型！

---

## 阶段1：基础知识

**时间：2-3周**

### Week 1: 理论基础

#### 学习目标
- [ ] 理解什么是机器学习
- [ ] 区分监督、非监督、强化学习
- [ ] 理解训练集、验证集、测试集
- [ ] 理解过拟合和欠拟合
- [ ] 理解偏差-方差权衡

#### 核心概念

**1. 机器学习定义**
- 从数据中学习模式
- 无需明确编程
- 基于经验改进

**2. 学习类型**
```
监督学习（Supervised Learning）
├─ 分类（Classification）
│   └─ 输出：离散类别（猫/狗，垃圾邮件/正常）
└─ 回归（Regression）
    └─ 输出：连续值（价格、温度）

非监督学习（Unsupervised Learning）
├─ 聚类（Clustering）
│   └─ 发现数据分组
└─ 降维（Dimensionality Reduction）
    └─ 减少特征数量

强化学习（Reinforcement Learning）
└─ 通过奖励学习策略
```

**3. 过拟合 vs 欠拟合**
```python
# 欠拟合：模型太简单
simple_model = LinearRegression()  # 对非线性数据

# 过拟合：模型太复杂
complex_model = DecisionTreeClassifier(max_depth=None)  # 无限深度

# 良好拟合：适中复杂度
good_model = DecisionTreeClassifier(max_depth=5)
```

#### 实践任务
```python
# 任务1：可视化过拟合
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 生成数据
X = np.linspace(0, 1, 20).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 不同阶数的多项式拟合
for degree in [1, 3, 15]:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)

    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X_plot, y_plot, label=f'Degree {degree}')
    plt.legend()
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.show()
```

### Week 2: 数据处理

#### 学习目标
- [ ] 掌握数据加载
- [ ] 掌握数据清洗
- [ ] 理解数据分割
- [ ] 掌握数据可视化
- [ ] 理解EDA（探索性数据分析）

#### 核心技能

**1. 数据加载**
```python
# CSV文件
df = pd.read_csv('data.csv')

# Excel
df = pd.read_excel('data.xlsx')

# SQL数据库
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table", conn)

# Scikit-learn内置数据集
from sklearn.datasets import load_boston, load_iris
data = load_iris()
```

**2. EDA流程**
```python
# 1. 基本信息
print(df.shape)
print(df.info())
print(df.describe())

# 2. 缺失值
print(df.isnull().sum())

# 3. 数据分布
df.hist(bins=50, figsize=(20, 15))
plt.show()

# 4. 相关性
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# 5. 目标变量分布
print(y.value_counts())
y.value_counts().plot(kind='bar')
```

**3. 数据清洗**
```python
# 处理缺失值
df = df.dropna(subset=['important_column'])
df['age'] = df['age'].fillna(df['age'].median())

# 处理异常值
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)]

# 处理重复值
df = df.drop_duplicates()

# 数据类型转换
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')
```

#### 实践项目
**项目：泰坦尼克数据分析**
```python
# 加载数据
titanic = pd.read_csv('titanic.csv')

# EDA
print(titanic.head())
print(titanic.info())
print(titanic.describe())

# 分析生存率
survival_rate = titanic.groupby('Pclass')['Survived'].mean()
print(survival_rate)

# 可视化
sns.barplot(x='Pclass', y='Survived', data=titanic)
plt.title('Survival Rate by Class')
plt.show()

# 特征工程
titanic['Age_Group'] = pd.cut(titanic['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])
titanic['Family_Size'] = titanic['SibSp'] + titanic['Parch'] + 1
```

### Week 3: 工具使用

#### 学习目标
- [ ] 掌握Scikit-learn基本API
- [ ] 理解fit和transform
- [ ] 掌握Pipeline
- [ ] 理解交叉验证
- [ ] 掌握网格搜索

#### Scikit-learn核心API

**1. 一致的API**
```python
# 所有模型都遵循相同的接口

# 训练
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评分
score = model.score(X_test, y_test)

# 概率（如果支持）
y_proba = model.predict_proba(X_test)
```

**2. Transformer**
```python
from sklearn.preprocessing import StandardScaler

# 创建transformer
scaler = StandardScaler()

# 在训练集上fit
scaler.fit(X_train)

# 在训练集和测试集上transform
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 或组合操作
X_train_scaled = scaler.fit_transform(X_train)
```

**3. Pipeline**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# 创建pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier())
])

# 训练
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)
```

#### 实践项目
**项目：构建完整的ML Pipeline**
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 网格搜索
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 评估
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 阶段2：监督学习

**时间：4-6周**

### Week 1-2: 分类算法

#### K-Nearest Neighbors (KNN)

**学习目标：**
- [ ] 理解KNN原理（距离度量）
- [ ] 实现KNN分类器
- [ ] 理解K值的影响
- [ ] 掌握距离计算方法
- [ ] 了解KNN的优缺点

**核心概念：**
```python
from sklearn.neighbors import KNeighborsClassifier

# KNN分类器
knn = KNeighborsClassifier(
    n_neighbors=5,        # K值
    weights='uniform',    # 权重：uniform或distance
    metric='minkowski',   # 距离度量
    p=2                   # p=2为欧氏距离
)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

**实践任务：**
```python
# 任务：找到最优K值
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5).mean()
    scores.append(score)

# 可视化
plt.plot(k_values, scores)
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN: Accuracy vs K Value')
plt.show()

best_k = k_values[np.argmax(scores)]
print(f"Best K: {best_k}")
```

**练习题：**
1. 为什么KNN是lazy learning？
2. K值过大/过小有什么影响？
3. 如何选择距离度量？
4. KNN在高维数据上的问题是什么？

#### Logistic Regression

**学习目标：**
- [ ] 理解Sigmoid函数
- [ ] 理解对数几率
- [ ] 掌握概率输出
- [ ] 理解正则化
- [ ] 掌握多分类策略

**核心概念：**
```python
from sklearn.linear_model import LogisticRegression

# Logistic回归
lr = LogisticRegression(
    C=1.0,                # 正则化强度（越小越强）
    penalty='l2',         # l1或l2正则化
    solver='lbfgs',       # 优化器
    max_iter=100,
    random_state=42
)

lr.fit(X_train, y_train)

# 获取概率
y_proba = lr.predict_proba(X_test)
print(f"Probability for class 1: {y_proba[0, 1]:.4f}")

# 获取系数
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_}")
```

**实践任务：**
```python
# 任务：理解决策边界
import numpy as np
import matplotlib.pyplot as plt

# 训练模型
lr = LogisticRegression()
lr.fit(X_train[:, :2], y_train)  # 使用两个特征便于可视化

# 绘制决策边界
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.8)
plt.title('Logistic Regression Decision Boundary')
plt.show()
```

#### Support Vector Machine (SVM)

**学习目标：**
- [ ] 理解最大间隔
- [ ] 理解支持向量
- [ ] 掌握核技巧
- [ ] 理解软间隔
- [ ] 调整C和gamma参数

**核心概念：**
```python
from sklearn.svm import SVC

# SVM分类器
svm = SVC(
    C=1.0,                # 惩罚参数
    kernel='rbf',         # 核函数：linear, poly, rbf, sigmoid
    gamma='scale',        # 核系数
    probability=True,     # 启用概率输出
    random_state=42
)

svm.fit(X_train, y_train)

# 查看支持向量
print(f"Number of support vectors: {len(svm.support_vectors_)}")
```

**实践任务：**
```python
# 任务：比较不同核函数
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
scores = []

for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    score = cross_val_score(svm, X_train, y_train, cv=5).mean()
    scores.append(score)
    print(f"{kernel}: {score:.4f}")

# 可视化
plt.bar(kernels, scores)
plt.ylabel('Cross-Validation Accuracy')
plt.title('SVM: Different Kernels')
plt.show()
```

#### Decision Tree

**学习目标：**
- [ ] 理解决策树构建
- [ ] 理解信息增益和基尼指数
- [ ] 掌握剪枝技术
- [ ] 可视化决策树
- [ ] 特征重要性分析

**核心概念：**
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 决策树
dt = DecisionTreeClassifier(
    criterion='gini',         # gini或entropy
    max_depth=5,              # 最大深度
    min_samples_split=10,     # 分割所需最小样本数
    min_samples_leaf=5,       # 叶节点最小样本数
    random_state=42
)

dt.fit(X_train, y_train)

# 可视化
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, feature_names=feature_names, class_names=class_names)
plt.show()

# 特征重要性
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)
```

**实践任务：**
```python
# 任务：理解过拟合
depths = range(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

plt.plot(depths, train_scores, label='Train')
plt.plot(depths, test_scores, label='Test')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Decision Tree: Depth vs Accuracy')
plt.show()
```

### Week 3-4: 回归算法

#### Linear Regression

**学习目标：**
- [ ] 理解最小二乘法
- [ ] 理解假设检验
- [ ] 掌握多重共线性
- [ ] 理解R²和调整R²
- [ ] 残差分析

**核心概念：**
```python
from sklearn.linear_model import LinearRegression

# 线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)

# 系数
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_}")

# R² Score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")

# 调整R²
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Adjusted R²: {adj_r2:.4f}")
```

**实践任务：**
```python
# 任务：残差分析
y_pred = lr.predict(X_test)
residuals = y_test - y_pred

# 1. 残差分布
plt.hist(residuals, bins=30)
plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.show()

# 2. 残差 vs 预测值
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.show()

# 3. Q-Q plot
from scipy import stats
stats.probplot(residuals, plot=plt)
plt.title('Q-Q Plot')
plt.show()
```

#### Ridge / Lasso / ElasticNet

**学习目标：**
- [ ] 理解L1和L2正则化
- [ ] 掌握正则化强度选择
- [ ] 理解特征选择（Lasso）
- [ ] 掌握ElasticNet

**核心概念：**
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge (L2)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso (L1)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# ElasticNet (L1 + L2)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)

# 比较系数
print(f"Ridge coef: {ridge.coef_[:5]}")
print(f"Lasso coef: {lasso.coef_[:5]}")
print(f"ElasticNet coef: {elastic.coef_[:5]}")
```

**实践任务：**
```python
# 任务：选择最优alpha
from sklearn.linear_model import RidgeCV, LassoCV

# Ridge with CV
alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Best alpha for Ridge: {ridge_cv.alpha_:.4f}")

# Lasso with CV
lasso_cv = LassoCV(alphas=alphas, cv=5)
lasso_cv.fit(X_train, y_train)
print(f"Best alpha for Lasso: {lasso_cv.alpha_:.4f}")

# 可视化系数路径
from sklearn.linear_model import lasso_path

alphas, coefs, _ = lasso_path(X_train, y_train, alphas=alphas)

for coef in coefs:
    plt.plot(alphas, coef)

plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Lasso Coefficient Path')
plt.show()
```

### Week 5-6: 模型评估

#### 分类评估指标

**学习目标：**
- [ ] 理解混淆矩阵
- [ ] 掌握精确率、召回率、F1
- [ ] 理解ROC-AUC
- [ ] 掌握PR曲线
- [ ] 理解多分类评估

**核心概念：**
```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 分类报告
print(classification_report(y_test, y_pred))

# ROC曲线
y_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# PR曲线
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

#### 回归评估指标

**学习目标：**
- [ ] 理解MSE、RMSE、MAE
- [ ] 理解R²
- [ ] 掌握MAPE
- [ ] 理解残差分析

**核心概念：**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MSE和RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# MAE
mae = mean_absolute_error(y_test, y_pred)

# R²
r2 = r2_score(y_test, y_pred)

# MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
```

#### 交叉验证

**学习目标：**
- [ ] 理解K折交叉验证
- [ ] 掌握分层交叉验证
- [ ] 理解留一法
- [ ] 掌握时间序列交叉验证

**核心概念：**
```python
from sklearn.model_selection import (
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, TimeSeriesSplit
)

# K折交叉验证
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# 分层K折
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=skf)

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(reg, X, y, cv=tscv)

# 多指标评估
scoring = ['accuracy', 'precision', 'recall', 'f1']
scores = cross_validate(clf, X, y, cv=5, scoring=scoring)
for metric in scoring:
    print(f"{metric}: {scores['test_' + metric].mean():.4f}")
```

---

## 阶段3：非监督学习

**时间：2-3周**

### Week 1: 聚类

#### K-Means

**学习目标：**
- [ ] 理解K-Means算法
- [ ] 掌握肘部法则
- [ ] 理解轮廓系数
- [ ] 掌握K值选择
- [ ] 可视化聚类结果

**核心概念：**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K-Means聚类
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)

labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# 轮廓系数
silhouette = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette:.4f}")
```

**实践任务：**
```python
# 任务：使用肘部法则选择K
from sklearn.cluster import KMeans

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# 肘部图
plt.plot(K_range, inertias, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 轮廓系数
plt.plot(K_range, silhouettes, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.show()
```

#### DBSCAN

**学习目标：**
- [ ] 理解基于密度的聚类
- [ ] 掌握eps和min_samples参数
- [ ] 识别噪声点
- [ ] 处理任意形状的簇

**核心概念：**
```python
from sklearn.cluster import DBSCAN

# DBSCAN
dbscan = DBSCAN(
    eps=0.5,              # 邻域半径
    min_samples=5,        # 核心点的最小邻居数
    metric='euclidean'
)

labels = dbscan.fit_predict(X)

# 噪声点
n_noise = list(labels).count(-1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(f"Clusters: {n_clusters}")
print(f"Noise points: {n_noise}")
```

### Week 2: 降维

#### PCA

**学习目标：**
- [ ] 理解主成分分析
- [ ] 掌握方差解释比例
- [ ] 选择成分数量
- [ ] 数据可视化
- [ ] 理解白化

**核心概念：**
```python
from sklearn.decomposition import PCA

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 方差解释比例
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total: {pca.explained_variance_ratio_.sum():.4f}")

# 可视化
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('PCA Visualization')
plt.colorbar()
plt.show()
```

**实践任务：**
```python
# 任务：选择成分数量
pca = PCA()
pca.fit(X)

# 累积方差解释比例
cumsum = np.cumsum(pca.explained_variance_ratio_)

plt.plot(cumsum)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.title('PCA: Choosing Number of Components')
plt.show()

# 选择保留95%方差的成分数
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"Components for 95% variance: {n_components_95}")
```

#### t-SNE

**学习目标：**
- [ ] 理解t-SNE原理
- [ ] 掌握困惑度参数
- [ ] 高维数据可视化
- [ ] 理解t-SNE的局限性

**核心概念：**
```python
from sklearn.manifold import TSNE

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)

X_tsne = tsne.fit_transform(X)

# 可视化
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title('t-SNE Visualization')
plt.colorbar()
plt.show()
```

---

## 阶段4：集成学习

**时间：2-3周**

### Week 1: Bagging

#### Random Forest

**学习目标：**
- [ ] 理解Bagging原理
- [ ] 理解随机森林
- [ ] 掌握特征重要性
- [ ] 超参数调优
- [ ] Out-of-Bag评估

**核心概念：**
```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,        # OOB评估
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# OOB Score
print(f"OOB Score: {rf.oob_score_:.4f}")

# 特征重要性
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.barh(importance['feature'][:10], importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.show()
```

### Week 2: Boosting

#### Gradient Boosting / XGBoost

**学习目标：**
- [ ] 理解Boosting原理
- [ ] 理解梯度提升
- [ ] 掌握XGBoost
- [ ] 超参数调优
- [ ] 早停策略

**核心概念：**
```python
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

gb.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 早停
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=False
)

print(f"Best iteration: {xgb.best_iteration}")
```

**实践任务：**
```python
# 任务：学习曲线
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)

eval_set = [(X_train, y_train), (X_val, y_val)]
xgb.fit(
    X_train, y_train,
    eval_set=eval_set,
    eval_metric='logloss',
    verbose=False
)

# 绘制学习曲线
results = xgb.evals_result()

plt.plot(results['validation_0']['logloss'], label='Train')
plt.plot(results['validation_1']['logloss'], label='Validation')
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
plt.legend()
plt.title('XGBoost Learning Curve')
plt.show()
```

---

## 阶段5：深度学习基础

**时间：4-6周**

### Week 1-2: 神经网络基础

**学习目标：**
- [ ] 理解神经元和激活函数
- [ ] 理解前向传播
- [ ] 理解反向传播
- [ ] 掌握损失函数
- [ ] 实现简单神经网络

**核心概念：**
```python
from sklearn.neural_network import MLPClassifier

# MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42
)

mlp.fit(X_train, y_train)
```

### Week 3-4: Keras/TensorFlow

**学习目标：**
- [ ] 掌握Keras Sequential API
- [ ] 构建深度神经网络
- [ ] 掌握Dropout和BatchNorm
- [ ] 使用回调函数
- [ ] 保存和加载模型

**核心概念：**
```python
from tensorflow import keras
from tensorflow.keras import layers

# 构建模型
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation='softmax')
])

# 编译
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3)
    ],
    verbose=1
)

# 评估
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

### Week 5-6: CNN基础

**学习目标：**
- [ ] 理解卷积操作
- [ ] 理解池化
- [ ] 构建CNN模型
- [ ] 图像分类任务

**核心概念：**
```python
# CNN for image classification
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

---

## 阶段6：实战项目

**时间：持续**

### 项目1：房价预测（回归）

**目标：**
- [ ] 完整的回归项目流程
- [ ] EDA和特征工程
- [ ] 模型比较和调优
- [ ] 结果分析

### 项目2：客户流失预测（分类）

**目标：**
- [ ] 不平衡数据处理
- [ ] 特征选择
- [ ] 模型集成
- [ ] 业务指标

### 项目3：Kaggle竞赛

**目标：**
- [ ] 参与真实竞赛
- [ ] 学习竞赛技巧
- [ ] 代码组织
- [ ] 团队协作

---

## 进阶方向

### 自然语言处理 (NLP)
- Word Embeddings
- Transformers
- BERT
- GPT

### 计算机视觉 (CV)
- Object Detection
- Semantic Segmentation
- Transfer Learning
- GANs

### 强化学习 (RL)
- Q-Learning
- Deep Q-Network
- Policy Gradient
- Actor-Critic

### MLOps
- Model Deployment
- CI/CD
- Monitoring
- A/B Testing

---

## 学习资源推荐

### 在线课程
1. **Andrew Ng - Machine Learning (Coursera)**
   - 经典入门课程
   - 理论扎实

2. **Fast.ai - Practical Deep Learning**
   - 自上而下学习
   - 注重实践

3. **Google - Machine Learning Crash Course**
   - 免费资源
   - 实用工具

### 书籍
1. **《机器学习》- 周志华**
   - 中文教材
   - 理论全面

2. **《Hands-On Machine Learning》- Aurélien Géron**
   - 实践导向
   - Scikit-learn和TensorFlow

3. **《Deep Learning》- Ian Goodfellow**
   - 深度学习圣经
   - 理论深入

### 实践平台
1. **Kaggle**
   - 竞赛平台
   - 学习notebooks

2. **Google Colab**
   - 免费GPU
   - Jupyter环境

3. **GitHub**
   - 开源项目
   - 代码学习

### YouTube频道
1. **3Blue1Brown**
   - 数学可视化
   - 深度学习

2. **StatQuest**
   - 统计和ML
   - 通俗易懂

3. **Sentdex**
   - Python和ML
   - 项目实践

---

## 学习里程碑

### 1个月
- [ ] 完成前置知识学习
- [ ] 理解ML基本概念
- [ ] 掌握Scikit-learn基础
- [ ] 完成QuickStart项目

### 2个月
- [ ] 掌握主要监督学习算法
- [ ] 理解模型评估
- [ ] 完成2-3个小项目

### 3个月
- [ ] 掌握非监督学习
- [ ] 理解集成学习
- [ ] 参与Kaggle竞赛

### 6个月
- [ ] 深度学习基础
- [ ] 完成3个完整项目
- [ ] 建立个人作品集

### 1年
- [ ] 选择专业方向（NLP/CV/RL）
- [ ] 深入学习
- [ ] 贡献开源项目
- [ ] 找到ML相关工作

---

## 学习建议

### 1. 理论与实践结合
- 不要只看视频
- 动手写代码
- 做笔记总结

### 2. 项目驱动
- 从实际问题出发
- 完整的项目流程
- 注重代码质量

### 3. 持续学习
- 每天至少1小时
- 跟踪最新进展
- 阅读论文

### 4. 社区参与
- Kaggle讨论
- GitHub贡献
- 技术博客

### 5. 建立作品集
- GitHub项目
- Kaggle排名
- 技术文章

---

**坚持学习，持续实践，你一定能成为优秀的机器学习工程师！**

**最后更新：2025-11-18**
