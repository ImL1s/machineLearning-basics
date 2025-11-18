# Scikit-learn 速查表 (Scikit-learn Cheatsheet)

> 最全面的 Scikit-learn API 快速参考指南

## 目录
- [安装和导入](#安装和导入)
- [数据预处理](#数据预处理)
- [特征工程](#特征工程)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [交叉验证](#交叉验证)
- [超参数调优](#超参数调优)
- [Pipeline](#pipeline)
- [特征选择](#特征选择)
- [模型保存和加载](#模型保存和加载)
- [常用工作流](#常用工作流)

---

## 安装和导入

### 安装
```bash
# 基础安装
pip install scikit-learn

# 包含所有依赖
pip install scikit-learn numpy pandas matplotlib seaborn
```

### 常用导入
```python
# 数据处理
import numpy as np
import pandas as pd

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 数据集
from sklearn import datasets
from sklearn.datasets import make_classification, make_regression

# 数据分割
from sklearn.model_selection import train_test_split

# 预处理
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 模型
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier

# 评估
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, r2_score

# 工具
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
```

---

## 数据预处理

### 1. 数据加载

```python
# 加载内置数据集
from sklearn.datasets import load_iris, load_boston, load_digits

# 分类数据集
iris = load_iris()
X, y = iris.data, iris.target

# 回归数据集
boston = load_boston()
X, y = boston.data, boston.target

# 生成数据集
from sklearn.datasets import make_classification, make_regression

# 生成分类数据
X, y = make_classification(
    n_samples=1000,      # 样本数
    n_features=20,       # 特征数
    n_informative=15,    # 有用特征数
    n_redundant=5,       # 冗余特征数
    n_classes=2,         # 类别数
    random_state=42
)

# 生成回归数据
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=0.1,
    random_state=42
)
```

### 2. 数据分割

```python
from sklearn.model_selection import train_test_split

# 基本分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 测试集比例
    random_state=42       # 随机种子
)

# 分层分割（分类任务推荐）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,           # 保持类别比例
    random_state=42
)

# 三分割（训练/验证/测试）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
# 结果：60% 训练，20% 验证，20% 测试
```

### 3. 缺失值处理

```python
from sklearn.impute import SimpleImputer

# 均值填充
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 其他策略
imputer = SimpleImputer(strategy='median')    # 中位数
imputer = SimpleImputer(strategy='most_frequent')  # 众数
imputer = SimpleImputer(strategy='constant', fill_value=0)  # 常数

# KNN填充（更高级）
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

### 4. 特征缩放

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 标准化（均值0，标准差1）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 归一化（范围[0, 1]）
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 自定义范围
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

# 鲁棒缩放（对异常值不敏感）
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 最大绝对值缩放
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)
```

### 5. 编码

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 标签编码（目标变量）
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# 解码
y_original = le.inverse_transform(y_encoded)

# One-Hot编码
from sklearn.preprocessing import OneHotEncoder

# 方法1：OneHotEncoder
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X_categorical)

# 方法2：get_dummies (pandas)
import pandas as pd
df_encoded = pd.get_dummies(df, columns=['category'])

# 方法3：OrdinalEncoder（有序类别）
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X_categorical)
```

### 6. 处理异常值

```python
# 使用IQR方法
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
X_filtered = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]

# 使用Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(X))
X_filtered = X[(z_scores < 3).all(axis=1)]
```

---

## 特征工程

### 1. 多项式特征

```python
from sklearn.preprocessing import PolynomialFeatures

# 创建多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# 例如：[a, b] -> [a, b, a^2, ab, b^2]
```

### 2. 特征交互

```python
# 手动创建交互特征
X_interaction = X[:, 0] * X[:, 1]

# 使用PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions = poly.fit_transform(X)
```

### 3. 特征变换

```python
from sklearn.preprocessing import PowerTransformer, FunctionTransformer

# Box-Cox变换（正数据）
transformer = PowerTransformer(method='box-cox')
X_transformed = transformer.fit_transform(X)

# Yeo-Johnson变换（任意数据）
transformer = PowerTransformer(method='yeo-johnson')
X_transformed = transformer.fit_transform(X)

# 对数变换
transformer = FunctionTransformer(np.log1p)
X_transformed = transformer.fit_transform(X)
```

### 4. 分箱

```python
from sklearn.preprocessing import KBinsDiscretizer

# 分箱
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_binned = binner.fit_transform(X)

# 策略选项
# - 'uniform': 等宽分箱
# - 'quantile': 等频分箱
# - 'kmeans': K-means分箱
```

---

## 模型训练

### 分类模型

```python
# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    C=1.0,                    # 正则化强度（越小越强）
    penalty='l2',             # 正则化类型：l1, l2, elasticnet, none
    solver='lbfgs',           # 优化器
    max_iter=100,             # 最大迭代次数
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# 2. K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(
    n_neighbors=5,            # K值
    weights='uniform',        # 'uniform' 或 'distance'
    metric='minkowski',       # 距离度量
    p=2                       # p=2为欧氏距离
)
clf.fit(X_train, y_train)

# 3. Support Vector Machine
from sklearn.svm import SVC

clf = SVC(
    C=1.0,                    # 惩罚参数
    kernel='rbf',             # 核函数：linear, poly, rbf, sigmoid
    gamma='scale',            # 核系数
    probability=True,         # 启用概率输出
    random_state=42
)
clf.fit(X_train, y_train)

# 4. Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion='gini',         # 分割标准：gini, entropy
    max_depth=None,           # 最大深度
    min_samples_split=2,      # 分割所需最小样本数
    min_samples_leaf=1,       # 叶节点最小样本数
    random_state=42
)
clf.fit(X_train, y_train)

# 5. Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=100,         # 树的数量
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    max_features='sqrt',      # 每次分割考虑的特征数
    bootstrap=True,           # 是否bootstrap采样
    random_state=42,
    n_jobs=-1                 # 并行数（-1为全部CPU）
)
clf.fit(X_train, y_train)

# 6. Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,        # 学习率
    max_depth=3,
    subsample=1.0,            # 样本采样比例
    random_state=42
)
clf.fit(X_train, y_train)

# 7. AdaBoost
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(
    base_estimator=None,      # 基学习器
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
clf.fit(X_train, y_train)

# 8. Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# 高斯朴素贝叶斯
clf = GaussianNB()
clf.fit(X_train, y_train)

# 多项式朴素贝叶斯（文本）
clf = MultinomialNB(alpha=1.0)
clf.fit(X_train, y_train)

# 伯努利朴素贝叶斯（二值特征）
clf = BernoulliNB(alpha=1.0)
clf.fit(X_train, y_train)
```

### 回归模型

```python
# 1. Linear Regression
from sklearn.linear_model import LinearRegression

reg = LinearRegression(
    fit_intercept=True,       # 是否拟合截距
    normalize=False           # 是否标准化
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# 访问系数
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

# 2. Ridge Regression
from sklearn.linear_model import Ridge

reg = Ridge(
    alpha=1.0,                # 正则化强度
    solver='auto',            # 优化器
    random_state=42
)
reg.fit(X_train, y_train)

# 3. Lasso Regression
from sklearn.linear_model import Lasso

reg = Lasso(
    alpha=1.0,
    max_iter=1000,
    random_state=42
)
reg.fit(X_train, y_train)

# 4. ElasticNet
from sklearn.linear_model import ElasticNet

reg = ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,             # L1惩罚的比例
    random_state=42
)
reg.fit(X_train, y_train)

# 5. Support Vector Regression
from sklearn.svm import SVR

reg = SVR(
    kernel='rbf',
    C=1.0,
    epsilon=0.1,              # epsilon-tube内的点不计入损失
    gamma='scale'
)
reg.fit(X_train, y_train)

# 6. Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
reg.fit(X_train, y_train)

# 7. Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
reg.fit(X_train, y_train)

# 8. Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
reg.fit(X_train, y_train)
```

### 聚类模型

```python
# 1. K-Means
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,             # 簇数量
    init='k-means++',         # 初始化方法
    n_init=10,                # 运行次数
    max_iter=300,
    random_state=42
)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# 2. DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.5,                  # 邻域半径
    min_samples=5,            # 核心点的最小邻居数
    metric='euclidean'
)
labels = dbscan.fit_predict(X)

# 3. Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'            # 链接方法：ward, complete, average
)
labels = agg.fit_predict(X)

# 4. Gaussian Mixture
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=3,           # 高斯分量数
    covariance_type='full',   # 协方差类型
    random_state=42
)
labels = gmm.fit_predict(X)
probas = gmm.predict_proba(X)
```

### 降维模型

```python
# 1. PCA
from sklearn.decomposition import PCA

pca = PCA(
    n_components=2,           # 保留成分数或方差比例
    whiten=False,             # 白化
    random_state=42
)
X_reduced = pca.fit_transform(X)

# 访问方差解释比例
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum()}")

# 2. t-SNE
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,            # 困惑度
    learning_rate=200,
    n_iter=1000,
    random_state=42
)
X_embedded = tsne.fit_transform(X)

# 3. LDA (Linear Discriminant Analysis)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# 4. Truncated SVD（稀疏数据）
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(
    n_components=2,
    random_state=42
)
X_reduced = svd.fit_transform(X)
```

---

## 模型评估

### 分类指标

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)

# 1. 准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 2. 精确率、召回率、F1
precision = precision_score(y_test, y_pred, average='binary')  # 二分类
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# 多分类
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# 3. 分类报告（综合）
print(classification_report(y_test, y_pred))
# 输出：precision, recall, f1-score, support

# 4. 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 可视化混淆矩阵
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 5. ROC-AUC
y_proba = clf.predict_proba(X_test)[:, 1]  # 正类概率
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 6. 多分类ROC-AUC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

y_bin = label_binarize(y_test, classes=[0, 1, 2])
y_proba = clf.predict_proba(X_test)
roc_auc = roc_auc_score(y_bin, y_proba, average='weighted', multi_class='ovr')
```

### 回归指标

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

# 1. 均方误差 (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

# 2. 均方根误差 (RMSE)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# 3. 平均绝对误差 (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")

# 4. R² Score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# 5. 平均绝对百分比误差 (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape:.4f}")

# 6. 调整R² (需手动计算)
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Adjusted R²: {adj_r2:.4f}")
```

### 聚类指标

```python
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score
)

# 1. 轮廓系数（-1到1，越大越好）
silhouette = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette:.4f}")

# 2. Calinski-Harabasz指数（越大越好）
ch_score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Score: {ch_score:.4f}")

# 3. Davies-Bouldin指数（越小越好）
db_score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Score: {db_score:.4f}")

# 4. 调整兰德指数（需要真实标签）
ari = adjusted_rand_score(y_true, labels)
print(f"Adjusted Rand Index: {ari:.4f}")
```

---

## 交叉验证

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, TimeSeriesSplit
)

# 1. 简单交叉验证
scores = cross_val_score(
    clf, X, y,
    cv=5,                     # 折数
    scoring='accuracy'        # 评分指标
)
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# 2. 多指标交叉验证
scoring = ['accuracy', 'precision', 'recall', 'f1']
scores = cross_validate(clf, X, y, cv=5, scoring=scoring)
print(f"Accuracy: {scores['test_accuracy'].mean():.4f}")
print(f"Precision: {scores['test_precision'].mean():.4f}")

# 3. K折交叉验证（手动）
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)
    scores.append(score)

print(f"Mean CV Score: {np.mean(scores):.4f}")

# 4. 分层K折（分类推荐）
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=skfold, scoring='accuracy')

# 5. 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(reg, X, y, cv=tscv, scoring='r2')

# 6. 留一法交叉验证 (LOOCV)
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(clf, X, y, cv=loo)
```

---

## 超参数调优

### Grid Search（网格搜索）

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 创建Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,                          # 交叉验证折数
    scoring='accuracy',            # 评分指标
    n_jobs=-1,                     # 并行数
    verbose=1,                     # 输出详细程度
    return_train_score=True
)

# 拟合
grid_search.fit(X_train, y_train)

# 最佳参数和分数
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# 使用最佳模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 查看所有结果
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df[['params', 'mean_test_score', 'std_test_score']])
```

### Random Search（随机搜索）

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 定义参数分布
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# 创建Random Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,                    # 采样次数
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

### Halving Grid Search（减半网格搜索，更快）

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

halving_search = HalvingGridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    factor=3,                      # 每次迭代保留的候选数
    cv=5,
    random_state=42
)

halving_search.fit(X_train, y_train)
```

---

## Pipeline

### 基本 Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 创建Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 训练
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)

# 评分
score = pipeline.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")
```

### ColumnTransformer（处理混合类型特征）

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 定义数值和类别列
numeric_features = ['age', 'income']
categorical_features = ['gender', 'occupation']

# 创建预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# 完整Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
```

### Pipeline + Grid Search

```python
# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 参数网格（注意命名：步骤名__参数名）
param_grid = {
    'pca__n_components': [5, 10, 15],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}

# Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
```

### 自定义 Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1):
        self.param = param

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 自定义转换逻辑
        return X * self.param

# 在Pipeline中使用
pipeline = Pipeline([
    ('custom', CustomTransformer(param=2)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

---

## 特征选择

### 1. 基于方差

```python
from sklearn.feature_selection import VarianceThreshold

# 移除方差小于阈值的特征
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)

print(f"Original features: {X.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")
```

### 2. 单变量特征选择

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# 选择K个最佳特征
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 获取特征分数
scores = selector.scores_
feature_importance = pd.DataFrame({
    'feature': range(X.shape[1]),
    'score': scores
}).sort_values('score', ascending=False)

# 其他评分函数
# - f_classif: ANOVA F-value（分类）
# - mutual_info_classif: 互信息（分类）
# - f_regression: F-value（回归）
# - mutual_info_regression: 互信息（回归）
```

### 3. 递归特征消除 (RFE)

```python
from sklearn.feature_selection import RFE

# 创建RFE选择器
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(
    estimator=estimator,
    n_features_to_select=10,   # 要选择的特征数
    step=1                     # 每次迭代移除的特征数
)

selector.fit(X_train, y_train)
X_selected = selector.transform(X_train)

# 查看选中的特征
selected_features = selector.support_
feature_ranking = selector.ranking_
print(f"Selected features: {selected_features}")
```

### 4. 从模型选择特征

```python
from sklearn.feature_selection import SelectFromModel

# 基于模型的特征选择
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
estimator.fit(X_train, y_train)

selector = SelectFromModel(
    estimator,
    threshold='median',        # 阈值：median, mean, 或具体数值
    prefit=True               # 使用已训练的模型
)

X_selected = selector.transform(X_train)

# 查看特征重要性
feature_importance = pd.DataFrame({
    'feature': range(X.shape[1]),
    'importance': estimator.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

### 5. L1正则化特征选择

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# 使用L1正则化
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
lasso.fit(X_train, y_train)

# 选择非零系数的特征
selector = SelectFromModel(lasso, prefit=True)
X_selected = selector.transform(X_train)
```

---

## 模型保存和加载

### 使用 joblib (推荐)

```python
import joblib

# 保存模型
joblib.dump(clf, 'model.pkl')

# 保存多个对象
joblib.dump({
    'model': clf,
    'scaler': scaler,
    'features': feature_names
}, 'model_package.pkl')

# 加载模型
clf = joblib.load('model.pkl')

# 加载多个对象
package = joblib.load('model_package.pkl')
clf = package['model']
scaler = package['scaler']
```

### 使用 pickle

```python
import pickle

# 保存
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# 加载
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)
```

### 保存 Pipeline

```python
# 保存整个Pipeline
joblib.dump(pipeline, 'pipeline.pkl')

# 加载并使用
pipeline = joblib.load('pipeline.pkl')
y_pred = pipeline.predict(X_new)
```

---

## 常用工作流

### 完整的分类工作流

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib

# 1. 加载数据
X, y = load_data()

# 2. 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. 创建Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 4. 超参数调优
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 5. 最佳模型
best_model = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# 6. 评估
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 7. 保存模型
joblib.dump(best_model, 'best_model.pkl')
```

### 完整的回归工作流

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1-2. 加载和分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# 4. 调参
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [3, 5]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 5-6. 评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# 7. 保存
joblib.dump(best_model, 'regression_model.pkl')
```

---

## 高级技巧

### 1. 处理不平衡数据

```python
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 方法1：类权重
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

clf = RandomForestClassifier(class_weight=class_weight_dict)

# 方法2：SMOTE过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 方法3：欠采样
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# 方法4：Pipeline中使用（需要imblearn）
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
```

### 2. 集成学习

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Voting
clf1 = LogisticRegression(random_state=42)
clf2 = RandomForestClassifier(random_state=42)
clf3 = GradientBoostingClassifier(random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)],
    voting='soft'              # 'hard' 或 'soft'
)
voting_clf.fit(X_train, y_train)

# Stacking
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking_clf.fit(X_train, y_train)
```

### 3. 增量学习

```python
from sklearn.linear_model import SGDClassifier

# 支持partial_fit的模型
clf = SGDClassifier(random_state=42)

# 批量训练
batch_size = 100
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    clf.partial_fit(X_batch, y_batch, classes=np.unique(y))
```

### 4. 模型校准

```python
from sklearn.calibration import CalibratedClassifierCV

# 校准分类器概率
clf = RandomForestClassifier(n_estimators=100, random_state=42)
calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
calibrated_clf.fit(X_train, y_train)

y_proba_calibrated = calibrated_clf.predict_proba(X_test)
```

---

## 总结

### 常用模式

1. **数据准备**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

2. **预处理**
   ```python
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

3. **模型训练**
   ```python
   clf = RandomForestClassifier()
   clf.fit(X_train, y_train)
   ```

4. **评估**
   ```python
   y_pred = clf.predict(X_test)
   print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
   ```

5. **使用Pipeline**
   ```python
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('clf', RandomForestClassifier())
   ])
   pipeline.fit(X_train, y_train)
   ```

### 最佳实践

1. **始终分割数据**：训练集/测试集
2. **使用Pipeline**：防止数据泄露
3. **交叉验证**：评估模型泛化能力
4. **超参数调优**：GridSearchCV或RandomizedSearchCV
5. **保存模型**：joblib.dump()
6. **设置random_state**：确保可重现性

---

**最后更新：2025-11-18**
