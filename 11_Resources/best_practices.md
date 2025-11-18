# 机器学习最佳实践 (Machine Learning Best Practices)

> 经过实战验证的机器学习开发规范和最佳实践

## 目录
- [数据准备最佳实践](#数据准备最佳实践)
- [特征工程最佳实践](#特征工程最佳实践)
- [模型训练最佳实践](#模型训练最佳实践)
- [模型评估最佳实践](#模型评估最佳实践)
- [代码组织最佳实践](#代码组织最佳实践)
- [生产部署最佳实践](#生产部署最佳实践)
- [避免常见陷阱](#避免常见陷阱)
- [性能优化建议](#性能优化建议)
- [团队协作最佳实践](#团队协作最佳实践)

---

## 数据准备最佳实践

### 1. 数据探索 (EDA)

#### ✅ 推荐做法

```python
# 1. 查看数据基本信息
print(df.shape)
print(df.info())
print(df.describe())

# 2. 检查缺失值
print(df.isnull().sum())
print(f"Missing percentage: {df.isnull().sum() / len(df) * 100}")

# 3. 检查数据类型
print(df.dtypes)

# 4. 检查目标变量分布
print(y.value_counts())
print(y.value_counts(normalize=True))

# 5. 可视化数据分布
import seaborn as sns
import matplotlib.pyplot as plt

# 数值特征分布
df.hist(bins=50, figsize=(20, 15))
plt.show()

# 类别特征分布
for col in categorical_columns:
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.show()

# 相关性矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.show()

# 6. 检查异常值
for col in numeric_columns:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    df[col].hist(bins=50)
    plt.subplot(1, 2, 2)
    df.boxplot(column=col)
    plt.show()
```

#### ❌ 避免做法

```python
# 不要直接开始建模，不了解数据
model.fit(X, y)  # 错误！

# 不要忽略数据质量问题
# 不要假设数据是干净的
```

### 2. 数据清洗

#### ✅ 推荐做法

```python
# 1. 处理缺失值 - 分析后再决定策略
# 分析缺失值模式
missing_analysis = pd.DataFrame({
    'column': df.columns,
    'missing_count': df.isnull().sum(),
    'missing_percent': df.isnull().sum() / len(df) * 100
}).sort_values('missing_percent', ascending=False)

print(missing_analysis)

# 根据缺失比例选择策略
# < 5%: 可以删除
# 5-20%: 填充（均值/中位数/众数）
# > 20%: 考虑是否保留该特征

# 数值特征：均值/中位数
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_numeric = imputer.fit_transform(X_numeric)

# 类别特征：众数或新类别'Unknown'
imputer = SimpleImputer(strategy='most_frequent')
X_categorical = imputer.fit_transform(X_categorical)

# 2. 处理重复数据
print(f"Duplicates: {df.duplicated().sum()}")
df = df.drop_duplicates()

# 3. 修正数据类型
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')

# 4. 处理异常值
# 使用IQR方法
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 选项1：移除异常值
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# 选项2：cap异常值
df['price'] = df['price'].clip(lower=lower_bound, upper=upper_bound)

# 5. 数据验证
assert df.isnull().sum().sum() == 0, "Still have missing values!"
assert df.duplicated().sum() == 0, "Still have duplicates!"
assert (df['age'] >= 0).all(), "Age should be non-negative!"
```

#### ❌ 避免做法

```python
# 不要盲目删除所有缺失值
df = df.dropna()  # 可能丢失大量数据

# 不要在测试集上fit
imputer.fit(X_test)  # 错误！数据泄露

# 不要忽略异常值
# 异常值可能是错误，也可能是重要信息
```

### 3. 数据分割

#### ✅ 推荐做法

```python
from sklearn.model_selection import train_test_split

# 1. 基本分割（80/20）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,    # 设置随机种子！
    stratify=y          # 分类任务保持类别比例
)

# 2. 三分割（训练/验证/测试）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
# 结果：70% 训练，15% 验证，15% 测试

# 3. 时间序列数据 - 按时间分割
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# 4. 验证分割
print(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"Train class distribution:\n{pd.Series(y_train).value_counts(normalize=True)}")
print(f"Test class distribution:\n{pd.Series(y_test).value_counts(normalize=True)}")
```

#### ❌ 避免做法

```python
# 不要随机打乱时间序列数据
X_train, X_test = train_test_split(time_series_data)  # 错误！

# 不要忘记设置random_state
X_train, X_test = train_test_split(X, y)  # 结果不可重现

# 不要在分割后才处理缺失值
# 这会导致数据泄露
```

---

## 特征工程最佳实践

### 1. 特征缩放

#### ✅ 推荐做法

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. 在训练集上fit，在测试集上transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 只transform！

# 2. 使用Pipeline防止数据泄露
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)

# 3. 根据算法选择是否缩放
# 需要缩放：Linear, Logistic, SVM, KNN, Neural Network
# 不需要缩放：Tree-based models (Decision Tree, Random Forest, XGBoost)

# 4. 选择合适的缩放方法
# StandardScaler: 数据近似正态分布
# MinMaxScaler: 需要固定范围[0, 1]
# RobustScaler: 有异常值时
```

#### ❌ 避免做法

```python
# 不要在测试集上fit
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)  # 错误！数据泄露

# 不要对整个数据集fit
scaler.fit(pd.concat([X_train, X_test]))  # 错误！

# 不要对树模型做缩放（浪费时间）
```

### 2. 编码类别特征

#### ✅ 推荐做法

```python
# 1. 二元特征：LabelEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])

# 2. 无序多类别：One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['city', 'occupation'], drop_first=True)

# 或使用sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = encoder.fit_transform(X_categorical)

# 3. 有序类别：OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder

# 定义顺序
education_order = [['High School', 'Bachelor', 'Master', 'PhD']]
encoder = OrdinalEncoder(categories=education_order)
df['education_encoded'] = encoder.fit_transform(df[['education']])

# 4. 高基数类别：Target Encoding（谨慎使用）
# 使用category_encoders库
from category_encoders import TargetEncoder

encoder = TargetEncoder()
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)
```

#### ❌ 避免做法

```python
# 不要对无序类别使用LabelEncoder
# 这会引入虚假的顺序关系
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])  # 错误！

# 不要在测试集上fit编码器
encoder.fit(X_test)  # 错误！

# 不要忘记处理测试集中的新类别
```

### 3. 特征创建

#### ✅ 推荐做法

```python
# 1. 领域知识特征
# 电商：购买频率、平均订单金额
df['purchase_frequency'] = df['total_purchases'] / df['days_since_first_purchase']
df['avg_order_value'] = df['total_spent'] / df['total_purchases']

# 2. 时间特征
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['timestamp'].dt.month
df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)

# 3. 聚合特征
# 按类别聚合
user_stats = df.groupby('user_id').agg({
    'price': ['mean', 'sum', 'count'],
    'rating': ['mean', 'std']
}).reset_index()

# 4. 交互特征
df['price_per_sqft'] = df['price'] / df['square_feet']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# 5. 多项式特征
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[['feature1', 'feature2']])

# 6. 文本特征
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100)
X_text = tfidf.fit_transform(df['description'])
```

#### ❌ 避免做法

```python
# 不要创建过多特征导致维度灾难
# 不要创建无意义的特征
df['random_feature'] = np.random.rand(len(df))  # 无意义

# 不要泄露目标信息到特征中
df['target_mean'] = y.mean()  # 错误！
```

### 4. 特征选择

#### ✅ 推荐做法

```python
# 1. 移除低方差特征
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# 2. 移除高度相关的特征
correlation_matrix = X.corr().abs()
upper = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X = X.drop(columns=to_drop)

# 3. 使用特征重要性
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# 保留重要特征
important_features = importance_df[importance_df['importance'] > 0.01]['feature']
X_train_selected = X_train[important_features]

# 4. 递归特征消除 (RFE)
from sklearn.feature_selection import RFE

rfe = RFE(estimator=rf, n_features_to_select=10)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]
```

#### ❌ 避免做法

```python
# 不要在包含测试集的数据上做特征选择
selector.fit(pd.concat([X_train, X_test]))  # 错误！

# 不要删除太多特征
# 保持领域知识重要的特征
```

---

## 模型训练最佳实践

### 1. 基线模型

#### ✅ 推荐做法

```python
# 1. 从简单模型开始
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# 最简单的baseline：多数类
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
baseline_score = dummy.score(X_test, y_test)
print(f"Baseline accuracy: {baseline_score:.4f}")

# 简单模型baseline
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
simple_score = lr.score(X_test, y_test)
print(f"Logistic Regression accuracy: {simple_score:.4f}")

# 2. 比较多个简单模型
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

#### ❌ 避免做法

```python
# 不要直接上复杂模型
model = ComplexDeepNeuralNetwork()  # 错误！先试简单的

# 不要没有baseline就评估模型
```

### 2. 训练配置

#### ✅ 推荐做法

```python
# 1. 设置随机种子确保可重现
import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 2. 使用交叉验证
from sklearn.model_selection import cross_val_score, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')

# 3. 监控训练和验证性能
train_scores = []
val_scores = []

for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    clf.fit(X_tr, y_tr)
    train_scores.append(clf.score(X_tr, y_tr))
    val_scores.append(clf.score(X_val, y_val))

print(f"Train: {np.mean(train_scores):.4f}")
print(f"Val: {np.mean(val_scores):.4f}")
print(f"Overfitting: {np.mean(train_scores) - np.mean(val_scores):.4f}")

# 4. 学习曲线
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    clf, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5
)

# 绘制学习曲线
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend()
plt.show()
```

#### ❌ 避免做法

```python
# 不要在全部训练集上评估
score = clf.score(X_train, y_train)  # 会过高！

# 不要忽略过拟合信号
# 如果训练准确率90%，验证准确率60%，模型过拟合了

# 不要忘记设置random_state
```

### 3. 正则化

#### ✅ 推荐做法

```python
# 1. L1正则化（Lasso）- 特征选择
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# 2. L2正则化（Ridge）- 权重衰减
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# 3. ElasticNet - 结合L1和L2
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)

# 4. 调整正则化强度
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(Ridge(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Best alpha: {grid.best_params_['alpha']}")

# 5. Random Forest正则化
rf = RandomForestClassifier(
    max_depth=10,              # 限制深度
    min_samples_split=10,      # 分割最小样本数
    min_samples_leaf=5,        # 叶节点最小样本数
    max_features='sqrt',       # 限制特征数
    random_state=42
)
```

#### ❌ 避免做法

```python
# 不要忘记正则化
# 特别是高维数据或小样本

# 不要使用固定的正则化强度
# 应该通过交叉验证选择
```

### 4. 处理不平衡数据

#### ✅ 推荐做法

```python
# 1. 检查类别不平衡
print(pd.Series(y_train).value_counts())
print(pd.Series(y_train).value_counts(normalize=True))

# 2. 方法1：类权重
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    class_weight='balanced',  # 自动平衡
    random_state=42
)

# 或手动设置
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
rf = RandomForestClassifier(class_weight=class_weight_dict)

# 3. 方法2：重采样
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 欠采样
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# 4. 方法3：调整阈值
y_proba = clf.predict_proba(X_test)[:, 1]
threshold = 0.3  # 降低阈值增加正类预测
y_pred = (y_proba >= threshold).astype(int)

# 5. 使用合适的评估指标
# 不要只看准确率！
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

print(f"F1: {f1_score(y_test, y_pred)}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba)}")
```

#### ❌ 避免做法

```python
# 不要忽略类别不平衡
# 不要只用准确率评估
# 不要在不平衡数据上使用默认阈值0.5
```

---

## 模型评估最佳实践

### 1. 选择合适的评估指标

#### ✅ 推荐做法

```python
# 分类任务
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# 1. 平衡数据 - 准确率
accuracy = accuracy_score(y_test, y_pred)

# 2. 不平衡数据 - F1、ROC-AUC
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# 3. 医疗诊断（假阴性代价高）- 召回率
recall = recall_score(y_test, y_pred)

# 4. 垃圾邮件检测（假阳性代价高）- 精确率
precision = precision_score(y_test, y_pred)

# 回归任务
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. 一般回归 - RMSE、R²
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 2. 对异常值不敏感 - MAE
mae = mean_absolute_error(y_test, y_pred)

# 3. 百分比误差 - MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
```

#### ❌ 避免做法

```python
# 不要对所有问题都用准确率
# 不要只看一个指标
# 不要忽略业务需求
```

### 2. 混淆矩阵分析

#### ✅ 推荐做法

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"TP: {cm[1, 1]}, FP: {cm[0, 1]}")
print(f"TN: {cm[0, 0]}, FN: {cm[1, 0]}")

# 2. 可视化
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 3. 归一化混淆矩阵
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')

# 4. 分析错误
# 找出假阳性和假阴性样本
fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]

print(f"False positives: {len(fp_indices)}")
print(f"False negatives: {len(fn_indices)}")

# 检查错误样本
print("False positive examples:")
print(X_test.iloc[fp_indices[:5]])
```

### 3. 交叉验证

#### ✅ 推荐做法

```python
from sklearn.model_selection import cross_val_score, cross_validate

# 1. 基本交叉验证
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# 2. 多指标评估
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

scores = cross_validate(clf, X, y, cv=5, scoring=scoring)
for metric, values in scores.items():
    if metric.startswith('test_'):
        print(f"{metric}: {values.mean():.4f} (+/- {values.std() * 2:.4f})")

# 3. 分层交叉验证（分类推荐）
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=skf)

# 4. 时间序列交叉验证
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(reg, X, y, cv=tscv)
```

### 4. 模型比较

#### ✅ 推荐做法

```python
# 1. 比较多个模型
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results.append({
        'Model': name,
        'Mean': scores.mean(),
        'Std': scores.std()
    })

results_df = pd.DataFrame(results).sort_values('Mean', ascending=False)
print(results_df)

# 2. 可视化比较
plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['Mean'])
plt.xlabel('Accuracy')
plt.title('Model Comparison')
plt.show()

# 3. 统计显著性检验
from scipy import stats

# 比较两个模型
model1_scores = cross_val_score(model1, X, y, cv=10)
model2_scores = cross_val_score(model2, X, y, cv=10)
t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("模型之间有显著差异")
```

---

## 代码组织最佳实践

### 1. 项目结构

#### ✅ 推荐做法

```
project/
│
├── data/
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后的数据
│   └── external/         # 外部数据
│
├── notebooks/            # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/                  # 源代码
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── models/               # 保存的模型
│   └── model_v1.pkl
│
├── outputs/              # 输出结果
│   ├── figures/
│   └── reports/
│
├── tests/                # 单元测试
│   └── test_preprocessing.py
│
├── requirements.txt      # 依赖
├── README.md             # 项目说明
└── config.py             # 配置文件
```

### 2. 代码风格

#### ✅ 推荐做法

```python
# 1. 使用有意义的变量名
# 好
X_train_scaled = scaler.fit_transform(X_train)
user_purchase_frequency = total_purchases / days_active

# 差
X1 = scaler.fit_transform(X)
a = b / c

# 2. 添加注释和文档字符串
def preprocess_data(df, target_column):
    """
    预处理数据集

    参数:
        df: pandas.DataFrame
            原始数据
        target_column: str
            目标变量列名

    返回:
        X: numpy.ndarray
            特征矩阵
        y: numpy.ndarray
            目标变量
    """
    # 分离特征和目标
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 处理缺失值
    X = X.fillna(X.median())

    return X, y

# 3. 模块化代码
# 将重复的逻辑提取为函数
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """训练和评估模型"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    return model, metrics

# 4. 使用配置文件
# config.py
class Config:
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

    # 模型参数
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10

# 使用
from config import Config

X_train, X_test = train_test_split(
    X, y,
    test_size=Config.TEST_SIZE,
    random_state=Config.RANDOM_SEED
)

# 5. 日志记录
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logger.info("开始训练模型")
model.fit(X_train, y_train)
logger.info(f"训练完成，准确率: {model.score(X_test, y_test):.4f}")
```

### 3. 版本控制

#### ✅ 推荐做法

```bash
# 1. 使用.gitignore
# .gitignore
data/raw/
*.pkl
*.h5
__pycache__/
.ipynb_checkpoints/
.env

# 2. 有意义的commit消息
git commit -m "feat: 添加特征工程pipeline"
git commit -m "fix: 修复缺失值处理bug"
git commit -m "refactor: 重构数据预处理代码"

# 3. 使用分支
git checkout -b feature/new-algorithm
git checkout -b fix/data-leakage

# 4. 版本化数据和模型
# 使用DVC (Data Version Control)
dvc add data/raw/dataset.csv
dvc add models/model_v1.pkl
```

---

## 生产部署最佳实践

### 1. 模型保存

#### ✅ 推荐做法

```python
import joblib
import pickle
from datetime import datetime

# 1. 保存完整的pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)

# 保存
joblib.dump(pipeline, 'model_pipeline.pkl')

# 2. 保存模型元数据
metadata = {
    'model_type': 'RandomForestClassifier',
    'train_date': datetime.now().isoformat(),
    'features': list(X_train.columns),
    'performance': {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    },
    'hyperparameters': pipeline.named_steps['classifier'].get_params()
}

import json
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

# 3. 版本化模型
model_version = 'v1.2.0'
model_path = f'models/model_{model_version}.pkl'
joblib.dump(pipeline, model_path)
```

### 2. 模型推理

#### ✅ 推荐做法

```python
class ModelPredictor:
    """模型推理类"""

    def __init__(self, model_path, metadata_path=None):
        self.model = joblib.load(model_path)

        if metadata_path:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

    def predict(self, X):
        """预测"""
        # 验证输入
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        # 验证特征
        expected_features = self.metadata['features']
        if list(X.columns) != expected_features:
            raise ValueError(f"Expected features: {expected_features}")

        # 预测
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return predictions, probabilities

    def predict_single(self, features_dict):
        """单样本预测"""
        X = pd.DataFrame([features_dict])
        return self.predict(X)

# 使用
predictor = ModelPredictor('model_pipeline.pkl', 'model_metadata.json')
predictions, probas = predictor.predict(X_new)
```

### 3. API部署

#### ✅ 推荐做法

```python
# app.py - Flask API
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 加载模型
model = joblib.load('model_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取数据
        data = request.get_json()

        # 转换为DataFrame
        X = pd.DataFrame([data])

        # 预测
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].tolist()

        # 返回结果
        return jsonify({
            'prediction': int(prediction),
            'probability': probability,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 避免常见陷阱

### 1. 数据泄露 (Data Leakage)

#### ❌ 常见错误

```python
# 错误1：在分割前做缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 错误！
X_train, X_test = train_test_split(X_scaled, y)

# 错误2：在全部数据上做特征选择
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # 错误！
X_train, X_test = train_test_split(X_selected, y)

# 错误3：目标泄露
# 使用未来信息
df['next_month_sales'] = df['sales'].shift(-1)  # 泄露！
```

#### ✅ 正确做法

```python
# 1. 先分割，再预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 只transform

# 2. 使用Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('classifier', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)

# 3. 时间序列：不要使用未来信息
```

### 2. 过拟合

#### ❌ 常见错误

```python
# 过于复杂的模型
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,  # 无限深度
    min_samples_split=2,  # 最小分割
    min_samples_leaf=1    # 最小叶节点
)

# 训练集上评估
score = rf.score(X_train, y_train)  # 会很高但无意义
```

#### ✅ 正确做法

```python
# 1. 使用正则化
rf = RandomForestClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# 2. 交叉验证
scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"CV Score: {scores.mean():.4f}")

# 3. 监控训练和验证误差
train_score = rf.score(X_train, y_train)
val_score = rf.score(X_val, y_val)
print(f"Train: {train_score:.4f}, Val: {val_score:.4f}")

if train_score - val_score > 0.1:
    print("警告：可能过拟合！")

# 4. 学习曲线
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    rf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)
```

### 3. 其他常见陷阱

```python
# 陷阱1：忘记设置random_state
# ❌ 错误
X_train, X_test = train_test_split(X, y)  # 每次结果不同

# ✅ 正确
X_train, X_test = train_test_split(X, y, random_state=42)

# 陷阱2：不平衡数据只看准确率
# ❌ 错误
accuracy = accuracy_score(y_test, y_pred)  # 可能误导

# ✅ 正确
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba)}")

# 陷阱3：在测试集上多次调参
# ❌ 错误
# 反复在测试集上试参数，实际上在拟合测试集

# ✅ 正确
# 使用验证集或交叉验证调参，最后在测试集上评估一次

# 陷阱4：忽略特征缩放
# ❌ 错误（对于某些算法）
clf = LogisticRegression()
clf.fit(X_train, y_train)  # 未缩放

# ✅ 正确
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
clf.fit(X_train_scaled, y_train)

# 陷阱5：删除太多数据
# ❌ 错误
df = df.dropna()  # 可能删除80%的数据

# ✅ 正确
# 分析缺失值，选择合适的填充策略
```

---

## 性能优化建议

### 1. 训练速度优化

```python
# 1. 并行化
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1  # 使用所有CPU核心
)

# 2. 减少数据量（大数据集）
# 采样训练
from sklearn.utils import resample

X_sample, y_sample = resample(
    X_train, y_train,
    n_samples=10000,
    random_state=42
)

# 3. 使用更快的算法
# XGBoost比Gradient Boosting快
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,
    tree_method='hist',  # 更快的树构建方法
    n_jobs=-1
)

# 4. 早停
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=1000,
    n_iter_no_change=10,  # 10轮不改进则停止
    validation_fraction=0.1
)
```

### 2. 内存优化

```python
# 1. 使用稀疏矩阵
from scipy.sparse import csr_matrix

X_sparse = csr_matrix(X)

# 2. 分批处理
def process_in_batches(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        yield batch

# 3. 降低数据精度
df = df.astype({
    'int64_col': 'int32',
    'float64_col': 'float32'
})

# 4. 删除不需要的列
df = df[relevant_columns]
```

---

## 团队协作最佳实践

### 1. 实验跟踪

```python
# 使用MLflow跟踪实验
import mlflow
import mlflow.sklearn

mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    # 记录参数
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # 训练模型
    rf = RandomForestClassifier(n_estimators=100, max_depth=10)
    rf.fit(X_train, y_train)

    # 记录指标
    accuracy = rf.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # 记录模型
    mlflow.sklearn.log_model(rf, "model")
```

### 2. 文档

```python
# README.md
"""
# 客户流失预测模型

## 项目描述
预测电信客户是否会流失

## 数据
- 训练集：10,000条记录
- 测试集：2,000条记录
- 特征：20个

## 模型
- Random Forest Classifier
- 准确率：85%
- F1: 0.82

## 使用方法
```python
from model import predict_churn

result = predict_churn(customer_data)
```

## 依赖
见 requirements.txt
"""
```

---

## 总结检查清单

### 数据准备
- [ ] 进行EDA了解数据
- [ ] 检查和处理缺失值
- [ ] 检查和处理异常值
- [ ] 先分割数据再预处理
- [ ] 设置random_state

### 特征工程
- [ ] 在训练集上fit，在测试集上transform
- [ ] 使用Pipeline防止数据泄露
- [ ] 根据算法选择是否缩放
- [ ] 正确编码类别特征
- [ ] 创建有意义的特征

### 模型训练
- [ ] 从简单模型开始
- [ ] 使用交叉验证
- [ ] 监控过拟合
- [ ] 使用正则化
- [ ] 处理不平衡数据

### 模型评估
- [ ] 使用合适的评估指标
- [ ] 分析混淆矩阵
- [ ] 不要在测试集上调参
- [ ] 比较多个模型
- [ ] 分析错误案例

### 代码质量
- [ ] 使用有意义的变量名
- [ ] 添加注释和文档
- [ ] 模块化代码
- [ ] 版本控制
- [ ] 编写测试

### 生产部署
- [ ] 保存完整的pipeline
- [ ] 版本化模型和数据
- [ ] 记录元数据
- [ ] 监控模型性能
- [ ] 设置日志

---

**遵循这些最佳实践，可以构建更可靠、可维护、可重现的机器学习系统！**

**最后更新：2025-11-18**
