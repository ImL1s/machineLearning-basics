# 机器学习面试题集 (ML Interview Questions)

> 涵盖150+常见机器学习面试题及详细答案

## 目录
- [基础概念题](#基础概念题)
- [算法原理题](#算法原理题)
- [实践问题题](#实践问题题)
- [数学基础题](#数学基础题)
- [代码实现题](#代码实现题)
- [深度学习题](#深度学习题)
- [案例分析题](#案例分析题)
- [系统设计题](#系统设计题)

---

## 基础概念题

### 1. 什么是过拟合？如何防止？

<details>
<summary>点击查看答案</summary>

**过拟合（Overfitting）**是指模型在训练集上表现很好，但在测试集或新数据上表现较差。模型"记住"了训练数据的噪声和细节，而没有学到数据的一般规律。

**识别过拟合的标志：**
- 训练误差很低，但验证/测试误差很高
- 训练准确率90%，测试准确率60%（差距过大）

**防止过拟合的方法：**

1. **正则化（Regularization）**
   - L1正则化（Lasso）：`loss + λ∑|w|`
   - L2正则化（Ridge）：`loss + λ∑w²`
   - ElasticNet：结合L1和L2

2. **交叉验证（Cross-Validation）**
   - K折交叉验证
   - 更好地评估模型泛化能力

3. **增加训练数据**
   - 更多数据帮助模型学习一般规律
   - 数据增强（Data Augmentation）

4. **降低模型复杂度**
   - 减少特征数量
   - 简化模型结构
   - 决策树：限制深度、最小样本数

5. **Early Stopping**
   - 监控验证误差
   - 当验证误差不再下降时停止训练

6. **Dropout（神经网络）**
   - 随机丢弃一些神经元
   - 防止过度依赖某些神经元

7. **集成方法**
   - Bagging（Random Forest）
   - 多个模型平均降低方差

**示例代码：**
```python
# 1. 正则化
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)  # alpha控制正则化强度

# 2. 交叉验证
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)

# 3. 早停
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(
    n_iter_no_change=10,  # 10轮不改进则停止
    validation_fraction=0.1
)

# 4. 限制模型复杂度
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(
    max_depth=5,              # 限制深度
    min_samples_split=20,     # 增加分割所需样本
    min_samples_leaf=10       # 增加叶节点最小样本
)
```

</details>

### 2. 监督学习 vs 非监督学习的区别？

<details>
<summary>点击查看答案</summary>

**监督学习（Supervised Learning）**

**定义：**
- 训练数据有标签（label）
- 学习输入到输出的映射关系
- 目标：预测新数据的标签

**类型：**
1. **分类（Classification）**
   - 输出：离散类别
   - 例子：垃圾邮件检测、图像分类、疾病诊断
   - 算法：Logistic Regression、SVM、Random Forest

2. **回归（Regression）**
   - 输出：连续数值
   - 例子：房价预测、股票预测、温度预测
   - 算法：Linear Regression、Ridge、XGBoost

**示例：**
```python
# 分类
X = [[特征1, 特征2], ...]
y = [0, 1, 0, 1, ...]  # 有标签

clf = LogisticRegression()
clf.fit(X, y)  # 学习X到y的映射
y_pred = clf.predict(X_new)
```

**非监督学习（Unsupervised Learning）**

**定义：**
- 训练数据没有标签
- 自动发现数据中的模式和结构
- 目标：理解数据的内在结构

**类型：**
1. **聚类（Clustering）**
   - 将相似数据分组
   - 例子：客户细分、文档分类、图像压缩
   - 算法：K-Means、DBSCAN、层次聚类

2. **降维（Dimensionality Reduction）**
   - 减少特征数量
   - 例子：数据可视化、特征提取、去噪
   - 算法：PCA、t-SNE、UMAP

3. **异常检测（Anomaly Detection）**
   - 识别异常样本
   - 例子：欺诈检测、系统故障检测
   - 算法：Isolation Forest、One-Class SVM

**示例：**
```python
# 聚类
X = [[特征1, 特征2], ...]
# 没有y标签！

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)  # 自动发现3个簇
```

**对比表格：**

| 特征 | 监督学习 | 非监督学习 |
|------|---------|-----------|
| 标签 | 有 | 无 |
| 目标 | 预测 | 发现模式 |
| 难度 | 相对简单 | 相对困难 |
| 评估 | 明确指标（准确率、RMSE） | 模糊指标（轮廓系数） |
| 应用 | 分类、回归 | 聚类、降维 |
| 例子 | 垃圾邮件检测 | 客户细分 |

**半监督学习：**
- 部分数据有标签，部分无标签
- 利用无标签数据提升性能
- 例子：Self-training、Co-training

**强化学习：**
- 通过与环境交互学习
- 获得奖励或惩罚
- 例子：AlphaGo、自动驾驶、游戏AI

</details>

### 3. 什么是偏差（Bias）和方差（Variance）？

<details>
<summary>点击查看答案</summary>

**偏差（Bias）**

**定义：**
- 模型预测值的期望与真实值之间的差距
- 衡量模型的"准确性"
- 反映模型的假设是否合理

**高偏差的特征：**
- 欠拟合（Underfitting）
- 模型过于简单
- 训练误差和测试误差都很高
- 例子：用线性模型拟合非线性数据

**方差（Variance）**

**定义：**
- 模型预测值的变化程度
- 衡量模型的"稳定性"
- 反映模型对训练数据的敏感度

**高方差的特征：**
- 过拟合（Overfitting）
- 模型过于复杂
- 训练误差很低，测试误差很高
- 对训练数据中的噪声过度敏感

**偏差-方差权衡（Bias-Variance Tradeoff）**

```
总误差 = 偏差² + 方差 + 不可约误差

低复杂度模型：高偏差、低方差
高复杂度模型：低偏差、高方差
最优模型：平衡偏差和方差
```

**可视化：**
```
                靶心（真实值）
                    ○

高偏差、低方差        低偏差、低方差
    ●●●                  ○○○
    ●●●         vs       ○●○
    ●●●                  ○○○
（偏离中心，但集中）   （靠近中心，且集中）✓

高偏差、高方差        低偏差、高方差
    ●  ●               ○ ● ○
  ●      ●      vs       ● ●
    ●  ●                 ○ ●
（偏离且分散）        （靠近但分散）
```

**如何诊断：**

```python
# 1. 学习曲线
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# 高偏差：训练和验证误差都高，且接近
# 高方差：训练误差低，验证误差高，差距大
```

**如何解决：**

**高偏差（欠拟合）：**
1. 增加模型复杂度
2. 添加更多特征
3. 减少正则化
4. 使用更复杂的模型

**高方差（过拟合）：**
1. 增加训练数据
2. 特征选择
3. 增加正则化
4. 降低模型复杂度
5. Early stopping
6. Dropout

**示例：**
```python
# 高偏差问题
simple_model = LinearRegression()  # 太简单
# 解决：使用多项式特征
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# 高方差问题
complex_model = DecisionTreeClassifier(max_depth=None)  # 太复杂
# 解决：正则化
better_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20
)
```

</details>

### 4. 解释精确率（Precision）和召回率（Recall）

<details>
<summary>点击查看答案</summary>

**混淆矩阵（Confusion Matrix）**

```
                 预测
             正类    负类
实际  正类    TP     FN
      负类    FP     TN

TP (True Positive): 真正例 - 正确预测为正类
TN (True Negative): 真负例 - 正确预测为负类
FP (False Positive): 假正例 - 错误预测为正类（Type I Error）
FN (False Negative): 假负例 - 错误预测为负类（Type II Error）
```

**精确率（Precision）**

**定义：**
```
Precision = TP / (TP + FP)
```

**含义：**
- 预测为正类的样本中，真正为正类的比例
- "查准率"
- 回答问题："预测的正类中有多少是对的？"

**何时重要：**
- 假阳性（FP）代价高
- 例子：
  - 垃圾邮件检测：误判正常邮件为垃圾邮件很糟糕
  - 推荐系统：推荐不喜欢的内容影响用户体验

**召回率（Recall / Sensitivity / True Positive Rate）**

**定义：**
```
Recall = TP / (TP + FN)
```

**含义：**
- 实际为正类的样本中，被正确预测的比例
- "查全率"
- 回答问题："实际的正类中有多少被找到了？"

**何时重要：**
- 假阴性（FN）代价高
- 例子：
  - 疾病诊断：漏诊重病很危险
  - 欺诈检测：漏掉欺诈交易损失大

**F1 Score**

**定义：**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**含义：**
- 精确率和召回率的调和平均
- 平衡精确率和召回率
- [0, 1]，越大越好

**Precision-Recall权衡：**
```
提高阈值 → 精确率↑，召回率↓
降低阈值 → 精确率↓，召回率↑
```

**实例：**

**场景：癌症检测**
- 100个患者，10个患癌（正类）

**模型A：保守**
- 预测2个患癌，都正确
- TP=2, FP=0, FN=8
- Precision = 2/2 = 100% （预测的都对）
- Recall = 2/10 = 20% （漏了很多）
- 问题：漏诊太多，不可接受

**模型B：激进**
- 预测20个患癌，10个正确
- TP=10, FP=10, FN=0
- Precision = 10/20 = 50%（误诊多）
- Recall = 10/10 = 100%（全找到）
- 问题：误诊多，但至少不漏诊

**代码示例：**
```python
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, precision_recall_curve

# 计算指标
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 分类报告（综合）
print(classification_report(y_true, y_pred))

# PR曲线
y_proba = clf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# 调整阈值
threshold = 0.3  # 降低阈值提高召回率
y_pred_adjusted = (y_proba >= threshold).astype(int)
```

**其他相关指标：**

**特异度（Specificity）**
```
Specificity = TN / (TN + FP)
```
- 实际为负类的样本中，被正确预测的比例

**准确率（Accuracy）**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- 所有预测中正确的比例
- 不平衡数据时不可靠

**选择指标的建议：**

| 场景 | 优先指标 | 原因 |
|------|---------|------|
| 垃圾邮件检测 | Precision | 误判正常邮件代价高 |
| 疾病诊断 | Recall | 漏诊代价高 |
| 欺诈检测 | Recall | 漏掉欺诈损失大 |
| 推荐系统 | Precision | 推荐不相关内容影响体验 |
| 平衡数据 | F1 Score | 综合考虑 |
| 不平衡数据 | F1 / ROC-AUC | 不要用Accuracy |

</details>

### 5. 什么是交叉验证？为什么需要它？

<details>
<summary>点击查看答案</summary>

**交叉验证（Cross-Validation）**是一种评估模型泛化能力的统计方法。

**为什么需要交叉验证？**

1. **单次分割的问题：**
   - 测试集可能不代表总体
   - 结果依赖于随机分割
   - 浪费数据（测试集不用于训练）

2. **交叉验证的优势：**
   - 更可靠的性能估计
   - 充分利用数据
   - 减少方差

**K折交叉验证（K-Fold CV）**

**流程：**
```
1. 将数据分为K个相等的折（fold）
2. 每次使用K-1个折训练，1个折验证
3. 重复K次，每个折都作为验证集一次
4. 平均K次结果
```

**可视化：**
```
数据：[1][2][3][4][5]  (5折)

迭代1: Train[2][3][4][5]  Test[1]
迭代2: Train[1][3][4][5]  Test[2]
迭代3: Train[1][2][4][5]  Test[3]
迭代4: Train[1][2][3][5]  Test[4]
迭代5: Train[1][2][3][4]  Test[5]

最终分数 = 平均(5次测试分数)
```

**代码示例：**
```python
from sklearn.model_selection import cross_val_score, KFold

# 方法1：简单使用
scores = cross_val_score(
    model, X, y,
    cv=5,              # 5折
    scoring='accuracy'
)

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# 方法2：手动控制
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"Fold {fold}: {score:.4f}")
```

**分层K折交叉验证（Stratified K-Fold）**

**用途：**
- 分类任务
- 保持每个折中的类别比例

**代码：**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
```

**留一法交叉验证（Leave-One-Out CV）**

**定义：**
- K = N（样本数量）
- 每次只用一个样本验证

**优点：**
- 最大化训练数据
- 无偏估计

**缺点：**
- 计算成本高（需训练N次）
- 方差大

**代码：**
```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

**时间序列交叉验证（Time Series CV）**

**特点：**
- 不能随机打乱数据
- 保持时间顺序
- 只用过去预测未来

**可视化：**
```
数据：[1][2][3][4][5]

迭代1: Train[1]         Test[2]
迭代2: Train[1][2]      Test[3]
迭代3: Train[1][2][3]   Test[4]
迭代4: Train[1][2][3][4] Test[5]
```

**代码：**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
```

**多指标交叉验证**

```python
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

scores = cross_validate(model, X, y, cv=5, scoring=scoring)

for metric in scoring:
    print(f"{metric}: {scores['test_' + metric].mean():.4f}")
```

**选择K值的建议：**

- **K=5 或 K=10**：常用选择，平衡偏差和方差
- **小数据集**：大K（如K=10或LOO）
- **大数据集**：小K（如K=3或K=5）节省时间
- **计算受限**：小K

**交叉验证 vs 训练/验证/测试分割：**

```python
# 方法1：简单分割
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# 训练
model.fit(X_train, y_train)

# 调参（在验证集）
best_model = tune_hyperparameters(X_train, y_train, X_val, y_val)

# 最终评估（在测试集，只用一次！）
final_score = best_model.score(X_test, y_test)

# 方法2：交叉验证
# 用于模型选择和超参数调优
scores = cross_val_score(model, X_train, y_train, cv=5)

# 最终评估仍在独立测试集
final_score = model.score(X_test, y_test)
```

**注意事项：**

1. **数据泄露：**
   - 在CV分割前不要做预处理
   - 使用Pipeline确保正确

2. **分层：**
   - 分类任务使用Stratified K-Fold
   - 保持类别分布

3. **时间序列：**
   - 不要随机打乱
   - 使用TimeSeriesSplit

4. **计算成本：**
   - CV增加K倍计算时间
   - 大数据集可能很慢

</details>

---

## 算法原理题

### 6. 解释逻辑回归的工作原理

<details>
<summary>点击查看答案</summary>

**逻辑回归（Logistic Regression）**是一种用于二分类的线性模型。

**核心思想：**
- 使用Sigmoid函数将线性组合映射到[0, 1]区间
- 输出可解释为概率
- 通过阈值（通常0.5）进行分类

**数学原理：**

**1. 线性组合：**
```
z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
  = w^T x + b
```

**2. Sigmoid函数：**
```
σ(z) = 1 / (1 + e^(-z))

特点：
- 输出范围：(0, 1)
- σ(0) = 0.5
- z → +∞时，σ(z) → 1
- z → -∞时，σ(z) → 0
```

**3. 概率预测：**
```
P(y=1|x) = σ(w^T x + b)
P(y=0|x) = 1 - P(y=1|x)
```

**4. 决策边界：**
```
预测类别 = 1  if P(y=1|x) ≥ 0.5
         = 0  otherwise

当 w^T x + b = 0 时，P = 0.5（决策边界）
```

**损失函数（对数损失/交叉熵）：**

```
单个样本：
L(y, ŷ) = -[y log(ŷ) + (1-y) log(1-ŷ)]

全部样本：
J(w) = -(1/m) Σ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

其中 ŷᵢ = σ(w^T xᵢ + b)
```

**为什么用对数损失？**
- 凸函数，便于优化
- 惩罚错误的自信预测
- 概率解释性强

**优化（梯度下降）：**

```
梯度：
∂J/∂w = (1/m) X^T (σ(Xw) - y)

更新规则：
w := w - α × ∂J/∂w

其中α是学习率
```

**正则化：**

**L2正则化（Ridge）：**
```
J(w) = -(1/m) Σ[...] + (λ/2m) Σwᵢ²
```

**L1正则化（Lasso）：**
```
J(w) = -(1/m) Σ[...] + (λ/m) Σ|wᵢ|
```

**多分类（One-vs-Rest或Softmax）：**

**One-vs-Rest：**
- 训练K个二分类器（K类）
- 每个分类器：该类 vs 其他类
- 预测时选择概率最高的类

**Softmax Regression：**
```
P(y=k|x) = e^(w_k^T x) / Σⱼ e^(w_j^T x)

损失函数（交叉熵）：
J(w) = -(1/m) Σᵢ Σₖ 1{yᵢ=k} log P(y=k|xᵢ)
```

**代码实现：**

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# 创建模型
lr = LogisticRegression(
    C=1.0,                    # 正则化强度的倒数（C越小越强）
    penalty='l2',             # 'l1', 'l2', 'elasticnet', 'none'
    solver='lbfgs',           # 优化器
    max_iter=100,
    random_state=42
)

# 训练
lr.fit(X_train, y_train)

# 预测类别
y_pred = lr.predict(X_test)

# 预测概率
y_proba = lr.predict_proba(X_test)
print(f"Probability of class 1: {y_proba[0, 1]:.4f}")

# 查看参数
print(f"Coefficients: {lr.coef_}")        # 权重
print(f"Intercept: {lr.intercept_}")      # 截距

# 手动预测（验证）
z = np.dot(X_test[0], lr.coef_[0]) + lr.intercept_[0]
prob = 1 / (1 + np.exp(-z))
print(f"Manual probability: {prob:.4f}")
```

**从头实现：**

```python
import numpy as np

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降
        for _ in range(self.n_iterations):
            # 前向传播
            linear = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear)

            # 计算梯度
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear)

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

# 使用
model = LogisticRegressionFromScratch()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**优点：**
1. 简单、快速
2. 输出概率
3. 可解释性强（系数表示特征重要性）
4. 适合线性可分数据
5. 在线学习

**缺点：**
1. 假设线性可分
2. 对特征工程要求高
3. 不能处理复杂的非线性关系
4. 对异常值敏感

**应用场景：**
- 垃圾邮件检测
- 疾病诊断
- 信用评分
- 点击率预测
- 客户流失预测

**决策边界可视化：**

```python
import matplotlib.pyplot as plt

# 训练模型（使用2个特征便于可视化）
lr = LogisticRegression()
lr.fit(X_train[:, :2], y_train)

# 创建网格
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 预测网格上的每个点
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', edgecolor='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
```

**与线性回归的区别：**

| 特征 | 线性回归 | 逻辑回归 |
|------|---------|---------|
| 任务 | 回归 | 分类 |
| 输出 | 连续值 | 概率[0,1] |
| 激活函数 | 无（线性） | Sigmoid |
| 损失函数 | MSE | 交叉熵 |
| 应用 | 价格预测 | 类别预测 |

</details>

### 7. SVM如何处理非线性数据？

<details>
<summary>点击查看答案</summary>

**支持向量机（SVM）**通过**核技巧（Kernel Trick）**处理非线性数据。

**基本思想：**
1. 将数据映射到高维空间
2. 在高维空间中找线性分隔超平面
3. 映射回原空间成为非线性边界

**核技巧（Kernel Trick）：**

**问题：**
- 显式映射到高维空间计算成本高
- 高维空间的点积计算困难

**解决方案：**
- 使用核函数直接计算高维空间的点积
- 无需显式计算映射

**数学表示：**
```
传统：φ(x) · φ(y)  （先映射再点积，慢）
核技巧：K(x, y)     （直接计算，快）

其中 K(x, y) = φ(x) · φ(y)
```

**常用核函数：**

**1. 线性核（Linear Kernel）**
```
K(x, y) = x · y

用途：
- 线性可分数据
- 高维数据（特征数 >> 样本数）
- 文本分类

特点：
- 最快
- 无参数
- 等价于线性SVM
```

**2. 多项式核（Polynomial Kernel）**
```
K(x, y) = (γ × x · y + r)^d

参数：
- d：多项式阶数（通常2或3）
- γ：核系数
- r：独立项

用途：
- 图像处理
- 自然语言处理

示例（d=2）：
输入：x = [x₁, x₂]
映射：[x₁², √2·x₁·x₂, x₂², √2·r·x₁, √2·r·x₂, r²]
```

**3. RBF核（Radial Basis Function / 高斯核）**
```
K(x, y) = exp(-γ ||x - y||²)

参数：
- γ (gamma)：核系数，控制影响范围
  - γ大：影响范围小，复杂边界（可能过拟合）
  - γ小：影响范围大，平滑边界（可能欠拟合）

用途：
- **最常用的核**
- 适合大多数非线性问题
- 通用性强

特点：
- 无限维映射
- 相似度度量
- 高斯分布权重
```

**4. Sigmoid核**
```
K(x, y) = tanh(γ × x · y + r)

用途：
- 神经网络替代
- 较少使用

特点：
- 类似神经网络激活函数
- 不一定是正定的
```

**代码示例：**

```python
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
import matplotlib.pyplot as plt
import numpy as np

# 生成非线性数据
X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)

# 1. 线性核（会失败）
svm_linear = SVC(kernel='linear')
svm_linear.fit(X, y)
print(f"Linear kernel accuracy: {svm_linear.score(X, y):.4f}")

# 2. 多项式核
svm_poly = SVC(kernel='poly', degree=3, gamma='auto')
svm_poly.fit(X, y)
print(f"Polynomial kernel accuracy: {svm_poly.score(X, y):.4f}")

# 3. RBF核（最常用）
svm_rbf = SVC(kernel='rbf', gamma='auto')
svm_rbf.fit(X, y)
print(f"RBF kernel accuracy: {svm_rbf.score(X, y):.4f}")

# 可视化决策边界
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='black')

    # 支持向量
    plt.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=200, linewidths=1, facecolors='none', edgecolors='green')

    plt.title(title)
    plt.show()

# 对比不同核
plot_decision_boundary(svm_linear, X, y, 'Linear Kernel (Poor)')
plot_decision_boundary(svm_rbf, X, y, 'RBF Kernel (Good)')
```

**核函数选择：**

```python
# 比较不同核
from sklearn.model_selection import cross_val_score

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
scores = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, gamma='auto')
    score = cross_val_score(svm, X, y, cv=5).mean()
    scores[kernel] = score
    print(f"{kernel}: {score:.4f}")

# 选择最佳核
best_kernel = max(scores, key=scores.get)
print(f"\nBest kernel: {best_kernel}")
```

**Gamma参数的影响：**

```python
# 不同gamma值
gammas = [0.001, 0.01, 0.1, 1, 10]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, gamma in enumerate(gammas):
    svm = SVC(kernel='rbf', gamma=gamma)
    svm.fit(X, y)

    # 绘制决策边界
    ax = axes[i]
    # ... (绘图代码)
    ax.set_title(f'gamma = {gamma}')

plt.show()

# 观察：
# gamma小：平滑边界（可能欠拟合）
# gamma大：复杂边界（可能过拟合）
```

**C参数的影响：**

```python
# C控制错分类的惩罚
# C大：硬间隔，严格分类（可能过拟合）
# C小：软间隔，容忍错分类（可能欠拟合）

Cs = [0.1, 1, 10, 100]

for C in Cs:
    svm = SVC(kernel='rbf', C=C, gamma='auto')
    svm.fit(X, y)
    print(f"C={C}: accuracy={svm.score(X, y):.4f}, "
          f"support vectors={len(svm.support_vectors_)}")
```

**超参数调优：**

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

# 网格搜索
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# 使用最佳模型
best_svm = grid_search.best_estimator_
test_score = best_svm.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

**自定义核函数：**

```python
# 自定义核
def my_kernel(X, Y):
    """
    自定义核函数示例：指数核
    """
    return np.exp(-0.1 * np.linalg.norm(X[:, np.newaxis] - Y, axis=2))

# 使用
svm_custom = SVC(kernel=my_kernel)
svm_custom.fit(X_train, y_train)
```

**核技巧的直观理解：**

```python
# 示例：圆形数据（非线性可分）
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3)

# 原始空间（2D）：不可分
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Original Space (2D) - Not Linearly Separable')
plt.show()

# 手动映射到3D
# φ(x) = [x₁, x₂, x₁² + x₂²]
X_3d = np.c_[X, X[:, 0]**2 + X[:, 1]**2]

# 3D空间：线性可分！
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y)
ax.set_title('Mapped Space (3D) - Linearly Separable!')
plt.show()

# RBF核自动完成类似映射（但在无限维）
svm = SVC(kernel='rbf')
svm.fit(X, y)
```

**核函数选择指南：**

| 核函数 | 使用场景 | 优点 | 缺点 |
|--------|---------|------|------|
| Linear | 线性可分、高维数据、文本 | 快速、可解释 | 只能处理线性 |
| Polynomial | 图像、NLP | 灵活性好 | 参数多、过拟合风险 |
| RBF | **通用首选** | 强大、参数少 | 需要调参、计算慢 |
| Sigmoid | 神经网络替代 | 类似NN | 不稳定、少用 |

**总结：**
1. **RBF核是默认首选**，适合大多数情况
2. **数据量大**时考虑LinearSVC（更快）
3. **特征数远大于样本数**时用Linear kernel
4. **先用RBF试验**，再考虑其他核
5. **一定要调整C和gamma参数**

</details>

### 8. Random Forest vs Gradient Boosting的区别？

<details>
<summary>点击查看答案</summary>

**Random Forest（随机森林）**和**Gradient Boosting（梯度提升）**都是基于决策树的集成学习方法，但策略完全不同。

**核心区别：**

| 特征 | Random Forest | Gradient Boosting |
|------|--------------|-------------------|
| **集成策略** | Bagging（并行） | Boosting（串行） |
| **树的构建** | 独立并行 | 顺序依赖 |
| **训练目标** | 降低方差 | 降低偏差 |
| **基学习器** | 深树（低偏差高方差） | 浅树（高偏差低方差） |
| **最终预测** | 平均/投票 | 加权求和 |
| **过拟合风险** | 低 | 高（需调参） |
| **训练速度** | 快（可并行） | 慢（串行） |
| **可解释性** | 低 | 低 |

---

**Random Forest（随机森林）**

**算法流程：**
```
1. Bootstrap采样：从训练集抽取N个子集（有放回）
2. 特征随机：每次分割只考虑随机子集特征
3. 并行训练：独立训练N棵深决策树
4. 聚合预测：
   - 分类：投票
   - 回归：平均
```

**核心特点：**

**1. Bootstrap Aggregating (Bagging)**
```python
# 每棵树使用不同的随机子集
样本1 = bootstrap_sample(原始数据)
样本2 = bootstrap_sample(原始数据)
样本3 = bootstrap_sample(原始数据)
...

树1.fit(样本1)
树2.fit(样本2)
树3.fit(样本3)

预测 = average([树1.predict(), 树2.predict(), 树3.predict()])
```

**2. 特征随机性**
```python
# 每次分割只考虑部分特征
总特征数 = n
每次分割考虑 = sqrt(n)  # 分类
             = n/3      # 回归
```

**3. Out-of-Bag (OOB) 评估**
```python
# 每棵树约有37%数据未被使用
# 可用这些数据评估，无需额外验证集
rf = RandomForestClassifier(oob_score=True)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")
```

**代码示例：**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,        # 树的数量
    max_depth=None,          # 树的深度（通常不限制）
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',     # 每次分割考虑的特征数
    bootstrap=True,          # 是否Bootstrap采样
    oob_score=True,          # 计算OOB分数
    random_state=42,
    n_jobs=-1                # 并行训练
)

rf.fit(X_train, y_train)

# 特征重要性
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(10))
```

---

**Gradient Boosting（梯度提升）**

**算法流程：**
```
1. 初始化：F₀(x) = 初始预测（如均值）
2. 对于 m=1 到 M：
   a. 计算负梯度（伪残差）：rₘ = -∂L/∂F
   b. 训练新树拟合残差：hₘ
   c. 更新模型：Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)
3. 最终模型：F(x) = F₀(x) + η∑hₘ(x)
```

**核心特点：**

**1. 串行训练，拟合残差**
```python
# 伪代码
predictions = [initial_prediction] * n_samples

for tree in range(n_trees):
    # 计算当前残差
    residuals = y_true - predictions

    # 训练树拟合残差
    new_tree = DecisionTree()
    new_tree.fit(X, residuals)

    # 更新预测
    predictions += learning_rate * new_tree.predict(X)
```

**2. 学习率（Learning Rate）**
```python
# 控制每棵树的贡献
# 小学习率 + 多树 = 更好的泛化
F(x) = F₀ + η₁h₁ + η₂h₂ + ... + ηₘhₘ
```

**3. 浅树 + 多轮迭代**
```python
# 通常使用浅树（max_depth=3-5）
# 通过多轮迭代逐步改进
```

**代码示例：**
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,        # 树的数量
    learning_rate=0.1,       # 学习率（shrinkage）
    max_depth=3,             # 树的深度（通常3-5）
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,           # 样本采样比例
    max_features='sqrt',
    random_state=42
)

gb.fit(X_train, y_train)

# 学习曲线
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    gb, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# 绘制
plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.legend()
plt.show()
```

---

**详细对比：**

**1. 并行 vs 串行**

```python
# Random Forest: 并行训练
from joblib import Parallel, delayed

def train_tree(data):
    tree = DecisionTree()
    tree.fit(data)
    return tree

# 所有树同时训练
trees = Parallel(n_jobs=-1)(
    delayed(train_tree)(bootstrap_sample(data))
    for _ in range(n_trees)
)

# Gradient Boosting: 串行训练
trees = []
predictions = initial_predictions

for _ in range(n_trees):
    residuals = y - predictions
    tree = DecisionTree()
    tree.fit(X, residuals)  # 依赖前面的预测
    trees.append(tree)
    predictions += learning_rate * tree.predict(X)
```

**2. 偏差-方差权衡**

```python
# Random Forest: 降低方差
# - 深树（低偏差，高方差）
# - 通过平均降低方差
单棵树：高方差（易过拟合）
多棵树平均：低方差（鲁棒）

# Gradient Boosting: 降低偏差
# - 浅树（高偏差，低方差）
# - 通过累加降低偏差
初始：高偏差（欠拟合）
逐步迭代：偏差降低
但可能增加方差（过拟合风险）
```

**3. 过拟合风险**

```python
# Random Forest: 抗过拟合
rf = RandomForestClassifier(
    n_estimators=1000,  # 树越多越好（几乎不过拟合）
    max_depth=None      # 可以不限制深度
)

# Gradient Boosting: 易过拟合
gb = GradientBoostingClassifier(
    n_estimators=1000,  # 树太多会过拟合！
    learning_rate=0.1,  # 需要小心调整
    max_depth=3,        # 需要限制深度
    subsample=0.8,      # 需要采样
    early_stopping=True # 需要早停
)
```

**4. 超参数调优**

```python
# Random Forest: 相对简单
param_grid_rf = {
    'n_estimators': [100, 200, 500],  # 通常越多越好
    'max_depth': [10, 20, None],      # 影响不大
    'max_features': ['sqrt', 'log2']  # sqrt通常最好
}

# Gradient Boosting: 复杂且关键
param_grid_gb = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],  # 关键！
    'max_depth': [3, 5, 7],              # 关键！
    'subsample': [0.6, 0.8, 1.0],
    'min_samples_split': [10, 20],
}

# GB需要平衡：
# learning_rate ↓ → n_estimators ↑
```

---

**XGBoost：Gradient Boosting的改进**

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,  # 列采样（类似RF）
    reg_alpha=0.1,         # L1正则化
    reg_lambda=1.0,        # L2正则化
    random_state=42,
    n_jobs=-1              # XGBoost支持并行！
)

xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,  # 早停
    verbose=False
)

print(f"Best iteration: {xgb.best_iteration}")
```

**XGBoost的优势：**
1. **速度更快**：并行化、近似算法
2. **正则化**：L1、L2防过拟合
3. **处理缺失值**：自动学习最佳方向
4. **列采样**：类似RF的特征随机
5. **早停**：避免过拟合

---

**选择建议：**

**使用 Random Forest 当：**
- 追求稳定性和鲁棒性
- 不想花太多时间调参
- 需要并行训练（大数据）
- 过拟合风险高
- 需要特征重要性分析

**使用 Gradient Boosting / XGBoost 当：**
- 追求最高精度（Kaggle竞赛）
- 愿意花时间调参
- 结构化表格数据
- 计算资源足够
- 特征工程做得好

**实际对比：**

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"{name}:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std: {scores.std():.4f}")
    print()

# 通常结果：
# Random Forest: 稳定，不错的准确率
# Gradient Boosting: 更高准确率，但训练慢
# XGBoost: 最高准确率，训练快
```

**总结：**
- **Random Forest**: 可靠的默认选择，易用
- **Gradient Boosting**: 高精度，需要调参
- **XGBoost**: Kaggle竞赛首选，综合最优

</details>

---

## 实践问题题

### 9. 如何处理不平衡数据？

<details>
<summary>点击查看答案</summary>

**不平衡数据（Imbalanced Data）**是指不同类别的样本数量差异很大，如欺诈检测（99%正常，1%欺诈）、疾病诊断等。

**问题：**
- 模型倾向于预测多数类
- 准确率高但无意义（预测全部为多数类也有99%准确率）
- 少数类（通常是重要的）被忽略

---

**解决方法：**

### 1. 重采样方法

**1.1 过采样（Over-sampling）**

**随机过采样：**
```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print(f"Original: {pd.Series(y_train).value_counts()}")
print(f"Resampled: {pd.Series(y_resampled).value_counts()}")
```

**SMOTE（Synthetic Minority Over-sampling Technique）：**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    sampling_strategy='auto',  # 平衡到1:1
    k_neighbors=5,             # 最近邻数量
    random_state=42
)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# SMOTE原理：
# 1. 对每个少数类样本
# 2. 找到k个最近邻（同类）
# 3. 随机选择一个邻居
# 4. 在两点之间随机生成新样本
# x_new = x + λ × (x_neighbor - x), λ ∈ [0, 1]
```

**ADASYN（Adaptive Synthetic Sampling）：**
```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# 根据难度生成样本
# 难分类的区域生成更多样本
```

**1.2 欠采样（Under-sampling）**

**随机欠采样：**
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# 缺点：丢失信息
```

**Tomek Links：**
```python
from imblearn.under_sampling import TomekLinks

tomek = TomekLinks()
X_resampled, y_resampled = tomek.fit_resample(X_train, y_train)

# 移除边界附近的多数类样本
# 使决策边界更清晰
```

**NearMiss：**
```python
from imblearn.under_sampling import NearMiss

nm = NearMiss(version=1)
X_resampled, y_resampled = nm.fit_resample(X_train, y_train)

# 保留接近少数类的多数类样本
```

**1.3 组合采样**

```python
from imblearn.combine import SMOTEENN, SMOTETomek

# SMOTE + ENN（清理）
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

# SMOTE + Tomek Links
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
```

---

### 2. 算法层面

**2.1 类权重（Class Weights）**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# 方法1：自动平衡
rf = RandomForestClassifier(
    class_weight='balanced',  # 自动计算权重
    random_state=42
)

# 方法2：手动设置
# 计算类权重
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))
# 例如：{0: 0.52, 1: 5.2}（少数类权重更高）

rf = RandomForestClassifier(
    class_weight=class_weight_dict,
    random_state=42
)

# 方法3：自定义权重
rf = RandomForestClassifier(
    class_weight={0: 1, 1: 10},  # 少数类权重10倍
    random_state=42
)

rf.fit(X_train, y_train)
```

**支持class_weight的算法：**
- LogisticRegression
- RandomForestClassifier
- GradientBoostingClassifier
- SVC

**2.2 阈值调整**

```python
# 默认阈值0.5可能不适合不平衡数据
y_proba = clf.predict_proba(X_test)[:, 1]

# 降低阈值增加召回率
threshold = 0.3
y_pred = (y_proba >= threshold).astype(int)

# 找到最优阈值
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# F1最大的阈值
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {best_threshold:.4f}")

# 使用最优阈值
y_pred = (y_proba >= best_threshold).astype(int)
```

**2.3 XGBoost的scale_pos_weight**

```python
from xgboost import XGBClassifier

# 计算scale_pos_weight
# = 负类样本数 / 正类样本数
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

xgb.fit(X_train, y_train)
```

---

### 3. 评估指标

**不要使用准确率！**

```python
# 错误示例
# 99% 负类，1% 正类
# 全预测为负类 → 准确率99%（但完全没用）

# 使用以下指标：
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# 1. 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 2. 分类报告
print(classification_report(y_test, y_pred))

# 3. F1 Score（推荐）
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

# 4. ROC-AUC
y_proba = clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

# 5. PR-AUC（不平衡数据更好）
pr_auc = average_precision_score(y_test, y_proba)
print(f"PR-AUC: {pr_auc:.4f}")

# 6. Balanced Accuracy
from sklearn.metrics import balanced_accuracy_score
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.4f}")
```

---

### 4. 集成方法

**4.1 Balanced Random Forest**

```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

brf.fit(X_train, y_train)
```

**4.2 EasyEnsemble**

```python
from imblearn.ensemble import EasyEnsembleClassifier

eec = EasyEnsembleClassifier(
    n_estimators=10,
    random_state=42,
    n_jobs=-1
)

eec.fit(X_train, y_train)
```

---

### 5. 异常检测方法

**将问题视为异常检测：**

```python
from sklearn.ensemble import IsolationForest

# 只在正常类上训练
X_normal = X_train[y_train == 0]

iso_forest = IsolationForest(
    contamination=0.01,  # 预期异常比例
    random_state=42
)

iso_forest.fit(X_normal)

# 预测（1:正常，-1:异常）
y_pred = iso_forest.predict(X_test)
y_pred = (y_pred == -1).astype(int)  # 转换为0/1
```

---

### 6. Pipeline示例

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 创建Pipeline
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 训练
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)

# 注意：SMOTE只应用于训练集！
# Pipeline自动处理
```

---

### 7. 实战策略

```python
# 完整的不平衡数据处理流程

# 1. 检查数据分布
print(pd.Series(y_train).value_counts())
print(pd.Series(y_train).value_counts(normalize=True))

# 2. 数据分割（分层）
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # 保持类别比例！
    random_state=42
)

# 3. 尝试多种方法
from sklearn.model_selection import cross_val_score

methods = {
    'Baseline': RandomForestClassifier(random_state=42),
    'Class Weight': RandomForestClassifier(class_weight='balanced', random_state=42),
    'SMOTE': ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'Under-sampling': ImbPipeline([
        ('rus', RandomUnderSampler(random_state=42)),
        ('clf', RandomForestClassifier(random_state=42))
    ])
}

results = {}
for name, model in methods.items():
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='f1'  # 使用F1而不是accuracy！
    )
    results[name] = scores.mean()
    print(f"{name}: F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

# 4. 选择最佳方法
best_method = max(results, key=results.get)
print(f"\nBest method: {best_method}")

# 5. 最终评估
best_model = methods[best_method]
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nTest Set Results:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

---

**总结：**

**推荐策略（按优先级）：**

1. **首先尝试class_weight**（简单有效）
2. **SMOTE + 分类器**（通常效果好）
3. **调整阈值**（灵活控制precision/recall）
4. **XGBoost + scale_pos_weight**（Kaggle常用）
5. **集成方法**（追求高精度）

**评估指标：**
- 不要用Accuracy
- 使用F1、ROC-AUC、PR-AUC
- 根据业务需求选择precision或recall

**注意事项：**
- 采样只在训练集！
- 使用分层分割
- 交叉验证时也要分层

</details>

### 10. 如何检测和防止数据泄露？

<details>
<summary>点击查看答案</summary>

**数据泄露（Data Leakage）**是指训练过程中使用了测试时不可得的信息，导致模型在训练时表现很好，但实际部署时表现很差。

---

**常见数据泄露类型：**

### 1. 时间泄露（Temporal Leakage）

**错误示例：**
```python
# 错误：使用未来信息
df['next_month_sales'] = df['sales'].shift(-1)  # 泄露！

# 错误：时间序列随机分割
X_train, X_test = train_test_split(time_series_data)  # 错误！

# 错误：使用全部数据做特征工程
df['sales_mean'] = df.groupby('product')['sales'].transform('mean')  # 泄露！
# 测试集的平均值包含了测试集本身的信息
```

**正确做法：**
```python
# 正确：按时间分割
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# 正确：只使用训练集的统计信息
train_stats = train_data.groupby('product')['sales'].mean()
test_data['sales_mean'] = test_data['product'].map(train_stats)

# 正确：使用滞后特征
df['last_month_sales'] = df['sales'].shift(1)  # 只用过去
```

---

### 2. 预处理泄露

**错误示例：**
```python
# 错误：在分割前做缩放
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 用了所有数据的均值/标准差！

X_train, X_test = train_test_split(X_scaled, y)
# 测试集的缩放使用了测试集本身的信息

# 错误：在分割前填充缺失值
X_filled = X.fillna(X.mean())  # 用了所有数据的均值！
X_train, X_test = train_test_split(X_filled, y)
```

**正确做法：**
```python
# 方法1：先分割，再预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 在训练集上fit
scaler = StandardScaler()
scaler.fit(X_train)

# 分别transform
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 方法2：使用Pipeline（推荐）
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 自动先fit再transform
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
pipeline.predict(X_test)
```

---

### 3. 特征工程泄露

**错误示例：**
```python
# 错误：使用整个数据集做特征选择
from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # 用了所有数据！

X_train, X_test = train_test_split(X_selected, y)

# 错误：使用目标变量的统计信息
# 例如：Target Encoding在全部数据上计算
df['category_mean'] = df.groupby('category')['target'].transform('mean')
```

**正确做法：**
```python
# 正确：只在训练集上做特征选择
X_train, X_test, y_train, y_test = train_test_split(X, y)

selector = SelectKBest(k=10)
selector.fit(X_train, y_train)  # 只用训练集

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# 正确：Target Encoding只用训练集
# 训练集
train_means = df_train.groupby('category')['target'].mean()

# 测试集
df_test['category_mean'] = df_test['category'].map(train_means)
```

---

### 4. 交叉验证泄露

**错误示例：**
```python
# 错误：在CV前做预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 泄露！

scores = cross_val_score(clf, X_scaled, y, cv=5)
# 每个fold的验证集使用了验证集本身的信息
```

**正确做法：**
```python
# 正确：使用Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

scores = cross_val_score(pipeline, X, y, cv=5)
# Pipeline确保每个fold独立做预处理
```

---

### 5. 目标泄露

**错误示例：**
```python
# 错误：特征包含目标信息
# 例如：预测信用卡欺诈
X['is_fraudulent_transaction'] = y  # 完全泄露！

# 微妙的泄露：
# 特征：'total_purchase_amount'
# 目标：是否购买
# 问题：如果 total_purchase_amount > 0，必定购买
#       这个特征在预测时不可得
```

**检测方法：**
```python
# 检查特征与目标的相关性
from sklearn.metrics import mutual_info_score

for col in X.columns:
    mi = mutual_info_score(X[col], y)
    if mi > 0.9:  # 异常高的相关性
        print(f"Warning: {col} might be leaking! MI = {mi:.4f}")

# 检查特征重要性
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# 如果某特征重要性异常高（如>0.9），可能泄露
print(importance.head(10))
```

---

### 6. 测试集污染

**错误示例：**
```python
# 错误：测试集参与训练
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 错误地使用了测试集
X_all = pd.concat([X_train, X_test])
clf.fit(X_all, pd.concat([y_train, y_test]))  # 严重泄露！

# 错误：在测试集上调参
for param in param_grid:
    clf.set_params(**param)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)  # 用测试集调参！
    if score > best_score:
        best_param = param
```

**正确做法：**
```python
# 正确：测试集只用一次
# 1. 训练集 + 验证集调参
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25)

# 2. 在验证集上调参
best_score = 0
for param in param_grid:
    clf.set_params(**param)
    clf.fit(X_tr, y_tr)
    score = clf.score(X_val, y_val)  # 验证集
    if score > best_score:
        best_score = score
        best_param = param

# 3. 用最佳参数在全部训练集上训练
clf.set_params(**best_param)
clf.fit(X_train, y_train)

# 4. 测试集只用一次做最终评估
final_score = clf.score(X_test, y_test)

# 或使用交叉验证
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(clf, param_grid, cv=5)  # CV调参
grid_search.fit(X_train, y_train)

# 测试集只用一次
final_score = grid_search.score(X_test, y_test)
```

---

**检测数据泄露的方法：**

### 1. 异常高的性能

```python
# 如果准确率接近100%，可能泄露
accuracy = clf.score(X_test, y_test)
if accuracy > 0.99:
    print("Warning: Too good to be true! Check for leakage.")

# 训练/测试性能几乎相同也可疑
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
if abs(train_score - test_score) < 0.01:
    print("Warning: Train and test scores too similar!")
```

### 2. 特征重要性分析

```python
# 检查是否有单个特征主导
importance = rf.feature_importances_
if importance.max() > 0.9:
    dominant_feature = X.columns[importance.argmax()]
    print(f"Warning: {dominant_feature} dominates (importance={importance.max():.4f})")

    # 检查该特征
    print(f"Feature values:\n{X[dominant_feature].describe()}")
    print(f"Correlation with target: {X[dominant_feature].corr(y):.4f}")
```

### 3. 时间一致性检查

```python
# 对时间序列，检查特征是否使用未来信息
def check_temporal_consistency(df, feature, date_col):
    # 特征值不应该依赖未来
    for i in range(len(df) - 1):
        current_value = df.loc[i, feature]
        # 只使用当前和过去的数据重新计算
        recalculated = calculate_feature(df.loc[:i])

        if current_value != recalculated:
            print(f"Warning: {feature} uses future information!")
            break
```

### 4. 数据流程审查

```python
# 代码审查清单
checklist = """
数据泄露检查清单：

1. 数据分割
   □ 是否先分割，再预处理？
   □ 时间序列是否按时间分割？
   □ 是否使用stratify（分类）？

2. 预处理
   □ fit只在训练集？
   □ transform分别应用？
   □ 使用Pipeline？

3. 特征工程
   □ 统计特征只用训练集？
   □ Target encoding只用训练集？
   □ 特征选择只用训练集？

4. 时间特征
   □ 只使用过去信息？
   □ 没有shift(-n)？
   □ 滞后特征正确？

5. 交叉验证
   □ 使用Pipeline？
   □ 时间序列用TimeSeriesSplit？

6. 测试集
   □ 测试集完全隔离？
   □ 调参不用测试集？
   □ 最后才评估测试集？

7. 特征合理性
   □ 特征在预测时可得？
   □ 没有未来信息？
   □ 业务逻辑合理？
"""
print(checklist)
```

---

**完整的无泄露工作流：**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 数据加载
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. 数据分割（第一步！）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 3. 特征工程（只用训练集）
# 计算统计特征
train_stats = X_train.groupby('category')['value'].mean()

# 应用到训练集和测试集
X_train['category_mean'] = X_train['category'].map(train_stats)
X_test['category_mean'] = X_test['category'].map(train_stats)

# 填充测试集中新类别（用整体均值）
overall_mean = X_train['value'].mean()
X_train['category_mean'].fillna(overall_mean, inplace=True)
X_test['category_mean'].fillna(overall_mean, inplace=True)

# 4. 创建Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 自动处理
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. 超参数调优（交叉验证，不用测试集）
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,  # 5折交叉验证
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)  # 只用训练集

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# 6. 最终评估（测试集只用一次！）
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nFinal Test Set Results:")
print(classification_report(y_test, y_pred))

# 7. 保存模型和预处理器
import joblib

joblib.dump(best_model, 'model.pkl')
joblib.dump(train_stats, 'train_stats.pkl')  # 保存统计信息

# 8. 生产环境预测
# 加载
model = joblib.load('model.pkl')
stats = joblib.load('train_stats.pkl')

# 新数据
X_new = pd.DataFrame(...)
X_new['category_mean'] = X_new['category'].map(stats)
X_new['category_mean'].fillna(overall_mean, inplace=True)

# 预测
predictions = model.predict(X_new)
```

---

**总结：**

**防止数据泄露的黄金法则：**

1. **先分割，再预处理**
2. **fit只在训练集，transform分别应用**
3. **使用Pipeline**
4. **时间序列按时间分割，只用过去信息**
5. **测试集完全隔离，只用一次**
6. **交叉验证时每个fold独立处理**
7. **检查特征在预测时是否可得**

**记住：如果性能好得不真实，那可能确实不真实！**

</details>

---

## 数学基础题

### 11. 解释梯度下降算法

<details>
<summary>点击查看答案</summary>

**梯度下降（Gradient Descent）**是一种优化算法，用于找到函数的最小值点。在机器学习中，用于最小化损失函数。

**核心思想：**
- 沿着梯度（导数）的反方向移动
- 逐步接近最小值
- 类比：下山找最低点，每次朝最陡的方向走

---

**数学原理：**

**目标：**最小化损失函数 J(θ)

**更新规则：**
```
θ := θ - α × ∂J/∂θ

其中：
- θ: 参数（权重）
- α: 学习率（步长）
- ∂J/∂θ: 损失函数对参数的梯度（导数）
```

**直观理解：**
```
当前位置
    ↓
    θ
    |
梯度指向上坡 →  ∂J/∂θ
我们走反方向 →  -∂J/∂θ
步长由α控制 →  α × ∂J/∂θ

新位置 = 当前位置 - 学习率 × 梯度
```

---

**类型：**

### 1. 批量梯度下降（Batch GD）

**特点：**
- 每次迭代使用**全部训练数据**
- 计算整个数据集的梯度
- 稳定但慢

**算法：**
```python
# 伪代码
for epoch in range(n_epochs):
    # 计算全部数据的梯度
    gradient = compute_gradient(X_all, y_all, theta)

    # 更新参数
    theta = theta - learning_rate * gradient
```

**优点：**
- 收敛稳定
- 理论上保证收敛到最优（凸函数）

**缺点：**
- 大数据集很慢
- 内存消耗大
- 可能卡在局部最优

**代码实现：**
```python
import numpy as np

def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)  # 初始化参数
    cost_history = []

    for iteration in range(n_iterations):
        # 预测
        predictions = X.dot(theta)

        # 计算梯度（全部数据）
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)

        # 更新参数
        theta = theta - learning_rate * gradient

        # 记录损失
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Cost = {cost:.4f}")

    return theta, cost_history

# 使用
X_train_with_bias = np.c_[np.ones(len(X_train)), X_train]
theta, costs = batch_gradient_descent(X_train_with_bias, y_train)

# 可视化收敛
import matplotlib.pyplot as plt
plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Convergence of Batch GD')
plt.show()
```

---

### 2. 随机梯度下降（Stochastic GD / SGD）

**特点：**
- 每次迭代使用**一个样本**
- 更新频繁
- 快速但波动大

**算法：**
```python
# 伪代码
for epoch in range(n_epochs):
    for i in range(m):  # 遍历每个样本
        # 计算单个样本的梯度
        gradient = compute_gradient(X[i], y[i], theta)

        # 更新参数
        theta = theta - learning_rate * gradient
```

**优点：**
- 速度快
- 可在线学习
- 可能跳出局部最优

**缺点：**
- 收敛不稳定，波动大
- 不能并行化
- 可能不收敛

**代码实现：**
```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=50):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for epoch in range(n_epochs):
        # 随机打乱数据
        indices = np.random.permutation(m)

        for i in indices:
            # 单个样本
            xi = X[i:i+1]
            yi = y[i:i+1]

            # 预测
            prediction = xi.dot(theta)

            # 计算梯度（单个样本）
            error = prediction - yi
            gradient = xi.T.dot(error)

            # 更新参数
            theta = theta - learning_rate * gradient

        # 每个epoch记录一次损失
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

        print(f"Epoch {epoch}: Cost = {cost:.4f}")

    return theta, cost_history
```

---

### 3. 小批量梯度下降（Mini-Batch GD）

**特点：**
- 每次使用**一小批样本**（如32、64、128）
- 平衡速度和稳定性
- **最常用的方法**

**算法：**
```python
# 伪代码
for epoch in range(n_epochs):
    for batch in get_batches(X, y, batch_size):
        # 计算batch的梯度
        gradient = compute_gradient(batch_X, batch_y, theta)

        # 更新参数
        theta = theta - learning_rate * gradient
```

**优点：**
- 比Batch GD快
- 比SGD稳定
- 可以利用矢量化加速
- 适合GPU并行

**缺点：**
- 需要调整batch size

**代码实现：**
```python
def mini_batch_gradient_descent(X, y, batch_size=32, learning_rate=0.01, n_epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for epoch in range(n_epochs):
        # 随机打乱
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # 分批
        for i in range(0, m, batch_size):
            # 获取batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # 预测
            predictions = X_batch.dot(theta)

            # 计算梯度（batch）
            errors = predictions - y_batch
            gradient = (1/len(X_batch)) * X_batch.T.dot(errors)

            # 更新参数
            theta = theta - learning_rate * gradient

        # 记录损失
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.4f}")

    return theta, cost_history
```

---

**学习率的影响：**

```python
# 学习率太小：收敛慢
learning_rates = [0.001, 0.01, 0.1, 1.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, lr in enumerate(learning_rates):
    ax = axes[i//2, i%2]

    theta, costs = mini_batch_gradient_descent(
        X, y,
        learning_rate=lr,
        n_epochs=100
    )

    ax.plot(costs)
    ax.set_title(f'Learning Rate = {lr}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')

plt.tight_layout()
plt.show()

# 观察：
# lr太小（0.001）：收敛很慢
# lr适中（0.01, 0.1）：稳定收敛
# lr太大（1.0）：震荡不收敛
```

---

**改进的梯度下降：**

### 1. 动量（Momentum）

**思想：**
- 累积过去的梯度
- 加速收敛
- 减少震荡

**公式：**
```
v := β × v + (1-β) × gradient
θ := θ - α × v

其中 β ≈ 0.9（动量系数）
```

**代码：**
```python
def gradient_descent_with_momentum(X, y, learning_rate=0.01, beta=0.9, n_epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    velocity = np.zeros(n)  # 速度

    for epoch in range(n_epochs):
        # 计算梯度
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)

        # 更新速度（累积）
        velocity = beta * velocity + (1 - beta) * gradient

        # 更新参数
        theta = theta - learning_rate * velocity

    return theta
```

### 2. Adam（Adaptive Moment Estimation）

**思想：**
- 自适应学习率
- 结合动量和RMSProp
- **最常用的优化器**

**公式：**
```
m := β₁ × m + (1-β₁) × gradient        # 一阶矩估计
v := β₂ × v + (1-β₂) × gradient²       # 二阶矩估计

m_hat := m / (1 - β₁^t)                 # 偏差修正
v_hat := v / (1 - β₂^t)

θ := θ - α × m_hat / (√v_hat + ε)

通常：β₁=0.9, β₂=0.999, ε=1e-8
```

**代码：**
```python
def adam_optimizer(X, y, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, n_epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    m_t = np.zeros(n)  # 一阶矩
    v_t = np.zeros(n)  # 二阶矩

    for t in range(1, n_epochs + 1):
        # 计算梯度
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)

        # 更新矩估计
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * gradient**2

        # 偏差修正
        m_hat = m_t / (1 - beta1**t)
        v_hat = v_t / (1 - beta2**t)

        # 更新参数
        theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return theta
```

---

**在Scikit-learn中使用：**

```python
from sklearn.linear_model import SGDRegressor, SGDClassifier

# 回归
sgd_reg = SGDRegressor(
    loss='squared_error',  # 损失函数
    penalty='l2',          # 正则化
    alpha=0.0001,          # 正则化强度
    learning_rate='invscaling',  # 学习率策略
    eta0=0.01,             # 初始学习率
    max_iter=1000,
    random_state=42
)

sgd_reg.fit(X_train, y_train)

# 分类
sgd_clf = SGDClassifier(
    loss='log_loss',       # Logistic loss
    learning_rate='adaptive',
    eta0=0.01,
    max_iter=1000,
    random_state=42
)

sgd_clf.fit(X_train, y_train)
```

---

**学习率调度（Learning Rate Scheduling）：**

```python
# 1. 固定学习率
lr = 0.01

# 2. 时间衰减
lr_t = lr_0 / (1 + decay_rate × epoch)

# 3. 指数衰减
lr_t = lr_0 × decay_rate^epoch

# 4. 阶梯衰减
if epoch % step_size == 0:
    lr = lr × decay_factor

# 实现
def learning_rate_decay(initial_lr, epoch, decay_rate, decay_step):
    return initial_lr * (decay_rate ** (epoch // decay_step))

# 使用
for epoch in range(n_epochs):
    lr = learning_rate_decay(initial_lr=0.1, epoch=epoch, decay_rate=0.95, decay_step=10)

    # 使用当前学习率更新
    theta = theta - lr * gradient
```

---

**梯度下降的挑战：**

1. **局部最优**
   - 非凸函数可能卡在局部最优
   - 解决：动量、随机性

2. **鞍点**
   - 梯度为0但不是最优点
   - 解决：动量

3. **梯度消失/爆炸**
   - 深度网络中梯度过小或过大
   - 解决：BatchNorm、梯度裁剪

4. **学习率选择**
   - 太小：慢
   - 太大：不收敛
   - 解决：学习率调度、Adam

---

**总结：**

**选择指南：**
- **小数据**：Batch GD
- **大数据**：Mini-Batch GD
- **在线学习**：SGD
- **通用首选**：Adam + Mini-Batch

**关键参数：**
- 学习率：0.001 - 0.1
- Batch size：32、64、128
- Adam参数：默认值通常很好

</details>

---

## 代码实现题

### 12. 实现一个简单的KNN分类器

<details>
<summary>点击查看答案</summary>

**从头实现K-Nearest Neighbors分类器**

```python
import numpy as np
from collections import Counter

class KNNClassifierFromScratch:
    """
    K-Nearest Neighbors分类器

    参数:
        k: int, default=3
            邻居数量

        distance_metric: str, default='euclidean'
            距离度量：'euclidean', 'manhattan', 'minkowski'

        p: int, default=2
            Minkowski距离的参数（p=2为欧氏距离）
    """

    def __init__(self, k=3, distance_metric='euclidean', p=2):
        self.k = k
        self.distance_metric = distance_metric
        self.p = p

    def fit(self, X, y):
        """
        训练（实际上只是存储数据）

        参数:
            X: numpy.ndarray, shape (n_samples, n_features)
                训练数据
            y: numpy.ndarray, shape (n_samples,)
                标签
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _calculate_distance(self, x1, x2):
        """
        计算两点之间的距离

        参数:
            x1, x2: numpy.ndarray
                两个数据点

        返回:
            distance: float
                距离值
        """
        if self.distance_metric == 'euclidean':
            # 欧氏距离：√(Σ(x1 - x2)²)
            return np.sqrt(np.sum((x1 - x2) ** 2))

        elif self.distance_metric == 'manhattan':
            # 曼哈顿距离：Σ|x1 - x2|
            return np.sum(np.abs(x1 - x2))

        elif self.distance_metric == 'minkowski':
            # 闵可夫斯基距离：(Σ|x1 - x2|^p)^(1/p)
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1/self.p)

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _get_neighbors(self, x):
        """
        找到k个最近邻居

        参数:
            x: numpy.ndarray
                测试样本

        返回:
            neighbors: numpy.ndarray
                k个最近邻居的标签
        """
        # 计算到所有训练样本的距离
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._calculate_distance(x, x_train)
            distances.append((dist, self.y_train[i]))

        # 排序并取前k个
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]

        # 返回邻居的标签
        return np.array([label for _, label in k_nearest])

    def predict_single(self, x):
        """
        预测单个样本

        参数:
            x: numpy.ndarray
                单个测试样本

        返回:
            prediction: int/str
                预测的类别
        """
        # 找到k个最近邻居
        neighbors = self._get_neighbors(x)

        # 投票：选择出现次数最多的类别
        counter = Counter(neighbors)
        most_common = counter.most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """
        预测多个样本

        参数:
            X: numpy.ndarray, shape (n_samples, n_features)
                测试数据

        返回:
            predictions: numpy.ndarray, shape (n_samples,)
                预测结果
        """
        predictions = []
        for x in X:
            pred = self.predict_single(x)
            predictions.append(pred)
        return np.array(predictions)

    def predict_proba(self, X):
        """
        预测概率

        参数:
            X: numpy.ndarray, shape (n_samples, n_features)
                测试数据

        返回:
            probabilities: numpy.ndarray, shape (n_samples, n_classes)
                每个类别的概率
        """
        # 获取所有类别
        classes = np.unique(self.y_train)
        n_classes = len(classes)

        probabilities = []

        for x in X:
            # 找到k个最近邻居
            neighbors = self._get_neighbors(x)

            # 计算每个类别的概率
            proba = np.zeros(n_classes)
            counter = Counter(neighbors)

            for i, cls in enumerate(classes):
                proba[i] = counter[cls] / self.k

            probabilities.append(proba)

        return np.array(probabilities)

    def score(self, X, y):
        """
        计算准确率

        参数:
            X: numpy.ndarray
                测试数据
            y: numpy.ndarray
                真实标签

        返回:
            accuracy: float
                准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ==================== 优化版本（向量化） ====================

class KNNClassifierVectorized:
    """
    向量化的KNN分类器（速度更快）
    """

    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes = np.unique(y)
        return self

    def _calculate_distances(self, X):
        """
        向量化计算距离（速度快）

        参数:
            X: numpy.ndarray, shape (n_test, n_features)
                测试数据

        返回:
            distances: numpy.ndarray, shape (n_test, n_train)
                距离矩阵
        """
        if self.distance_metric == 'euclidean':
            # 使用广播计算欧氏距离
            # |X - X_train|² = X² - 2·X·X_train^T + X_train²
            X_squared = np.sum(X**2, axis=1).reshape(-1, 1)
            X_train_squared = np.sum(self.X_train**2, axis=1).reshape(1, -1)
            cross_term = -2 * np.dot(X, self.X_train.T)

            distances_squared = X_squared + cross_term + X_train_squared
            return np.sqrt(np.maximum(distances_squared, 0))  # 避免负数

        elif self.distance_metric == 'manhattan':
            # 曼哈顿距离
            return np.sum(np.abs(X[:, np.newaxis] - self.X_train), axis=2)

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def predict(self, X):
        X = np.array(X)

        # 计算所有距离
        distances = self._calculate_distances(X)

        # 找到k个最近邻的索引
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]

        # 获取k个最近邻的标签
        k_nearest_labels = self.y_train[k_nearest_indices]

        # 投票
        predictions = []
        for labels in k_nearest_labels:
            counter = Counter(labels)
            predictions.append(counter.most_common(1)[0][0])

        return np.array(predictions)

    def predict_proba(self, X):
        X = np.array(X)
        distances = self._calculate_distances(X)
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]

        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))

        for i, labels in enumerate(k_nearest_labels):
            counter = Counter(labels)
            for j, cls in enumerate(self.classes):
                probabilities[i, j] = counter[cls] / self.k

        return probabilities

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 生成测试数据
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # 生成数据
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ==================== 测试基础版本 ====================
    print("=== Basic KNN ===")
    knn_basic = KNNClassifierFromScratch(k=5, distance_metric='euclidean')
    knn_basic.fit(X_train, y_train)

    # 预测
    y_pred = knn_basic.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # 概率
    y_proba = knn_basic.predict_proba(X_test[:5])
    print(f"Probabilities (first 5):\n{y_proba}")

    # ==================== 测试向量化版本 ====================
    print("\n=== Vectorized KNN ===")
    knn_vec = KNNClassifierVectorized(k=5, distance_metric='euclidean')
    knn_vec.fit(X_train, y_train)

    y_pred_vec = knn_vec.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_vec):.4f}")

    # ==================== 速度比较 ====================
    import time

    # 基础版本
    start = time.time()
    knn_basic.predict(X_test)
    basic_time = time.time() - start

    # 向量化版本
    start = time.time()
    knn_vec.predict(X_test)
    vec_time = time.time() - start

    print(f"\nSpeed comparison:")
    print(f"Basic: {basic_time:.4f}s")
    print(f"Vectorized: {vec_time:.4f}s")
    print(f"Speedup: {basic_time/vec_time:.2f}x")

    # ==================== 与Scikit-learn对比 ====================
    from sklearn.neighbors import KNeighborsClassifier

    sklearn_knn = KNeighborsClassifier(n_neighbors=5)
    sklearn_knn.fit(X_train, y_train)
    sklearn_score = sklearn_knn.score(X_test, y_test)

    print(f"\nScikit-learn KNN accuracy: {sklearn_score:.4f}")
    print(f"Our KNN accuracy: {knn_vec.score(X_test, y_test):.4f}")

    # ==================== 可视化决策边界 ====================
    import matplotlib.pyplot as plt

    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测网格上的每个点
    Z = knn_vec.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                cmap='RdYlBu', edgecolor='black', marker='o',
                s=50, label='Train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                cmap='RdYlBu', edgecolor='black', marker='s',
                s=50, label='Test')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'KNN Decision Boundary (k={knn_vec.k})')
    plt.legend()
    plt.colorbar()
    plt.show()

    # ==================== K值的影响 ====================
    k_values = range(1, 31)
    train_scores = []
    test_scores = []

    for k in k_values:
        knn = KNNClassifierVectorized(k=k)
        knn.fit(X_train, y_train)

        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_scores, label='Train Accuracy', marker='o')
    plt.plot(k_values, test_scores, label='Test Accuracy', marker='s')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('KNN: Accuracy vs K Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    best_k = k_values[np.argmax(test_scores)]
    print(f"\nBest K value: {best_k}")
    print(f"Best test accuracy: {max(test_scores):.4f}")
```

**关键要点：**

1. **KNN是懒惰学习**：不训练，只存储数据
2. **距离计算是核心**：欧氏、曼哈顿、闵可夫斯基
3. **K值的选择**：通过交叉验证
4. **向量化提速**：使用NumPy广播
5. **应用**：分类、回归、异常检测

</details>

---

**由于篇幅限制，这里只展示了部分面试题。完整版本包含150+题目。**

**更多题目包括：**
- 深度学习题（神经网络、CNN、RNN、Transformer等）
- 案例分析题（推荐系统、客户流失、欺诈检测等）
- 系统设计题（ML系统架构、A/B测试、模型部署等）
- NLP题（词嵌入、BERT、GPT等）
- 计算机视觉题（目标检测、图像分割等）
- 强化学习题（Q-learning、Policy Gradient等）

---

## 学习建议

1. **理解原理比记答案更重要**
2. **能用自己的话解释概念**
3. **准备代码实现**
4. **准备实际项目案例**
5. **关注最新进展**

## 面试准备清单

- [ ] 理解所有基础概念
- [ ] 掌握常用算法原理
- [ ] 能手写核心算法
- [ ] 准备3个项目深入讲解
- [ ] 了解公司业务和技术栈
- [ ] 准备提问面试官的问题

---

**祝面试顺利！**

**最后更新：2025-11-18**
