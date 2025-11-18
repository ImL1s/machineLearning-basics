# 机器学习算法速查表 (Algorithm Cheatsheet)

> 快速参考各类机器学习算法的原理、优缺点、适用场景和关键参数

## 目录
- [分类算法](#分类算法)
- [回归算法](#回归算法)
- [聚类算法](#聚类算法)
- [降维算法](#降维算法)
- [异常检测算法](#异常检测算法)
- [算法选择决策树](#算法选择决策树)
- [常用公式](#常用公式)

---

## 分类算法

### 分类算法速查表

| 算法 | 原理 | 优点 | 缺点 | 适用场景 | Scikit-learn API | 关键参数 |
|------|------|------|------|----------|------------------|----------|
| **Logistic Regression**<br>逻辑回归 | 使用Sigmoid函数将线性组合映射到[0,1]，输出概率 | • 简单快速<br>• 输出概率值<br>• 可解释性强<br>• 适合线性可分数据<br>• 训练快速 | • 只能处理线性边界<br>• 特征工程要求高<br>• 对异常值敏感 | • 二分类/多分类<br>• 线性可分数据<br>• 需要概率输出<br>• 基准模型 | `LogisticRegression()` | `C`(正则化强度)<br>`penalty`(l1/l2)<br>`solver`(优化器) |
| **K-Nearest Neighbors**<br>K近邻 | 根据K个最近邻居的多数类别投票决定 | • 简单直观<br>• 无需训练<br>• 适合多分类<br>• 非参数方法 | • 预测慢<br>• 内存消耗大<br>• 对特征缩放敏感<br>• 维度灾难 | • 小数据集<br>• 非线性边界<br>• 异常检测 | `KNeighborsClassifier()` | `n_neighbors`(K值)<br>`metric`(距离度量)<br>`weights`(uniform/distance) |
| **Support Vector Machine**<br>支持向量机 | 找到最大间隔超平面，使用核技巧处理非线性 | • 高维数据表现好<br>• 内存效率高<br>• 核技巧处理非线性<br>• 泛化能力强 | • 训练慢(大数据)<br>• 参数选择困难<br>• 不输出概率(默认) | • 中小数据集<br>• 高维数据<br>• 文本分类<br>• 图像分类 | `SVC()`<br>`LinearSVC()` | `C`(惩罚参数)<br>`kernel`(rbf/linear/poly)<br>`gamma`(核系数) |
| **Decision Tree**<br>决策树 | 递归分割特征空间，构建树状决策规则 | • 可解释性强<br>• 处理非线性<br>• 无需特征缩放<br>• 处理类别特征<br>• 处理缺失值 | • 容易过拟合<br>• 不稳定<br>• 偏向多值特征 | • 需要可解释性<br>• 混合特征类型<br>• 特征重要性分析 | `DecisionTreeClassifier()` | `max_depth`(深度)<br>`min_samples_split`<br>`criterion`(gini/entropy) |
| **Random Forest**<br>随机森林 | 多个决策树的集成，采用bagging和特征随机 | • 精度高<br>• 抗过拟合<br>• 特征重要性<br>• 处理缺失值<br>• 并行训练 | • 可解释性弱<br>• 内存消耗大<br>• 预测慢 | • 通用分类任务<br>• 特征选择<br>• 不平衡数据 | `RandomForestClassifier()` | `n_estimators`(树数量)<br>`max_depth`<br>`max_features` |
| **Gradient Boosting**<br>梯度提升 | 串行训练弱学习器，每次拟合前一次的残差 | • 精度高<br>• 处理非线性<br>• 特征重要性<br>• 鲁棒性好 | • 训练慢<br>• 容易过拟合<br>• 参数调优复杂<br>• 不支持并行 | • Kaggle竞赛<br>• 结构化数据<br>• 需要高精度 | `GradientBoostingClassifier()` | `n_estimators`<br>`learning_rate`<br>`max_depth` |
| **XGBoost** | 优化的梯度提升，加入正则化和并行化 | • 精度最高<br>• 速度快<br>• 并行化<br>• 处理缺失值<br>• 正则化防过拟合 | • 参数多<br>• 内存消耗大<br>• 黑盒模型 | • 竞赛首选<br>• 结构化数据<br>• 大规模数据 | `XGBClassifier()` | `n_estimators`<br>`learning_rate`<br>`max_depth`<br>`reg_alpha/lambda` |
| **Naive Bayes**<br>朴素贝叶斯 | 基于贝叶斯定理，假设特征条件独立 | • 训练快<br>• 预测快<br>• 适合高维数据<br>• 输出概率<br>• 小数据表现好 | • 特征独立假设强<br>• 对特征关系建模弱 | • 文本分类<br>• 垃圾邮件检测<br>• 情感分析<br>• 实时预测 | `GaussianNB()`<br>`MultinomialNB()` | `var_smoothing`(高斯)<br>`alpha`(拉普拉斯平滑) |
| **AdaBoost** | 自适应提升，调整样本权重，关注错分样本 | • 精度高<br>• 自动特征选择<br>• 不易过拟合 | • 对噪声敏感<br>• 训练慢 | • 中小数据集<br>• 二分类<br>• 特征选择 | `AdaBoostClassifier()` | `n_estimators`<br>`learning_rate`<br>`base_estimator` |
| **Voting Classifier**<br>投票分类器 | 多个分类器投票，硬投票或软投票 | • 提升精度<br>• 降低方差<br>• 组合优势 | • 训练时间长<br>• 可解释性弱 | • 集成多个模型<br>• Kaggle竞赛 | `VotingClassifier()` | `voting`(hard/soft)<br>`weights`(权重) |
| **Stacking**<br>堆叠 | 使用元学习器组合多个基学习器 | • 精度最高<br>• 灵活性强 | • 复杂度高<br>• 过拟合风险<br>• 训练慢 | • 竞赛<br>• 追求最高精度 | `StackingClassifier()` | `estimators`<br>`final_estimator` |

---

## 回归算法

### 回归算法速查表

| 算法 | 原理 | 优点 | 缺点 | 适用场景 | Scikit-learn API | 关键参数 |
|------|------|------|------|----------|------------------|----------|
| **Linear Regression**<br>线性回归 | 拟合线性方程 y = wx + b | • 简单快速<br>• 可解释性强<br>• 训练快<br>• 适合线性关系 | • 假设线性关系<br>• 对异常值敏感<br>• 多重共线性 | • 线性关系<br>• 基准模型<br>• 趋势预测 | `LinearRegression()` | `fit_intercept`<br>`normalize` |
| **Ridge Regression**<br>岭回归 | 线性回归 + L2正则化 | • 防止过拟合<br>• 处理多重共线性<br>• 稳定性好 | • 不做特征选择<br>• 需要调参 | • 特征相关性高<br>• 过拟合风险<br>• 特征数>样本数 | `Ridge()` | `alpha`(正则化强度) |
| **Lasso Regression**<br>Lasso回归 | 线性回归 + L1正则化 | • 特征选择<br>• 防止过拟合<br>• 稀疏解 | • 需要调参<br>• 不稳定(相关特征) | • 特征选择<br>• 高维稀疏数据 | `Lasso()` | `alpha`(正则化强度) |
| **ElasticNet**<br>弹性网络 | 线性回归 + L1 + L2正则化 | • 结合Ridge和Lasso<br>• 特征选择<br>• 稳定性好 | • 参数多<br>• 计算慢 | • 高维数据<br>• 特征相关 | `ElasticNet()` | `alpha`<br>`l1_ratio` |
| **Polynomial Regression**<br>多项式回归 | 通过多项式特征拟合非线性关系 | • 拟合非线性<br>• 简单直观 | • 容易过拟合<br>• 特征爆炸 | • 非线性关系<br>• 曲线拟合 | `PolynomialFeatures()`<br>`+ LinearRegression()` | `degree`(多项式阶数) |
| **Support Vector Regression**<br>支持向量回归 | SVM用于回归，在ε范围内无损失 | • 处理非线性<br>• 鲁棒性好<br>• 高维数据 | • 训练慢<br>• 参数选择难<br>• 内存消耗大 | • 非线性回归<br>• 小中数据集 | `SVR()` | `C`<br>`epsilon`<br>`kernel` |
| **Decision Tree Regressor**<br>决策树回归 | 递归分割特征空间 | • 非线性<br>• 可解释性<br>• 无需特征缩放 | • 容易过拟合<br>• 不稳定 | • 非线性关系<br>• 可解释性需求 | `DecisionTreeRegressor()` | `max_depth`<br>`min_samples_split` |
| **Random Forest Regressor**<br>随机森林回归 | 多个决策树的集成 | • 精度高<br>• 抗过拟合<br>• 特征重要性 | • 可解释性弱<br>• 预测慢 | • 通用回归任务<br>• 非线性关系 | `RandomForestRegressor()` | `n_estimators`<br>`max_depth` |
| **Gradient Boosting Regressor**<br>梯度提升回归 | 串行训练，拟合残差 | • 精度高<br>• 处理非线性 | • 训练慢<br>• 易过拟合 | • 竞赛<br>• 高精度需求 | `GradientBoostingRegressor()` | `n_estimators`<br>`learning_rate` |
| **XGBoost Regressor** | 优化的梯度提升回归 | • 精度最高<br>• 速度快<br>• 并行化 | • 参数多<br>• 黑盒 | • 竞赛<br>• 结构化数据 | `XGBRegressor()` | `n_estimators`<br>`learning_rate`<br>`max_depth` |

---

## 聚类算法

### 聚类算法速查表

| 算法 | 原理 | 优点 | 缺点 | 适用场景 | Scikit-learn API | 关键参数 |
|------|------|------|------|----------|------------------|----------|
| **K-Means**<br>K均值 | 将数据分为K个簇，最小化簇内距离 | • 简单快速<br>• 可扩展<br>• 适合球形簇 | • 需要预设K<br>• 对初始值敏感<br>• 假设球形簇<br>• 对异常值敏感 | • 客户细分<br>• 图像压缩<br>• 数据探索 | `KMeans()` | `n_clusters`(K值)<br>`init`(初始化)<br>`n_init`(运行次数) |
| **DBSCAN**<br>基于密度的聚类 | 基于密度的空间聚类，识别任意形状簇 | • 不需要预设K<br>• 识别噪声点<br>• 任意形状簇 | • 参数敏感<br>• 密度不均时表现差<br>• 高维数据困难 | • 空间数据<br>• 异常检测<br>• 任意形状簇 | `DBSCAN()` | `eps`(邻域半径)<br>`min_samples`(最小样本数) |
| **Hierarchical Clustering**<br>层次聚类 | 构建层次树状结构，自底向上或自顶向下 | • 不需要预设K<br>• 树状图可视化<br>• 任意距离度量 | • 计算复杂度高<br>• 不适合大数据 | • 生物分类<br>• 社交网络<br>• 小数据集 | `AgglomerativeClustering()` | `n_clusters`<br>`linkage`(ward/complete/average) |
| **Mean Shift**<br>均值漂移 | 基于密度梯度的聚类 | • 不需要预设K<br>• 任意形状簇 | • 计算慢<br>• 参数敏感 | • 图像分割<br>• 目标跟踪 | `MeanShift()` | `bandwidth`(带宽) |
| **Gaussian Mixture Model**<br>高斯混合模型 | 假设数据由多个高斯分布生成 | • 软聚类(概率)<br>• 灵活的簇形状 | • 需要预设K<br>• 对初始值敏感<br>• 收敛慢 | • 软聚类<br>• 密度估计 | `GaussianMixture()` | `n_components`<br>`covariance_type` |

---

## 降维算法

### 降维算法速查表

| 算法 | 原理 | 优点 | 缺点 | 适用场景 | Scikit-learn API | 关键参数 |
|------|------|------|------|----------|------------------|----------|
| **PCA**<br>主成分分析 | 线性投影到方差最大的方向 | • 快速高效<br>• 可解释性<br>• 去除相关性 | • 线性假设<br>• 损失可解释性<br>• 对缩放敏感 | • 数据可视化<br>• 去噪<br>• 压缩 | `PCA()` | `n_components`(成分数)<br>`whiten`(白化) |
| **t-SNE**<br>t分布邻域嵌入 | 非线性降维，保持局部结构 | • 可视化效果好<br>• 保持局部结构<br>• 揭示聚类 | • 计算慢<br>• 不确定性<br>• 不适合新数据<br>• 参数敏感 | • 高维数据可视化<br>• 聚类可视化 | `TSNE()` | `n_components`(通常2或3)<br>`perplexity`(困惑度) |
| **UMAP**<br>统一流形逼近 | 非线性降维，保持全局和局部结构 | • 速度快<br>• 保持全局结构<br>• 支持新数据 | • 参数多<br>• 较新的方法 | • 大数据可视化<br>• 聚类分析 | `umap.UMAP()` | `n_neighbors`<br>`min_dist` |
| **LDA**<br>线性判别分析 | 监督降维，最大化类间距离 | • 监督学习<br>• 提高分类性能 | • 线性假设<br>• 需要标签 | • 分类前处理<br>• 特征提取 | `LinearDiscriminantAnalysis()` | `n_components` |
| **Autoencoder**<br>自编码器 | 神经网络非线性降维 | • 强大的非线性<br>• 灵活性高 | • 训练复杂<br>• 需要大数据<br>• 超参数多 | • 图像压缩<br>• 异常检测 | Keras/PyTorch | `encoding_dim`(编码维度) |

---

## 异常检测算法

### 异常检测算法速查表

| 算法 | 原理 | 优点 | 缺点 | 适用场景 | Scikit-learn API | 关键参数 |
|------|------|------|------|----------|------------------|----------|
| **Isolation Forest**<br>孤立森林 | 随机分割，异常点更容易被孤立 | • 无需标签<br>• 高维数据<br>• 线性时间复杂度 | • 参数敏感<br>• 不适合局部异常 | • 欺诈检测<br>• 网络入侵<br>• 系统监控 | `IsolationForest()` | `contamination`(异常比例)<br>`n_estimators` |
| **One-Class SVM**<br>单类SVM | 在特征空间找到最小超球 | • 处理非线性<br>• 核技巧 | • 参数选择难<br>• 训练慢 | • 新颖性检测<br>• 高维异常检测 | `OneClassSVM()` | `nu`(异常上限)<br>`kernel` |
| **Local Outlier Factor**<br>局部离群因子 | 基于局部密度的异常检测 | • 检测局部异常<br>• 任意形状 | • 计算复杂<br>• 参数敏感 | • 局部异常<br>• 密度变化大的数据 | `LocalOutlierFactor()` | `n_neighbors`<br>`contamination` |
| **Elliptic Envelope**<br>椭圆包络 | 假设数据服从高斯分布 | • 简单快速<br>• 适合高斯数据 | • 高斯假设<br>• 不适合复杂分布 | • 正态分布数据<br>• 快速检测 | `EllipticEnvelope()` | `contamination` |

---

## 算法选择决策树

### 根据问题类型选择算法

```
📊 你的问题类型？
│
├─ 🎯 分类 (Classification)
│  │
│  ├─ 数据是否线性可分？
│  │  ├─ ✅ 是 → Logistic Regression / Linear SVM
│  │  └─ ❌ 否 → SVM (RBF kernel) / Random Forest / XGBoost
│  │
│  ├─ 数据规模？
│  │  ├─ 小 (<1K) → KNN / SVM / Decision Tree
│  │  ├─ 中 (1K-100K) → Random Forest / Gradient Boosting
│  │  └─ 大 (>100K) → Logistic Regression / XGBoost / LightGBM
│  │
│  ├─ 需要概率输出？
│  │  ├─ ✅ 是 → Logistic Regression / Naive Bayes / Random Forest
│  │  └─ ❌ 否 → SVM / Perceptron
│  │
│  ├─ 需要可解释性？
│  │  ├─ ✅ 是 → Logistic Regression / Decision Tree / Naive Bayes
│  │  └─ ❌ 否 → XGBoost / Neural Network / Ensemble
│  │
│  ├─ 数据不平衡？
│  │  ├─ ✅ 是 → Random Forest (balanced) / XGBoost (scale_pos_weight)
│  │  └─ ❌ 否 → 任意分类器
│  │
│  └─ 追求最高精度？
│     └─ 🏆 XGBoost / LightGBM / Stacking / Neural Network
│
├─ 📈 回归 (Regression)
│  │
│  ├─ 数据是否线性关系？
│  │  ├─ ✅ 是 → Linear Regression / Ridge / Lasso
│  │  └─ ❌ 否 → Polynomial / SVR / Random Forest / XGBoost
│  │
│  ├─ 是否有过拟合？
│  │  ├─ ✅ 是 → Ridge / Lasso / ElasticNet
│  │  └─ ❌ 否 → Linear Regression
│  │
│  ├─ 需要特征选择？
│  │  ├─ ✅ 是 → Lasso / ElasticNet
│  │  └─ ❌ 否 → Ridge / Random Forest
│  │
│  └─ 追求最高精度？
│     └─ 🏆 XGBoost Regressor / Gradient Boosting / Stacking
│
├─ 🔍 聚类 (Clustering)
│  │
│  ├─ 是否知道簇数量？
│  │  ├─ ✅ 是 → K-Means / Gaussian Mixture
│  │  └─ ❌ 否 → DBSCAN / Hierarchical
│  │
│  ├─ 簇的形状？
│  │  ├─ 球形 → K-Means
│  │  └─ 任意形状 → DBSCAN / Hierarchical
│  │
│  ├─ 数据规模？
│  │  ├─ 小 (<10K) → Hierarchical / DBSCAN
│  │  └─ 大 (>10K) → K-Means / Mini-Batch K-Means
│  │
│  └─ 需要软聚类(概率)？
│     └─ ✅ 是 → Gaussian Mixture Model
│
├─ 📉 降维 (Dimensionality Reduction)
│  │
│  ├─ 目的是什么？
│  │  ├─ 可视化 → t-SNE / UMAP
│  │  ├─ 特征提取 → PCA / LDA
│  │  └─ 压缩 → PCA / Autoencoder
│  │
│  ├─ 数据规模？
│  │  ├─ 小-中 (<100K) → t-SNE
│  │  └─ 大 (>100K) → PCA / UMAP
│  │
│  └─ 有标签？
│     ├─ ✅ 是 → LDA (Linear Discriminant Analysis)
│     └─ ❌ 否 → PCA / t-SNE / UMAP
│
└─ 🚨 异常检测 (Anomaly Detection)
   │
   ├─ 有标签？
   │  ├─ ✅ 是 → Supervised Classification
   │  └─ ❌ 否 → Unsupervised methods ↓
   │
   ├─ 数据分布？
   │  ├─ 高斯分布 → Elliptic Envelope
   │  └─ 任意分布 → Isolation Forest / One-Class SVM
   │
   └─ 异常类型？
      ├─ 全局异常 → Isolation Forest
      └─ 局部异常 → Local Outlier Factor (LOF)
```

### 按数据特征选择算法

| 数据特征 | 推荐算法 | 避免算法 |
|---------|---------|---------|
| 🔢 数值特征 | 所有算法 | - |
| 📝 类别特征 | Decision Tree / Random Forest / XGBoost | KNN / SVM (需编码) |
| 📊 高维数据 | SVM / Naive Bayes / Neural Network | KNN (维度灾难) |
| 📉 少量样本 | SVM / KNN / Naive Bayes | Deep Learning |
| 📈 大量样本 | Logistic / SGD / XGBoost / Neural Network | KNN / SVM |
| ⚖️ 不平衡数据 | XGBoost / Random Forest (调参) / SMOTE | Logistic (默认) |
| 🎭 多重共线性 | Ridge / Lasso / PCA + Regression | Linear Regression |
| 🌟 非线性关系 | SVM (RBF) / Random Forest / XGBoost | Linear / Logistic |
| ⚡ 需要实时预测 | Logistic / Naive Bayes / Linear | KNN / Deep Learning |

---

## 常用公式

### 评估指标公式

#### 分类指标

**1. 准确率 (Accuracy)**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**2. 精确率 (Precision)**
```
Precision = TP / (TP + FP)
```
含义：预测为正的样本中，真正为正的比例

**3. 召回率 (Recall / Sensitivity)**
```
Recall = TP / (TP + FN)
```
含义:实际为正的样本中，被正确预测的比例

**4. F1分数 (F1 Score)**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
含义：精确率和召回率的调和平均

**5. 特异度 (Specificity)**
```
Specificity = TN / (TN + FP)
```

**6. ROC-AUC**
- ROC曲线：以FPR为横轴，TPR为纵轴
- AUC：ROC曲线下面积，[0.5, 1.0]

#### 回归指标

**1. 均方误差 (MSE - Mean Squared Error)**
```
MSE = (1/n) × Σ(yi - ŷi)²
```

**2. 均方根误差 (RMSE)**
```
RMSE = √MSE
```

**3. 平均绝对误差 (MAE - Mean Absolute Error)**
```
MAE = (1/n) × Σ|yi - ŷi|
```

**4. R² 分数 (决定系数)**
```
R² = 1 - (SS_res / SS_tot)
SS_res = Σ(yi - ŷi)²
SS_tot = Σ(yi - ȳ)²
```
- R² = 1：完美预测
- R² = 0：预测等同于均值
- R² < 0：比均值更差

**5. 调整R² (Adjusted R²)**
```
R²_adj = 1 - [(1-R²)(n-1) / (n-p-1)]
```
n：样本数，p：特征数

### 损失函数

**1. 均方误差损失 (MSE Loss)**
```
L = (1/n) × Σ(yi - ŷi)²
```
用于：线性回归

**2. 交叉熵损失 (Cross-Entropy Loss)**

二分类：
```
L = -(1/n) × Σ[yi log(ŷi) + (1-yi) log(1-ŷi)]
```

多分类：
```
L = -(1/n) × Σ Σ yic log(ŷic)
```
用于：逻辑回归、神经网络

**3. Hinge Loss**
```
L = max(0, 1 - yi × ŷi)
```
用于：SVM

**4. Huber Loss**
```
L = {
  0.5 × (y - ŷ)²           if |y - ŷ| ≤ δ
  δ × |y - ŷ| - 0.5 × δ²   otherwise
}
```
用于：鲁棒回归

### 正则化

**1. L1正则化 (Lasso)**
```
L = Loss + λ × Σ|wi|
```
- 产生稀疏解
- 特征选择

**2. L2正则化 (Ridge)**
```
L = Loss + λ × Σwi²
```
- 权重衰减
- 防止过拟合

**3. ElasticNet**
```
L = Loss + λ1 × Σ|wi| + λ2 × Σwi²
```
- 结合L1和L2

### 距离度量

**1. 欧氏距离 (Euclidean Distance)**
```
d(x, y) = √(Σ(xi - yi)²)
```

**2. 曼哈顿距离 (Manhattan Distance)**
```
d(x, y) = Σ|xi - yi|
```

**3. 余弦相似度 (Cosine Similarity)**
```
cos(θ) = (x · y) / (||x|| × ||y||)
```

**4. 闵可夫斯基距离 (Minkowski Distance)**
```
d(x, y) = (Σ|xi - yi|^p)^(1/p)
```
- p=1: 曼哈顿距离
- p=2: 欧氏距离

### 其他重要公式

**1. 信息增益 (Information Gain)**
```
IG(T, a) = H(T) - H(T|a)
H(T) = -Σ p(c) log p(c)  (熵)
```

**2. 基尼指数 (Gini Index)**
```
Gini = 1 - Σ pi²
```

**3. Sigmoid函数**
```
σ(x) = 1 / (1 + e^(-x))
```

**4. Softmax函数**
```
softmax(xi) = e^xi / Σe^xj
```

**5. 学习率衰减**
```
lr_t = lr_0 / (1 + decay × epoch)
```

---

## 算法调参建议

### 通用调参策略

1. **从默认参数开始**
2. **一次调一个参数**
3. **使用网格搜索或随机搜索**
4. **使用交叉验证**
5. **监控训练和验证误差**

### 关键参数调参范围

| 算法 | 参数 | 建议范围 | 调参技巧 |
|------|------|---------|---------|
| Random Forest | `n_estimators` | [100, 200, 500, 1000] | 越大越好，但边际收益递减 |
|  | `max_depth` | [10, 20, 30, None] | 控制过拟合 |
|  | `min_samples_split` | [2, 5, 10] | 太小易过拟合 |
| XGBoost | `n_estimators` | [100, 200, 500] | 配合learning_rate |
|  | `learning_rate` | [0.01, 0.05, 0.1, 0.3] | 小lr需要多轮 |
|  | `max_depth` | [3, 5, 7, 9] | 控制复杂度 |
|  | `subsample` | [0.6, 0.8, 1.0] | 防过拟合 |
| SVM | `C` | [0.1, 1, 10, 100] | 惩罚参数 |
|  | `gamma` | [0.001, 0.01, 0.1, 1] | RBF核参数 |
| KNN | `n_neighbors` | [3, 5, 7, 11] | 奇数避免平票 |
| Logistic | `C` | [0.01, 0.1, 1, 10] | 正则化强度 |

---

## 快速参考卡片

### 什么时候用什么算法？

| 场景 | 首选算法 |
|------|---------|
| 🚀 快速原型 | Logistic Regression / Random Forest |
| 🏆 Kaggle竞赛 | XGBoost / LightGBM / Stacking |
| 📊 可解释性 | Logistic Regression / Decision Tree |
| 📈 大数据 | SGD / Logistic / XGBoost |
| 📉 小数据 | SVM / KNN / Naive Bayes |
| 🎯 高精度 | Ensemble (XGBoost / Stacking) |
| ⚡ 实时预测 | Logistic / Naive Bayes / Linear |
| 🌟 非线性 | SVM (RBF) / Random Forest / XGBoost |
| 📝 文本分类 | Naive Bayes / SVM / BERT |
| 🖼️ 图像分类 | CNN / Vision Transformer |
| ⚖️ 不平衡数据 | XGBoost / SMOTE + Classifier |

---

## 总结

### 推荐学习顺序

1. **基础算法** (1-2周)
   - Linear Regression
   - Logistic Regression
   - KNN

2. **树模型** (2-3周)
   - Decision Tree
   - Random Forest

3. **支持向量机** (1-2周)
   - SVM

4. **集成学习** (2-3周)
   - Boosting (AdaBoost, Gradient Boosting)
   - XGBoost

5. **聚类和降维** (1-2周)
   - K-Means
   - PCA

6. **深度学习** (4+周)
   - Neural Network
   - CNN
   - RNN

### 实战建议

1. **总是从简单模型开始**（Baseline）
2. **理解数据比选择算法更重要**
3. **特征工程通常比算法选择更重要**
4. **使用交叉验证评估模型**
5. **尝试集成多个模型**
6. **监控过拟合**
7. **保存和版本化模型**

---

**最后更新：2025-11-18**
