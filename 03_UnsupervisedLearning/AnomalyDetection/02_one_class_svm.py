"""
One-Class SVM - 单类支持向量机异常检测
One-Class SVM - Anomaly Detection with Support Vector Machines

原理详解 / Algorithm Principle:
----------------------------------
One-Class SVM 是一种用于异常检测的无监督学习算法。
One-Class SVM is an unsupervised learning algorithm for anomaly detection.

核心思想 / Core Idea:
1. 在特征空间中寻找一个最小超球面或超平面
   Find a minimal hypersphere or hyperplane in feature space

2. 使正常数据点都在超球面内部或超平面一侧
   Ensure normal data points are inside the hypersphere or on one side of hyperplane

3. 利用核技巧处理非线性边界
   Use kernel trick to handle non-linear boundaries

4. nu 参数控制异常比例的上界
   nu parameter controls upper bound on anomaly fraction

数学原理 / Mathematical Principle:
min (1/2)||w||² + (1/νn)Σξᵢ - ρ
s.t. wᵀφ(xᵢ) ≥ ρ - ξᵢ, ξᵢ ≥ 0

其中 φ(x) 是核函数映射
where φ(x) is kernel function mapping

优势 / Advantages:
- 理论基础扎实（基于统计学习理论）
  Solid theoretical foundation (based on statistical learning theory)
- 支持多种核函数，灵活性高
  Supports multiple kernel functions, highly flexible
- 对噪声和离群点鲁棒
  Robust to noise and outliers

局限 / Limitations:
- 计算复杂度较高 O(n²)到O(n³)
  High computational complexity O(n²) to O(n³)
- 需要仔细调参（核函数、gamma、nu）
  Requires careful parameter tuning (kernel, gamma, nu)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_curve, auc, accuracy_score,
                            precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

# 导入工具模块 / Import utility modules
from utils import RANDOM_STATE, setup_chinese_fonts, create_subplots

# 设置中文字体 / Setup Chinese fonts
setup_chinese_fonts()

print("=" * 80)
print("One-Class SVM 单类支持向量机异常检测".center(80))
print("=" * 80)


# ============================================================================
# 1. 数据生成函数 / Data Generation Function
# ============================================================================

def generate_outlier_data(n_samples=300, outliers_fraction=0.1, random_state=RANDOM_STATE):
    """
    生成包含异常点的数据集
    Generate dataset with outliers

    Parameters:
    -----------
    n_samples : int
        总样本数 / Total number of samples
    outliers_fraction : float
        异常点比例 / Fraction of outliers
    random_state : int
        随机种子 / Random seed

    Returns:
    --------
    X : ndarray, shape (n_samples, 2)
        特征数据 / Feature data
    y : ndarray, shape (n_samples,)
        标签 (1: 正常, -1: 异常) / Labels (1: normal, -1: anomaly)
    """
    np.random.seed(random_state)

    # 计算样本数量 / Calculate sample counts
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    # 生成正常数据（两个高斯分布）/ Generate normal data (two Gaussian distributions)
    X_inliers = 0.3 * np.random.randn(n_inliers, 2)
    X_inliers = np.r_[X_inliers + 2, X_inliers - 2][:n_inliers]

    # 生成异常数据（均匀分布）/ Generate outliers (uniform distribution)
    X_outliers = np.random.uniform(low=-6, high=6, size=(n_outliers, 2))

    # 合并数据 / Combine data
    X = np.vstack([X_inliers, X_outliers])
    y = np.ones(n_samples, dtype=int)
    y[-n_outliers:] = -1  # -1 表示异常 / -1 indicates anomaly

    # 打乱顺序 / Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]

    return X, y


def plot_decision_boundary_svm(X, y, clf, ax, title):
    """
    绘制 One-Class SVM 决策边界
    Plot One-Class SVM decision boundary

    Parameters:
    -----------
    X : ndarray
        特征数据 / Feature data
    y : ndarray
        真实标签 / True labels
    clf : OneClassSVM
        训练好的模型 / Trained model
    ax : matplotlib axis
        绘图轴 / Plot axis
    title : str
        图表标题 / Plot title
    """
    # 创建网格 / Create mesh grid
    h = 0.1  # 步长 / Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测网格点 / Predict on mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界 / Plot decision boundary
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3,
                colors=['red', 'blue'], antialiased=True)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', alpha=0.7)

    # 绘制数据点 / Plot data points
    inliers = y == 1
    outliers = y == -1
    ax.scatter(X[inliers, 0], X[inliers, 1], c='blue',
              label='正常点 Normal', edgecolors='k', s=50, alpha=0.7)
    ax.scatter(X[outliers, 0], X[outliers, 1], c='red',
              label='异常点 Outlier', edgecolors='k', s=50, marker='^', alpha=0.7)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('特征 1 / Feature 1', fontsize=10)
    ax.set_ylabel('特征 2 / Feature 2', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)


# ============================================================================
# 2. 基础演示 - 不同核函数对比 / Basic Demo - Kernel Comparison
# ============================================================================

print("\n" + "=" * 80)
print("不同核函数对比 / Kernel Function Comparison")
print("=" * 80)

# 生成数据 / Generate data
X, y_true = generate_outlier_data(n_samples=300, outliers_fraction=0.15)

print(f"\n数据集信息 / Dataset Info:")
print(f"  总样本数 / Total samples: {len(X)}")
print(f"  正常样本 / Normal samples: {np.sum(y_true == 1)}")
print(f"  异常样本 / Anomaly samples: {np.sum(y_true == -1)}")

# 测试不同核函数 / Test different kernels
kernels = ['rbf', 'linear', 'poly']
kernel_names = ['RBF (径向基)', 'Linear (线性)', 'Polynomial (多项式)']

# 可视化1：不同核函数的决策边界 / Visualization 1: Decision boundaries of different kernels
fig1, axes = create_subplots(1, 3, figsize=(18, 5))

results = {}
for idx, (kernel, kernel_name) in enumerate(zip(kernels, kernel_names)):
    print(f"\n训练 {kernel_name} 核函数 / Training {kernel} kernel...")

    # 训练模型 / Train model
    if kernel == 'poly':
        clf = OneClassSVM(kernel=kernel, nu=0.15, gamma='auto', degree=3)
    else:
        clf = OneClassSVM(kernel=kernel, nu=0.15, gamma='auto')

    start_time = time.time()
    clf.fit(X)
    y_pred = clf.predict(X)
    training_time = time.time() - start_time

    # 计算指标 / Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=-1)
    f1 = f1_score(y_true, y_pred, pos_label=-1)

    results[kernel] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': training_time
    }

    print(f"  准确率 / Accuracy: {accuracy:.4f}")
    print(f"  精确率 / Precision: {precision:.4f}")
    print(f"  召回率 / Recall: {recall:.4f}")
    print(f"  F1分数 / F1-Score: {f1:.4f}")
    print(f"  训练时间 / Training time: {training_time:.4f}s")

    # 绘制决策边界 / Plot decision boundary
    plot_decision_boundary_svm(X, y_true, clf, axes[idx],
                              f'{kernel_name} 核函数\n{kernel.upper()} Kernel\n'
                              f'F1={f1:.3f}, Time={training_time:.3f}s')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/oc_01_kernels.png',
            dpi=150, bbox_inches='tight')
print("\n✓ 图1已保存: oc_01_kernels.png")


# ============================================================================
# 3. 参数分析 - nu 参数影响 / Parameter Analysis - nu Parameter
# ============================================================================

print("\n" + "=" * 80)
print("参数分析：nu（异常比例上界）/ Parameter Analysis: nu")
print("=" * 80)

fig2, axes = create_subplots(2, 2, figsize=(16, 12))
nu_values = [0.05, 0.10, 0.15, 0.20]

for idx, nu in enumerate(nu_values):
    ax = axes[idx // 2, idx % 2]

    # 训练模型 / Train model
    clf_temp = OneClassSVM(kernel='rbf', nu=nu, gamma='auto')
    clf_temp.fit(X)
    y_pred_temp = clf_temp.predict(X)

    # 计算指标 / Calculate metrics
    acc = accuracy_score(y_true, y_pred_temp)
    prec = precision_score(y_true, y_pred_temp, pos_label=-1, zero_division=0)
    rec = recall_score(y_true, y_pred_temp, pos_label=-1)
    f1 = f1_score(y_true, y_pred_temp, pos_label=-1)

    print(f"\nnu={nu}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    # 绘制 / Plot
    plot_decision_boundary_svm(X, y_true, clf_temp, ax,
                              f'nu={nu}\n'
                              f'Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/oc_02_nu_param.png',
            dpi=150, bbox_inches='tight')
print("✓ 图2已保存: oc_02_nu_param.png")


# ============================================================================
# 4. 参数分析 - gamma 参数影响（RBF kernel）/ Parameter Analysis - gamma
# ============================================================================

print("\n" + "=" * 80)
print("参数分析：gamma（RBF 核函数参数）/ Parameter Analysis: gamma")
print("=" * 80)

fig3, axes = create_subplots(2, 2, figsize=(16, 12))
gamma_values = [0.01, 0.1, 0.5, 1.0]

for idx, gamma in enumerate(gamma_values):
    ax = axes[idx // 2, idx % 2]

    # 训练模型 / Train model
    clf_temp = OneClassSVM(kernel='rbf', nu=0.15, gamma=gamma)
    clf_temp.fit(X)
    y_pred_temp = clf_temp.predict(X)

    # 计算指标 / Calculate metrics
    acc = accuracy_score(y_true, y_pred_temp)
    prec = precision_score(y_true, y_pred_temp, pos_label=-1, zero_division=0)
    rec = recall_score(y_true, y_pred_temp, pos_label=-1)
    f1 = f1_score(y_true, y_pred_temp, pos_label=-1)

    print(f"\ngamma={gamma}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    # 绘制 / Plot
    plot_decision_boundary_svm(X, y_true, clf_temp, ax,
                              f'gamma={gamma}\n'
                              f'Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/oc_03_gamma_param.png',
            dpi=150, bbox_inches='tight')
print("✓ 图3已保存: oc_03_gamma_param.png")


# ============================================================================
# 5. 高维数据应用 - Iris 数据集 / High-Dimensional Application - Iris Dataset
# ============================================================================

print("\n" + "=" * 80)
print("高维数据应用：Iris 数据集异常检测 / High-D Application: Iris Dataset")
print("=" * 80)

# 加载 Iris 数据集 / Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"\nIris 数据集信息 / Iris Dataset Info:")
print(f"  样本数 / Samples: {X_iris.shape[0]}")
print(f"  特征数 / Features: {X_iris.shape[1]}")
print(f"  类别数 / Classes: {len(np.unique(y_iris))}")
print(f"  类别名称 / Class names: {iris.target_names}")

# 使用类别0（Setosa）作为正常类，其他类别作为异常
# Use class 0 (Setosa) as normal, others as anomalies
X_normal = X_iris[y_iris == 0]  # Setosa
X_test = X_iris  # 全部数据用于测试 / All data for testing
y_test = np.ones(len(X_test))
y_test[y_iris != 0] = -1  # 非Setosa为异常 / Non-Setosa as anomalies

print(f"\n训练数据：仅使用 Setosa 类 / Training: Only Setosa class")
print(f"  训练样本数 / Training samples: {len(X_normal)}")
print(f"\n测试数据 / Test data:")
print(f"  总样本数 / Total samples: {len(X_test)}")
print(f"  正常样本（Setosa） / Normal (Setosa): {np.sum(y_test == 1)}")
print(f"  异常样本（非Setosa） / Anomaly (Non-Setosa): {np.sum(y_test == -1)}")

# 标准化 / Standardize
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)
X_test_scaled = scaler.transform(X_test)

# 训练 One-Class SVM / Train One-Class SVM
clf_iris = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
clf_iris.fit(X_normal_scaled)
y_pred_iris = clf_iris.predict(X_test_scaled)

# 评估 / Evaluate
acc_iris = accuracy_score(y_test, y_pred_iris)
prec_iris = precision_score(y_test, y_pred_iris, pos_label=-1, zero_division=0)
rec_iris = recall_score(y_test, y_pred_iris, pos_label=-1)
f1_iris = f1_score(y_test, y_pred_iris, pos_label=-1)

print(f"\nIris 异常检测结果 / Iris Anomaly Detection Results:")
print(f"  准确率 / Accuracy: {acc_iris:.4f}")
print(f"  精确率 / Precision: {prec_iris:.4f}")
print(f"  召回率 / Recall: {rec_iris:.4f}")
print(f"  F1分数 / F1-Score: {f1_iris:.4f}")

# 可视化4：Iris异常检测结果（使用前2个主要特征）/ Visualization 4: Iris results
fig4, axes = create_subplots(1, 2, figsize=(16, 6))

# 使用前两个特征可视化 / Visualize using first two features
X_iris_2d = X_iris[:, :2]
X_normal_2d = X_normal[:, :2]

# 左图：原始数据 / Left: Original data
colors = ['blue', 'red', 'green']
for i, (color, name) in enumerate(zip(colors, iris.target_names)):
    mask = y_iris == i
    axes[0].scatter(X_iris_2d[mask, 0], X_iris_2d[mask, 1],
                   c=color, label=name, edgecolors='k', s=50, alpha=0.7)
axes[0].set_xlabel(f'{iris.feature_names[0]}', fontsize=11)
axes[0].set_ylabel(f'{iris.feature_names[1]}', fontsize=11)
axes[0].set_title('Iris 数据集 - 真实标签 / Iris Dataset - True Labels',
                 fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：异常检测结果 / Right: Anomaly detection results
axes[1].scatter(X_iris_2d[y_pred_iris == 1, 0], X_iris_2d[y_pred_iris == 1, 1],
               c='blue', label='预测正常 Predicted Normal',
               edgecolors='k', s=50, alpha=0.7)
axes[1].scatter(X_iris_2d[y_pred_iris == -1, 0], X_iris_2d[y_pred_iris == -1, 1],
               c='red', label='预测异常 Predicted Anomaly',
               edgecolors='k', s=50, marker='^', alpha=0.7)
axes[1].set_xlabel(f'{iris.feature_names[0]}', fontsize=11)
axes[1].set_ylabel(f'{iris.feature_names[1]}', fontsize=11)
axes[1].set_title(f'异常检测结果 / Anomaly Detection Results\n'
                 f'F1={f1_iris:.3f}, Precision={prec_iris:.3f}, Recall={rec_iris:.3f}',
                 fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/oc_04_iris.png',
            dpi=150, bbox_inches='tight')
print("✓ 图4已保存: oc_04_iris.png")


# ============================================================================
# 6. 混淆矩阵 / Confusion Matrix
# ============================================================================

print("\n" + "=" * 80)
print("混淆矩阵分析 / Confusion Matrix Analysis")
print("=" * 80)

cm = confusion_matrix(y_test, y_pred_iris, labels=[1, -1])
print("\n混淆矩阵 / Confusion Matrix:")
print("              预测正常  预测异常")
print("              Pred Normal  Pred Anomaly")
print(f"真实正常 True Normal     {cm[0, 0]:4d}        {cm[0, 1]:4d}")
print(f"真实异常 True Anomaly    {cm[1, 0]:4d}        {cm[1, 1]:4d}")

# 可视化5：混淆矩阵 / Visualization 5: Confusion matrix
fig5, ax = create_subplots(1, 1, figsize=(8, 6))

im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)

# 设置标签 / Set labels
classes = ['正常 Normal', '异常 Anomaly']
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# 添加数值 / Add values
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
               ha="center", va="center",
               color="white" if cm[i, j] > thresh else "black",
               fontsize=14, fontweight='bold')

ax.set_ylabel('真实标签 / True Label', fontsize=11)
ax.set_xlabel('预测标签 / Predicted Label', fontsize=11)
ax.set_title('One-Class SVM 混淆矩阵 / Confusion Matrix',
            fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/oc_05_confusion_matrix.png',
            dpi=150, bbox_inches='tight')
print("✓ 图5已保存: oc_05_confusion_matrix.png")


# ============================================================================
# 7. ROC 曲线分析 / ROC Curve Analysis
# ============================================================================

print("\n" + "=" * 80)
print("ROC 曲线分析 / ROC Curve Analysis")
print("=" * 80)

# 获取决策函数值（距离边界的距离）/ Get decision function values
decision_scores = clf_iris.decision_function(X_test_scaled)

# 计算 ROC 曲线 / Calculate ROC curve
y_test_binary = (y_test == -1).astype(int)  # 异常为1 / Anomaly as 1
scores_inverted = -decision_scores  # 距离越负越异常 / More negative = more anomalous

fpr, tpr, thresholds = roc_curve(y_test_binary, scores_inverted)
roc_auc = auc(fpr, tpr)

print(f"\nROC AUC 分数 / ROC AUC Score: {roc_auc:.4f}")

# 可视化6：ROC曲线 / Visualization 6: ROC curve
fig6, axes = create_subplots(1, 2, figsize=(16, 6))

# 左图：ROC 曲线 / Left: ROC curve
axes[0].plot(fpr, tpr, color='darkorange', linewidth=2,
            label=f'One-Class SVM (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--',
            label='随机猜测 Random Guess')
axes[0].set_xlabel('假阳性率 / False Positive Rate', fontsize=11)
axes[0].set_ylabel('真阳性率 / True Positive Rate', fontsize=11)
axes[0].set_title('ROC 曲线 / ROC Curve', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# 右图：决策函数分布 / Right: Decision function distribution
axes[1].hist(decision_scores[y_test == 1], bins=30, alpha=0.7,
            label='正常点 Normal', color='blue', edgecolor='black')
axes[1].hist(decision_scores[y_test == -1], bins=30, alpha=0.7,
            label='异常点 Anomaly', color='red', edgecolor='black')
axes[1].axvline(x=0, color='green', linestyle='--', linewidth=2,
               label='决策阈值 Threshold')
axes[1].set_xlabel('决策函数值 / Decision Function Value', fontsize=11)
axes[1].set_ylabel('样本数量 / Number of Samples', fontsize=11)
axes[1].set_title('决策函数分布 / Decision Function Distribution',
                 fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/oc_06_roc.png',
            dpi=150, bbox_inches='tight')
print("✓ 图6已保存: oc_06_roc.png")


# ============================================================================
# 8. 算法对比：One-Class SVM vs Isolation Forest / Algorithm Comparison
# ============================================================================

print("\n" + "=" * 80)
print("算法对比：One-Class SVM vs Isolation Forest")
print("=" * 80)

# 使用之前的2D数据 / Use previous 2D data
X_compare, y_compare = generate_outlier_data(n_samples=500, outliers_fraction=0.15)

# 训练两个模型 / Train both models
print("\n训练模型 / Training models...")

# One-Class SVM
start_time = time.time()
clf_svm = OneClassSVM(kernel='rbf', nu=0.15, gamma='auto')
clf_svm.fit(X_compare)
y_pred_svm = clf_svm.predict(X_compare)
time_svm = time.time() - start_time

# Isolation Forest
start_time = time.time()
clf_if = IsolationForest(contamination=0.15, random_state=RANDOM_STATE, n_estimators=100)
y_pred_if = clf_if.fit_predict(X_compare)
time_if = time.time() - start_time

# 计算性能指标 / Calculate performance metrics
metrics = {
    'One-Class SVM': {
        'accuracy': accuracy_score(y_compare, y_pred_svm),
        'precision': precision_score(y_compare, y_pred_svm, pos_label=-1),
        'recall': recall_score(y_compare, y_pred_svm, pos_label=-1),
        'f1': f1_score(y_compare, y_pred_svm, pos_label=-1),
        'time': time_svm
    },
    'Isolation Forest': {
        'accuracy': accuracy_score(y_compare, y_pred_if),
        'precision': precision_score(y_compare, y_pred_if, pos_label=-1),
        'recall': recall_score(y_compare, y_pred_if, pos_label=-1),
        'f1': f1_score(y_compare, y_pred_if, pos_label=-1),
        'time': time_if
    }
}

print("\n性能对比 / Performance Comparison:")
print("-" * 80)
print(f"{'算法 / Algorithm':<20} {'准确率/Acc':<12} {'精确率/Prec':<12} "
      f"{'召回率/Rec':<12} {'F1分数/F1':<12} {'时间/Time(s)':<12}")
print("-" * 80)
for algo, m in metrics.items():
    print(f"{algo:<20} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
          f"{m['recall']:<12.4f} {m['f1']:<12.4f} {m['time']:<12.4f}")
print("-" * 80)

# 可视化7：性能对比柱状图 / Visualization 7: Performance comparison bar chart
fig7, axes = create_subplots(1, 2, figsize=(16, 6))

# 左图：性能指标对比 / Left: Performance metrics comparison
metric_names = ['准确率\nAccuracy', '精确率\nPrecision', '召回率\nRecall', 'F1分数\nF1-Score']
svm_scores = [metrics['One-Class SVM'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
if_scores = [metrics['Isolation Forest'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]

x = np.arange(len(metric_names))
width = 0.35

bars1 = axes[0].bar(x - width/2, svm_scores, width, label='One-Class SVM',
                    color='steelblue', edgecolor='black', alpha=0.8)
bars2 = axes[0].bar(x + width/2, if_scores, width, label='Isolation Forest',
                    color='coral', edgecolor='black', alpha=0.8)

# 添加数值标签 / Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

axes[0].set_ylabel('分数 / Score', fontsize=11)
axes[0].set_title('性能指标对比 / Performance Metrics Comparison',
                 fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metric_names, fontsize=10)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0, 1.1])

# 右图：训练时间对比 / Right: Training time comparison
time_data = [metrics['One-Class SVM']['time'], metrics['Isolation Forest']['time']]
colors_time = ['steelblue', 'coral']
bars = axes[1].bar(['One-Class SVM', 'Isolation Forest'], time_data,
                   color=colors_time, edgecolor='black', alpha=0.8)

# 添加数值标签 / Add value labels
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

axes[1].set_ylabel('训练时间（秒）/ Training Time (s)', fontsize=11)
axes[1].set_title('训练时间对比 / Training Time Comparison',
                 fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/oc_07_comparison.png',
            dpi=150, bbox_inches='tight')
print("✓ 图7已保存: oc_07_comparison.png")


# ============================================================================
# 9. 决策边界可视化对比 / Decision Boundary Comparison
# ============================================================================

print("\n" + "=" * 80)
print("决策边界对比可视化 / Decision Boundary Comparison Visualization")
print("=" * 80)

# 可视化8：决策边界对比 / Visualization 8: Decision boundary comparison
fig8, axes = create_subplots(1, 2, figsize=(16, 6))

# 左图：One-Class SVM
plot_decision_boundary_svm(X_compare, y_compare, clf_svm, axes[0],
                          f'One-Class SVM 决策边界\nDecision Boundary\n'
                          f'F1={metrics["One-Class SVM"]["f1"]:.3f}')

# 右图：Isolation Forest
# 需要重新实现 plot_decision_boundary 以支持 Isolation Forest
h = 0.1
x_min, x_max = X_compare[:, 0].min() - 1, X_compare[:, 0].max() + 1
y_min, y_max = X_compare[:, 1].min() - 1, X_compare[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z_if = clf_if.predict(np.c_[xx.ravel(), yy.ravel()])
Z_if = Z_if.reshape(xx.shape)

axes[1].contourf(xx, yy, Z_if, levels=[-1, 0, 1], alpha=0.3,
                colors=['red', 'blue'], antialiased=True)
axes[1].contour(xx, yy, Z_if, levels=[0], linewidths=2, colors='black', alpha=0.7)

inliers = y_compare == 1
outliers = y_compare == -1
axes[1].scatter(X_compare[inliers, 0], X_compare[inliers, 1], c='blue',
               label='正常点 Normal', edgecolors='k', s=50, alpha=0.7)
axes[1].scatter(X_compare[outliers, 0], X_compare[outliers, 1], c='red',
               label='异常点 Outlier', edgecolors='k', s=50, marker='^', alpha=0.7)

axes[1].set_title(f'Isolation Forest 决策边界\nDecision Boundary\n'
                 f'F1={metrics["Isolation Forest"]["f1"]:.3f}',
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('特征 1 / Feature 1', fontsize=10)
axes[1].set_ylabel('特征 2 / Feature 2', fontsize=10)
axes[1].legend(loc='best', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/oc_08_boundary_comparison.png',
            dpi=150, bbox_inches='tight')
print("✓ 图8已保存: oc_08_boundary_comparison.png")


# ============================================================================
# 10. 不同数据规模下的性能对比 / Performance Comparison at Different Scales
# ============================================================================

print("\n" + "=" * 80)
print("不同数据规模下的性能对比 / Performance at Different Data Scales")
print("=" * 80)

sample_sizes = [100, 200, 500, 1000, 2000]
time_svm_list = []
time_if_list = []
f1_svm_list = []
f1_if_list = []

for n_samples in sample_sizes:
    print(f"\n测试样本数 / Testing sample size: {n_samples}")

    # 生成数据 / Generate data
    X_scale, y_scale = generate_outlier_data(n_samples=n_samples, outliers_fraction=0.15)

    # One-Class SVM
    start_time = time.time()
    clf_svm_scale = OneClassSVM(kernel='rbf', nu=0.15, gamma='auto')
    clf_svm_scale.fit(X_scale)
    y_pred_svm_scale = clf_svm_scale.predict(X_scale)
    time_svm_list.append(time.time() - start_time)
    f1_svm_list.append(f1_score(y_scale, y_pred_svm_scale, pos_label=-1))

    # Isolation Forest
    start_time = time.time()
    clf_if_scale = IsolationForest(contamination=0.15, random_state=RANDOM_STATE, n_estimators=100)
    y_pred_if_scale = clf_if_scale.fit_predict(X_scale)
    time_if_list.append(time.time() - start_time)
    f1_if_list.append(f1_score(y_scale, y_pred_if_scale, pos_label=-1))

    print(f"  One-Class SVM: F1={f1_svm_list[-1]:.4f}, Time={time_svm_list[-1]:.4f}s")
    print(f"  Isolation Forest: F1={f1_if_list[-1]:.4f}, Time={time_if_list[-1]:.4f}s")

# 可视化9：不同规模性能对比 / Visualization 9: Performance at different scales
fig9, axes = create_subplots(1, 2, figsize=(16, 6))

# 左图：F1分数 / Left: F1 scores
axes[0].plot(sample_sizes, f1_svm_list, marker='o', linewidth=2,
            label='One-Class SVM', color='steelblue')
axes[0].plot(sample_sizes, f1_if_list, marker='s', linewidth=2,
            label='Isolation Forest', color='coral')
axes[0].set_xlabel('样本数量 / Sample Size', fontsize=11)
axes[0].set_ylabel('F1 分数 / F1 Score', fontsize=11)
axes[0].set_title('不同数据规模的 F1 分数 / F1 Score at Different Scales',
                 fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：训练时间 / Right: Training time
axes[1].plot(sample_sizes, time_svm_list, marker='o', linewidth=2,
            label='One-Class SVM', color='steelblue')
axes[1].plot(sample_sizes, time_if_list, marker='s', linewidth=2,
            label='Isolation Forest', color='coral')
axes[1].set_xlabel('样本数量 / Sample Size', fontsize=11)
axes[1].set_ylabel('训练时间（秒）/ Training Time (s)', fontsize=11)
axes[1].set_title('不同数据规模的训练时间 / Training Time at Different Scales',
                 fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/oc_09_scaling.png',
            dpi=150, bbox_inches='tight')
print("✓ 图9已保存: oc_09_scaling.png")


# ============================================================================
# 11. 总结和建议 / Summary and Recommendations
# ============================================================================

print("\n" + "=" * 80)
print("One-Class SVM 总结 / Summary")
print("=" * 80)

summary_text = """
【算法原理 / Algorithm Principle】
One-Class SVM 通过在高维特征空间中寻找一个最小超球面或超平面，
将正常数据与异常数据分离。利用核技巧处理非线性边界。

【主要优势 / Main Advantages】
✓ 理论基础扎实：基于统计学习理论和凸优化
  Solid theoretical foundation: Based on SLT and convex optimization
✓ 灵活性高：支持多种核函数（RBF、Linear、Poly等）
  Highly flexible: Supports multiple kernel functions
✓ 适合小样本：在样本量较小时仍能获得较好效果
  Suitable for small samples: Good performance even with limited data
✓ 鲁棒性强：对噪声和离群点具有较强的鲁棒性
  Robust: Strong robustness to noise and outliers

【主要劣势 / Main Disadvantages】
✗ 计算复杂：时间复杂度 O(n²)到O(n³)，不适合大规模数据
  Computationally expensive: O(n²)-O(n³), not suitable for large datasets
✗ 参数敏感：需要仔细调整 nu、gamma、kernel 等参数
  Parameter sensitive: Requires careful tuning of nu, gamma, kernel
✗ 内存消耗：需要存储支持向量，内存需求较大
  Memory intensive: Needs to store support vectors

【关键参数 / Key Parameters】
• nu: 异常比例的上界，范围 (0, 1]（推荐0.01-0.2）
  Upper bound on anomaly fraction, range (0, 1] (recommended 0.01-0.2)
• kernel: 核函数类型（'rbf', 'linear', 'poly', 'sigmoid'）
  Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
• gamma: RBF核的参数，控制决策边界的平滑度
  RBF kernel parameter, controls smoothness of decision boundary
  - gamma大 → 复杂边界 / Large gamma → complex boundary
  - gamma小 → 平滑边界 / Small gamma → smooth boundary

【核函数选择 / Kernel Selection】
• RBF（径向基）: 最常用，适合大多数情况，非线性边界
  RBF (Radial Basis): Most common, suitable for most cases, non-linear
• Linear（线性）: 计算快，适合线性可分数据
  Linear: Fast computation, suitable for linearly separable data
• Polynomial（多项式）: 适合具有多项式关系的数据
  Polynomial: Suitable for data with polynomial relationships

【与 Isolation Forest 对比 / Comparison with Isolation Forest】

┌────────────────┬─────────────────────┬─────────────────────┐
│    特性        │   One-Class SVM     │  Isolation Forest   │
│   Feature      │                     │                     │
├────────────────┼─────────────────────┼─────────────────────┤
│ 计算复杂度     │  O(n²)-O(n³) 慢     │  O(n) 快            │
│ Complexity     │  Slow               │  Fast               │
├────────────────┼─────────────────────┼─────────────────────┤
│ 内存消耗       │  高（支持向量）     │  低                 │
│ Memory         │  High (SVs)         │  Low                │
├────────────────┼─────────────────────┼─────────────────────┤
│ 参数调优       │  复杂（多参数）     │  简单（少参数）     │
│ Tuning         │  Complex            │  Simple             │
├────────────────┼─────────────────────┼─────────────────────┤
│ 小样本表现     │  好                 │  一般               │
│ Small Data     │  Good               │  Fair               │
├────────────────┼─────────────────────┼─────────────────────┤
│ 大数据表现     │  差（慢）           │  好（快）           │
│ Large Data     │  Poor (Slow)        │  Good (Fast)        │
├────────────────┼─────────────────────┼─────────────────────┤
│ 高维数据       │  一般               │  好                 │
│ High-D Data    │  Fair               │  Good               │
├────────────────┼─────────────────────┼─────────────────────┤
│ 理论基础       │  强（SVM理论）      │  直观（隔离思想）   │
│ Theory         │  Strong (SVM)       │  Intuitive          │
└────────────────┴─────────────────────┴─────────────────────┘

【使用场景建议 / Use Case Recommendations】

使用 One-Class SVM 当：
Use One-Class SVM when:
✓ 数据量较小（< 10,000 样本）
  Small dataset (< 10,000 samples)
✓ 需要精确的决策边界
  Need precise decision boundary
✓ 数据具有明显的非线性结构
  Data has clear non-linear structure
✓ 对计算时间要求不高
  Computational time is not critical
✓ 需要理论可解释性
  Need theoretical interpretability

使用 Isolation Forest 当：
Use Isolation Forest when:
✓ 数据量大（> 10,000 样本）
  Large dataset (> 10,000 samples)
✓ 高维稀疏数据
  High-dimensional sparse data
✓ 需要快速训练和预测
  Need fast training and prediction
✓ 参数调优时间有限
  Limited time for parameter tuning
✓ 实时异常检测应用
  Real-time anomaly detection

【实用技巧 / Practical Tips】
1. 始终标准化数据（StandardScaler）
   Always standardize data (StandardScaler)
2. 从 nu=0.1 开始，根据验证集调整
   Start with nu=0.1, adjust based on validation
3. RBF 核的 gamma 可从 'auto' 开始尝试
   Start with gamma='auto' for RBF kernel
4. 使用交叉验证选择最佳参数组合
   Use cross-validation to select best parameter combination
5. 可视化决策边界帮助理解模型行为
   Visualize decision boundary to understand model behavior
"""

print(summary_text)

print("=" * 80)
print("所有可视化图表已保存完成！")
print("All visualizations saved successfully!")
print("=" * 80)
