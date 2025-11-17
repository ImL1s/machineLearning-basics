"""
Isolation Forest 孤立森林 - 异常检测算法
Isolation Forest - Anomaly Detection Algorithm

原理详解 / Algorithm Principle:
----------------------------------
Isolation Forest（孤立森林）是一种专门用于异常检测的机器学习算法。
Isolation Forest is a machine learning algorithm specifically designed for anomaly detection.

核心思想 / Core Idea:
1. 异常点的特征是"容易被孤立"
   Anomalies have the characteristic of being "easily isolated"

2. 通过随机选择特征和分割值来构建隔离树
   Build isolation trees by randomly selecting features and split values

3. 异常点需要更少的分割次数就能被隔离
   Anomalies require fewer splits to be isolated

4. 异常分数 = 平均路径长度的函数
   Anomaly score = function of average path length

优势 / Advantages:
- 高效：线性时间复杂度 O(n)
  Efficient: Linear time complexity O(n)
- 无需定义距离度量
  No need to define distance metrics
- 适合高维数据
  Suitable for high-dimensional data
- 对异常比例不敏感
  Insensitive to anomaly ratio
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

# 导入工具模块 / Import utility modules
from utils import RANDOM_STATE, setup_chinese_fonts, create_subplots

# 设置中文字体 / Setup Chinese fonts
setup_chinese_fonts()

print("=" * 80)
print("Isolation Forest 孤立森林异常检测".center(80))
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

    # 生成正常数据（多个高斯分布）/ Generate normal data (multiple Gaussian distributions)
    X_inliers = np.random.randn(n_inliers, 2)
    X_inliers[:, 0] = X_inliers[:, 0] * 2 + 1
    X_inliers[:, 1] = X_inliers[:, 1] * 2 - 1

    # 生成异常数据（均匀分布在较大范围）/ Generate outliers (uniform distribution in larger range)
    X_outliers = np.random.uniform(low=-8, high=8, size=(n_outliers, 2))

    # 合并数据 / Combine data
    X = np.vstack([X_inliers, X_outliers])
    y = np.ones(n_samples, dtype=int)
    y[-n_outliers:] = -1  # -1 表示异常 / -1 indicates anomaly

    # 打乱顺序 / Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]

    return X, y


def plot_decision_boundary(X, y, clf, ax, title):
    """
    绘制决策边界
    Plot decision boundary

    Parameters:
    -----------
    X : ndarray
        特征数据 / Feature data
    y : ndarray
        真实标签 / True labels
    clf : IsolationForest
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
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', alpha=0.5)

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
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


# ============================================================================
# 2. 简单2D数据演示 / Simple 2D Data Demonstration
# ============================================================================

print("\n" + "=" * 80)
print("2D 数据异常检测演示 / 2D Data Anomaly Detection Demo")
print("=" * 80)

# 生成数据 / Generate data
X, y_true = generate_outlier_data(n_samples=300, outliers_fraction=0.15)

print(f"\n数据集信息 / Dataset Info:")
print(f"  总样本数 / Total samples: {len(X)}")
print(f"  正常样本 / Normal samples: {np.sum(y_true == 1)}")
print(f"  异常样本 / Anomaly samples: {np.sum(y_true == -1)}")
print(f"  异常比例 / Anomaly ratio: {np.sum(y_true == -1) / len(X):.2%}")

# 训练 Isolation Forest / Train Isolation Forest
clf = IsolationForest(contamination=0.15, random_state=RANDOM_STATE, n_estimators=100)
y_pred = clf.fit_predict(X)

# 获取异常分数（越负越异常）/ Get anomaly scores (more negative = more anomalous)
scores = clf.score_samples(X)

print(f"\n模型预测结果 / Model Prediction Results:")
print(f"  预测正常 / Predicted normal: {np.sum(y_pred == 1)}")
print(f"  预测异常 / Predicted anomaly: {np.sum(y_pred == -1)}")

# 评估指标 / Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label=-1)
recall = recall_score(y_true, y_pred, pos_label=-1)
f1 = f1_score(y_true, y_pred, pos_label=-1)

print(f"\n性能指标 / Performance Metrics:")
print(f"  准确率 / Accuracy: {accuracy:.4f}")
print(f"  精确率 / Precision: {precision:.4f}")
print(f"  召回率 / Recall: {recall:.4f}")
print(f"  F1分数 / F1-Score: {f1:.4f}")

# 可视化1：原始数据和检测结果 / Visualization 1: Original data and detection results
fig1, axes = create_subplots(1, 2, figsize=(16, 6))

# 左图：原始数据标签 / Left: Original data labels
axes[0].scatter(X[y_true == 1, 0], X[y_true == 1, 1],
               c='blue', label='真实正常 True Normal', edgecolors='k', s=50, alpha=0.7)
axes[0].scatter(X[y_true == -1, 0], X[y_true == -1, 1],
               c='red', label='真实异常 True Outlier', edgecolors='k', s=50, marker='^', alpha=0.7)
axes[0].set_title('原始数据标签 / True Labels', fontsize=14, fontweight='bold')
axes[0].set_xlabel('特征 1 / Feature 1', fontsize=11)
axes[0].set_ylabel('特征 2 / Feature 2', fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：决策边界 / Right: Decision boundary
plot_decision_boundary(X, y_true, clf, axes[1],
                      'Isolation Forest 决策边界 / Decision Boundary')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/if_01_basic_demo.png',
            dpi=150, bbox_inches='tight')
print("\n✓ 图1已保存: if_01_basic_demo.png")


# ============================================================================
# 3. 参数分析 - contamination 参数影响 / Parameter Analysis - contamination
# ============================================================================

print("\n" + "=" * 80)
print("参数分析：contamination（预期异常比例）/ Parameter Analysis: contamination")
print("=" * 80)

fig2, axes = create_subplots(2, 2, figsize=(16, 12))
contamination_values = [0.05, 0.10, 0.15, 0.20]

for idx, contamination in enumerate(contamination_values):
    ax = axes[idx // 2, idx % 2]

    # 训练模型 / Train model
    clf_temp = IsolationForest(contamination=contamination,
                               random_state=RANDOM_STATE,
                               n_estimators=100)
    y_pred_temp = clf_temp.fit_predict(X)

    # 计算指标 / Calculate metrics
    acc = accuracy_score(y_true, y_pred_temp)
    prec = precision_score(y_true, y_pred_temp, pos_label=-1, zero_division=0)
    rec = recall_score(y_true, y_pred_temp, pos_label=-1)

    # 绘制 / Plot
    plot_decision_boundary(X, y_true, clf_temp, ax,
                          f'contamination={contamination}\n'
                          f'Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/if_02_contamination.png',
            dpi=150, bbox_inches='tight')
print("✓ 图2已保存: if_02_contamination.png")


# ============================================================================
# 4. 参数分析 - n_estimators 影响 / Parameter Analysis - n_estimators
# ============================================================================

print("\n" + "=" * 80)
print("参数分析：n_estimators（树的数量）/ Parameter Analysis: n_estimators")
print("=" * 80)

estimators_range = [10, 50, 100, 200, 300]
performance_metrics = {'estimators': [], 'accuracy': [], 'precision': [],
                      'recall': [], 'f1': [], 'time': []}

for n_est in estimators_range:
    start_time = time.time()

    clf_temp = IsolationForest(n_estimators=n_est,
                               contamination=0.15,
                               random_state=RANDOM_STATE)
    y_pred_temp = clf_temp.fit_predict(X)

    elapsed_time = time.time() - start_time

    # 计算指标 / Calculate metrics
    performance_metrics['estimators'].append(n_est)
    performance_metrics['accuracy'].append(accuracy_score(y_true, y_pred_temp))
    performance_metrics['precision'].append(precision_score(y_true, y_pred_temp, pos_label=-1))
    performance_metrics['recall'].append(recall_score(y_true, y_pred_temp, pos_label=-1))
    performance_metrics['f1'].append(f1_score(y_true, y_pred_temp, pos_label=-1))
    performance_metrics['time'].append(elapsed_time)

    print(f"n_estimators={n_est:3d}: Acc={performance_metrics['accuracy'][-1]:.4f}, "
          f"F1={performance_metrics['f1'][-1]:.4f}, Time={elapsed_time:.4f}s")

# 可视化3：n_estimators 影响 / Visualization 3: n_estimators effect
fig3, axes = create_subplots(1, 2, figsize=(16, 6))

# 左图：性能指标 / Left: Performance metrics
axes[0].plot(performance_metrics['estimators'], performance_metrics['accuracy'],
            marker='o', label='准确率 Accuracy', linewidth=2)
axes[0].plot(performance_metrics['estimators'], performance_metrics['precision'],
            marker='s', label='精确率 Precision', linewidth=2)
axes[0].plot(performance_metrics['estimators'], performance_metrics['recall'],
            marker='^', label='召回率 Recall', linewidth=2)
axes[0].plot(performance_metrics['estimators'], performance_metrics['f1'],
            marker='d', label='F1分数 F1-Score', linewidth=2)
axes[0].set_xlabel('n_estimators（树的数量）/ Number of Trees', fontsize=11)
axes[0].set_ylabel('性能指标 / Performance Metrics', fontsize=11)
axes[0].set_title('n_estimators 对性能的影响 / Effect on Performance',
                 fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：训练时间 / Right: Training time
axes[1].plot(performance_metrics['estimators'], performance_metrics['time'],
            marker='o', color='red', linewidth=2)
axes[1].set_xlabel('n_estimators（树的数量）/ Number of Trees', fontsize=11)
axes[1].set_ylabel('训练时间（秒）/ Training Time (s)', fontsize=11)
axes[1].set_title('n_estimators 对训练时间的影响 / Effect on Training Time',
                 fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/if_03_n_estimators.png',
            dpi=150, bbox_inches='tight')
print("✓ 图3已保存: if_03_n_estimators.png")


# ============================================================================
# 5. 异常分数分布 / Anomaly Score Distribution
# ============================================================================

print("\n" + "=" * 80)
print("异常分数分析 / Anomaly Score Analysis")
print("=" * 80)

# 重新训练获取分数 / Retrain to get scores
clf_score = IsolationForest(contamination=0.15, random_state=RANDOM_STATE, n_estimators=100)
clf_score.fit(X)
scores = clf_score.score_samples(X)
y_pred_score = clf_score.predict(X)

print(f"\n异常分数统计 / Score Statistics:")
print(f"  最小分数 / Min score: {scores.min():.4f}")
print(f"  最大分数 / Max score: {scores.max():.4f}")
print(f"  平均分数 / Mean score: {scores.mean():.4f}")
print(f"  分数标准差 / Std score: {scores.std():.4f}")

# 可视化4：异常分数分布 / Visualization 4: Score distribution
fig4, axes = create_subplots(1, 2, figsize=(16, 6))

# 左图：异常分数直方图 / Left: Score histogram
axes[0].hist(scores[y_true == 1], bins=50, alpha=0.7, label='正常点 Normal', color='blue', edgecolor='black')
axes[0].hist(scores[y_true == -1], bins=50, alpha=0.7, label='异常点 Outlier', color='red', edgecolor='black')
axes[0].axvline(x=0, color='green', linestyle='--', linewidth=2, label='决策阈值 Threshold')
axes[0].set_xlabel('异常分数 / Anomaly Score', fontsize=11)
axes[0].set_ylabel('样本数量 / Number of Samples', fontsize=11)
axes[0].set_title('异常分数分布 / Anomaly Score Distribution', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：散点图着色by分数 / Right: Scatter plot colored by score
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=scores, cmap='RdYlBu',
                         edgecolors='k', s=50, alpha=0.7)
axes[1].set_xlabel('特征 1 / Feature 1', fontsize=11)
axes[1].set_ylabel('特征 2 / Feature 2', fontsize=11)
axes[1].set_title('异常分数可视化 / Anomaly Score Visualization',
                 fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[1])
cbar.set_label('异常分数 / Anomaly Score', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/if_04_scores.png',
            dpi=150, bbox_inches='tight')
print("✓ 图4已保存: if_04_scores.png")


# ============================================================================
# 6. ROC 曲线和 AUC / ROC Curve and AUC
# ============================================================================

print("\n" + "=" * 80)
print("ROC 曲线分析 / ROC Curve Analysis")
print("=" * 80)

# 计算 ROC 曲线 / Calculate ROC curve
# 注意：需要将标签转换为0/1，异常为1 / Note: Convert labels to 0/1, anomaly as 1
y_true_binary = (y_true == -1).astype(int)
scores_inverted = -scores  # 分数越低越异常，需要取反 / Lower score = more anomalous, need to invert

fpr, tpr, thresholds = roc_curve(y_true_binary, scores_inverted)
roc_auc = auc(fpr, tpr)

# 计算 Precision-Recall 曲线 / Calculate Precision-Recall curve
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true_binary, scores_inverted)

print(f"\nROC AUC 分数 / ROC AUC Score: {roc_auc:.4f}")

# 可视化5：ROC和PR曲线 / Visualization 5: ROC and PR curves
fig5, axes = create_subplots(1, 2, figsize=(16, 6))

# 左图：ROC 曲线 / Left: ROC curve
axes[0].plot(fpr, tpr, color='darkorange', linewidth=2,
            label=f'ROC 曲线 (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--',
            label='随机猜测 Random Guess')
axes[0].set_xlabel('假阳性率 / False Positive Rate', fontsize=11)
axes[0].set_ylabel('真阳性率 / True Positive Rate', fontsize=11)
axes[0].set_title('ROC 曲线 / ROC Curve', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# 右图：Precision-Recall 曲线 / Right: Precision-Recall curve
axes[1].plot(recall_curve, precision_curve, color='green', linewidth=2,
            label='PR 曲线 / PR Curve')
axes[1].set_xlabel('召回率 / Recall', fontsize=11)
axes[1].set_ylabel('精确率 / Precision', fontsize=11)
axes[1].set_title('Precision-Recall 曲线 / Precision-Recall Curve',
                 fontsize=13, fontweight='bold')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/if_05_roc_pr.png',
            dpi=150, bbox_inches='tight')
print("✓ 图5已保存: if_05_roc_pr.png")


# ============================================================================
# 7. 混淆矩阵 / Confusion Matrix
# ============================================================================

print("\n" + "=" * 80)
print("混淆矩阵分析 / Confusion Matrix Analysis")
print("=" * 80)

cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
print("\n混淆矩阵 / Confusion Matrix:")
print("              预测正常  预测异常")
print("              Pred Normal  Pred Anomaly")
print(f"真实正常 True Normal     {cm[0, 0]:4d}        {cm[0, 1]:4d}")
print(f"真实异常 True Anomaly    {cm[1, 0]:4d}        {cm[1, 1]:4d}")

# 可视化6：混淆矩阵 / Visualization 6: Confusion matrix
fig6, ax = create_subplots(1, 1, figsize=(8, 6))

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
ax.set_title('混淆矩阵 / Confusion Matrix', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/if_06_confusion_matrix.png',
            dpi=150, bbox_inches='tight')
print("✓ 图6已保存: if_06_confusion_matrix.png")


# ============================================================================
# 8. 实际应用案例1：信用卡欺诈检测 / Application 1: Credit Card Fraud
# ============================================================================

print("\n" + "=" * 80)
print("实际应用案例1：信用卡欺诈检测模拟 / Application 1: Credit Card Fraud Detection")
print("=" * 80)

# 生成模拟信用卡交易数据 / Generate simulated credit card transaction data
np.random.seed(RANDOM_STATE)
n_transactions = 1000
fraud_rate = 0.02

# 正常交易：金额较小，时间分布均匀 / Normal: small amount, uniform time
normal_amount = np.random.gamma(2, 50, int(n_transactions * (1 - fraud_rate)))
normal_time = np.random.uniform(0, 24, int(n_transactions * (1 - fraud_rate)))

# 欺诈交易：金额较大，时间集中在深夜 / Fraud: large amount, late night
fraud_amount = np.random.gamma(5, 100, int(n_transactions * fraud_rate))
fraud_time = np.random.uniform(22, 4, int(n_transactions * fraud_rate)) % 24

# 合并数据 / Combine data
X_credit = np.vstack([
    np.column_stack([normal_amount, normal_time]),
    np.column_stack([fraud_amount, fraud_time])
])
y_credit = np.hstack([
    np.ones(len(normal_amount)),
    -np.ones(len(fraud_amount))
])

# 标准化 / Standardize
scaler = StandardScaler()
X_credit_scaled = scaler.fit_transform(X_credit)

# 训练模型 / Train model
clf_credit = IsolationForest(contamination=fraud_rate, random_state=RANDOM_STATE, n_estimators=100)
y_pred_credit = clf_credit.fit_predict(X_credit_scaled)

# 评估 / Evaluate
acc_credit = accuracy_score(y_credit, y_pred_credit)
prec_credit = precision_score(y_credit, y_pred_credit, pos_label=-1)
rec_credit = recall_score(y_credit, y_pred_credit, pos_label=-1)
f1_credit = f1_score(y_credit, y_pred_credit, pos_label=-1)

print(f"\n信用卡欺诈检测结果 / Credit Card Fraud Detection Results:")
print(f"  总交易数 / Total transactions: {len(X_credit)}")
print(f"  欺诈交易 / Fraud transactions: {np.sum(y_credit == -1)}")
print(f"  准确率 / Accuracy: {acc_credit:.4f}")
print(f"  精确率 / Precision: {prec_credit:.4f}")
print(f"  召回率 / Recall: {rec_credit:.4f}")
print(f"  F1分数 / F1-Score: {f1_credit:.4f}")

# 可视化7：信用卡欺诈检测 / Visualization 7: Credit card fraud detection
fig7, axes = create_subplots(1, 2, figsize=(16, 6))

# 左图：原始数据 / Left: Original data
axes[0].scatter(X_credit[y_credit == 1, 0], X_credit[y_credit == 1, 1],
               c='blue', label='正常交易 Normal', edgecolors='k', s=30, alpha=0.6)
axes[0].scatter(X_credit[y_credit == -1, 0], X_credit[y_credit == -1, 1],
               c='red', label='欺诈交易 Fraud', edgecolors='k', s=50, marker='^', alpha=0.8)
axes[0].set_xlabel('交易金额 / Transaction Amount', fontsize=11)
axes[0].set_ylabel('交易时间（小时）/ Transaction Time (hour)', fontsize=11)
axes[0].set_title('信用卡交易数据 / Credit Card Transactions',
                 fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：检测结果 / Right: Detection results
correct = (y_credit == y_pred_credit)
axes[1].scatter(X_credit[correct & (y_credit == 1), 0],
               X_credit[correct & (y_credit == 1), 1],
               c='blue', label='正确识别正常 True Normal',
               edgecolors='k', s=30, alpha=0.6)
axes[1].scatter(X_credit[correct & (y_credit == -1), 0],
               X_credit[correct & (y_credit == -1), 1],
               c='red', label='正确识别欺诈 True Fraud',
               edgecolors='k', s=50, marker='^', alpha=0.8)
axes[1].scatter(X_credit[~correct, 0], X_credit[~correct, 1],
               c='orange', label='识别错误 Misclassified',
               edgecolors='k', s=80, marker='x', linewidths=2, alpha=0.9)
axes[1].set_xlabel('交易金额 / Transaction Amount', fontsize=11)
axes[1].set_ylabel('交易时间（小时）/ Transaction Time (hour)', fontsize=11)
axes[1].set_title(f'欺诈检测结果 / Fraud Detection Results\n'
                 f'Precision={prec_credit:.3f}, Recall={rec_credit:.3f}',
                 fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/if_07_credit_fraud.png',
            dpi=150, bbox_inches='tight')
print("✓ 图7已保存: if_07_credit_fraud.png")


# ============================================================================
# 9. 实际应用案例2：网络入侵检测 / Application 2: Network Intrusion Detection
# ============================================================================

print("\n" + "=" * 80)
print("实际应用案例2：网络入侵检测模拟 / Application 2: Network Intrusion Detection")
print("=" * 80)

# 生成模拟网络流量数据 / Generate simulated network traffic data
np.random.seed(RANDOM_STATE)
n_packets = 800
intrusion_rate = 0.05

# 正常流量：包大小中等，连接数少 / Normal: medium packet size, few connections
normal_packet_size = np.random.normal(500, 100, int(n_packets * (1 - intrusion_rate)))
normal_connections = np.random.poisson(5, int(n_packets * (1 - intrusion_rate)))

# 入侵流量：包大小异常，连接数多 / Intrusion: abnormal packet size, many connections
intrusion_packet_size = np.random.choice([
    np.random.normal(100, 20, int(n_packets * intrusion_rate // 2)),  # 小包攻击
    np.random.normal(2000, 300, int(n_packets * intrusion_rate // 2))  # 大包攻击
]).flatten()
intrusion_connections = np.random.poisson(50, int(n_packets * intrusion_rate))

# 合并数据 / Combine data
X_network = np.vstack([
    np.column_stack([normal_packet_size, normal_connections]),
    np.column_stack([intrusion_packet_size, intrusion_connections])
])
y_network = np.hstack([
    np.ones(len(normal_packet_size)),
    -np.ones(len(intrusion_packet_size))
])

# 标准化 / Standardize
X_network_scaled = scaler.fit_transform(X_network)

# 训练模型 / Train model
clf_network = IsolationForest(contamination=intrusion_rate, random_state=RANDOM_STATE, n_estimators=100)
y_pred_network = clf_network.fit_predict(X_network_scaled)

# 评估 / Evaluate
acc_network = accuracy_score(y_network, y_pred_network)
prec_network = precision_score(y_network, y_pred_network, pos_label=-1)
rec_network = recall_score(y_network, y_pred_network, pos_label=-1)
f1_network = f1_score(y_network, y_pred_network, pos_label=-1)

print(f"\n网络入侵检测结果 / Network Intrusion Detection Results:")
print(f"  总数据包 / Total packets: {len(X_network)}")
print(f"  入侵数据包 / Intrusion packets: {np.sum(y_network == -1)}")
print(f"  准确率 / Accuracy: {acc_network:.4f}")
print(f"  精确率 / Precision: {prec_network:.4f}")
print(f"  召回率 / Recall: {rec_network:.4f}")
print(f"  F1分数 / F1-Score: {f1_network:.4f}")

# 可视化8：网络入侵检测 / Visualization 8: Network intrusion detection
fig8, axes = create_subplots(1, 2, figsize=(16, 6))

# 左图：原始数据 / Left: Original data
axes[0].scatter(X_network[y_network == 1, 0], X_network[y_network == 1, 1],
               c='green', label='正常流量 Normal', edgecolors='k', s=30, alpha=0.6)
axes[0].scatter(X_network[y_network == -1, 0], X_network[y_network == -1, 1],
               c='red', label='入侵流量 Intrusion', edgecolors='k', s=50, marker='^', alpha=0.8)
axes[0].set_xlabel('数据包大小 / Packet Size', fontsize=11)
axes[0].set_ylabel('连接数 / Number of Connections', fontsize=11)
axes[0].set_title('网络流量数据 / Network Traffic Data',
                 fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：检测结果 / Right: Detection results
correct_network = (y_network == y_pred_network)
axes[1].scatter(X_network[correct_network & (y_network == 1), 0],
               X_network[correct_network & (y_network == 1), 1],
               c='green', label='正确识别正常 True Normal',
               edgecolors='k', s=30, alpha=0.6)
axes[1].scatter(X_network[correct_network & (y_network == -1), 0],
               X_network[correct_network & (y_network == -1), 1],
               c='red', label='正确识别入侵 True Intrusion',
               edgecolors='k', s=50, marker='^', alpha=0.8)
axes[1].scatter(X_network[~correct_network, 0], X_network[~correct_network, 1],
               c='orange', label='识别错误 Misclassified',
               edgecolors='k', s=80, marker='x', linewidths=2, alpha=0.9)
axes[1].set_xlabel('数据包大小 / Packet Size', fontsize=11)
axes[1].set_ylabel('连接数 / Number of Connections', fontsize=11)
axes[1].set_title(f'入侵检测结果 / Intrusion Detection Results\n'
                 f'Precision={prec_network:.3f}, Recall={rec_network:.3f}',
                 fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/if_08_network_intrusion.png',
            dpi=150, bbox_inches='tight')
print("✓ 图8已保存: if_08_network_intrusion.png")


# ============================================================================
# 10. 实际应用案例3：设备故障检测 / Application 3: Equipment Fault Detection
# ============================================================================

print("\n" + "=" * 80)
print("实际应用案例3：设备故障检测模拟 / Application 3: Equipment Fault Detection")
print("=" * 80)

# 生成模拟设备传感器数据 / Generate simulated equipment sensor data
np.random.seed(RANDOM_STATE)
n_readings = 500
fault_rate = 0.08

# 正常运行：温度和振动在正常范围 / Normal: temperature and vibration in normal range
normal_temp = np.random.normal(70, 5, int(n_readings * (1 - fault_rate)))
normal_vibration = np.random.normal(0.5, 0.1, int(n_readings * (1 - fault_rate)))

# 故障状态：温度过高或振动异常 / Fault: high temperature or abnormal vibration
fault_temp = np.random.normal(95, 10, int(n_readings * fault_rate))
fault_vibration = np.random.normal(2.0, 0.5, int(n_readings * fault_rate))

# 合并数据 / Combine data
X_equipment = np.vstack([
    np.column_stack([normal_temp, normal_vibration]),
    np.column_stack([fault_temp, fault_vibration])
])
y_equipment = np.hstack([
    np.ones(len(normal_temp)),
    -np.ones(len(fault_temp))
])

# 标准化 / Standardize
X_equipment_scaled = scaler.fit_transform(X_equipment)

# 训练模型 / Train model
clf_equipment = IsolationForest(contamination=fault_rate, random_state=RANDOM_STATE, n_estimators=100)
y_pred_equipment = clf_equipment.fit_predict(X_equipment_scaled)

# 评估 / Evaluate
acc_equipment = accuracy_score(y_equipment, y_pred_equipment)
prec_equipment = precision_score(y_equipment, y_pred_equipment, pos_label=-1)
rec_equipment = recall_score(y_equipment, y_pred_equipment, pos_label=-1)
f1_equipment = f1_score(y_equipment, y_pred_equipment, pos_label=-1)

print(f"\n设备故障检测结果 / Equipment Fault Detection Results:")
print(f"  总读数 / Total readings: {len(X_equipment)}")
print(f"  故障读数 / Fault readings: {np.sum(y_equipment == -1)}")
print(f"  准确率 / Accuracy: {acc_equipment:.4f}")
print(f"  精确率 / Precision: {prec_equipment:.4f}")
print(f"  召回率 / Recall: {rec_equipment:.4f}")
print(f"  F1分数 / F1-Score: {f1_equipment:.4f}")

# 可视化9：设备故障检测 / Visualization 9: Equipment fault detection
fig9, axes = create_subplots(1, 2, figsize=(16, 6))

# 左图：原始数据 / Left: Original data
axes[0].scatter(X_equipment[y_equipment == 1, 0], X_equipment[y_equipment == 1, 1],
               c='green', label='正常运行 Normal', edgecolors='k', s=30, alpha=0.6)
axes[0].scatter(X_equipment[y_equipment == -1, 0], X_equipment[y_equipment == -1, 1],
               c='red', label='故障状态 Fault', edgecolors='k', s=50, marker='^', alpha=0.8)
axes[0].set_xlabel('温度 (°C) / Temperature (°C)', fontsize=11)
axes[0].set_ylabel('振动 (mm/s) / Vibration (mm/s)', fontsize=11)
axes[0].set_title('设备传感器数据 / Equipment Sensor Data',
                 fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：检测结果 / Right: Detection results
correct_equipment = (y_equipment == y_pred_equipment)
axes[1].scatter(X_equipment[correct_equipment & (y_equipment == 1), 0],
               X_equipment[correct_equipment & (y_equipment == 1), 1],
               c='green', label='正确识别正常 True Normal',
               edgecolors='k', s=30, alpha=0.6)
axes[1].scatter(X_equipment[correct_equipment & (y_equipment == -1), 0],
               X_equipment[correct_equipment & (y_equipment == -1), 1],
               c='red', label='正确识别故障 True Fault',
               edgecolors='k', s=50, marker='^', alpha=0.8)
axes[1].scatter(X_equipment[~correct_equipment, 0], X_equipment[~correct_equipment, 1],
               c='orange', label='识别错误 Misclassified',
               edgecolors='k', s=80, marker='x', linewidths=2, alpha=0.9)
axes[1].set_xlabel('温度 (°C) / Temperature (°C)', fontsize=11)
axes[1].set_ylabel('振动 (mm/s) / Vibration (mm/s)', fontsize=11)
axes[1].set_title(f'故障检测结果 / Fault Detection Results\n'
                 f'Precision={prec_equipment:.3f}, Recall={rec_equipment:.3f}',
                 fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/03_UnsupervisedLearning/AnomalyDetection/if_09_equipment_fault.png',
            dpi=150, bbox_inches='tight')
print("✓ 图9已保存: if_09_equipment_fault.png")


# ============================================================================
# 11. 总结和建议 / Summary and Recommendations
# ============================================================================

print("\n" + "=" * 80)
print("Isolation Forest 总结 / Summary")
print("=" * 80)

summary_text = """
【算法原理 / Algorithm Principle】
Isolation Forest 基于"异常点容易被孤立"的假设，通过随机构建隔离树来检测异常。
异常点比正常点需要更少的分割次数就能被隔离出来。

【主要优势 / Main Advantages】
✓ 高效：线性时间复杂度，适合大规模数据
  Efficient: Linear time complexity, suitable for large-scale data
✓ 无监督：不需要标签数据
  Unsupervised: No need for labeled data
✓ 适合高维：不依赖距离度量，适合高维稀疏数据
  High-dimensional friendly: No distance metrics, suitable for sparse data
✓ 少量参数：只需调整 contamination 和 n_estimators
  Few parameters: Only need to tune contamination and n_estimators

【关键参数 / Key Parameters】
• contamination: 预期异常比例（默认0.1）
  Expected proportion of outliers (default 0.1)
• n_estimators: 隔离树数量（推荐100-300）
  Number of isolation trees (recommended 100-300)
• max_samples: 训练每棵树的样本数（默认256或'auto'）
  Number of samples for training each tree (default 256 or 'auto')

【适用场景 / Use Cases】
✓ 信用卡欺诈检测 / Credit card fraud detection
✓ 网络入侵检测 / Network intrusion detection
✓ 设备故障检测 / Equipment fault detection
✓ 日志异常分析 / Log anomaly analysis
✓ 传感器数据监控 / Sensor data monitoring

【使用建议 / Recommendations】
1. 根据实际情况估计 contamination 参数
   Estimate contamination based on actual situation
2. 增加 n_estimators 可提高稳定性，但会增加计算时间
   Increasing n_estimators improves stability but increases computation time
3. 对于大数据集，可以减小 max_samples 加速训练
   For large datasets, reduce max_samples to speed up training
4. 结合异常分数进行阈值调整，获得更好的性能
   Combine anomaly scores for threshold tuning to get better performance
"""

print(summary_text)

print("=" * 80)
print("所有可视化图表已保存完成！")
print("All visualizations saved successfully!")
print("=" * 80)
