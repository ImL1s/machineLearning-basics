"""
文本分类（Text Classification）
NLP 最常见的应用之一

Text Classification - One of the Most Common NLP Applications

从垃圾邮件过滤到情感分析，文本分类无处不在
From spam filtering to sentiment analysis, text classification is everywhere
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, precision_recall_fscore_support,
                            roc_curve, auc, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from utils import RANDOM_STATE, setup_chinese_fonts, create_subplots, DPI, save_figure, get_output_path

# 尝试导入深度学习库
try:
    from sklearn.neural_network import MLPClassifier
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False

setup_chinese_fonts()

print("=" * 80)
print("文本分类（Text Classification）教程".center(80))
print("=" * 80)

# ============================================================================
# 1. 文本分类任务介绍 / Text Classification Introduction
# ============================================================================
print("\n【1】文本分类任务介绍")
print("-" * 80)
print("""
什么是文本分类？
What is Text Classification?

文本分类是将文本自动分配到预定义类别的任务
Automatically assigning text to predefined categories

常见应用场景：
Common Applications:

📧 垃圾邮件检测 (Spam Detection)
   • 判断邮件是否为垃圾邮件
   • 二分类问题

😊 情感分析 (Sentiment Analysis)
   • 分析评论的情感倾向（正面/负面/中性）
   • 多分类问题

📰 新闻分类 (News Classification)
   • 将新闻分类到不同主题
   • 多分类问题

🏷️ 主题标注 (Topic Labeling)
   • 为文档打上主题标签
   • 多标签分类

💬 意图识别 (Intent Classification)
   • 理解用户查询意图
   • 聊天机器人核心功能

文本分类流程：
Classification Pipeline:

1. 数据收集与标注 → 2. 文本预处理 → 3. 特征提取 →
4. 模型训练 → 5. 模型评估 → 6. 模型部署
""")

# ============================================================================
# 2. 数据准备 / Data Preparation
# ============================================================================
print("\n【2】数据准备 - 20 Newsgroups 数据集")
print("-" * 80)
print("正在加载数据集...")

# 为了演示，我们只使用 4 个类别
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

# 加载训练数据
try:
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=RANDOM_STATE,
        remove=('headers', 'footers', 'quotes')
    )

    # 加载测试数据
    newsgroups_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=RANDOM_STATE,
        remove=('headers', 'footers', 'quotes')
    )

    X_train_text = newsgroups_train.data
    y_train = newsgroups_train.target
    X_test_text = newsgroups_test.data
    y_test = newsgroups_test.target

    target_names = newsgroups_train.target_names

    print(f"✓ 数据集加载成功")
    print(f"\n数据集信息:")
    print(f"  训练集大小: {len(X_train_text)}")
    print(f"  测试集大小: {len(X_test_text)}")
    print(f"  类别数: {len(target_names)}")
    print(f"  类别名称: {target_names}")

    # 类别分布
    print(f"\n训练集类别分布:")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, count in zip(unique, counts):
        print(f"  {target_names[i]}: {count} ({count/len(y_train)*100:.1f}%)")

    # 查看示例文档
    print(f"\n示例文档（类别: {target_names[y_train[0]]}):")
    print(f"{X_train_text[0][:300]}...")

    DATA_LOADED = True

except Exception as e:
    print(f"✗ 数据集加载失败: {e}")
    print("使用自定义示例数据...")

    # 创建简单的示例数据
    X_train_text = [
        "Python is a great programming language for machine learning",
        "I love using Python for data science projects",
        "Machine learning algorithms are fascinating",
        "Deep learning neural networks are powerful",
        "This movie was absolutely amazing and wonderful",
        "I really enjoyed watching this film",
        "The weather is sunny and beautiful today",
        "What a lovely day with clear blue skies",
        "This product is terrible and poorly made",
        "Very disappointed with this purchase",
        "Python programming is my favorite hobby",
        "Data science is an exciting field",
        "The acting in this movie was superb",
        "Beautiful cinematography and great story",
        "Awful experience, would not recommend",
        "Complete waste of money and time",
    ]

    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])

    # 创建测试集
    X_test_text = [
        "Learning Python is fun and rewarding",
        "Machine learning is the future",
        "This film was excellent and entertaining",
        "Beautiful weather for outdoor activities",
        "Disappointing quality and poor service",
        "Data analysis with Python is efficient",
    ]

    y_test = np.array([0, 0, 1, 1, 2, 0])

    target_names = ['Technology', 'Positive', 'Negative']

    print(f"✓ 使用自定义数据")
    print(f"  训练集大小: {len(X_train_text)}")
    print(f"  测试集大小: {len(X_test_text)}")
    print(f"  类别: {target_names}")

    DATA_LOADED = False

# ============================================================================
# 3. 数据探索和可视化 / Data Exploration
# ============================================================================
print("\n【3】数据探索")
print("-" * 80)

# 文本长度统计
text_lengths = [len(text.split()) for text in X_train_text]

print(f"文本长度统计:")
print(f"  平均长度: {np.mean(text_lengths):.1f} 词")
print(f"  最短: {np.min(text_lengths)} 词")
print(f"  最长: {np.max(text_lengths)} 词")
print(f"  中位数: {np.median(text_lengths):.1f} 词")

# ============================================================================
# 4. 特征提取 / Feature Extraction
# ============================================================================
print("\n【4】特征提取")
print("-" * 80)

# TF-IDF 特征提取
print("使用 TF-IDF 进行特征提取...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

print(f"✓ TF-IDF 特征提取完成")
print(f"  训练集形状: {X_train_tfidf.shape}")
print(f"  测试集形状: {X_test_tfidf.shape}")
print(f"  词汇表大小: {len(tfidf_vectorizer.vocabulary_)}")
print(f"  稀疏度: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100:.2f}%")

# BoW 特征提取（用于 Naive Bayes）
bow_vectorizer = CountVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    stop_words='english'
)

X_train_bow = bow_vectorizer.fit_transform(X_train_text)
X_test_bow = bow_vectorizer.transform(X_test_text)

print(f"\n✓ BoW 特征提取完成")
print(f"  训练集形状: {X_train_bow.shape}")

# ============================================================================
# 5. 传统机器学习模型 / Traditional ML Models
# ============================================================================
print("\n【5】训练传统机器学习模型")
print("=" * 80)

# 存储模型和结果
models = {}
results = {}
training_times = {}
prediction_times = {}

# 5.1 Logistic Regression + TF-IDF
print("\n【5.1】Logistic Regression + TF-IDF")
print("-" * 80)

start_time = time.time()
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
lr_model.fit(X_train_tfidf, y_train)
training_times['Logistic Regression'] = time.time() - start_time

start_time = time.time()
y_pred_lr = lr_model.predict(X_test_tfidf)
prediction_times['Logistic Regression'] = time.time() - start_time

accuracy_lr = accuracy_score(y_test, y_pred_lr)
models['Logistic Regression'] = lr_model
results['Logistic Regression'] = {
    'predictions': y_pred_lr,
    'accuracy': accuracy_lr
}

print(f"✓ 训练完成")
print(f"  训练时间: {training_times['Logistic Regression']:.3f} 秒")
print(f"  预测时间: {prediction_times['Logistic Regression']:.3f} 秒")
print(f"  准确率: {accuracy_lr:.4f}")

print(f"\n分类报告:")
print(classification_report(y_test, y_pred_lr, target_names=target_names))

# 5.2 Naive Bayes + BoW
print("\n【5.2】Naive Bayes + BoW")
print("-" * 80)

start_time = time.time()
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_bow, y_train)
training_times['Naive Bayes'] = time.time() - start_time

start_time = time.time()
y_pred_nb = nb_model.predict(X_test_bow)
prediction_times['Naive Bayes'] = time.time() - start_time

accuracy_nb = accuracy_score(y_test, y_pred_nb)
models['Naive Bayes'] = nb_model
results['Naive Bayes'] = {
    'predictions': y_pred_nb,
    'accuracy': accuracy_nb
}

print(f"✓ 训练完成")
print(f"  训练时间: {training_times['Naive Bayes']:.3f} 秒")
print(f"  预测时间: {prediction_times['Naive Bayes']:.3f} 秒")
print(f"  准确率: {accuracy_nb:.4f}")

print(f"\n分类报告:")
print(classification_report(y_test, y_pred_nb, target_names=target_names))

# 5.3 SVM (Linear) + TF-IDF
print("\n【5.3】SVM (Linear) + TF-IDF")
print("-" * 80)

start_time = time.time()
svm_model = LinearSVC(
    random_state=RANDOM_STATE,
    max_iter=2000
)
svm_model.fit(X_train_tfidf, y_train)
training_times['SVM'] = time.time() - start_time

start_time = time.time()
y_pred_svm = svm_model.predict(X_test_tfidf)
prediction_times['SVM'] = time.time() - start_time

accuracy_svm = accuracy_score(y_test, y_pred_svm)
models['SVM'] = svm_model
results['SVM'] = {
    'predictions': y_pred_svm,
    'accuracy': accuracy_svm
}

print(f"✓ 训练完成")
print(f"  训练时间: {training_times['SVM']:.3f} 秒")
print(f"  预测时间: {prediction_times['SVM']:.3f} 秒")
print(f"  准确率: {accuracy_svm:.4f}")

print(f"\n分类报告:")
print(classification_report(y_test, y_pred_svm, target_names=target_names))

# 5.4 Random Forest + TF-IDF
print("\n【5.4】Random Forest + TF-IDF")
print("-" * 80)

start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_tfidf, y_train)
training_times['Random Forest'] = time.time() - start_time

start_time = time.time()
y_pred_rf = rf_model.predict(X_test_tfidf)
prediction_times['Random Forest'] = time.time() - start_time

accuracy_rf = accuracy_score(y_test, y_pred_rf)
models['Random Forest'] = rf_model
results['Random Forest'] = {
    'predictions': y_pred_rf,
    'accuracy': accuracy_rf
}

print(f"✓ 训练完成")
print(f"  训练时间: {training_times['Random Forest']:.3f} 秒")
print(f"  预测时间: {prediction_times['Random Forest']:.3f} 秒")
print(f"  准确率: {accuracy_rf:.4f}")

print(f"\n分类报告:")
print(classification_report(y_test, y_pred_rf, target_names=target_names))

# 特征重要性（Random Forest）
if hasattr(rf_model, 'feature_importances_'):
    feature_importances = rf_model.feature_importances_
    top_indices = feature_importances.argsort()[-10:][::-1]
    top_features = [tfidf_vectorizer.get_feature_names_out()[i] for i in top_indices]
    top_importance = feature_importances[top_indices]

    print(f"\nTop 10 重要特征:")
    for feature, importance in zip(top_features, top_importance):
        print(f"  {feature}: {importance:.4f}")

# 5.5 简单神经网络（MLP）
if MLP_AVAILABLE:
    print("\n【5.5】多层感知器 (MLP) + TF-IDF")
    print("-" * 80)

    start_time = time.time()
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=300,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp_model.fit(X_train_tfidf, y_train)
    training_times['MLP'] = time.time() - start_time

    start_time = time.time()
    y_pred_mlp = mlp_model.predict(X_test_tfidf)
    prediction_times['MLP'] = time.time() - start_time

    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    models['MLP'] = mlp_model
    results['MLP'] = {
        'predictions': y_pred_mlp,
        'accuracy': accuracy_mlp
    }

    print(f"✓ 训练完成")
    print(f"  训练时间: {training_times['MLP']:.3f} 秒")
    print(f"  预测时间: {prediction_times['MLP']:.3f} 秒")
    print(f"  准确率: {accuracy_mlp:.4f}")
    print(f"  迭代次数: {mlp_model.n_iter_}")

    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred_mlp, target_names=target_names))

# ============================================================================
# 6. 模型对比 / Model Comparison
# ============================================================================
print("\n【6】模型性能对比")
print("=" * 80)

# 性能对比表格
print("\n所有模型性能总结:")
print("-" * 80)
print(f"{'模型':<20} {'准确率':<12} {'训练时间(s)':<15} {'预测时间(s)':<15}")
print("-" * 80)

for model_name in results.keys():
    accuracy = results[model_name]['accuracy']
    train_time = training_times[model_name]
    pred_time = prediction_times[model_name]
    print(f"{model_name:<20} {accuracy:<12.4f} {train_time:<15.3f} {pred_time:<15.3f}")

# 找出最佳模型
best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
print(f"\n🏆 最佳模型: {best_model_name}")
print(f"   准确率: {results[best_model_name]['accuracy']:.4f}")

# ============================================================================
# 7. 实际应用示例 / Practical Examples
# ============================================================================
print("\n【7】实际应用示例")
print("=" * 80)

# 使用最佳模型进行预测
best_model = models[best_model_name]

# 新文本示例
new_texts = [
    "Python machine learning libraries are very powerful",
    "The documentary was inspiring and well-made",
    "Terrible customer service and low quality product"
]

print("\n使用最佳模型预测新文本:")
print("-" * 80)

if best_model_name == 'Naive Bayes':
    new_features = bow_vectorizer.transform(new_texts)
else:
    new_features = tfidf_vectorizer.transform(new_texts)

new_predictions = best_model.predict(new_features)

# 获取预测概率
if hasattr(best_model, 'predict_proba'):
    new_probas = best_model.predict_proba(new_features)
elif hasattr(best_model, 'decision_function'):
    # SVM 使用 decision_function
    decision_scores = best_model.decision_function(new_features)
    # 简单归一化
    new_probas = np.exp(decision_scores) / np.exp(decision_scores).sum(axis=1, keepdims=True)
else:
    new_probas = None

for i, text in enumerate(new_texts):
    pred_class = target_names[new_predictions[i]]
    print(f"\n文本 {i+1}: {text}")
    print(f"  预测类别: {pred_class}")

    if new_probas is not None:
        print(f"  类别概率:")
        for j, class_name in enumerate(target_names):
            print(f"    {class_name}: {new_probas[i][j]:.4f}")

# ============================================================================
# 8. 错误分析 / Error Analysis
# ============================================================================
print("\n【8】错误分析")
print("=" * 80)

# 找出分类错误的样本
y_pred_best = results[best_model_name]['predictions']
errors = np.where(y_pred_best != y_test)[0]

print(f"错误样本数: {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")

if len(errors) > 0:
    print(f"\n前 3 个错误样本:")
    for i, error_idx in enumerate(errors[:3], 1):
        true_label = target_names[y_test[error_idx]]
        pred_label = target_names[y_pred_best[error_idx]]
        text = X_test_text[error_idx][:150]

        print(f"\n错误 {i}:")
        print(f"  文本: {text}...")
        print(f"  真实类别: {true_label}")
        print(f"  预测类别: {pred_label}")

# ============================================================================
# 9. 可视化 / Visualization
# ============================================================================
print("\n【9】模型性能可视化")
print("=" * 80)

# 9.1 数据集类别分布
fig, axes = create_subplots(2, 2, figsize=(16, 12))

# 类别分布
unique_train, counts_train = np.unique(y_train, return_counts=True)
axes[0, 0].bar([target_names[i] for i in unique_train], counts_train,
              color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('类别 / Category', fontsize=10)
axes[0, 0].set_ylabel('样本数 / Count', fontsize=10)
axes[0, 0].set_title('训练集类别分布\nTraining Set Class Distribution',
                     fontsize=12, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)

for i, v in enumerate(counts_train):
    axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

# 文本长度分布
axes[0, 1].hist(text_lengths, bins=30, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('文本长度（词数）/ Text Length (words)', fontsize=10)
axes[0, 1].set_ylabel('频数 / Frequency', fontsize=10)
axes[0, 1].set_title('文本长度分布\nText Length Distribution',
                     fontsize=12, fontweight='bold')
axes[0, 1].axvline(np.mean(text_lengths), color='red', linestyle='--',
                   label=f'平均值: {np.mean(text_lengths):.1f}')
axes[0, 1].legend()

# 模型准确率对比
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]

colors_bar = ['steelblue', 'coral', 'lightgreen', 'mediumpurple', 'gold'][:len(model_names)]
bars = axes[1, 0].bar(model_names, accuracies, color=colors_bar,
                      alpha=0.7, edgecolor='black')

# 高亮最佳模型
best_idx = accuracies.index(max(accuracies))
bars[best_idx].set_edgecolor('red')
bars[best_idx].set_linewidth(3)

axes[1, 0].set_xlabel('模型 / Model', fontsize=10)
axes[1, 0].set_ylabel('准确率 / Accuracy', fontsize=10)
axes[1, 0].set_title('模型准确率对比\nModel Accuracy Comparison',
                     fontsize=12, fontweight='bold')
axes[1, 0].set_ylim([0, 1.1])
axes[1, 0].tick_params(axis='x', rotation=45)

for i, v in enumerate(accuracies):
    axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

# 训练/预测时间对比
x_pos = np.arange(len(model_names))
width = 0.35

train_times = [training_times[name] for name in model_names]
pred_times = [prediction_times[name] for name in model_names]

axes[1, 1].bar(x_pos - width/2, train_times, width, label='训练时间',
              color='skyblue', alpha=0.8)
axes[1, 1].bar(x_pos + width/2, pred_times, width, label='预测时间',
              color='lightcoral', alpha=0.8)

axes[1, 1].set_xlabel('模型 / Model', fontsize=10)
axes[1, 1].set_ylabel('时间 (秒) / Time (seconds)', fontsize=10)
axes[1, 1].set_title('训练/预测时间对比\nTraining/Prediction Time Comparison',
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('nlp_classification_overview.png'))
print("✓ 图表已保存: output/nlp_classification_overview.png")
plt.show()

# 9.2 混淆矩阵对比
n_models = len(model_names)
n_rows = (n_models + 1) // 2
n_cols = 2

fig, axes = create_subplots(n_rows, n_cols, figsize=(14, 6*n_rows))

if n_models == 1:
    axes = np.array([axes])

axes_flat = axes.flatten() if n_models > 1 else axes

for idx, model_name in enumerate(model_names):
    y_pred = results[model_name]['predictions']
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': '样本数 / Count'},
                ax=axes_flat[idx])

    axes_flat[idx].set_title(f'{model_name}\n准确率: {results[model_name]["accuracy"]:.3f}',
                            fontsize=12, fontweight='bold')
    axes_flat[idx].set_xlabel('预测类别 / Predicted', fontsize=10)
    axes_flat[idx].set_ylabel('真实类别 / Actual', fontsize=10)

# 隐藏多余的子图
for idx in range(n_models, len(axes_flat)):
    axes_flat[idx].axis('off')

plt.tight_layout()
save_figure(fig, get_output_path('nlp_confusion_matrices.png'))
print("✓ 图表已保存: output/nlp_confusion_matrices.png")
plt.show()

# 9.3 特征重要性（Random Forest）
if 'Random Forest' in models and hasattr(models['Random Forest'], 'feature_importances_'):
    fig, ax = create_subplots(1, 1, figsize=(12, 8))

    feature_importances = models['Random Forest'].feature_importances_
    top_indices = feature_importances.argsort()[-20:][::-1]
    top_features = [tfidf_vectorizer.get_feature_names_out()[i] for i in top_indices]
    top_importance = feature_importances[top_indices]

    ax.barh(range(len(top_features)), top_importance, color='mediumpurple', alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('重要性 / Importance', fontsize=11)
    ax.set_title('Random Forest - Top 20 特征重要性\nRandom Forest - Top 20 Feature Importances',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_figure(fig, get_output_path('nlp_feature_importance.png'))
    print("✓ 图表已保存: output/nlp_feature_importance.png")
    plt.show()

# 9.4 模型性能雷达图
fig, ax = create_subplots(1, 1, figsize=(10, 10))

# 计算各模型的精确率、召回率、F1
metrics = {}
for model_name in model_names:
    y_pred = results[model_name]['predictions']
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = results[model_name]['accuracy']

    # 归一化训练时间（越快越好，所以用倒数）
    max_train_time = max(training_times.values())
    normalized_speed = 1 - (training_times[model_name] / max_train_time)

    metrics[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Speed': normalized_speed
    }

# 绘制雷达图（仅绘制前3个模型，避免过于拥挤）
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, projection='polar')

colors_radar = ['steelblue', 'coral', 'lightgreen', 'mediumpurple', 'gold']

for idx, model_name in enumerate(model_names[:3]):  # 只显示前3个
    values = [metrics[model_name][cat] for cat in categories]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=model_name,
            color=colors_radar[idx])
    ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('模型性能雷达图（前3个模型）\nModel Performance Radar Chart',
             fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
ax.grid(True)

plt.tight_layout()
save_figure(fig, get_output_path('nlp_performance_radar.png'))
print("✓ 图表已保存: output/nlp_performance_radar.png")
plt.show()

# 9.5 学习曲线（最佳模型）
if DATA_LOADED and len(X_train_text) > 100:
    print("\n生成学习曲线（可能需要一些时间）...")

    fig, ax = create_subplots(1, 1, figsize=(12, 8))

    # 使用 Pipeline
    if best_model_name == 'Naive Bayes':
        pipeline = Pipeline([
            ('vectorizer', bow_vectorizer),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        X_for_curve = X_train_text
    else:
        pipeline = Pipeline([
            ('vectorizer', tfidf_vectorizer),
            ('classifier', models[best_model_name])
        ])
        X_for_curve = X_train_text

    train_sizes = np.linspace(0.1, 1.0, 10)

    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            pipeline, X_for_curve, y_train,
            train_sizes=train_sizes,
            cv=3,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        ax.plot(train_sizes_abs, train_mean, 'o-', color='steelblue',
                label='训练集得分', linewidth=2)
        ax.fill_between(train_sizes_abs, train_mean - train_std,
                        train_mean + train_std, alpha=0.2, color='steelblue')

        ax.plot(train_sizes_abs, val_mean, 'o-', color='coral',
                label='验证集得分', linewidth=2)
        ax.fill_between(train_sizes_abs, val_mean - val_std,
                        val_mean + val_std, alpha=0.2, color='coral')

        ax.set_xlabel('训练样本数 / Training Set Size', fontsize=11)
        ax.set_ylabel('得分 / Score', fontsize=11)
        ax.set_title(f'学习曲线 - {best_model_name}\nLearning Curve',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, get_output_path('nlp_learning_curve.png'))
        print("✓ 图表已保存: output/nlp_learning_curve.png")
        plt.show()
    except Exception as e:
        print(f"✗ 学习曲线生成失败: {e}")

# 9.6 综合性能对比图
fig, ax = create_subplots(1, 1, figsize=(14, 8))

# 准备数据
performance_data = []
for model_name in model_names:
    y_pred = results[model_name]['predictions']
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    performance_data.append({
        'Model': model_name,
        'Accuracy': results[model_name]['accuracy'],
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

df_performance = pd.DataFrame(performance_data)

x = np.arange(len(model_names))
width = 0.2

ax.bar(x - 1.5*width, df_performance['Accuracy'], width, label='Accuracy',
       color='steelblue', alpha=0.8)
ax.bar(x - 0.5*width, df_performance['Precision'], width, label='Precision',
       color='coral', alpha=0.8)
ax.bar(x + 0.5*width, df_performance['Recall'], width, label='Recall',
       color='lightgreen', alpha=0.8)
ax.bar(x + 1.5*width, df_performance['F1-Score'], width, label='F1-Score',
       color='mediumpurple', alpha=0.8)

ax.set_xlabel('模型 / Model', fontsize=11)
ax.set_ylabel('得分 / Score', fontsize=11)
ax.set_title('模型综合性能对比\nComprehensive Model Performance Comparison',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('nlp_comprehensive_performance.png'))
print("✓ 图表已保存: output/nlp_comprehensive_performance.png")
plt.show()

# ============================================================================
# 10. 总结与最佳实践 / Summary and Best Practices
# ============================================================================
print("\n【10】总结与最佳实践")
print("=" * 80)
print(f"""
本教程涵盖的内容：
Topics Covered:

✓ 文本分类任务介绍
  • 应用场景：垃圾邮件检测、情感分析、新闻分类等

✓ 数据准备与探索
  • 20 Newsgroups 数据集
  • 数据分布分析

✓ 特征提取
  • TF-IDF 和 Bag of Words
  • 词汇表构建和参数调优

✓ 传统机器学习模型
  • Logistic Regression
  • Naive Bayes
  • SVM
  • Random Forest
  {'• MLP (Multi-Layer Perceptron)' if MLP_AVAILABLE else ''}

✓ 模型评估与对比
  • 准确率、精确率、召回率、F1-score
  • 混淆矩阵
  • 学习曲线

✓ 实际应用与错误分析
  • 新文本预测
  • 错误样本分析

最佳实践建议：
Best Practices:

📝 数据预处理
  1. 去除 HTML 标签和特殊字符
  2. 统一文本格式（小写）
  3. 去除停用词（根据任务决定）
  4. 词形归一化（Lemmatization）

🔧 特征工程
  1. 使用 TF-IDF 而非简单的词频
  2. 考虑 n-grams（bigrams, trigrams）
  3. 设置合理的 min_df 和 max_df
  4. 限制特征数量（max_features）

🎯 模型选择
  1. 快速原型：Naive Bayes
  2. 平衡性能：Logistic Regression 或 SVM
  3. 需要可解释性：Logistic Regression
  4. 追求性能：集成方法或深度学习

📊 模型评估
  1. 使用多个评估指标
  2. 分析混淆矩阵
  3. 进行交叉验证
  4. 检查学习曲线

⚡ 性能优化
  1. Pipeline 化工作流程
  2. 使用 n_jobs=-1 并行计算
  3. 增量学习（大数据集）
  4. 特征选择减少维度

🔍 实际应用
  1. 保存训练好的模型和向量化器
  2. 设置合理的预测阈值
  3. 持续监控模型性能
  4. 定期更新模型

模型性能总结：
Model Performance Summary:
""")

for model_name in model_names:
    print(f"\n{model_name}:")
    print(f"  ✓ 准确率: {results[model_name]['accuracy']:.4f}")
    print(f"  ✓ 训练时间: {training_times[model_name]:.3f} 秒")
    print(f"  ✓ 预测时间: {prediction_times[model_name]:.3f} 秒")

print(f"\n🏆 推荐模型: {best_model_name}")
print(f"   理由: 在准确率和效率之间取得最佳平衡")

print("""
下一步学习：
Next Steps:

1. 尝试深度学习方法（LSTM, BERT）
2. 多标签分类
3. 不平衡数据处理
4. 模型部署和服务化
5. 在线学习和增量更新
""")

print("\n" + "=" * 80)
print("文本分类教程完成！".center(80))
print("Text Classification Tutorial Complete!".center(80))
print("=" * 80)
