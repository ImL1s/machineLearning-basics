"""
樸素貝葉斯（Naive Bayes）
基於貝葉斯定理的簡單而高效的分類算法

"樸素"：假設特徵之間相互獨立
原理：利用貝葉斯定理計算後驗概率
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("樸素貝葉斯（Naive Bayes）教程".center(80))
print("=" * 80)

# ============================================================================
# 1. Naive Bayes 基本概念
# ============================================================================
print("\n【1】Naive Bayes 基本概念")
print("-" * 80)
print("""
樸素貝葉斯核心思想：
• 基於貝葉斯定理：P(y|x) = P(x|y) * P(y) / P(x)
• 假設特徵獨立：P(x1,x2,...,xn|y) = P(x1|y) * P(x2|y) * ... * P(xn|y)

三種主要類型：

1. GaussianNB（高斯樸素貝葉斯）：
   • 假設特徵服從正態分布
   • 適用於連續特徵
   • 常用於一般分類問題

2. MultinomialNB（多項式樸素貝葉斯）：
   • 假設特徵服從多項式分布
   • 適用於離散特徵（計數數據）
   • 常用於文本分類（詞頻）

3. BernoulliNB（伯努利樸素貝葉斯）：
   • 假設特徵為二值（0/1）
   • 適用於二值特徵
   • 常用於文本分類（詞出現與否）

優點：
✓ 訓練速度極快
✓ 預測速度快
✓ 對小樣本數據效果好
✓ 適合高維數據
✓ 對缺失數據不敏感
✓ 可以處理多分類問題

缺點：
✗ 假設特徵獨立（現實中很少成立）
✗ 對特徵關係建模能力弱
✗ 對數值特徵需要假設分布
""")

# ============================================================================
# 2. 實例1：高斯樸素貝葉斯 - 鳶尾花分類
# ============================================================================
print("\n【2】實例1：高斯樸素貝葉斯 - 鳶尾花分類")
print("-" * 80)

# 加載數據
iris = load_iris()
X, y = iris.data, iris.target

# 數據分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 訓練高斯樸素貝葉斯
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 預測
y_pred_gnb = gnb.predict(X_test)
y_pred_proba_gnb = gnb.predict_proba(X_test)

# 評估
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print(f"高斯樸素貝葉斯準確率：{accuracy_gnb:.4f}")

print("\n分類報告：")
print(classification_report(y_test, y_pred_gnb, target_names=iris.target_names))

# ============================================================================
# 3. 實例2：文本分類 - 新聞分類
# ============================================================================
print("\n【3】實例2：多項式樸素貝葉斯 - 文本分類")
print("-" * 80)

# 創建簡單的文本數據集
documents = [
    "machine learning is great",
    "deep learning is powerful",
    "python is easy to learn",
    "data science is interesting",
    "artificial intelligence is the future",
    "neural networks are complex",
    "I love programming in python",
    "statistics is important for data science",
    "natural language processing is challenging",
    "computer vision uses deep learning"
]

labels = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  # 0: ML/AI, 1: Programming

# 文本向量化
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(documents)

print(f"詞彙表大小：{len(vectorizer.get_feature_names_out())}")
print(f"特徵矩陣形狀：{X_text.shape}")

# 訓練多項式樸素貝葉斯
mnb = MultinomialNB()
mnb.fit(X_text, labels)

# 預測新文本
test_docs = [
    "python programming is fun",
    "deep neural networks",
    "coding in python"
]
test_X = vectorizer.transform(test_docs)
predictions = mnb.predict(test_X)
probabilities = mnb.predict_proba(test_X)

print("\n測試預測：")
for doc, pred, prob in zip(test_docs, predictions, probabilities):
    category = "ML/AI" if pred == 0 else "Programming"
    print(f"文本: '{doc}'")
    print(f"  預測: {category}")
    print(f"  概率: ML/AI={prob[0]:.3f}, Programming={prob[1]:.3f}\n")

# ============================================================================
# 4. 實例3：伯努利樸素貝葉斯
# ============================================================================
print("\n【4】實例3：伯努利樸素貝葉斯 - 二值特徵")
print("-" * 80)

# 使用二值化的文本特徵
X_binary = (X_text > 0).astype(int)

bnb = BernoulliNB()
bnb.fit(X_binary, labels)

accuracy_bnb = bnb.score(X_binary, labels)
print(f"伯努利樸素貝葉斯準確率：{accuracy_bnb:.4f}")

# ============================================================================
# 5. 三種樸素貝葉斯比較
# ============================================================================
print("\n【5】三種樸素貝葉斯在鳶尾花數據上的比較")
print("-" * 80)

# GaussianNB
gnb_score = cross_val_score(GaussianNB(), X, y, cv=5).mean()

# MultinomialNB（需要非負特徵，先縮放）
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
mnb_score = cross_val_score(MultinomialNB(), X_scaled, y, cv=5).mean()

# BernoulliNB
X_binary_iris = (X > X.mean()).astype(int)
bnb_score = cross_val_score(BernoulliNB(), X_binary_iris, y, cv=5).mean()

print(f"GaussianNB 平均準確率：{gnb_score:.4f}")
print(f"MultinomialNB 平均準確率：{mnb_score:.4f}")
print(f"BernoulliNB 平均準確率：{bnb_score:.4f}")

# ============================================================================
# 可視化
# ============================================================================

fig = plt.figure(figsize=(16, 10))

# 1. 混淆矩陣（高斯樸素貝葉斯）
ax1 = plt.subplot(2, 3, 1)
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names, ax=ax1)
ax1.set_title('GaussianNB Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

# 2. 預測概率（高斯樸素貝葉斯）
ax2 = plt.subplot(2, 3, 2)
for i, target_name in enumerate(iris.target_names):
    mask = y_test == i
    ax2.scatter(range(np.sum(mask)), y_pred_proba_gnb[mask, i],
               label=target_name, alpha=0.6)

ax2.set_xlabel('Sample Index (grouped by class)')
ax2.set_ylabel('Prediction Probability')
ax2.set_title('GaussianNB Prediction Probabilities', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 三種樸素貝葉斯比較
ax3 = plt.subplot(2, 3, 3)
models = ['GaussianNB', 'MultinomialNB', 'BernoulliNB']
scores = [gnb_score, mnb_score, bnb_score]
colors = ['blue', 'green', 'orange']

bars = ax3.bar(models, scores, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Cross-Validation Accuracy')
ax3.set_title('Naive Bayes Variants Comparison', fontsize=12, fontweight='bold')
ax3.set_ylim([0.8, 1.0])
ax3.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.3f}', ha='center', va='bottom')

# 4. 特徵概率（高斯樸素貝葉斯）
ax4 = plt.subplot(2, 3, 4)
for i, target_name in enumerate(iris.target_names):
    ax4.plot(gnb.theta_[i], marker='o', label=target_name, linewidth=2)

ax4.set_xlabel('Feature Index')
ax4.set_ylabel('Mean Value')
ax4.set_title('GaussianNB: Feature Means per Class', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(len(iris.feature_names)))
ax4.set_xticklabels(['F1', 'F2', 'F3', 'F4'])

# 5. 文本特徵重要性（多項式樸素貝葉斯）
ax5 = plt.subplot(2, 3, 5)
feature_log_prob = mnb.feature_log_prob_[0] - mnb.feature_log_prob_[1]
feature_names = vectorizer.get_feature_names_out()

# 選擇最重要的特徵
top_indices = np.argsort(np.abs(feature_log_prob))[-10:]
top_features = [feature_names[i] for i in top_indices]
top_scores = feature_log_prob[top_indices]

colors_bar = ['red' if x < 0 else 'green' for x in top_scores]
ax5.barh(range(len(top_features)), top_scores, color=colors_bar, alpha=0.7)
ax5.set_yticks(range(len(top_features)))
ax5.set_yticklabels(top_features)
ax5.set_xlabel('Log Probability Difference')
ax5.set_title('Text Feature Importance (MultinomialNB)', fontsize=12, fontweight='bold')
ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax5.grid(True, alpha=0.3, axis='x')

# 6. 類別先驗概率
ax6 = plt.subplot(2, 3, 6)
class_prior = np.exp(gnb.class_log_prior_)
ax6.bar(iris.target_names, class_prior, alpha=0.7, edgecolor='black', color='purple')
ax6.set_ylabel('Prior Probability')
ax6.set_title('Class Prior Probabilities', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

for i, (name, prob) in enumerate(zip(iris.target_names, class_prior)):
    ax6.text(i, prob, f'{prob:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('02_SupervisedLearning/Classification/05_naive_bayes_results.png',
            dpi=150, bbox_inches='tight')
print("\n已保存結果圖表")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("Naive Bayes 要點總結")
print("=" * 80)
print("""
1. 三種變體選擇：
   • GaussianNB：連續特徵，正態分布
   • MultinomialNB：離散特徵，計數數據（如詞頻）
   • BernoulliNB：二值特徵（如詞是否出現）

2. 適用場景：
   • 文本分類（垃圾郵件、情感分析）
   • 文檔分類
   • 實時預測（速度快）
   • 多分類問題
   • 高維稀疏數據

3. 優化建議：
   • 文本分類：使用 TF-IDF 替代詞頻
   • 平滑參數：MultinomialNB 的 alpha（默認1.0）
   • 特徵選擇：去除低頻詞
   • 特徵工程：n-grams、詞性標註

4. 與其他算法比較：
   • vs Logistic Regression：NB 更快，LR 更準確
   • vs SVM：NB 更適合高維數據
   • vs Random Forest：NB 更快，RF 更強大
   • vs 深度學習：NB 更簡單，DL 更適合大數據

5. 注意事項：
   • 特徵獨立假設很強（但實際效果往往不錯）
   • 需要足夠的訓練數據估計概率
   • 對特徵分布假設要合理
   • MultinomialNB 需要非負特徵
   • 零頻率問題（使用平滑）

6. 實際應用：
   • 垃圾郵件過濾
   • 新聞分類
   • 情感分析
   • 醫療診斷
   • 推薦系統（協同過濾）
""")
