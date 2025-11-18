"""
文本特征提取（Text Feature Extraction）
将文本转换为数值特征向量

Text Feature Extraction - Converting Text to Numerical Features

特征提取是 NLP 的核心步骤，决定了模型能否理解文本语义
Feature extraction is crucial for enabling machines to understand text
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from utils import RANDOM_STATE, setup_chinese_fonts, create_subplots

# 尝试导入 Gensim（Word2Vec）
try:
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠ Gensim 未安装。运行: pip install gensim")

# 尝试导入 NLTK
try:
    import nltk
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠ NLTK 未安装。运行: pip install nltk")

setup_chinese_fonts()

print("=" * 80)
print("文本特征提取（Text Feature Extraction）教程".center(80))
print("=" * 80)

# ============================================================================
# 1. 特征提取方法概述 / Feature Extraction Overview
# ============================================================================
print("\n【1】特征提取方法概述")
print("-" * 80)
print("""
为什么需要特征提取？
Why Feature Extraction?
• 机器学习算法只能处理数值数据
  Machine learning algorithms work with numbers, not text
• 需要将文本转换为向量表示
  Need to convert text into vector representations

主要方法：
Main Methods:

1. Bag of Words (BoW) - 词袋模型
   • 统计词频
   • 忽略词序和语法
   • 简单高效

2. TF-IDF (Term Frequency-Inverse Document Frequency)
   • 考虑词的重要性
   • 降低高频常用词的权重
   • 提升区分性词汇的权重

3. Word Embeddings - 词嵌入
   • Word2Vec, GloVe, FastText
   • 捕捉词义和上下文
   • 低维稠密向量

方法对比：
Comparison:

特性          | BoW      | TF-IDF   | Word2Vec
-------------|----------|----------|----------
维度         | 高（稀疏） | 高（稀疏） | 低（稠密）
语义信息     | 无        | 少        | 丰富
训练时间     | 快        | 快        | 慢
内存占用     | 大        | 大        | 小
适用场景     | 分类      | 检索/分类 | 语义任务
""")

# ============================================================================
# 2. 准备示例数据 / Prepare Sample Data
# ============================================================================
print("\n【2】准备示例数据")
print("-" * 80)

# 创建文档语料库
documents = [
    "Machine learning is a fascinating field of artificial intelligence.",
    "Deep learning uses neural networks to learn from data.",
    "Natural language processing helps computers understand human language.",
    "Text mining extracts useful information from text data.",
    "Data science combines statistics and machine learning.",
    "Neural networks are the foundation of deep learning.",
    "Artificial intelligence is transforming many industries.",
    "Text classification is a common NLP task.",
    "Machine learning algorithms learn patterns from data.",
    "Deep neural networks can solve complex problems.",
]

print("文档语料库:")
for i, doc in enumerate(documents, 1):
    print(f"{i}. {doc}")

print(f"\n总文档数: {len(documents)}")

# ============================================================================
# 3. 词袋模型（Bag of Words）
# ============================================================================
print("\n【3】词袋模型（Bag of Words）")
print("-" * 80)
print("""
词袋模型原理：
Bag of Words Principle:

1. 建立词汇表（所有文档的唯一词汇）
   Build vocabulary (all unique words in all documents)

2. 统计每个文档中每个词的出现次数
   Count word occurrences in each document

3. 生成文档-词频矩阵
   Generate document-term matrix

特点：
• 简单直观
• 忽略词序
• 高维稀疏
""")

# 3.1 基础 CountVectorizer
print("\n【3.1】基础 CountVectorizer")

vectorizer_basic = CountVectorizer()
bow_matrix = vectorizer_basic.fit_transform(documents)

print(f"词汇表大小: {len(vectorizer_basic.vocabulary_)}")
print(f"矩阵形状: {bow_matrix.shape}")
print(f"稀疏度: {(1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])) * 100:.2f}%")

# 查看部分词汇表
vocab = vectorizer_basic.get_feature_names_out()
print(f"\n部分词汇表（前20个）:")
print(vocab[:20])

# 查看第一个文档的向量表示
print(f"\n第一个文档的向量表示:")
print(f"文档: {documents[0]}")
print(f"向量（非零元素）:")
doc_vector = bow_matrix[0].toarray()[0]
for idx, count in enumerate(doc_vector):
    if count > 0:
        print(f"  {vocab[idx]}: {count}")

# 3.2 参数调优
print("\n【3.2】CountVectorizer 参数调优")

# 限制词汇表大小
vectorizer_limited = CountVectorizer(max_features=20)
bow_limited = vectorizer_limited.fit_transform(documents)
print(f"\n限制 max_features=20:")
print(f"词汇表大小: {len(vectorizer_limited.vocabulary_)}")

# 设置 min_df 和 max_df
vectorizer_filtered = CountVectorizer(min_df=2, max_df=0.8)
bow_filtered = vectorizer_filtered.fit_transform(documents)
print(f"\nmin_df=2, max_df=0.8:")
print(f"词汇表大小: {len(vectorizer_filtered.vocabulary_)}")
print(f"过滤掉的词: {len(vectorizer_basic.vocabulary_) - len(vectorizer_filtered.vocabulary_)}")

# 3.3 N-grams
print("\n【3.3】N-grams")

# Bigrams
vectorizer_bigram = CountVectorizer(ngram_range=(1, 2), max_features=30)
bow_bigram = vectorizer_bigram.fit_transform(documents)
print(f"\nBigrams (1-2 grams):")
print(f"特征数: {bow_bigram.shape[1]}")
bigram_features = vectorizer_bigram.get_feature_names_out()
print(f"示例特征:")
print(bigram_features[:20])

# ============================================================================
# 4. TF-IDF
# ============================================================================
print("\n【4】TF-IDF (Term Frequency-Inverse Document Frequency)")
print("-" * 80)
print("""
TF-IDF 原理：
TF-IDF Formula:

TF-IDF(t, d) = TF(t, d) × IDF(t)

其中：
• TF(t, d) = 词 t 在文档 d 中的频率
  Term Frequency: frequency of term t in document d

• IDF(t) = log(总文档数 / 包含词 t 的文档数)
  Inverse Document Frequency: log(total docs / docs containing t)

直觉：
Intuition:
• 词在文档中出现越多，TF 越大（重要）
  More frequent in document → higher TF → important

• 词在越多文档中出现，IDF 越小（不重要）
  Appears in more documents → lower IDF → less distinctive

• TF-IDF 平衡了词频和稀有度
  TF-IDF balances frequency and rarity
""")

# 4.1 基础 TF-IDF
print("\n【4.1】基础 TfidfVectorizer")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print(f"TF-IDF 矩阵形状: {tfidf_matrix.shape}")
print(f"词汇表大小: {len(tfidf_vectorizer.vocabulary_)}")

# 查看第一个文档的 TF-IDF 值
print(f"\n第一个文档的 TF-IDF 值（非零）:")
doc_tfidf = tfidf_matrix[0].toarray()[0]
tfidf_vocab = tfidf_vectorizer.get_feature_names_out()
word_tfidf = [(tfidf_vocab[i], doc_tfidf[i]) for i in range(len(doc_tfidf)) if doc_tfidf[i] > 0]
word_tfidf_sorted = sorted(word_tfidf, key=lambda x: x[1], reverse=True)

for word, score in word_tfidf_sorted:
    print(f"  {word}: {score:.4f}")

# 4.2 提取每个文档的重要词汇
print("\n【4.2】每个文档的 Top 3 重要词汇")

for i, doc in enumerate(documents):
    doc_tfidf_vector = tfidf_matrix[i].toarray()[0]
    word_scores = [(tfidf_vocab[j], doc_tfidf_vector[j])
                   for j in range(len(doc_tfidf_vector)) if doc_tfidf_vector[j] > 0]
    word_scores_sorted = sorted(word_scores, key=lambda x: x[1], reverse=True)

    print(f"\n文档 {i+1}: {doc[:50]}...")
    print(f"  Top 3: {[(w, f'{s:.3f}') for w, s in word_scores_sorted[:3]]}")

# 4.3 文档相似度计算
print("\n【4.3】文档相似度计算（Cosine Similarity）")

# 计算所有文档对的余弦相似度
similarity_matrix = cosine_similarity(tfidf_matrix)

print(f"相似度矩阵形状: {similarity_matrix.shape}")
print(f"\n文档 1 与其他文档的相似度:")
for i in range(len(documents)):
    if i != 0:
        print(f"  文档 {i+1}: {similarity_matrix[0, i]:.4f}")

# 找出最相似的文档对
print("\n最相似的文档对（Top 5）:")
similarities = []
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        similarities.append((i, j, similarity_matrix[i, j]))

similarities_sorted = sorted(similarities, key=lambda x: x[2], reverse=True)
for i, j, sim in similarities_sorted[:5]:
    print(f"\n文档 {i+1} vs 文档 {j+1}: {sim:.4f}")
    print(f"  文档 {i+1}: {documents[i][:60]}...")
    print(f"  文档 {j+1}: {documents[j][:60]}...")

# ============================================================================
# 5. Word2Vec
# ============================================================================
print("\n【5】Word2Vec 词嵌入")
print("-" * 80)
print("""
Word2Vec 原理：
Word2Vec Principle:

两种训练模式：
Two Training Architectures:

1. CBOW (Continuous Bag of Words)
   • 通过上下文预测中心词
   • 适合小数据集
   • 训练速度快

2. Skip-gram
   • 通过中心词预测上下文
   • 适合大数据集
   • 对低频词效果好

优势：
Advantages:
• 捕捉词义相似性
  Captures semantic similarity
• 支持词汇类比（king - man + woman = queen）
  Supports word analogies
• 低维稠密向量（通常 100-300 维）
  Low-dimensional dense vectors
""")

if GENSIM_AVAILABLE:
    print("\n【5.1】训练 Word2Vec 模型")

    # 准备训练数据（需要分词）
    if NLTK_AVAILABLE:
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    else:
        tokenized_docs = [doc.lower().split() for doc in documents]

    print(f"训练数据示例:")
    for i, tokens in enumerate(tokenized_docs[:3], 1):
        print(f"  文档 {i}: {tokens}")

    # 训练 Word2Vec 模型
    # 参数说明:
    # - vector_size: 词向量维度
    # - window: 上下文窗口大小
    # - min_count: 最小词频
    # - sg: 0=CBOW, 1=Skip-gram
    # - workers: 线程数
    w2v_model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=50,  # 词向量维度
        window=5,        # 上下文窗口
        min_count=1,     # 最小词频
        sg=0,            # CBOW
        workers=4,
        seed=RANDOM_STATE
    )

    print(f"\n✓ Word2Vec 模型训练完成")
    print(f"词汇表大小: {len(w2v_model.wv)}")
    print(f"词向量维度: {w2v_model.wv.vector_size}")

    # 5.2 词向量查看
    print("\n【5.2】词向量示例")

    # 查看 'learning' 的词向量
    if 'learning' in w2v_model.wv:
        learning_vector = w2v_model.wv['learning']
        print(f"\n'learning' 的词向量（前10维）:")
        print(learning_vector[:10])

    # 5.3 词汇相似度
    print("\n【5.3】词汇相似度")

    test_words = ['learning', 'data', 'neural', 'text']

    for word in test_words:
        if word in w2v_model.wv:
            print(f"\n'{word}' 的最相似词:")
            similar_words = w2v_model.wv.most_similar(word, topn=5)
            for similar_word, score in similar_words:
                print(f"  {similar_word}: {score:.4f}")

    # 5.4 词汇类比
    print("\n【5.4】词汇类比（Word Analogies）")
    print("尝试: learning - machine + deep = ?")

    try:
        if all(word in w2v_model.wv for word in ['learning', 'machine', 'deep']):
            result = w2v_model.wv.most_similar(
                positive=['learning', 'deep'],
                negative=['machine'],
                topn=3
            )
            print("结果:")
            for word, score in result:
                print(f"  {word}: {score:.4f}")
    except:
        print("  词汇不足以进行类比")

    # 5.5 文档向量化（平均词向量）
    print("\n【5.5】文档向量化")

    def document_vector(doc_tokens, model):
        """
        通过平均词向量获得文档向量
        Get document vector by averaging word vectors
        """
        vectors = [model.wv[word] for word in doc_tokens if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(model.wv.vector_size)
        return np.mean(vectors, axis=0)

    # 计算所有文档的向量
    doc_vectors = np.array([document_vector(tokens, w2v_model) for tokens in tokenized_docs])

    print(f"文档向量矩阵形状: {doc_vectors.shape}")

    # 计算文档相似度
    doc_similarity_w2v = cosine_similarity(doc_vectors)

    print(f"\nWord2Vec 文档相似度示例:")
    print(f"文档 1 与文档 2: {doc_similarity_w2v[0, 1]:.4f}")
    print(f"文档 1 与文档 3: {doc_similarity_w2v[0, 2]:.4f}")

else:
    print("⚠ Gensim 未安装，跳过 Word2Vec 部分")

# ============================================================================
# 6. 特征提取方法对比 / Feature Extraction Comparison
# ============================================================================
print("\n【6】特征提取方法对比")
print("-" * 80)

# 创建对比表格
comparison_data = {
    '特征': ['BoW', 'TF-IDF', 'Word2Vec'],
    '维度': [
        bow_matrix.shape[1],
        tfidf_matrix.shape[1],
        50 if GENSIM_AVAILABLE else 'N/A'
    ],
    '稀疏度': [
        f'{(1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])) * 100:.1f}%',
        f'{(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.1f}%',
        '0%' if GENSIM_AVAILABLE else 'N/A'
    ],
    '语义信息': ['无', '少', '丰富'],
    '适用场景': ['分类', '检索/分类', '语义任务']
}

comparison_df = pd.DataFrame(comparison_data)
print("\n特征提取方法对比:")
print(comparison_df.to_string(index=False))

# ============================================================================
# 7. 可视化 / Visualization
# ============================================================================
print("\n【7】特征提取可视化")
print("-" * 80)

# 7.1 BoW 词频矩阵热力图
fig, ax = create_subplots(1, 1, figsize=(14, 8))

# 选择前15个词和前5个文档
bow_dense = bow_matrix[:5, :15].toarray()
vocab_subset = vocab[:15]

sns.heatmap(bow_dense, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=vocab_subset, yticklabels=[f'Doc {i+1}' for i in range(5)],
            cbar_kws={'label': '词频 / Word Count'}, ax=ax)
ax.set_title('词袋模型 - 词频矩阵热力图\nBag of Words - Word Frequency Heatmap',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('词汇 / Vocabulary', fontsize=11)
ax.set_ylabel('文档 / Documents', fontsize=11)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/output/nlp_bow_heatmap.png',
            dpi=150, bbox_inches='tight')
print("✓ 图表已保存: output/nlp_bow_heatmap.png")
plt.show()

# 7.2 TF-IDF 权重分布
fig, axes = create_subplots(2, 2, figsize=(16, 12))

# 7.2.1 TF-IDF 热力图
tfidf_dense = tfidf_matrix[:5, :15].toarray()
sns.heatmap(tfidf_dense, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=vocab_subset, yticklabels=[f'Doc {i+1}' for i in range(5)],
            cbar_kws={'label': 'TF-IDF 权重'}, ax=axes[0, 0])
axes[0, 0].set_title('TF-IDF 权重热力图\nTF-IDF Weight Heatmap',
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('词汇 / Vocabulary', fontsize=10)
axes[0, 0].set_ylabel('文档 / Documents', fontsize=10)
plt.sca(axes[0, 0])
plt.xticks(rotation=45, ha='right')

# 7.2.2 全局词汇重要性（平均 TF-IDF）
avg_tfidf = tfidf_matrix.mean(axis=0).A1
top_indices = avg_tfidf.argsort()[-15:][::-1]
top_words = [tfidf_vocab[i] for i in top_indices]
top_scores = [avg_tfidf[i] for i in top_indices]

axes[0, 1].barh(range(len(top_words)), top_scores, color='steelblue')
axes[0, 1].set_yticks(range(len(top_words)))
axes[0, 1].set_yticklabels(top_words)
axes[0, 1].set_xlabel('平均 TF-IDF 权重', fontsize=10)
axes[0, 1].set_title('全局重要词汇 (Top 15)\nGlobally Important Words',
                     fontsize=12, fontweight='bold')
axes[0, 1].invert_yaxis()

# 7.2.3 BoW vs TF-IDF 对比
sample_doc_idx = 0
bow_sample = bow_matrix[sample_doc_idx].toarray()[0]
tfidf_sample = tfidf_matrix[sample_doc_idx].toarray()[0]

# 选择非零词汇
nonzero_indices = np.where(bow_sample > 0)[0][:10]
sample_words = [vocab[i] for i in nonzero_indices]
bow_values = [bow_sample[i] for i in nonzero_indices]
tfidf_values = [tfidf_sample[i] for i in nonzero_indices]

x = np.arange(len(sample_words))
width = 0.35

axes[1, 0].bar(x - width/2, bow_values, width, label='BoW', color='skyblue', alpha=0.8)
axes[1, 0].bar(x + width/2, tfidf_values, width, label='TF-IDF', color='coral', alpha=0.8)
axes[1, 0].set_xlabel('词汇 / Words', fontsize=10)
axes[1, 0].set_ylabel('权重 / Weight', fontsize=10)
axes[1, 0].set_title(f'文档 1: BoW vs TF-IDF 对比\nDocument 1: BoW vs TF-IDF Comparison',
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(sample_words, rotation=45, ha='right')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(axis='y', alpha=0.3)

# 7.2.4 文档相似度矩阵
im = axes[1, 1].imshow(similarity_matrix, cmap='YlOrRd', aspect='auto')
axes[1, 1].set_xticks(range(len(documents)))
axes[1, 1].set_yticks(range(len(documents)))
axes[1, 1].set_xticklabels([f'D{i+1}' for i in range(len(documents))])
axes[1, 1].set_yticklabels([f'D{i+1}' for i in range(len(documents))])
axes[1, 1].set_title('文档相似度矩阵 (TF-IDF)\nDocument Similarity Matrix',
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('文档 / Documents', fontsize=10)
axes[1, 1].set_ylabel('文档 / Documents', fontsize=10)

# 添加数值标注
for i in range(len(documents)):
    for j in range(len(documents)):
        text = axes[1, 1].text(j, i, f'{similarity_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=7)

plt.colorbar(im, ax=axes[1, 1], label='余弦相似度 / Cosine Similarity')

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/output/nlp_tfidf_analysis.png',
            dpi=150, bbox_inches='tight')
print("✓ 图表已保存: output/nlp_tfidf_analysis.png")
plt.show()

# 7.3 Word2Vec 可视化
if GENSIM_AVAILABLE:
    fig, axes = create_subplots(2, 2, figsize=(16, 12))

    # 7.3.1 词向量 t-SNE 降维可视化
    # 获取所有词向量
    word_vectors = []
    words_list = []
    for word in w2v_model.wv.index_to_key:
        word_vectors.append(w2v_model.wv[word])
        words_list.append(word)

    word_vectors_array = np.array(word_vectors)

    # t-SNE 降维
    if len(word_vectors_array) > 1:
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(5, len(word_vectors_array)-1))
        word_vectors_2d = tsne.fit_transform(word_vectors_array)

        axes[0, 0].scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1],
                          c=range(len(words_list)), cmap='viridis', alpha=0.6, s=100)

        # 标注词汇
        for i, word in enumerate(words_list):
            axes[0, 0].annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]),
                               xytext=(5, 2), textcoords='offset points',
                               fontsize=8, alpha=0.7)

        axes[0, 0].set_title('Word2Vec 词向量 t-SNE 可视化\nWord2Vec t-SNE Visualization',
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('t-SNE 维度 1', fontsize=10)
        axes[0, 0].set_ylabel('t-SNE 维度 2', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

    # 7.3.2 词汇相似度网络
    # 选择几个关键词
    key_words = ['learning', 'data', 'neural', 'text']
    key_words = [w for w in key_words if w in w2v_model.wv]

    if len(key_words) > 0:
        # 为每个关键词找到最相似的词
        network_data = []
        for word in key_words[:3]:  # 限制数量
            similar = w2v_model.wv.most_similar(word, topn=3)
            for sim_word, score in similar:
                network_data.append((word, sim_word, score))

        # 简化的网络可视化（使用散点图）
        axes[0, 1].text(0.5, 0.5, '词汇相似度网络\nWord Similarity Network',
                       ha='center', va='center', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # 显示相似词对
        y_pos = 0.8
        for word, sim_word, score in network_data[:8]:
            axes[0, 1].text(0.5, y_pos, f'{word} ↔ {sim_word}: {score:.3f}',
                           ha='center', va='center', fontsize=9)
            y_pos -= 0.1

    # 7.3.3 文档向量相似度
    doc_sim_w2v = cosine_similarity(doc_vectors)

    im = axes[1, 0].imshow(doc_sim_w2v, cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_xticks(range(len(documents)))
    axes[1, 0].set_yticks(range(len(documents)))
    axes[1, 0].set_xticklabels([f'D{i+1}' for i in range(len(documents))])
    axes[1, 0].set_yticklabels([f'D{i+1}' for i in range(len(documents))])
    axes[1, 0].set_title('文档相似度矩阵 (Word2Vec)\nDocument Similarity Matrix',
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('文档 / Documents', fontsize=10)
    axes[1, 0].set_ylabel('文档 / Documents', fontsize=10)

    for i in range(len(documents)):
        for j in range(len(documents)):
            text = axes[1, 0].text(j, i, f'{doc_sim_w2v[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=7)

    plt.colorbar(im, ax=axes[1, 0], label='余弦相似度 / Cosine Similarity')

    # 7.3.4 特征维度对比
    methods = ['BoW', 'TF-IDF', 'Word2Vec']
    dimensions = [bow_matrix.shape[1], tfidf_matrix.shape[1], 50]
    colors_dim = ['skyblue', 'coral', 'lightgreen']

    axes[1, 1].bar(methods, dimensions, color=colors_dim, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('特征维度 / Feature Dimensions', fontsize=10)
    axes[1, 1].set_title('特征维度对比\nFeature Dimensions Comparison',
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_yscale('log')  # 对数刻度

    for i, v in enumerate(dimensions):
        axes[1, 1].text(i, v * 1.1, str(v), ha='center', va='bottom',
                       fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/user/machineLearning-basics/output/nlp_word2vec_visualization.png',
                dpi=150, bbox_inches='tight')
    print("✓ 图表已保存: output/nlp_word2vec_visualization.png")
    plt.show()

# 7.4 方法性能对比
fig, axes = create_subplots(2, 2, figsize=(16, 12))

# 7.4.1 稀疏度对比
methods_sparse = ['BoW', 'TF-IDF', 'Word2Vec']
sparsity = [
    (1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])) * 100,
    (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100,
    0 if GENSIM_AVAILABLE else 0
]

axes[0, 0].bar(methods_sparse, sparsity, color=['steelblue', 'coral', 'lightgreen'],
              alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('稀疏度 / Sparsity (%)', fontsize=10)
axes[0, 0].set_title('特征稀疏度对比\nFeature Sparsity Comparison',
                    fontsize=12, fontweight='bold')
for i, v in enumerate(sparsity):
    axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

# 7.4.2 词汇表大小对比
vocab_sizes = [
    len(vectorizer_basic.vocabulary_),
    len(tfidf_vectorizer.vocabulary_),
    len(w2v_model.wv) if GENSIM_AVAILABLE else 0
]

axes[0, 1].bar(methods_sparse, vocab_sizes, color=['steelblue', 'coral', 'lightgreen'],
              alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('词汇表大小 / Vocabulary Size', fontsize=10)
axes[0, 1].set_title('词汇表大小对比\nVocabulary Size Comparison',
                    fontsize=12, fontweight='bold')
for i, v in enumerate(vocab_sizes):
    axes[0, 1].text(i, v + 0.5, str(v), ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

# 7.4.3 特征向量示例（第一个文档）
feature_comparison = pd.DataFrame({
    'BoW': bow_matrix[0].toarray()[0][:10],
    'TF-IDF': tfidf_matrix[0].toarray()[0][:10],
    'Word2Vec': doc_vectors[0][:10] if GENSIM_AVAILABLE else np.zeros(10)
})

feature_comparison.plot(kind='bar', ax=axes[1, 0], color=['steelblue', 'coral', 'lightgreen'],
                       alpha=0.7, width=0.8)
axes[1, 0].set_xlabel('特征维度 / Feature Dimension', fontsize=10)
axes[1, 0].set_ylabel('特征值 / Feature Value', fontsize=10)
axes[1, 0].set_title('文档 1 的特征向量对比 (前10维)\nDocument 1 Feature Vectors Comparison',
                    fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(axis='y', alpha=0.3)

# 7.4.4 适用场景总结
axes[1, 1].axis('off')

summary_text = """
特征提取方法适用场景总结
Feature Extraction Methods Summary

📊 Bag of Words (BoW)
  ✓ 文本分类（简单任务）
  ✓ 垃圾邮件过滤
  ✓ 文档聚类

📈 TF-IDF
  ✓ 信息检索
  ✓ 文档排序
  ✓ 关键词提取
  ✓ 文本分类（中等复杂度）

🧠 Word2Vec
  ✓ 语义相似度计算
  ✓ 词汇类比任务
  ✓ 命名实体识别
  ✓ 情感分析
  ✓ 问答系统

💡 选择建议：
  • 小数据集，简单任务 → BoW
  • 需要关键词提取 → TF-IDF
  • 需要语义理解 → Word2Vec
"""

axes[1, 1].text(0.1, 0.95, summary_text, ha='left', va='top',
               fontsize=9, family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('/home/user/machineLearning-basics/output/nlp_methods_comparison.png',
            dpi=150, bbox_inches='tight')
print("✓ 图表已保存: output/nlp_methods_comparison.png")
plt.show()

# ============================================================================
# 8. 总结 / Summary
# ============================================================================
print("\n【8】总结")
print("=" * 80)
print("""
本教程涵盖的内容：
Topics Covered:

✓ 词袋模型（Bag of Words）
  • CountVectorizer 使用
  • 参数调优（max_features, min_df, max_df）
  • N-grams 生成

✓ TF-IDF
  • 原理和计算公式
  • 重要词汇提取
  • 文档相似度计算

✓ Word2Vec
  • CBOW vs Skip-gram
  • 词向量训练
  • 词汇相似度和类比
  • 文档向量化

✓ 方法对比
  • 维度、稀疏度、语义信息
  • 适用场景分析

最佳实践建议：
Best Practices:

1. 根据任务选择合适的特征提取方法
   Choose the right method for your task

2. TF-IDF 适合大多数传统 ML 任务
   TF-IDF works well for most traditional ML tasks

3. Word2Vec 适合需要语义理解的任务
   Word2Vec is better for semantic understanding

4. 可以组合多种方法（集成学习）
   Can combine multiple methods (ensemble learning)

5. 大规模数据考虑使用预训练词向量
   Use pre-trained embeddings for large-scale data

下一步学习：
Next Steps:
• 03_text_classification.py - 文本分类实战
  将特征提取应用到实际分类任务
""")

print("\n" + "=" * 80)
print("文本特征提取教程完成！".center(80))
print("Text Feature Extraction Tutorial Complete!".center(80))
print("=" * 80)
