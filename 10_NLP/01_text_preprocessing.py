"""
文本预处理（Text Preprocessing）
自然语言处理的基础步骤

Text Preprocessing - Foundation of Natural Language Processing

文本预处理是 NLP 任务的第一步，也是最重要的步骤之一
Proper text preprocessing is crucial for successful NLP applications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
from utils import RANDOM_STATE, setup_chinese_fonts, create_subplots, DPI, save_figure, get_output_path

# 尝试导入 NLP 相关库 / Try importing NLP libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
    from nltk import ngrams
    NLTK_AVAILABLE = True

    # 下载必要的 NLTK 数据 / Download necessary NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("正在下载 NLTK 数据包...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠ NLTK 未安装。运行: pip install nltk")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("⚠ jieba 未安装（中文分词）。运行: pip install jieba")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("⚠ wordcloud 未安装。运行: pip install wordcloud")

setup_chinese_fonts()

print("=" * 80)
print("文本预处理（Text Preprocessing）教程".center(80))
print("=" * 80)

# ============================================================================
# 1. 文本预处理概述 / Text Preprocessing Overview
# ============================================================================
print("\n【1】文本预处理概述")
print("-" * 80)
print("""
NLP 任务流程 / NLP Pipeline:
1. 文本收集（Text Collection）
2. 文本预处理（Text Preprocessing）← 我们在这里
3. 特征提取（Feature Extraction）
4. 模型训练（Model Training）
5. 模型评估（Model Evaluation）

为什么需要文本预处理？
Why Text Preprocessing?
• 降低噪声（Remove noise）
• 统一格式（Normalize format）
• 减少词汇量（Reduce vocabulary size）
• 提高模型性能（Improve model performance）

常见预处理步骤：
Common preprocessing steps:
1. 小写转换（Lowercasing）
2. 去除标点符号（Remove punctuation）
3. 去除数字（Remove numbers）
4. 去除 HTML 标签（Remove HTML tags）
5. 去除 URL（Remove URLs）
6. 分词（Tokenization）
7. 去除停用词（Remove stopwords）
8. 词干提取/词形还原（Stemming/Lemmatization）
""")

# ============================================================================
# 2. 基础文本清洗 / Basic Text Cleaning
# ============================================================================
print("\n【2】基础文本清洗")
print("-" * 80)

# 示例文本 / Sample texts
sample_text_en = """
    Hello World! This is a SAMPLE text for NLP preprocessing.
    It contains UPPERCASE, lowercase, numbers like 123 and 456,
    punctuation marks!!!, and special characters @#$%.
    Visit https://www.example.com for more info.
    Email: test@example.com

    Here's another sentence with    multiple   spaces.
"""

sample_text_cn = """
    你好世界！这是一个自然语言处理的示例文本。
    它包含了中文、English混合，标点符号！！！
    还有数字123和456，以及特殊字符@#$%。
    访问 https://www.example.com 获取更多信息。
"""

print("原始英文文本:")
print(sample_text_en)
print("\n原始中文文本:")
print(sample_text_cn)

# 2.1 小写转换 / Lowercasing
def to_lowercase(text):
    """
    转换为小写
    Convert to lowercase
    """
    return text.lower()

# 2.2 去除 HTML 标签 / Remove HTML tags
def remove_html_tags(text):
    """
    去除 HTML 标签
    Remove HTML tags
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# 2.3 去除 URL / Remove URLs
def remove_urls(text):
    """
    去除 URL
    Remove URLs
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

# 2.4 去除邮箱 / Remove emails
def remove_emails(text):
    """
    去除邮箱地址
    Remove email addresses
    """
    email_pattern = re.compile(r'\S+@\S+')
    return email_pattern.sub('', text)

# 2.5 去除数字 / Remove numbers
def remove_numbers(text):
    """
    去除数字
    Remove numbers
    """
    return re.sub(r'\d+', '', text)

# 2.6 去除标点符号 / Remove punctuation
def remove_punctuation(text):
    """
    去除标点符号
    Remove punctuation
    """
    # 英文标点
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # 中文标点
    cn_punctuation = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏.'
    for char in cn_punctuation:
        text = text.replace(char, '')

    return text

# 2.7 去除多余空格 / Remove extra whitespace
def remove_extra_whitespace(text):
    """
    去除多余空格
    Remove extra whitespace
    """
    return ' '.join(text.split())

# 2.8 综合清洗函数 / Combined cleaning function
def clean_text(text, lowercase=True, remove_url=True, remove_email=True,
               remove_num=False, remove_punct=True, remove_whitespace=True):
    """
    文本清洗主函数
    Main text cleaning function

    Parameters:
    -----------
    text : str
        输入文本 / Input text
    lowercase : bool
        是否转小写 / Whether to lowercase
    remove_url : bool
        是否去除 URL / Whether to remove URLs
    remove_email : bool
        是否去除邮箱 / Whether to remove emails
    remove_num : bool
        是否去除数字 / Whether to remove numbers
    remove_punct : bool
        是否去除标点 / Whether to remove punctuation
    remove_whitespace : bool
        是否去除多余空格 / Whether to remove extra whitespace

    Returns:
    --------
    str
        清洗后的文本 / Cleaned text
    """
    if remove_url:
        text = remove_urls(text)
    if remove_email:
        text = remove_emails(text)
    if lowercase:
        text = to_lowercase(text)
    if remove_num:
        text = remove_numbers(text)
    if remove_punct:
        text = remove_punctuation(text)
    if remove_whitespace:
        text = remove_extra_whitespace(text)

    return text

# 演示清洗效果
cleaned_en = clean_text(sample_text_en)
print("\n清洗后的英文文本:")
print(cleaned_en)

cleaned_cn = clean_text(sample_text_cn)
print("\n清洗后的中文文本:")
print(cleaned_cn)

# ============================================================================
# 3. 分词（Tokenization）
# ============================================================================
print("\n【3】分词（Tokenization）")
print("-" * 80)

def tokenize_text(text, language='en'):
    """
    文本分词
    Text tokenization

    Parameters:
    -----------
    text : str
        输入文本 / Input text
    language : str
        语言类型 ('en' 或 'cn') / Language type

    Returns:
    --------
    list
        词语列表 / List of tokens
    """
    if language == 'en':
        if NLTK_AVAILABLE:
            # 使用 NLTK 分词
            return word_tokenize(text)
        else:
            # 简单的空格分割
            return text.split()

    elif language == 'cn':
        if JIEBA_AVAILABLE:
            # 使用 jieba 分词
            return list(jieba.cut(text))
        else:
            # 按字符分割（不推荐）
            return list(text.replace(' ', ''))

    return text.split()

# 英文分词示例
if NLTK_AVAILABLE:
    en_tokens = tokenize_text(cleaned_en, 'en')
    print(f"\n英文分词结果（前20个）:")
    print(en_tokens[:20])
    print(f"总词数: {len(en_tokens)}")

# 中文分词示例
if JIEBA_AVAILABLE:
    cn_tokens = tokenize_text(cleaned_cn, 'cn')
    print(f"\n中文分词结果（前20个）:")
    print(cn_tokens[:20])
    print(f"总词数: {len(cn_tokens)}")

# 句子分割 / Sentence tokenization
if NLTK_AVAILABLE:
    sentences = sent_tokenize(sample_text_en)
    print(f"\n句子分割结果:")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent.strip()}")

# ============================================================================
# 4. 停用词处理 / Stopwords Removal
# ============================================================================
print("\n【4】停用词处理")
print("-" * 80)
print("""
停用词（Stopwords）是什么？
• 高频但对文本意义贡献小的词
• 例如：the, is, at, which, on（英文）
• 例如：的、是、在、和、了（中文）

为什么要去除停用词？
• 减少特征维度
• 提高计算效率
• 聚焦重要词汇
""")

# 加载停用词
if NLTK_AVAILABLE:
    try:
        stop_words_en = set(stopwords.words('english'))
        print(f"\n英文停用词数量: {len(stop_words_en)}")
        print(f"示例: {list(stop_words_en)[:20]}")
    except:
        stop_words_en = set(['the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but'])
        print("使用默认英文停用词")
else:
    stop_words_en = set(['the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but'])

# 中文停用词（简化版）
stop_words_cn = set(['的', '是', '在', '和', '了', '有', '个', '也', '就', '都', '这', '那', '要', '会', '能'])

def remove_stopwords(tokens, stop_words):
    """
    去除停用词
    Remove stopwords

    Parameters:
    -----------
    tokens : list
        词语列表 / List of tokens
    stop_words : set
        停用词集合 / Set of stopwords

    Returns:
    --------
    list
        过滤后的词语列表 / Filtered tokens
    """
    return [token for token in tokens if token.lower() not in stop_words]

# 演示停用词去除
if NLTK_AVAILABLE:
    en_tokens_no_stop = remove_stopwords(en_tokens, stop_words_en)
    print(f"\n英文去除停用词前: {len(en_tokens)} 词")
    print(f"去除停用词后: {len(en_tokens_no_stop)} 词")
    print(f"减少了 {len(en_tokens) - len(en_tokens_no_stop)} 词 ({(1-len(en_tokens_no_stop)/len(en_tokens))*100:.1f}%)")

# ============================================================================
# 5. 词形归一化 / Word Normalization
# ============================================================================
print("\n【5】词形归一化")
print("-" * 80)
print("""
词形归一化的两种方法：

1. Stemming（词干提取）
   • 简单粗暴，去除词缀
   • 结果可能不是真实单词
   • 速度快
   • 例如: running → run, studies → studi

2. Lemmatization（词形还原）
   • 基于词典，还原为词根
   • 结果是真实单词
   • 速度慢，需要词性标注
   • 例如: running → run, studies → study, better → good
""")

if NLTK_AVAILABLE:
    # 5.1 Stemming
    print("\n【5.1】Stemming（词干提取）")

    # Porter Stemmer
    porter = PorterStemmer()

    # Snowball Stemmer
    snowball = SnowballStemmer('english')

    # 示例词汇
    words = ['running', 'runs', 'ran', 'runner', 'easily', 'fairly',
             'studies', 'studying', 'studied', 'cats', 'dogs', 'better', 'good']

    print("\n词干提取对比:")
    print(f"{'原词':<15} {'Porter':<15} {'Snowball':<15}")
    print("-" * 45)
    for word in words:
        porter_stem = porter.stem(word)
        snowball_stem = snowball.stem(word)
        print(f"{word:<15} {porter_stem:<15} {snowball_stem:<15}")

    # 5.2 Lemmatization
    print("\n【5.2】Lemmatization（词形还原）")

    lemmatizer = WordNetLemmatizer()

    print("\n词形还原示例:")
    print(f"{'原词':<15} {'还原结果':<15} {'词性':<10}")
    print("-" * 40)

    # 不同词性的还原
    examples = [
        ('running', 'v'),  # 动词
        ('runs', 'v'),
        ('better', 'a'),   # 形容词
        ('studies', 'n'),  # 名词
        ('studies', 'v'),  # 动词
        ('cats', 'n'),
    ]

    for word, pos in examples:
        lemma = lemmatizer.lemmatize(word, pos=pos)
        print(f"{word:<15} {lemma:<15} {pos:<10}")

    # 5.3 对比
    print("\n【5.3】Stemming vs Lemmatization 对比")

    test_words = ['running', 'studies', 'better', 'fairly', 'cats', 'women']
    print(f"\n{'原词':<15} {'Stemming':<15} {'Lemmatization':<15}")
    print("-" * 45)
    for word in test_words:
        stem = porter.stem(word)
        lemma = lemmatizer.lemmatize(word, pos='v')  # 假设是动词
        print(f"{word:<15} {stem:<15} {lemma:<15}")

# ============================================================================
# 6. 高级处理 / Advanced Processing
# ============================================================================
print("\n【6】高级处理")
print("-" * 80)

# 6.1 N-grams 生成
print("\n【6.1】N-grams 生成")
print("""
N-grams: 连续的 N 个词的组合
• Unigrams (1-gram): 单个词
• Bigrams (2-gram): 两个词的组合
• Trigrams (3-gram): 三个词的组合
""")

if NLTK_AVAILABLE and len(en_tokens_no_stop) > 0:
    # Bigrams
    bigrams_list = list(ngrams(en_tokens_no_stop, 2))
    print(f"\nBigrams 示例（前10个）:")
    for i, bg in enumerate(bigrams_list[:10], 1):
        print(f"{i}. {' '.join(bg)}")

    # Trigrams
    trigrams_list = list(ngrams(en_tokens_no_stop, 3))
    print(f"\nTrigrams 示例（前10个）:")
    for i, tg in enumerate(trigrams_list[:10], 1):
        print(f"{i}. {' '.join(tg)}")

# 6.2 处理缩写 / Handle contractions
print("\n【6.2】处理英文缩写")

contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

def expand_contractions(text, contractions_dict=contractions_dict):
    """
    展开英文缩写
    Expand contractions
    """
    pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                         flags=re.IGNORECASE|re.DOTALL)

    def replace(match):
        return contractions_dict[match.group(0).lower()]

    return pattern.sub(replace, text)

# 示例
contraction_text = "I can't believe it's already 2024! We're gonna have a great time."
expanded_text = expand_contractions(contraction_text)
print(f"原文: {contraction_text}")
print(f"展开后: {expanded_text}")

# ============================================================================
# 7. 完整 Pipeline 示例 / Complete Pipeline Example
# ============================================================================
print("\n【7】完整预处理 Pipeline")
print("-" * 80)

class TextPreprocessor:
    """
    文本预处理器
    Text Preprocessor
    """

    def __init__(self, language='en', remove_stopwords_flag=True,
                 use_stemming=False, use_lemmatization=False):
        """
        初始化预处理器

        Parameters:
        -----------
        language : str
            语言 ('en' 或 'cn')
        remove_stopwords_flag : bool
            是否去除停用词
        use_stemming : bool
            是否使用词干提取
        use_lemmatization : bool
            是否使用词形还原
        """
        self.language = language
        self.remove_stopwords_flag = remove_stopwords_flag
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization

        # 加载工具
        if language == 'en':
            if NLTK_AVAILABLE:
                self.stop_words = set(stopwords.words('english')) if remove_stopwords_flag else set()
                if use_stemming:
                    self.stemmer = PorterStemmer()
                if use_lemmatization:
                    self.lemmatizer = WordNetLemmatizer()
            else:
                self.stop_words = stop_words_en if remove_stopwords_flag else set()
        else:
            self.stop_words = stop_words_cn if remove_stopwords_flag else set()

    def preprocess(self, text):
        """
        完整的预处理流程
        Complete preprocessing pipeline

        Parameters:
        -----------
        text : str
            输入文本

        Returns:
        --------
        list
            处理后的词语列表
        """
        # 1. 清洗文本
        text = clean_text(text)

        # 2. 展开缩写（仅英文）
        if self.language == 'en':
            text = expand_contractions(text)

        # 3. 分词
        tokens = tokenize_text(text, self.language)

        # 4. 去除停用词
        if self.remove_stopwords_flag:
            tokens = remove_stopwords(tokens, self.stop_words)

        # 5. 词干提取或词形还原（仅英文）
        if self.language == 'en' and NLTK_AVAILABLE:
            if self.use_stemming:
                tokens = [self.stemmer.stem(token) for token in tokens]
            elif self.use_lemmatization:
                tokens = [self.lemmatizer.lemmatize(token, pos='v') for token in tokens]

        return tokens

# 演示完整 Pipeline
preprocessor = TextPreprocessor(language='en', remove_stopwords_flag=True,
                                use_lemmatization=True)

demo_text = """
Natural Language Processing (NLP) is a fascinating field!
It's all about making computers understand human languages.
We're using various techniques like tokenization, stemming, and more.
"""

print("原始文本:")
print(demo_text)

processed_tokens = preprocessor.preprocess(demo_text)
print(f"\n预处理后的词语:")
print(processed_tokens)
print(f"\n词数: {len(processed_tokens)}")

# ============================================================================
# 8. 可视化 / Visualization
# ============================================================================
print("\n【8】文本预处理可视化")
print("-" * 80)

# 准备更多示例文本用于可视化
sample_texts = [
    "Natural language processing is amazing! It helps computers understand human language.",
    "Machine learning and deep learning are transforming the world of AI.",
    "Text preprocessing is the first and most important step in NLP.",
    "Tokenization, stemming, and lemmatization are key techniques.",
    "Stop words removal helps reduce noise in text data.",
]

# 合并所有文本
all_text = ' '.join(sample_texts)

# 预处理
preprocessor_vis = TextPreprocessor(language='en', remove_stopwords_flag=False,
                                    use_lemmatization=True)
all_tokens = preprocessor_vis.preprocess(all_text)

preprocessor_vis_no_stop = TextPreprocessor(language='en', remove_stopwords_flag=True,
                                            use_lemmatization=True)
all_tokens_no_stop = preprocessor_vis_no_stop.preprocess(all_text)

# 8.1 预处理前后对比
fig, axes = create_subplots(2, 2, figsize=(16, 12))

# 词频统计（预处理前）
if NLTK_AVAILABLE:
    tokens_before = word_tokenize(all_text.lower())
else:
    tokens_before = all_text.lower().split()

word_freq_before = Counter(tokens_before)
top_words_before = word_freq_before.most_common(15)

words_before, counts_before = zip(*top_words_before)
axes[0, 0].barh(range(len(words_before)), counts_before, color='steelblue')
axes[0, 0].set_yticks(range(len(words_before)))
axes[0, 0].set_yticklabels(words_before)
axes[0, 0].set_xlabel('频率 / Frequency', fontsize=10)
axes[0, 0].set_title('预处理前词频统计\nWord Frequency Before Preprocessing', fontsize=12, fontweight='bold')
axes[0, 0].invert_yaxis()

# 词频统计（预处理后）
word_freq_after = Counter(all_tokens_no_stop)
top_words_after = word_freq_after.most_common(15)

words_after, counts_after = zip(*top_words_after)
axes[0, 1].barh(range(len(words_after)), counts_after, color='coral')
axes[0, 1].set_yticks(range(len(words_after)))
axes[0, 1].set_yticklabels(words_after)
axes[0, 1].set_xlabel('频率 / Frequency', fontsize=10)
axes[0, 1].set_title('预处理后词频统计\nWord Frequency After Preprocessing', fontsize=12, fontweight='bold')
axes[0, 1].invert_yaxis()

# 停用词影响分析
labels = ['保留停用词\nWith Stopwords', '去除停用词\nWithout Stopwords']
sizes = [len(all_tokens), len(all_tokens_no_stop)]
colors = ['lightblue', 'lightcoral']
axes[1, 0].bar(labels, sizes, color=colors, alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('词数 / Token Count', fontsize=10)
axes[1, 0].set_title('停用词对词数的影响\nImpact of Stopwords Removal', fontsize=12, fontweight='bold')
for i, v in enumerate(sizes):
    axes[1, 0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

# 文本长度分布
text_lengths = [len(preprocessor_vis.preprocess(text)) for text in sample_texts]
axes[1, 1].hist(text_lengths, bins=5, color='mediumpurple', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('词数 / Token Count', fontsize=10)
axes[1, 1].set_ylabel('文本数 / Text Count', fontsize=10)
axes[1, 1].set_title('文本长度分布\nText Length Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
save_figure(fig, get_output_path('nlp_preprocessing_comparison.png'))
print("✓ 图表已保存: output/nlp_preprocessing_comparison.png")
plt.show()

# 8.2 Stemming vs Lemmatization 对比
if NLTK_AVAILABLE:
    fig, ax = create_subplots(1, 1, figsize=(14, 8))

    test_words_vis = ['running', 'runs', 'ran', 'studies', 'studying', 'better',
                      'fairly', 'cats', 'dogs', 'wolves']

    stemmer_vis = PorterStemmer()
    lemmatizer_vis = WordNetLemmatizer()

    stemmed = [stemmer_vis.stem(word) for word in test_words_vis]
    lemmatized = [lemmatizer_vis.lemmatize(word, pos='v') for word in test_words_vis]

    x = np.arange(len(test_words_vis))
    width = 0.35

    # 为了可视化，我们计算词长变化
    original_lengths = [len(w) for w in test_words_vis]
    stemmed_lengths = [len(w) for w in stemmed]
    lemmatized_lengths = [len(w) for w in lemmatized]

    ax.bar(x - width/2, stemmed_lengths, width, label='Stemming', color='skyblue', alpha=0.8)
    ax.bar(x + width/2, lemmatized_lengths, width, label='Lemmatization', color='lightcoral', alpha=0.8)

    ax.set_xlabel('词语 / Words', fontsize=11)
    ax.set_ylabel('字符长度 / Character Length', fontsize=11)
    ax.set_title('Stemming vs Lemmatization 词长对比\nStemming vs Lemmatization Character Length Comparison',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(test_words_vis, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_figure(fig, get_output_path('nlp_stemming_vs_lemmatization.png'))
    print("✓ 图表已保存: output/nlp_stemming_vs_lemmatization.png")
    plt.show()

# 8.3 词云可视化
if WORDCLOUD_AVAILABLE:
    fig, axes = create_subplots(1, 2, figsize=(18, 7))

    # 预处理前的词云
    text_for_cloud_before = ' '.join(tokens_before)
    wordcloud_before = WordCloud(width=800, height=400, background_color='white',
                                  colormap='Blues').generate(text_for_cloud_before)

    axes[0].imshow(wordcloud_before, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('预处理前词云\nWord Cloud Before Preprocessing',
                      fontsize=13, fontweight='bold', pad=20)

    # 预处理后的词云
    text_for_cloud_after = ' '.join(all_tokens_no_stop)
    wordcloud_after = WordCloud(width=800, height=400, background_color='white',
                                 colormap='Reds').generate(text_for_cloud_after)

    axes[1].imshow(wordcloud_after, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('预处理后词云\nWord Cloud After Preprocessing',
                      fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    save_figure(fig, get_output_path('nlp_wordcloud_comparison.png'))
    print("✓ 图表已保存: output/nlp_wordcloud_comparison.png")
    plt.show()

# 8.4 N-grams 频率分析
if NLTK_AVAILABLE and len(all_tokens_no_stop) > 0:
    fig, axes = create_subplots(2, 2, figsize=(16, 12))

    # Unigrams
    unigram_freq = Counter(all_tokens_no_stop)
    top_unigrams = unigram_freq.most_common(10)
    words_uni, counts_uni = zip(*top_unigrams)

    axes[0, 0].barh(range(len(words_uni)), counts_uni, color='steelblue')
    axes[0, 0].set_yticks(range(len(words_uni)))
    axes[0, 0].set_yticklabels(words_uni)
    axes[0, 0].set_xlabel('频率 / Frequency', fontsize=10)
    axes[0, 0].set_title('Top 10 Unigrams', fontsize=12, fontweight='bold')
    axes[0, 0].invert_yaxis()

    # Bigrams
    bigrams_all = list(ngrams(all_tokens_no_stop, 2))
    bigram_freq = Counter(bigrams_all)
    top_bigrams = bigram_freq.most_common(10)
    bigram_labels = [' '.join(bg) for bg, _ in top_bigrams]
    bigram_counts = [count for _, count in top_bigrams]

    axes[0, 1].barh(range(len(bigram_labels)), bigram_counts, color='coral')
    axes[0, 1].set_yticks(range(len(bigram_labels)))
    axes[0, 1].set_yticklabels(bigram_labels)
    axes[0, 1].set_xlabel('频率 / Frequency', fontsize=10)
    axes[0, 1].set_title('Top 10 Bigrams', fontsize=12, fontweight='bold')
    axes[0, 1].invert_yaxis()

    # Trigrams
    trigrams_all = list(ngrams(all_tokens_no_stop, 3))
    trigram_freq = Counter(trigrams_all)
    top_trigrams = trigram_freq.most_common(10)
    trigram_labels = [' '.join(tg) for tg, _ in top_trigrams]
    trigram_counts = [count for _, count in top_trigrams]

    axes[1, 0].barh(range(len(trigram_labels)), trigram_counts, color='mediumpurple')
    axes[1, 0].set_yticks(range(len(trigram_labels)))
    axes[1, 0].set_yticklabels(trigram_labels)
    axes[1, 0].set_xlabel('频率 / Frequency', fontsize=10)
    axes[1, 0].set_title('Top 10 Trigrams', fontsize=12, fontweight='bold')
    axes[1, 0].invert_yaxis()

    # N-grams 数量对比
    ngram_types = ['Unigrams', 'Bigrams', 'Trigrams']
    ngram_counts_total = [len(set(all_tokens_no_stop)), len(set(bigrams_all)), len(set(trigrams_all))]

    axes[1, 1].bar(ngram_types, ngram_counts_total, color=['steelblue', 'coral', 'mediumpurple'],
                   alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('唯一 N-gram 数量 / Unique N-gram Count', fontsize=10)
    axes[1, 1].set_title('N-grams 唯一数量对比\nUnique N-grams Count Comparison',
                         fontsize=12, fontweight='bold')
    for i, v in enumerate(ngram_counts_total):
        axes[1, 1].text(i, v + 0.5, str(v), ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, get_output_path('nlp_ngrams_analysis.png'))
    print("✓ 图表已保存: output/nlp_ngrams_analysis.png")
    plt.show()

# 8.5 预处理步骤流程图（文本可视化）
fig, ax = create_subplots(1, 1, figsize=(12, 10))
ax.axis('off')

steps = [
    '1. 原始文本\nRaw Text',
    '2. 去除 URL、邮箱\nRemove URLs, Emails',
    '3. 转换小写\nLowercase',
    '4. 去除标点符号\nRemove Punctuation',
    '5. 分词\nTokenization',
    '6. 去除停用词\nRemove Stopwords',
    '7. 词形归一化\nNormalization',
    '8. 最终结果\nFinal Tokens'
]

y_positions = np.linspace(0.9, 0.1, len(steps))
colors_flow = plt.cm.Blues(np.linspace(0.3, 0.9, len(steps)))

for i, (step, y_pos, color) in enumerate(zip(steps, y_positions, colors_flow)):
    # 绘制步骤框
    bbox = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='black', linewidth=2)
    ax.text(0.5, y_pos, step, ha='center', va='center', fontsize=11,
            bbox=bbox, fontweight='bold')

    # 绘制箭头
    if i < len(steps) - 1:
        ax.annotate('', xy=(0.5, y_positions[i+1] + 0.04), xytext=(0.5, y_pos - 0.04),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('文本预处理流程图\nText Preprocessing Pipeline',
             fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
save_figure(fig, get_output_path('nlp_preprocessing_pipeline.png'))
print("✓ 图表已保存: output/nlp_preprocessing_pipeline.png")
plt.show()

# ============================================================================
# 9. 总结 / Summary
# ============================================================================
print("\n【9】总结")
print("=" * 80)
print("""
本教程涵盖的内容：
Topics Covered:

✓ 文本清洗（小写、去除URL、标点等）
  Text cleaning (lowercase, remove URLs, punctuation, etc.)

✓ 分词（英文、中文）
  Tokenization (English, Chinese)

✓ 停用词处理
  Stopwords removal

✓ 词形归一化（Stemming & Lemmatization）
  Word normalization (Stemming & Lemmatization)

✓ N-grams 生成
  N-grams generation

✓ 完整预处理 Pipeline
  Complete preprocessing pipeline

最佳实践建议：
Best Practices:

1. 根据任务选择合适的预处理步骤
   Choose preprocessing steps based on your task

2. 对于情感分析，保留标点可能有用
   For sentiment analysis, punctuation might be useful

3. 对于主题分类，去除停用词很重要
   For topic classification, removing stopwords is important

4. Lemmatization 比 Stemming 更准确但更慢
   Lemmatization is more accurate but slower than Stemming

5. 保存预处理后的数据以提高效率
   Save preprocessed data for efficiency

下一步学习：
Next Steps:
• 02_feature_extraction.py - 特征提取（TF-IDF、Word2Vec）
• 03_text_classification.py - 文本分类实战
""")

print("\n" + "=" * 80)
print("文本预处理教程完成！".center(80))
print("Text Preprocessing Tutorial Complete!".center(80))
print("=" * 80)
