# 自然語言處理基礎 | Natural Language Processing Basics

## 📖 模塊概述

本模塊涵蓋自然語言處理（NLP）的基礎知識和實踐，從文本預處理到文本分類的完整流程。

## 📂 文件結構

- `01_text_preprocessing.py` - 文本預處理（943行）
- `02_feature_extraction.py` - 特徵提取（794行）
- `03_text_classification.py` - 文本分類（931行）

## 🎯 學習路徑

### 1. 文本預處理
- 文本清洗和標準化
- 中英文分詞
- 詞幹提取和詞形還原
- 停用詞處理

### 2. 特徵提取
- Bag of Words (BoW)
- TF-IDF 權重計算
- Word2Vec 詞嵌入
- 文檔相似度

### 3. 文本分類
- 20 Newsgroups 數據集
- 5種分類器比較
- 完整 NLP Pipeline

## 🚀 快速開始

```bash
# 基礎運行（不需要可選依賴）
python 01_text_preprocessing.py

# 完整功能（需要安裝可選依賴）
pip install -r ../requirements-advanced.txt
python 02_feature_extraction.py
```

## 📊 生成的圖表

運行完成後，會在 `output/` 目錄生成約30張圖表。

## ⚙️ 依賴說明

**必需依賴：**
- scikit-learn（核心）

**可選依賴：**
- nltk（英文文本處理）
- jieba（中文分詞）
- gensim（Word2Vec）
- wordcloud（詞雲生成）

## 💡 最佳實踐

1. 根據語言選擇合適的分詞工具
2. 處理停用詞時注意業務場景
3. TF-IDF 通常優於 BoW
4. Word2Vec 需要大量數據

## 🔗 相關資源

- NLTK 文檔：https://www.nltk.org/
- Gensim 文檔：https://radimrehurek.com/gensim/
