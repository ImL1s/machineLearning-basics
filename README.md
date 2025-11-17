# 機器學習完整教程 | Machine Learning Complete Tutorial

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🎓 從零開始的機器學習完整學習路徑
> 📚 涵蓋基礎概念、經典算法、深度學習、實戰項目
> 💡 理論與實踐結合，包含完整代碼示例

[中文](#中文文檔) | [English](#english-documentation)

---

## 中文文檔

### 📖 專案簡介

這是一個**從入門到深入**的機器學習完整教程，適合：
- 🔰 機器學習初學者
- 💻 有編程基礎想學習 ML 的開發者
- 📊 數據分析師想轉型機器學習
- 🎯 想系統學習機器學習的學生

### ✨ 特色

- ✅ **完整的學習路徑**：從基礎到深度學習
- ✅ **豐富的實例**：每個算法都有完整代碼和可視化
- ✅ **詳細的註釋**：中英文雙語註釋，易於理解
- ✅ **最新的工具**：使用最新版本的 scikit-learn、TensorFlow
- ✅ **實戰導向**：包含真實數據集和實戰項目

### 📂 專案結構

```
machineLearning-basics/
│
├── 01_Basics/                          # 機器學習基礎
│   ├── 01_introduction.py              # ML 基本概念和術語
│   └── 02_numpy_pandas_basics.py       # NumPy 和 Pandas 基礎
│
├── 02_SupervisedLearning/              # 監督學習
│   ├── Classification/                 # 分類算法
│   │   ├── 01_knn_classifier.py        # K-近鄰算法
│   │   ├── 02_svm_classifier.py        # 支持向量機
│   │   └── 03_random_forest.py         # 隨機森林
│   └── Regression/                     # 回歸算法
│       └── 01_linear_regression.py     # 線性回歸系列
│
├── 03_UnsupervisedLearning/            # 非監督學習
│   ├── Clustering/                     # 聚類
│   │   └── 01_kmeans.py                # K-Means 聚類
│   └── DimensionalityReduction/        # 降維
│       └── 01_pca.py                   # 主成分分析
│
├── 04_FeatureEngineering/              # 特徵工程
│   └── feature_engineering_guide.py    # 特徵工程完整指南
│
├── 05_ModelEvaluation/                 # 模型評估與調參
│   └── model_evaluation_guide.py       # 評估和調參指南
│
├── 06_DeepLearning/                    # 深度學習
│   └── 01_keras_basics.py              # Keras/TensorFlow 基礎
│
├── 07_Projects/                        # 實戰項目（待添加）
│
├── DecisionTree/                       # 決策樹（原始項目，已優化）
│   ├── main.py                         # 決策樹完整示例
│   └── data.csv                        # 示例數據
│
├── requirements.txt                    # 依賴套件
├── .gitignore                          # Git 忽略文件
└── README.md                           # 本文件
```

### 🚀 快速開始

#### 1. 克隆專案

```bash
git clone https://github.com/你的用戶名/machineLearning-basics.git
cd machineLearning-basics
```

#### 2. 創建虛擬環境（推薦）

```bash
# 使用 venv
python -m venv venv

# 激活虛擬環境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

#### 3. 安裝依賴

```bash
pip install -r requirements.txt
```

#### 4. 運行示例

```bash
# 運行機器學習基礎教程
python 01_Basics/01_introduction.py

# 運行 KNN 分類器
python 02_SupervisedLearning/Classification/01_knn_classifier.py

# 運行決策樹示例
python DecisionTree/main.py
```

### 📚 學習路徑

#### 階段 1：基礎知識（1-2週）
1. **機器學習概念** → `01_Basics/01_introduction.py`
   - 了解 ML 基本概念
   - 監督學習 vs 非監督學習
   - 過擬合與欠擬合

2. **工具基礎** → `01_Basics/02_numpy_pandas_basics.py`
   - NumPy 數組操作
   - Pandas 數據處理
   - 數據可視化

#### 階段 2：監督學習（2-3週）
3. **分類算法**
   - K-近鄰（KNN）→ `02_SupervisedLearning/Classification/01_knn_classifier.py`
   - 支持向量機（SVM）→ `02_SupervisedLearning/Classification/02_svm_classifier.py`
   - 決策樹 → `DecisionTree/main.py`
   - 隨機森林 → `02_SupervisedLearning/Classification/03_random_forest.py`

4. **回歸算法**
   - 線性回歸 → `02_SupervisedLearning/Regression/01_linear_regression.py`
   - Ridge、Lasso、ElasticNet

#### 階段 3：非監督學習（1-2週）
5. **聚類**
   - K-Means → `03_UnsupervisedLearning/Clustering/01_kmeans.py`

6. **降維**
   - PCA → `03_UnsupervisedLearning/DimensionalityReduction/01_pca.py`

#### 階段 4：進階技巧（2週）
7. **特徵工程** → `04_FeatureEngineering/feature_engineering_guide.py`
   - 數據預處理
   - 特徵縮放
   - 特徵選擇

8. **模型評估** → `05_ModelEvaluation/model_evaluation_guide.py`
   - 評估指標
   - 交叉驗證
   - 超參數調優

#### 階段 5：深度學習入門（2-3週）
9. **神經網絡基礎** → `06_DeepLearning/01_keras_basics.py`
   - 全連接神經網絡（MLP）
   - 卷積神經網絡（CNN）
   - Keras/TensorFlow 使用

### 🔧 依賴套件

主要依賴：
- **NumPy**：數值計算
- **Pandas**：數據處理
- **Matplotlib**：數據可視化
- **scikit-learn**：機器學習算法
- **TensorFlow/Keras**：深度學習
- **XGBoost**：梯度提升
- **Seaborn**：統計可視化

完整依賴列表請查看 `requirements.txt`

### 📊 數據集

本教程使用的數據集：
- **Iris（鳶尾花）**：經典分類數據集
- **Wine（葡萄酒）**：多分類數據集
- **Breast Cancer（乳腺癌）**：二分類數據集
- **Diabetes（糖尿病）**：回歸數據集
- **Digits（手寫數字）**：圖像分類數據集

所有數據集都來自 scikit-learn 內置數據集，無需額外下載。

### 💡 學習建議

1. **按順序學習**：從基礎到深入，不要跳過基礎部分
2. **動手實踐**：每個示例都要親自運行和修改
3. **理解原理**：不僅要會用，還要理解算法原理
4. **可視化數據**：多用圖表理解數據和模型
5. **參數調整**：嘗試不同的參數，觀察效果變化
6. **對比實驗**：同一問題嘗試不同算法，比較性能

### 🎯 核心算法總結

| 算法 | 類型 | 適用場景 | 優點 | 缺點 |
|------|------|----------|------|------|
| KNN | 分類/回歸 | 小規模數據 | 簡單直觀 | 預測慢 |
| SVM | 分類/回歸 | 中小規模，高維 | 泛化能力強 | 大數據慢 |
| 決策樹 | 分類/回歸 | 需要可解釋性 | 易理解 | 容易過擬合 |
| 隨機森林 | 分類/回歸 | 表格數據 | 準確率高 | 模型大 |
| XGBoost | 分類/回歸 | 競賽/生產 | 性能最優 | 調參複雜 |
| 線性回歸 | 回歸 | 線性關係 | 簡單快速 | 只能線性 |
| K-Means | 聚類 | 球形簇 | 快速 | 需指定K |
| PCA | 降維 | 高維可視化 | 去相關 | 難解釋 |

### 🤝 貢獻

歡迎貢獻！請隨時：
- 🐛 報告 Bug
- 💡 提出新功能建議
- 📝 改進文檔
- 🔨 提交 Pull Request

### 📄 許可證

MIT License - 詳見 LICENSE 文件

### 📧 聯繫方式

如有問題或建議，歡迎通過以下方式聯繫：
- 提交 Issue
- 發送 Pull Request

---

## English Documentation

### 📖 Project Introduction

This is a **comprehensive machine learning tutorial from beginner to advanced**, suitable for:
- 🔰 ML beginners
- 💻 Developers with programming background wanting to learn ML
- 📊 Data analysts transitioning to machine learning
- 🎯 Students wanting systematic ML learning

### ✨ Features

- ✅ **Complete Learning Path**: From basics to deep learning
- ✅ **Rich Examples**: Complete code and visualizations for each algorithm
- ✅ **Detailed Comments**: Bilingual (Chinese/English) comments
- ✅ **Latest Tools**: Using latest versions of scikit-learn, TensorFlow
- ✅ **Practice-Oriented**: Including real datasets and practical projects

### 🚀 Quick Start

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/machineLearning-basics.git
cd machineLearning-basics
```

#### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run Examples

```bash
# Run ML basics tutorial
python 01_Basics/01_introduction.py

# Run KNN classifier
python 02_SupervisedLearning/Classification/01_knn_classifier.py
```

### 📚 Learning Path

1. **Basics** (1-2 weeks)
   - ML concepts and terminology
   - NumPy and Pandas basics

2. **Supervised Learning** (2-3 weeks)
   - Classification: KNN, SVM, Decision Trees, Random Forest
   - Regression: Linear Regression, Ridge, Lasso

3. **Unsupervised Learning** (1-2 weeks)
   - Clustering: K-Means
   - Dimensionality Reduction: PCA

4. **Advanced Techniques** (2 weeks)
   - Feature Engineering
   - Model Evaluation and Tuning

5. **Deep Learning Introduction** (2-3 weeks)
   - Neural Networks (MLP, CNN)
   - Keras/TensorFlow basics

### 🎯 Algorithm Summary

| Algorithm | Type | Use Case | Pros | Cons |
|-----------|------|----------|------|------|
| KNN | Classification/Regression | Small datasets | Simple, intuitive | Slow prediction |
| SVM | Classification/Regression | Medium, high-dim | Strong generalization | Slow for big data |
| Decision Tree | Classification/Regression | Interpretability needed | Easy to understand | Overfitting |
| Random Forest | Classification/Regression | Tabular data | High accuracy | Large model |
| Linear Regression | Regression | Linear relationships | Simple, fast | Linear only |
| K-Means | Clustering | Spherical clusters | Fast | Need to specify K |
| PCA | Dimensionality Reduction | High-dim visualization | Decorrelation | Hard to interpret |

### 🤝 Contributing

Contributions are welcome! Feel free to:
- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🔨 Submit pull requests

### 📄 License

MIT License - see LICENSE file for details

---

## 🌟 Star History

如果這個專案對你有幫助，請給個 ⭐ Star！
If this project helps you, please give it a ⭐ Star!

---

**Happy Learning! 祝學習愉快！** 🎉
