# 機器學習完整教程 | Machine Learning Complete Tutorial

<div align="center">

**從零到深入的機器學習全攻略：理論、算法與實踐**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Build Status](https://github.com/ImL1s/machineLearning-basics/actions/workflows/ci.yml/badge.svg)](https://github.com/ImL1s/machineLearning-basics/actions/workflows/ci.yml)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 目錄 (Table of Contents)
- [✨ 專案特色](#-特色)
- [🚀 快速開始](#-快速開始)
- [🛠️ 依賴安裝指南](#-依賴安裝指南)
- [📚 學習路徑分階段詳解](#-學習路徑)
- [🕐 時間序列分析 (TimeSeries)](#🕐-時間序列分析模塊3-python--4-md2706-行代碼)
- [📝 自然語言處理 (NLP)](#📝-自然語言處理基礎模塊3-python2668-行代碼)
- [📊 算法摘要與對比](#-核心算法總結)

---

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
├── 00_QuickStart/                      # ⭐ 快速入門
│   └── quick_start_guide.py            # 5分鐘快速上手指南
│
├── 01_Basics/                          # 機器學習基礎
│   ├── 01_introduction.py              # ML 基本概念和術語
│   ├── 02_numpy_pandas_basics.py       # NumPy 和 Pandas 基礎
│   └── 03_data_visualization.py        # ⭐ 數據可視化完整教程
│
├── 02_SupervisedLearning/              # 監督學習
│   ├── Classification/                 # 分類算法
│   │   ├── 01_knn_classifier.py        # K-近鄰算法
│   │   ├── 02_svm_classifier.py        # 支持向量機
│   │   ├── 03_random_forest.py         # 隨機森林
│   │   ├── 04_logistic_regression.py   # 邏輯回歸
│   │   ├── 05_naive_bayes.py           # 樸素貝葉斯
│   │   └── 06_gradient_boosting_xgboost.py  # 梯度提升/XGBoost
│   ├── Regression/                     # 回歸算法
│   │   ├── 01_linear_regression.py     # 線性回歸系列
│   │   ├── 02_nonlinear_regression.py  # 🆕 非線性回歸（多項式、SVR、樣條）
│   │   └── 03_tree_based_regression.py # 🆕 樹模型回歸（RF、GBDT、XGBoost、LightGBM）
│   └── Ensemble/                       # 🆕 集成學習（新增）
│       ├── 01_voting_classifier.py     # 投票分類器
│       ├── 02_adaboost.py              # AdaBoost 算法
│       └── 03_stacking.py              # 堆疊集成
│
├── 03_UnsupervisedLearning/            # 非監督學習
│   ├── Clustering/                     # 聚類
│   │   ├── 01_kmeans.py                # K-Means 聚類
│   │   ├── 02_dbscan.py                # 🆕 DBSCAN 密度聚類（新增）
│   │   └── 03_hierarchical.py          # 🆕 層次聚類（新增）
│   ├── DimensionalityReduction/        # 降維
│   │   ├── 01_pca.py                   # 主成分分析
│   │   └── 02_tsne_umap.py             # 🆕 t-SNE 和 UMAP（新增）
│   └── AnomalyDetection/               # 🆕 異常檢測（新增）
│       ├── 01_isolation_forest.py      # 孤立森林
│       └── 02_one_class_svm.py         # One-Class SVM
│
├── 04_FeatureEngineering/              # 特徵工程
│   ├── feature_engineering_guide.py    # 特徵工程完整指南
│   └── handling_imbalanced_data.py     # ⭐ 處理不平衡數據
│
├── 05_ModelEvaluation/                 # 模型評估與調參
│   ├── model_evaluation_guide.py       # 評估和調參指南
│   ├── model_persistence.py            # 模型保存和加載
│   ├── pipeline_guide.py               # ⭐ Pipeline 完整使用指南
│   └── model_interpretability.py       # 🆕 模型解釋性（SHAP/LIME）（新增）
│
├── 06_DeepLearning/                    # 深度學習
│   └── 01_keras_basics.py              # Keras/TensorFlow 基礎
│
├── 07_Projects/                        # 實戰項目
│   ├── 01_titanic_survival_prediction.py  # 泰坦尼克號生存預測
│   ├── 02_house_price_prediction.py    # 🆕 房價預測（回歸項目）
│   └── 03_customer_churn_prediction.py # 🆕 客戶流失預測（不平衡分類）
│
├── 08_TipsAndTricks/                   # ⭐ 技巧與最佳實踐
│   └── common_mistakes_and_debugging.md    # 常見錯誤和調試指南
│
├── 09_TimeSeries/                      # 🆕 時間序列分析（新增）
│   ├── 01_time_series_basics.py        # 時間序列基礎（平穩性、ACF/PACF）
│   ├── 02_arima.py                     # ARIMA 模型（含 SARIMA）
│   ├── 03_forecasting_methods.py       # 預測方法大比拼（14種方法）
│   ├── README.md                       # 模塊完整指南
│   ├── QUICK_START.md                  # 快速入門
│   ├── PROJECT_SUMMARY.md              # 項目總結
│   └── VISUALIZATION_INDEX.md          # 可視化索引
│
├── 10_NLP/                             # 🆕 自然語言處理基礎（新增）
│   ├── 01_text_preprocessing.py        # 文本預處理（中英文支持）
│   ├── 02_feature_extraction.py        # 特徵提取（BoW、TF-IDF、Word2Vec）
│   └── 03_text_classification.py       # 文本分類（20 Newsgroups）
│
├── 11_Resources/                       # 🆕 學習資源（新增）
│   ├── algorithm_cheatsheet.md         # 算法速查表（20+算法對比）
│   ├── sklearn_cheatsheet.md           # Sklearn API 速查表
│   ├── best_practices.md               # 最佳實踐指南
│   ├── learning_roadmap.md             # 完整學習路線圖
│   └── interview_questions.md          # 面試題集（150+問題）
│
├── DecisionTree/                       # 決策樹（原始項目，已優化）
│   ├── main.py                         # 決策樹完整示例
│   └── data.csv                        # 示例數據
│
├── utils/                              # 🆕 工具模塊（新增）
│   ├── __init__.py                     # 模塊初始化
│   ├── config.py                       # 統一配置管理
│   ├── paths.py                        # 路徑管理
│   ├── plotting.py                     # 繪圖工具
│   └── README.md                       # 工具使用指南
│
├── requirements.txt                    # 核心依賴
├── requirements-ml.txt                 # 🆕 機器學習擴展依賴
├── requirements-dl.txt                 # 🆕 深度學習依賴
├── requirements-advanced.txt           # 🆕 高級工具依賴
├── requirements-dev.txt                # 🆕 開發工具依賴
├── .gitignore                          # Git 忽略文件
├── LICENSE                             # MIT 許可證
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

### 🛠️ 依賴安裝指南 (Installation Matrix)

我們提供了分層的依賴文件，請根據您的學習階段選擇：

| 安裝模式 (Mode) | 命令 (Command) | 包含功能 (Features) | 推薦對象 |
| :--- | :--- | :--- | :--- |
| **最小安裝** | `pip install -r requirements.txt` | NumPy, Pandas, Scikit-Learn | 初學者/基礎教程 |
| **機器學習加強** | `... -r requirements-ml.txt` | XGBoost, LightGBM, Imbal-learn | 參與競賽/進階者 |
| **深度學習擴充** | `... -r requirements-dl.txt` | TensorFlow / Keras (較大) | 深度學習研究者 |
| **專業開發模式** | `... -r requirements-dev.txt` | JupyterLab, Notebook, Pytest | 開發與筆記愛好者 |
| **全方位安裝** | `... -r requirements-advanced.txt` | SHAP, LIME, Optuna | 高級分析師 |

---

## 🧠 核心基礎與深層原理 (Deep Dive)

### 1. 機器學習的數學核心 (Mathematical Foundations)
要真正理解 ML，必須掌握以下三大支柱：
- **線性代數**：矩陣運算是一切算法的語言（如 PCA 的特徵值分解）。
- **微積分**：**梯度下降 (Gradient Descent)** 依賴於對損失函數求導，找出參數的最優解。
- **機率統計**：理解正則化、極大似然估計 (MLE) 以及貝葉斯推斷。

### 2. 反向傳播 (Backpropagation) 運作原理
在深度學習中，模型如何「學習」？
- **前向傳播**：數據經過神經網路得到預測值。
- **計算損失**：衡量預測與現實的差距。
- **鏈式法則 (Chain Rule)**：將誤差從輸出層反向傳回輸入層，更新每一層的權重。本專案的 `06_DeepLearning` 章節中有手動實現簡易神經網路的範例。

---

#### 4. 運行示例

```bash
# ⭐ 新手推薦：5分鐘快速入門
python 00_QuickStart/quick_start_guide.py

# 運行機器學習基礎教程
python 01_Basics/01_introduction.py

# 運行數據可視化教程
python 01_Basics/03_data_visualization.py

# 運行 KNN 分類器
python 02_SupervisedLearning/Classification/01_knn_classifier.py

# 運行決策樹示例
python DecisionTree/main.py
```

### 📚 學習路徑

#### 階段 0：快速入門（0.5小時）⭐ 新增
0. **5分鐘快速上手** → `00_QuickStart/quick_start_guide.py`
   - 完整機器學習工作流程
   - 從數據加載到模型預測
   - 快速體驗 ML 魅力

#### 階段 1：基礎知識（1-2週）
1. **機器學習概念** → `01_Basics/01_introduction.py`
   - 了解 ML 基本概念
   - 監督學習 vs 非監督學習
   - 過擬合與欠擬合

2. **工具基礎** → `01_Basics/02_numpy_pandas_basics.py`
   - NumPy 數組操作
   - Pandas 數據處理

3. **數據可視化** → `01_Basics/03_data_visualization.py` ⭐ 新增
   - Matplotlib 基礎繪圖
   - Seaborn 統計圖表
   - ML 專用可視化（決策邊界、學習曲線等）

#### 階段 2：監督學習（3-4週）
3. **分類算法**
   - K-近鄰（KNN）→ `02_SupervisedLearning/Classification/01_knn_classifier.py`
   - 支持向量機（SVM）→ `02_SupervisedLearning/Classification/02_svm_classifier.py`
   - 決策樹 → `DecisionTree/main.py`
   - 隨機森林 → `02_SupervisedLearning/Classification/03_random_forest.py`
   - 邏輯回歸 → `02_SupervisedLearning/Classification/04_logistic_regression.py`
   - 樸素貝葉斯 → `02_SupervisedLearning/Classification/05_naive_bayes.py`
   - 梯度提升/XGBoost → `02_SupervisedLearning/Classification/06_gradient_boosting_xgboost.py`

4. **回歸算法**
   - 線性回歸 → `02_SupervisedLearning/Regression/01_linear_regression.py`
   - Ridge、Lasso、ElasticNet
   - 🆕 非線性回歸 → `02_SupervisedLearning/Regression/02_nonlinear_regression.py`
   - 🆕 樹模型回歸 → `02_SupervisedLearning/Regression/03_tree_based_regression.py`

5. **🆕 集成學習**（新增）
   - 投票分類器 → `02_SupervisedLearning/Ensemble/01_voting_classifier.py`
   - AdaBoost → `02_SupervisedLearning/Ensemble/02_adaboost.py`
   - Stacking → `02_SupervisedLearning/Ensemble/03_stacking.py`

#### 階段 3：非監督學習（2-3週）
6. **聚類**
   - K-Means → `03_UnsupervisedLearning/Clustering/01_kmeans.py`
   - 🆕 DBSCAN（密度聚類）→ `03_UnsupervisedLearning/Clustering/02_dbscan.py`
   - 🆕 層次聚類 → `03_UnsupervisedLearning/Clustering/03_hierarchical.py`

7. **降維**
   - PCA → `03_UnsupervisedLearning/DimensionalityReduction/01_pca.py`
   - 🆕 t-SNE 和 UMAP → `03_UnsupervisedLearning/DimensionalityReduction/02_tsne_umap.py`

8. **🆕 異常檢測**（新增）
   - Isolation Forest → `03_UnsupervisedLearning/AnomalyDetection/01_isolation_forest.py`
   - One-Class SVM → `03_UnsupervisedLearning/AnomalyDetection/02_one_class_svm.py`

#### 階段 4：進階技巧（3-4週）
9. **特徵工程** → `04_FeatureEngineering/feature_engineering_guide.py`
   - 數據預處理
   - 特徵縮放
   - 特徵選擇
   - 處理不平衡數據 → `04_FeatureEngineering/handling_imbalanced_data.py` ⭐

10. **模型評估與調優**
    - 評估指標 → `05_ModelEvaluation/model_evaluation_guide.py`
    - 交叉驗證
    - 超參數調優
    - Pipeline 完整指南 → `05_ModelEvaluation/pipeline_guide.py` ⭐
    - 模型保存和加載 → `05_ModelEvaluation/model_persistence.py`
    - 🆕 模型解釋性（SHAP/LIME）→ `05_ModelEvaluation/model_interpretability.py`

#### 階段 5：深度學習入門（2-3週）
11. **神經網絡基礎** → `06_DeepLearning/01_keras_basics.py`
    - 全連接神經網絡（MLP）
    - 卷積神經網絡（CNN）
    - Keras/TensorFlow 使用

#### 階段 6：時間序列分析（2-3週）🆕 新增
12. **時間序列基礎** → `09_TimeSeries/01_time_series_basics.py`
    - 平穩性檢驗（ADF、KPSS）
    - ACF/PACF 分析
    - 差分和季節性分解

13. **ARIMA 模型** → `09_TimeSeries/02_arima.py`
    - ARIMA 參數選擇
    - SARIMA 季節性模型
    - Auto ARIMA 自動調參

14. **預測方法比較** → `09_TimeSeries/03_forecasting_methods.py`
    - 14種預測方法（統計方法 + ML方法）
    - 性能基準測試
    - 模型選擇指南

#### 階段 7：自然語言處理基礎（2-3週）🆕 新增
15. **文本預處理** → `10_NLP/01_text_preprocessing.py`
    - 文本清洗和標準化
    - 分詞（中英文）
    - 詞幹提取和詞形還原

16. **特徵提取** → `10_NLP/02_feature_extraction.py`
    - Bag of Words（BoW）
    - TF-IDF 權重
    - Word2Vec 詞嵌入

17. **文本分類** → `10_NLP/03_text_classification.py`
    - 20 Newsgroups 數據集
    - 5種分類器比較
    - 完整的NLP流程

#### 階段 8：最佳實踐與實戰（2-3週）⭐
18. **實戰項目**
    - 泰坦尼克號生存預測 → `07_Projects/01_titanic_survival_prediction.py`
    - 🆕 房價預測 → `07_Projects/02_house_price_prediction.py`
    - 🆕 客戶流失預測 → `07_Projects/03_customer_churn_prediction.py`

19. **技巧與避坑指南** → `08_TipsAndTricks/common_mistakes_and_debugging.md` ⭐
    - 常見錯誤和解決方案
    - 數據洩漏、過擬合等問題
    - 調試技巧和最佳實踐
    - 生產環境部署注意事項

20. **學習資源** → `11_Resources/` 🆕 新增
    - 算法速查表 → `algorithm_cheatsheet.md`
    - Sklearn API 參考 → `sklearn_cheatsheet.md`
    - 最佳實踐 → `best_practices.md`
    - 學習路線圖 → `learning_roadmap.md`
    - 面試題集 → `interview_questions.md`（150+問題）

### 🆕 本次更新亮點（Round 4）

**新增 15 個 Python 文件 + 9 個 Markdown 文檔，共 9,000+ 行代碼，80+ 張圖表**

#### 1. 🕐 時間序列分析模塊（3 Python + 4 MD，2,706 行代碼）

- **時間序列基礎** `09_TimeSeries/01_time_series_basics.py` (717 行)
  - 平穩性檢驗（ADF、KPSS）
  - ACF/PACF 自相關分析
  - 移動平均和指數平滑
  - 8 張專業圖表

- **ARIMA 模型** `09_TimeSeries/02_arima.py` (825 行)
  - ARIMA 模型實現和參數選擇
  - SARIMA 季節性建模
  - Auto ARIMA 自動調參
  - 9 張預測和診斷圖表

- **預測方法大比拼** `09_TimeSeries/03_forecasting_methods.py` (1,164 行)
  - 14+ 種預測方法（Naive、MA、ES、ARIMA、ML方法）
  - 完整性能基準測試
  - 模型選擇決策樹
  - 12 張對比圖表

- **完整文檔**
  - README.md（模塊指南）
  - QUICK_START.md（快速入門）
  - PROJECT_SUMMARY.md（項目總結）
  - VISUALIZATION_INDEX.md（可視化索引）

#### 2. 📝 自然語言處理基礎模塊（3 Python，2,668 行代碼）

- **文本預處理** `10_NLP/01_text_preprocessing.py` (943 行)
  - 文本清洗和標準化
  - 中英文分詞支持
  - 詞幹提取（Stemming）和詞形還原（Lemmatization）
  - 停用詞處理
  - 8 張文本分析圖表

- **特徵提取** `10_NLP/02_feature_extraction.py` (794 行)
  - Bag of Words（BoW）
  - TF-IDF 權重計算
  - Word2Vec 詞嵌入
  - 文檔相似度計算
  - 10 張特徵可視化圖表

- **文本分類** `10_NLP/03_text_classification.py` (931 行)
  - 使用 20 Newsgroups 數據集
  - 5 種分類器比較（Naive Bayes、Logistic Regression、SVM 等）
  - 完整的 NLP Pipeline
  - 12 張分類結果圖表

#### 3. 📈 回歸算法擴展（2 Python，~1,400 行代碼）

- **非線性回歸** `02_SupervisedLearning/Regression/02_nonlinear_regression.py` (~650 行)
  - 多項式回歸（不同階數對比）
  - 支持向量回歸（SVR）
  - 樣條回歸
  - 過擬合分析
  - 7 張回歸分析圖表

- **樹模型回歸** `02_SupervisedLearning/Regression/03_tree_based_regression.py` (~750 行)
  - Decision Tree 回歸
  - Random Forest 回歸
  - Gradient Boosting 回歸
  - XGBoost 回歸
  - LightGBM 回歸
  - 特徵重要性對比
  - 10 張模型對比圖表

#### 4. 🏆 新實戰項目（2 Python，2,306 行代碼）

- **房價預測** `07_Projects/02_house_price_prediction.py` (1,071 行)
  - 完整的回歸項目流程
  - 8 種回歸模型比較
  - 特徵工程（10+ 新特徵）
  - 模型融合
  - 業務洞察和建議
  - 21 張專業圖表

- **客戶流失預測** `07_Projects/03_customer_churn_prediction.py` (1,235 行)
  - 不平衡分類問題處理
  - 4 種採樣策略（原始、class weight、SMOTE、欠採樣）
  - 業務 ROI 分析
  - 客戶細分和保留策略
  - 21 張業務分析圖表

#### 5. 📚 學習資源（5 Markdown，203KB）

- **算法速查表** `11_Resources/algorithm_cheatsheet.md` (22KB)
  - 20+ 算法對比表
  - 算法選擇決策樹
  - 參數調優指南

- **Sklearn API 速查表** `11_Resources/sklearn_cheatsheet.md` (32KB)
  - 常用 API 快速參考
  - 完整代碼示例
  - Pipeline 最佳實踐

- **最佳實踐指南** `11_Resources/best_practices.md` (34KB)
  - 數據預處理最佳實踐
  - 模型選擇和調優
  - 生產環境部署
  - 常見陷阱和避坑指南

- **學習路線圖** `11_Resources/learning_roadmap.md` (36KB)
  - 完整學習路徑規劃
  - 每階段學習建議
  - 實踐項目推薦

- **面試題集** `11_Resources/interview_questions.md` (79KB)
  - 150+ 機器學習面試問題
  - 詳細答案和解釋
  - 代碼實現示例
  - 使用 `<details>` 標籤可折疊

---

### 📊 專案統計（更新後）

- **Python 文件**：44 個
- **代碼總量**：48,000+ 行
- **可視化圖表**：180+ 張
- **文檔**：12 份 Markdown 文檔
- **覆蓋算法**：30+ 種機器學習算法
- **實戰項目**：3 個完整項目
- **學習資源**：5 份速查表和指南（203KB）

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

**分類數據集**
- **Iris（鳶尾花）**：經典分類數據集
- **Wine（葡萄酒）**：多分類數據集
- **Breast Cancer（乳腺癌）**：二分類數據集
- **Digits（手寫數字）**：圖像分類數據集

**回歸數據集**
- **Diabetes（糖尿病）**：回歸數據集
- **California Housing（加州房價）**：🆕 房價預測數據集
- **Boston Housing（波士頓房價）**：房價分析數據集

**時間序列數據集** 🆕
- **航空乘客數據**：經典時間序列數據
- **CO2 數據**：季節性時間序列

**NLP 數據集** 🆕
- **20 Newsgroups**：文本分類數據集
- **SMS Spam Collection**：垃圾郵件分類

大部分數據集來自 scikit-learn 內置數據集，無需額外下載。

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
| **分類算法** |
| KNN | 分類/回歸 | 小規模數據 | 簡單直觀 | 預測慢 |
| Logistic Regression | 分類 | 線性可分 | 快速，可解釋 | 只能線性 |
| Naive Bayes | 分類 | 文本分類 | 極快，高維 | 假設特徵獨立 |
| SVM | 分類/回歸 | 中小規模，高維 | 泛化能力強 | 大數據慢 |
| 決策樹 | 分類/回歸 | 需要可解釋性 | 易理解 | 容易過擬合 |
| 隨機森林 | 分類/回歸 | 表格數據 | 準確率高 | 模型大 |
| XGBoost/Gradient Boosting | 分類/回歸 | 競賽/生產 | 性能最優 | 調參複雜 |
| **回歸算法** |
| 線性回歸 | 回歸 | 線性關係 | 簡單快速 | 只能線性 |
| Ridge/Lasso | 回歸 | 需要正則化 | 防止過擬合 | 需調參 |
| 🆕 多項式回歸 | 回歸 | 非線性關係 | 靈活 | 易過擬合 |
| 🆕 SVR | 回歸 | 非線性，高維 | 泛化能力強 | 計算慢 |
| **集成學習** |
| 🆕 Voting | 分類/回歸 | 模型融合 | 提高穩定性 | 計算開銷大 |
| 🆕 AdaBoost | 分類/回歸 | 弱學習器提升 | 高準確率 | 對噪聲敏感 |
| 🆕 Stacking | 分類/回歸 | 多層集成 | 性能優異 | 複雜度高 |
| **聚類算法** |
| K-Means | 聚類 | 球形簇 | 快速 | 需指定K |
| 🆕 DBSCAN | 聚類 | 任意形狀簇 | 自動檢測簇數 | 參數敏感 |
| 🆕 層次聚類 | 聚類 | 需要樹狀圖 | 不需指定K | 計算慢 |
| **降維算法** |
| PCA | 降維 | 高維可視化 | 去相關 | 難解釋 |
| 🆕 t-SNE | 降維可視化 | 非線性降維 | 保留局部結構 | 計算慢 |
| 🆕 UMAP | 降維可視化 | 大規模數據 | 快速，保留結構 | 參數多 |
| **異常檢測** |
| 🆕 Isolation Forest | 異常檢測 | 高維數據 | 快速，無監督 | 需調參 |
| 🆕 One-Class SVM | 異常檢測 | 小樣本 | 靈活 | 計算慢 |
| **時間序列** |
| 🆕 ARIMA | 時間序列預測 | 平穩序列 | 經典方法 | 需平穩性 |
| 🆕 SARIMA | 時間序列預測 | 季節性數據 | 處理季節性 | 參數複雜 |
| **NLP** |
| 🆕 TF-IDF | 文本特徵 | 文本分類 | 簡單有效 | 不考慮語義 |
| 🆕 Word2Vec | 詞嵌入 | 語義相似度 | 捕捉語義 | 需大量數據 |

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

We provide layered dependency files. Choose based on your needs:

**Option 1: Minimal Installation (Recommended for beginners, ~2 minutes)**
```bash
pip install -r requirements.txt
```
Includes: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn

**Option 2: Full ML Features (~3 minutes)**
```bash
pip install -r requirements.txt -r requirements-ml.txt
```
Additionally includes: XGBoost, LightGBM, imbalanced-learn

**Option 3: With Deep Learning (~10 minutes, large)**
```bash
pip install -r requirements.txt -r requirements-dl.txt
```
Additionally includes: TensorFlow/Keras (~500MB)

**Option 4: Development Environment (with Jupyter)**
```bash
pip install -r requirements.txt -r requirements-dev.txt
```
Additionally includes: JupyterLab, Notebook

**Option 5: Complete Installation (all features, ~15 minutes)**
```bash
pip install -r requirements.txt -r requirements-ml.txt -r requirements-dl.txt -r requirements-advanced.txt
```
Includes all tools (SHAP, LIME, Optuna, etc.)

#### 4. Run Examples

```bash
# ⭐ Recommended for beginners: 5-minute quick start
python 00_QuickStart/quick_start_guide.py

# Run ML basics tutorial
python 01_Basics/01_introduction.py

# Run data visualization tutorial
python 01_Basics/03_data_visualization.py

# Run KNN classifier
python 02_SupervisedLearning/Classification/01_knn_classifier.py
```

### 📚 Learning Path

**Stage 0: Quick Start** (0.5 hour) ⭐ New
- 5-minute ML workflow → `00_QuickStart/quick_start_guide.py`

**Stage 1: Basics** (1-2 weeks)
- ML concepts and terminology → `01_Basics/01_introduction.py`
- NumPy and Pandas basics → `01_Basics/02_numpy_pandas_basics.py`
- Data Visualization ⭐ New → `01_Basics/03_data_visualization.py`

**Stage 2: Supervised Learning** (2-3 weeks)
- Classification: KNN, SVM, Decision Trees, Random Forest, Logistic Regression, Naive Bayes, XGBoost
- Regression: Linear Regression, Ridge, Lasso

**Stage 3: Unsupervised Learning** (1-2 weeks)
- Clustering: K-Means
- Dimensionality Reduction: PCA

**Stage 4: Advanced Techniques** (2 weeks)
- Feature Engineering
- Handling Imbalanced Data ⭐ New → `04_FeatureEngineering/handling_imbalanced_data.py`
- Pipeline Guide ⭐ New → `05_ModelEvaluation/pipeline_guide.py`
- Model Evaluation and Tuning

**Stage 5: Deep Learning Introduction** (2-3 weeks)
- Neural Networks (MLP, CNN)
- Keras/TensorFlow basics

**Stage 6: Best Practices & Projects** (1-2 weeks) ⭐ New
- Real-world project: Titanic Survival Prediction
- Common Mistakes & Debugging ⭐ New → `08_TipsAndTricks/common_mistakes_and_debugging.md`

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
