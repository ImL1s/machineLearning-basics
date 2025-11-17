# 常見錯誤和調試指南
# Common Mistakes and Debugging Guide

機器學習新手常犯的錯誤和解決方案

---

## 📋 目錄

1. [數據相關錯誤](#1-數據相關錯誤)
2. [特徵工程錯誤](#2-特徵工程錯誤)
3. [模型訓練錯誤](#3-模型訓練錯誤)
4. [評估錯誤](#4-評估錯誤)
5. [代碼錯誤](#5-代碼錯誤)
6. [性能問題](#6-性能問題)
7. [部署問題](#7-部署問題)

---

## 1. 數據相關錯誤

### ❌ 錯誤 1.1：數據洩漏

**問題：**
```python
# 錯誤：在分割數據前進行標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 用所有數據 fit
X_train, X_test = train_test_split(X_scaled, y)
```

**為什麼錯誤：**
- 測試集信息洩漏到訓練過程
- 模型性能被高估
- 生產環境性能會下降

**✅ 正確做法：**
```python
# 正確：先分割，再標準化
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 只用訓練集 fit
X_test_scaled = scaler.transform(X_test)       # 測試集只 transform
```

---

### ❌ 錯誤 1.2：忘記處理缺失值

**問題：**
```python
# 錯誤：直接使用有缺失值的數據
model.fit(X_train, y_train)  # X_train 有 NaN
# 報錯：Input contains NaN, infinity or a value too large
```

**✅ 正確做法：**
```python
from sklearn.impute import SimpleImputer

# 檢查缺失值
print(df.isnull().sum())

# 處理缺失值
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

---

### ❌ 錯誤 1.3：沒有打亂數據

**問題：**
```python
# 錯誤：數據是排序的，直接分割
# 假設前 80% 都是類別 0，後 20% 都是類別 1
X_train, X_test = X[:800], X[800:]
```

**✅ 正確做法：**
```python
# 使用 train_test_split 自動打亂
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,    # 默認就是 True
    stratify=y       # 保持類別比例
)
```

---

## 2. 特徵工程錯誤

### ❌ 錯誤 2.1：忘記編碼類別特徵

**問題：**
```python
# 錯誤：直接使用字符串類別
df['gender'] = ['male', 'female', 'male', ...]
model.fit(df, y)  # 報錯
```

**✅ 正確做法：**
```python
# 方法1：Label Encoding（有序類別）
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])

# 方法2：One-Hot Encoding（無序類別，推薦）
df_encoded = pd.get_dummies(df, columns=['gender'])
```

---

### ❌ 錯誤 2.2：特徵縮放不一致

**問題：**
```python
# 錯誤：訓練和測試用不同的縮放器
scaler1 = StandardScaler().fit(X_train)
scaler2 = StandardScaler().fit(X_test)  # 錯誤！

X_train_scaled = scaler1.transform(X_train)
X_test_scaled = scaler2.transform(X_test)
```

**✅ 正確做法：**
```python
# 正確：使用同一個縮放器
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 只 transform

# 最佳：使用 Pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```

---

### ❌ 錯誤 2.3：創建目標相關的特徵

**問題：**
```python
# 錯誤：特徵包含目標信息
df['total_price'] = df['unit_price'] * df['quantity']
# 但 quantity 是我們要預測的目標！
```

**✅ 正確做法：**
- 仔細檢查特徵是否包含未來信息
- 使用時間戳確保特徵在預測時可用
- 避免使用目標變量的衍生特徵

---

## 3. 模型訓練錯誤

### ❌ 錯誤 3.1：過擬合

**症狀：**
```python
訓練準確率：99%
測試準確率：70%  # 差距太大！
```

**原因：**
- 模型過於複雜
- 訓練時間過長
- 數據量太小

**✅ 解決方案：**
```python
# 1. 使用正則化
model = LogisticRegression(C=0.1)  # 更強的正則化

# 2. 減少模型複雜度
rf = RandomForestClassifier(max_depth=5)  # 限制深度

# 3. 增加訓練數據
# 4. 使用 Dropout（深度學習）
# 5. Early Stopping
```

---

### ❌ 錯誤 3.2：欠擬合

**症狀：**
```python
訓練準確率：60%
測試準確率：58%  # 都很低
```

**原因：**
- 模型過於簡單
- 特徵不足
- 訓練時間不夠

**✅ 解決方案：**
```python
# 1. 增加模型複雜度
rf = RandomForestClassifier(max_depth=None, n_estimators=200)

# 2. 添加更多特徵
# 3. 特徵工程
# 4. 嘗試更複雜的模型
```

---

### ❌ 錯誤 3.3：忘記設置 random_state

**問題：**
```python
# 錯誤：每次運行結果不同
model = RandomForestClassifier()
model.fit(X_train, y_train)
# 每次分數都不一樣，無法復現
```

**✅ 正確做法：**
```python
# 設置隨機種子，保證可復現
model = RandomForestClassifier(random_state=42)
X_train, X_test = train_test_split(X, y, random_state=42)
```

---

## 4. 評估錯誤

### ❌ 錯誤 4.1：在訓練集上評估

**問題：**
```python
# 錯誤：在訓練集上評估
model.fit(X_train, y_train)
score = model.score(X_train, y_train)  # 錯誤！
print(f"模型準確率：{score}")  # 過於樂觀
```

**✅ 正確做法：**
```python
# 正確：在測試集上評估
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)  # 正確
train_score = model.score(X_train, y_train)  # 可選，檢查過擬合

print(f"訓練準確率：{train_score}")
print(f"測試準確率：{test_score}")
```

---

### ❌ 錯誤 4.2：使用錯誤的評估指標

**問題：**
```python
# 不平衡數據（95% vs 5%）只看準確率
accuracy = 0.95  # 看起來很好
# 但可能模型全預測為多數類！
```

**✅ 正確做法：**
```python
from sklearn.metrics import f1_score, precision_score, recall_score

# 不平衡數據使用多種指標
print(f"F1 Score: {f1_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")

# 查看混淆矩陣
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
```

---

### ❌ 錯誤 4.3：忽略交叉驗證

**問題：**
```python
# 錯誤：只用一次分割評估
X_train, X_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
# 結果不穩定，可能運氣好/壞
```

**✅ 正確做法：**
```python
from sklearn.model_selection import cross_val_score

# 使用交叉驗證
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

---

## 5. 代碼錯誤

### ❌ 錯誤 5.1：形狀不匹配

**問題：**
```python
# 錯誤：sklearn 需要 2D 數組
X = df['feature'].values  # 1D array
model.fit(X, y)  # ValueError: Expected 2D array
```

**✅ 正確做法：**
```python
# 方法1：使用 reshape
X = df['feature'].values.reshape(-1, 1)

# 方法2：使用 DataFrame
X = df[['feature']]  # 雙括號返回 DataFrame

# 方法3：使用 numpy
X = df['feature'].values[:, np.newaxis]
```

---

### ❌ 錯誤 5.2：內存洩漏

**問題：**
```python
# 錯誤：在循環中不斷創建大對象
for i in range(1000):
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(X_train, y_train)
    # 沒有釋放模型，內存不斷增加
```

**✅ 正確做法：**
```python
import gc

for i in range(1000):
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(X_train, y_train)
    # 使用後清理
    del model
    gc.collect()
```

---

### ❌ 錯誤 5.3：版本不兼容

**問題：**
```python
# 用 sklearn 1.0 訓練的模型
# 用 sklearn 0.24 加載
model = joblib.load('model.pkl')  # 可能出錯
```

**✅ 正確做法：**
```python
# 保存版本信息
import sklearn
metadata = {
    'sklearn_version': sklearn.__version__,
    'created_at': datetime.now(),
    'model_type': 'RandomForest'
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f)

# 使用相同版本加載
```

---

## 6. 性能問題

### ❌ 錯誤 6.1：沒有使用向量化

**問題：**
```python
# 慢：使用 Python 循環
result = []
for i in range(len(X)):
    result.append(X[i] ** 2)
```

**✅ 正確做法：**
```python
# 快：使用 NumPy 向量化
result = X ** 2  # 快 100 倍以上
```

---

### ❌ 錯誤 6.2：沒有並行處理

**問題：**
```python
# 慢：單線程訓練
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train, y_train)
```

**✅ 正確做法：**
```python
# 快：使用所有 CPU 核心
rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
grid = GridSearchCV(model, params, n_jobs=-1)
```

---

## 7. 部署問題

### ❌ 錯誤 7.1：只保存模型

**問題：**
```python
# 錯誤：只保存模型，忘記保存預處理器
joblib.dump(model, 'model.pkl')
# 生產環境不知道如何預處理新數據
```

**✅ 正確做法：**
```python
# 方法1：保存 Pipeline
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipe.fit(X_train, y_train)
joblib.dump(pipe, 'pipeline.pkl')

# 方法2：分別保存
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model.pkl')
```

---

### ❌ 錯誤 7.2：不處理新數據的特殊情況

**問題：**
```python
# 生產環境遇到訓練時沒見過的類別
# 模型崩潰
```

**✅ 正確做法：**
```python
from sklearn.preprocessing import OneHotEncoder

# 使用 handle_unknown
encoder = OneHotEncoder(handle_unknown='ignore')

# 添加輸入驗證
def predict_safe(X_new):
    if X_new.isnull().any().any():
        raise ValueError("Input contains missing values")
    if not isinstance(X_new, pd.DataFrame):
        raise TypeError("Input must be DataFrame")
    return model.predict(X_new)
```

---

## 🔍 調試技巧

### 1. 檢查數據形狀
```python
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
```

### 2. 檢查數據類型
```python
print(df.dtypes)
print(df.info())
```

### 3. 檢查缺失值
```python
print(df.isnull().sum())
```

### 4. 檢查類別分布
```python
print(y.value_counts())
```

### 5. 可視化數據
```python
import matplotlib.pyplot as plt
df.hist(figsize=(12, 10))
plt.show()
```

### 6. 檢查模型參數
```python
print(model.get_params())
```

### 7. 打印中間結果
```python
# Pipeline 中間結果
X_transformed = pipe[:-1].transform(X_test)
print(X_transformed[:5])
```

---

## 📚 快速檢查清單

訓練模型前：
- [ ] 檢查數據形狀
- [ ] 處理缺失值
- [ ] 編碼類別特徵
- [ ] 檢查類別平衡
- [ ] 分割訓練/測試集（stratify）
- [ ] 特徵縮放（在分割後）
- [ ] 設置 random_state

訓練模型時：
- [ ] 使用 Pipeline
- [ ] 設置合適的超參數
- [ ] 使用交叉驗證
- [ ] 監控訓練過程

評估模型時：
- [ ] 在測試集上評估
- [ ] 使用多種指標
- [ ] 查看混淆矩陣
- [ ] 檢查過擬合/欠擬合
- [ ] 分析特徵重要性

部署模型前：
- [ ] 保存完整 Pipeline
- [ ] 保存元數據
- [ ] 添加輸入驗證
- [ ] 測試邊緣情況
- [ ] 文檔化使用方法

---

## 💡 最佳實踐

1. **始終使用 Pipeline**
2. **設置 random_state 保證可復現**
3. **使用交叉驗證評估**
4. **檢查過擬合**
5. **選擇合適的評估指標**
6. **保存完整的訓練流程**
7. **記錄實驗結果**
8. **版本控制代碼和數據**

---

記住：**犯錯是學習的一部分！** 重要的是知道如何識別和修正錯誤。

**Happy Debugging! 🐛🔧**
