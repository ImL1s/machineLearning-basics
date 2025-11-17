# 工具模塊使用指南 / Utils Module Guide

這個目錄包含專案的共享工具函數，用於統一管理配置、路徑和繪圖。

This directory contains shared utility functions for configuration, path management, and plotting.

## 📁 模塊說明 / Module Description

### 1. `config.py` - 配置管理

集中管理所有常用參數，避免硬編碼。

**常用配置：**
- `RANDOM_STATE = 42` - 隨機種子
- `TEST_SIZE = 0.2` - 測試集比例
- `FIGURE_SIZE = (18, 12)` - 圖表大小
- `DPI = 150` - 圖像分辨率

**使用示例：**
```python
from utils.config import RANDOM_STATE, TEST_SIZE

# 數據分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)
```

---

### 2. `paths.py` - 路徑管理

提供統一的路徑管理，避免硬編碼相對路徑。

**主要目錄：**
- `PROJECT_ROOT` - 專案根目錄
- `DATA_DIR` - 數據目錄
- `OUTPUT_DIR` - 輸出目錄
- `MODELS_DIR` - 模型保存目錄

**主要函數：**

#### `get_data_path(filename)`
獲取數據文件路徑

```python
from utils.paths import get_data_path

# 而不是硬編碼相對路徑
# data_path = './data.csv'  # 錯誤！

# 使用工具函數
data_path = get_data_path('data.csv')  # 正確！
```

#### `get_output_path(filename, subfolder=None)`
獲取輸出文件路徑

```python
from utils.paths import get_output_path

# 保存圖表
figure_path = get_output_path('results.png', 'classification')
# 路徑: /path/to/project/output/classification/results.png
```

#### `get_model_path(model_name, version=None)`
獲取模型保存路徑

```python
from utils.paths import get_model_path
import joblib

# 保存模型
model_path = get_model_path('random_forest', version='1.0')
joblib.dump(model, model_path)
```

---

### 3. `plotting.py` - 繪圖工具

提供統一的繪圖配置和輔助函數。

**主要函數：**

#### `setup_chinese_fonts()`
設置中文字體支持

```python
from utils.plotting import setup_chinese_fonts
import matplotlib.pyplot as plt

setup_chinese_fonts()
plt.title('中文標題')  # 現在可以正常顯示中文
```

#### `save_figure(fig, filepath)`
安全地保存圖表

```python
from utils.plotting import save_figure
from utils.paths import get_output_path
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

# 保存圖表（自動創建目錄、錯誤處理）
save_figure(fig, get_output_path('plot.png'))
```

#### `create_subplots(nrows, ncols, figsize=None)`
創建子圖並自動設置中文字體

```python
from utils.plotting import create_subplots

# 自動設置中文字體和合適的大小
fig, axes = create_subplots(2, 2)
axes[0, 0].plot([1, 2, 3])
axes[0, 0].set_title('中文標題')  # 自動支持
```

---

## 🚀 完整示例 / Complete Example

```python
"""
完整的機器學習流程示例，使用工具模塊
Complete ML workflow example using utils module
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 導入工具模塊
from utils import (
    RANDOM_STATE, TEST_SIZE,  # 配置
    get_data_path, get_output_path, get_model_path,  # 路徑
    setup_chinese_fonts, save_figure  # 繪圖
)

# 1. 加載數據
iris = load_iris()
X, y = iris.data, iris.target

# 2. 數據分割（使用統一配置）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# 3. 訓練模型
model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# 4. 評估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"準確率: {accuracy:.4f}")

# 5. 可視化（使用繪圖工具）
setup_chinese_fonts()

fig, ax = plt.subplots(figsize=(10, 6))
feature_importance = model.feature_importances_
ax.barh(iris.feature_names, feature_importance)
ax.set_xlabel('特徵重要性')
ax.set_title('隨機森林特徵重要性分析')

# 6. 保存圖表（使用路徑工具）
save_figure(fig, get_output_path('feature_importance.png', 'examples'))

# 7. 保存模型（使用路徑工具）
import joblib
model_path = get_model_path('random_forest_iris', version='1.0')
joblib.dump(model, model_path)
print(f"✓ 模型已保存: {model_path}")
```

---

## 💡 最佳實踐 / Best Practices

### ✅ 推薦做法

1. **始終使用配置常量**
   ```python
   from utils.config import RANDOM_STATE
   model = RandomForestClassifier(random_state=RANDOM_STATE)
   ```

2. **始終使用路徑工具**
   ```python
   from utils.paths import get_data_path
   data = pd.read_csv(get_data_path('data.csv'))
   ```

3. **在每個繪圖文件開頭設置字體**
   ```python
   from utils.plotting import setup_chinese_fonts
   setup_chinese_fonts()
   ```

### ❌ 避免的做法

1. **硬編碼魔術數字**
   ```python
   # 錯誤
   random_state = 42  # 在每個文件中重複

   # 正確
   from utils.config import RANDOM_STATE
   ```

2. **硬編碼相對路徑**
   ```python
   # 錯誤
   df = pd.read_csv('./data/file.csv')  # 運行目錄不同會失敗

   # 正確
   from utils.paths import get_data_path
   df = pd.read_csv(get_data_path('file.csv'))
   ```

3. **在每個文件重複字體設置**
   ```python
   # 錯誤
   plt.rcParams['font.sans-serif'] = [...]  # 在每個文件中重複

   # 正確
   from utils.plotting import setup_chinese_fonts
   setup_chinese_fonts()
   ```

---

## 📚 更多資源 / More Resources

- 查看 `00_QuickStart/quick_start_guide.py` 了解基本用法
- 查看各個算法文件的完整示例
- 參考主 README.md 的完整專案說明

---

**Happy Coding! 編碼愉快！** 🎉
