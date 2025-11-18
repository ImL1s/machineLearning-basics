# 12. 模型部署 | Model Deployment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

## 📋 目錄 | Table of Contents

- [模塊概述](#模塊概述)
- [核心概念](#核心概念)
- [文件結構](#文件結構)
- [快速開始](#快速開始)
- [API 文檔](#api-文檔)
- [性能測試](#性能測試)
- [生產部署](#生產部署)
- [最佳實踐](#最佳實踐)
- [常見問題](#常見問題)

---

## 🎯 模塊概述

本模塊演示如何將訓練好的機器學習模型部署為生產級 REST API 服務，涵蓋從模型訓練、API 開發、測試到性能優化的完整流程。

### 學習目標

- ✅ 理解模型部署的基本概念和流程
- ✅ 掌握 Flask 框架構建 REST API
- ✅ 學習模型序列化和版本管理
- ✅ 實現健康檢查和監控端點
- ✅ 進行性能測試和優化
- ✅ 了解生產環境部署方案

### 技術棧

| 技術 | 用途 | 版本 |
|------|------|------|
| Flask | Web 框架 | 2.3+ |
| scikit-learn | 機器學習 | 1.3+ |
| joblib | 模型序列化 | 1.3+ |
| requests | HTTP 客戶端 | 2.31+ |
| gunicorn | WSGI 服務器 | 21.2+ |

---

## 📚 核心概念

### 1. 模型部署流程

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ 模型訓練    │───>│ 模型序列化   │───>│ API 開發    │───>│ 服務部署     │
│ Training    │    │ Serialization│    │ API Dev     │    │ Deployment   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
      ↓                    ↓                   ↓                   ↓
  數據預處理          保存模型文件         實現端點           配置服務器
  特徵工程            版本管理             錯誤處理           監控日誌
  模型評估            元數據記錄           性能優化           擴展部署
```

### 2. REST API 設計

- **端點 (Endpoints)**: 定義清晰的 URL 路徑
- **HTTP 方法**: GET (查詢), POST (創建/預測)
- **請求格式**: JSON
- **響應格式**: JSON with status code
- **錯誤處理**: 統一的錯誤響應格式

### 3. 模型序列化

使用 `joblib` 保存和加載模型:
- 模型文件 (`.pkl`)
- 預處理器 (scaler, encoder 等)
- 元數據 (特徵名稱、版本信息等)

---

## 📁 文件結構

```
12_ModelDeployment/
├── 01_flask_api_deployment.py      # Flask REST API 實現 (500行)
│   ├── Part 1: 訓練並保存模型
│   ├── Part 2: 創建 Flask API
│   ├── Part 3: API 端點定義
│   └── Part 4: 錯誤處理
│
├── 02_model_serving_example.py     # 客戶端和性能測試 (400行)
│   ├── Part 1: API 客戶端封裝
│   ├── Part 2: 基本調用示例
│   ├── Part 3: 批量預測示例
│   ├── Part 4: 性能測試
│   └── Part 5: 錯誤處理測試
│
├── requirements.txt                # 部署依賴
├── README.md                       # 本文件
│
├── saved_models/                   # 模型文件目錄 (自動生成)
│   ├── iris_model.pkl             # 訓練好的模型
│   ├── iris_scaler.pkl            # 特徵縮放器
│   └── metadata.json              # 模型元數據
│
└── api.log                        # API 日誌 (自動生成)
```

---

## 🚀 快速開始

### 步驟 1: 安裝依賴

```bash
# 進入模塊目錄
cd 12_ModelDeployment

# 安裝依賴
pip install -r requirements.txt

# 或使用項目根目錄的依賴文件
pip install -r ../requirements.txt
```

### 步驟 2: 啟動 API 服務器

```bash
# 運行 Flask 服務器
python 01_flask_api_deployment.py
```

輸出示例:
```
================================================================================
Part 1: 訓練並保存模型 | Training and Saving Model
================================================================================

[1/5] 加載數據...
數據集形狀: (150, 4)
訓練集準確率: 1.0000
測試集準確率: 1.0000

✓ 模型已保存到: saved_models/

================================================================================
啟動 Flask 服務器 | Starting Flask Server
================================================================================

服務器配置:
  - 地址: http://localhost:5000
  - 調試模式: 開啟

* Running on http://0.0.0.0:5000
```

### 步驟 3: 測試 API (新終端)

在另一個終端中運行測試腳本:

```bash
# 運行客戶端測試
python 02_model_serving_example.py
```

或使用 `curl` 命令:

```bash
# 健康檢查
curl http://localhost:5000/health

# 模型信息
curl http://localhost:5000/model_info

# 單個預測
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---

## 📡 API 文檔

### 基本信息

- **Base URL**: `http://localhost:5000`
- **Content-Type**: `application/json`
- **Response Format**: JSON

### 端點列表

#### 1. GET `/` - API 首頁

獲取 API 基本信息和可用端點列表。

**響應示例:**
```json
{
  "service": "Iris Classification API",
  "version": "1.0.0",
  "model_loaded": true,
  "endpoints": {
    "GET /": "API 信息",
    "GET /health": "健康檢查",
    "POST /predict": "單個預測"
  }
}
```

#### 2. GET `/health` - 健康檢查

檢查 API 服務和模型加載狀態。

**響應示例:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-18T10:00:00",
  "components": {
    "model": true,
    "scaler": true,
    "metadata": true
  }
}
```

**狀態碼:**
- `200`: 服務健康
- `503`: 服務不可用

#### 3. GET `/model_info` - 模型信息

獲取模型詳細信息和性能指標。

**響應示例:**
```json
{
  "model_type": "RandomForestClassifier",
  "model_version": "1.0.0",
  "trained_date": "2025-11-18T10:00:00",
  "test_score": 1.0,
  "n_features": 4,
  "features": [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
  ],
  "target_names": ["setosa", "versicolor", "virginica"]
}
```

#### 4. POST `/predict` - 單個預測

對單個樣本進行預測。

**請求體:**
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**響應示例:**
```json
{
  "prediction": 0,
  "prediction_label": "setosa",
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.01,
    "virginica": 0.01
  },
  "confidence": 0.98,
  "timestamp": "2025-11-18T10:00:00",
  "model_version": "1.0.0"
}
```

**錯誤響應:**
```json
{
  "error": "期望 4 個特徵，但收到 3 個"
}
```

#### 5. POST `/predict_batch` - 批量預測

對多個樣本進行批量預測。

**請求體:**
```json
{
  "samples": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.9, 4.3, 1.3],
    [7.3, 2.9, 6.3, 1.8]
  ]
}
```

**響應示例:**
```json
{
  "results": [
    {
      "sample_id": 0,
      "prediction": 0,
      "prediction_label": "setosa",
      "confidence": 0.98
    },
    ...
  ],
  "count": 3,
  "timestamp": "2025-11-18T10:00:00"
}
```

**限制:**
- 最大批量大小: 1000 個樣本

#### 6. POST `/predict_proba` - 概率預測

獲取詳細的概率分布。

**請求體:**
```json
{
  "features": [5.8, 2.7, 5.1, 1.9]
}
```

**響應示例:**
```json
{
  "probabilities": [
    {
      "class_id": 2,
      "class_name": "virginica",
      "probability": 0.92,
      "percentage": "92.00%"
    },
    ...
  ],
  "most_likely": "virginica",
  "confidence": 0.92
}
```

---

## 📊 性能測試

### 運行性能測試

```bash
python 02_model_serving_example.py
```

### 典型性能指標

基於 100 次請求的測試結果:

| 指標 | 值 |
|------|-----|
| 平均延遲 | ~15 ms |
| P95 延遲 | ~25 ms |
| P99 延遲 | ~35 ms |
| 吞吐量 | ~60 req/s |

### 批量預測性能

| 批量大小 | 總耗時 | 平均延遲 | 吞吐量 |
|---------|--------|---------|--------|
| 1 | 15 ms | 15.0 ms | 66 req/s |
| 10 | 25 ms | 2.5 ms | 400 req/s |
| 50 | 60 ms | 1.2 ms | 833 req/s |
| 100 | 100 ms | 1.0 ms | 1000 req/s |

**結論**: 批量預測可顯著提升吞吐量。

---

## 🚢 生產部署

### 1. 使用 Gunicorn (Linux/Mac)

```bash
# 安裝 Gunicorn
pip install gunicorn

# 啟動服務 (4個工作進程)
gunicorn -w 4 -b 0.0.0.0:5000 01_flask_api_deployment:app

# 使用配置文件
gunicorn -c gunicorn_config.py 01_flask_api_deployment:app
```

**gunicorn_config.py 示例:**
```python
# Gunicorn 配置文件
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
timeout = 30
keepalive = 2
errorlog = "logs/error.log"
accesslog = "logs/access.log"
loglevel = "info"
```

### 2. 使用 Waitress (Windows)

```bash
# 安裝 Waitress
pip install waitress

# 啟動服務
waitress-serve --host=0.0.0.0 --port=5000 01_flask_api_deployment:app
```

### 3. Docker 部署

**Dockerfile 示例:**
```dockerfile
FROM python:3.8-slim

WORKDIR /app

# 複製依賴文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用代碼
COPY . .

# 訓練模型 (可選，也可以掛載預訓練模型)
RUN python 01_flask_api_deployment.py --train-only

# 暴露端口
EXPOSE 5000

# 啟動服務
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "01_flask_api_deployment:app"]
```

**構建和運行:**
```bash
# 構建鏡像
docker build -t iris-api:1.0 .

# 運行容器
docker run -p 5000:5000 iris-api:1.0

# 使用 docker-compose
docker-compose up -d
```

**docker-compose.yml 示例:**
```yaml
version: '3.8'

services:
  iris-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./saved_models:/app/saved_models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 4. Kubernetes 部署

**deployment.yaml 示例:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iris-api
  template:
    metadata:
      labels:
        app: iris-api
    spec:
      containers:
      - name: iris-api
        image: iris-api:1.0
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: iris-api-service
spec:
  selector:
    app: iris-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

---

## 💡 最佳實踐

### 1. 模型版本管理

```python
# 使用語義化版本號
model_version = "1.2.3"  # MAJOR.MINOR.PATCH

# 在文件名中包含版本
model_path = f"models/iris_model_v{model_version}.pkl"

# 保存元數據
metadata = {
    'version': model_version,
    'trained_date': datetime.now().isoformat(),
    'git_commit': get_git_commit_hash()
}
```

### 2. 日誌和監控

```python
import logging

# 配置結構化日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

# 記錄關鍵事件
logger.info(f"Model loaded: {model_version}")
logger.info(f"Prediction made: {prediction}, confidence: {confidence}")
```

### 3. 錯誤處理

```python
# 統一的錯誤響應格式
@app.errorhandler(Exception)
def handle_error(error):
    response = {
        'error': str(error),
        'type': type(error).__name__,
        'timestamp': datetime.now().isoformat()
    }
    logger.error(f"Error: {response}")
    return jsonify(response), 500
```

### 4. 輸入驗證

```python
def validate_features(features, expected_count):
    """驗證輸入特徵"""
    if len(features) != expected_count:
        raise ValueError(f"Expected {expected_count} features")

    if any(np.isnan(features)) or any(np.isinf(features)):
        raise ValueError("Features contain invalid values")

    return True
```

### 5. 性能優化

- **使用批量預測**: 對多個樣本使用 `predict_batch`
- **模型緩存**: 在應用啟動時加載模型，避免重複加載
- **連接池**: 使用 `requests.Session()` 復用 HTTP 連接
- **異步處理**: 對於長時間運行的任務，使用任務隊列 (Celery)

### 6. 安全性

```python
from flask_limiter import Limiter
from flask_cors import CORS

# 啟用 CORS
CORS(app)

# 速率限制
limiter = Limiter(app, key_func=get_remote_address)

@app.route('/predict')
@limiter.limit("100 per minute")
def predict():
    # ...
```

---

## ❓ 常見問題

### Q1: 如何更新已部署的模型?

**A**: 使用以下策略之一:
1. **藍綠部署**: 同時運行新舊版本，逐步切換流量
2. **滾動更新**: 逐步替換實例
3. **影子部署**: 新模型接收流量但不返回結果，用於驗證

```python
# 實現模型熱更新
@app.route('/reload_model', methods=['POST'])
def reload_model():
    global MODEL
    MODEL = joblib.load('new_model.pkl')
    return jsonify({'status': 'Model reloaded'})
```

### Q2: 如何處理大批量請求?

**A**:
- 使用異步任務隊列 (Celery + Redis)
- 實現批處理端點
- 限制最大批量大小
- 使用流式響應

### Q3: 如何監控模型性能?

**A**:
```python
from prometheus_client import Counter, Histogram

# 定義指標
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

# 記錄指標
@prediction_latency.time()
def predict():
    prediction_counter.inc()
    # ...
```

### Q4: 模型文件太大怎麼辦?

**A**:
- 使用模型壓縮技術 (pruning, quantization)
- 將模型存儲在對象存儲 (S3, GCS)
- 使用模型服務框架 (TensorFlow Serving, TorchServe)

### Q5: 如何實現 A/B 測試?

**A**:
```python
import random

@app.route('/predict')
def predict():
    # 隨機選擇模型
    if random.random() < 0.5:
        model = model_a
        version = 'A'
    else:
        model = model_b
        version = 'B'

    prediction = model.predict(features)

    # 記錄使用的模型版本
    log_prediction(features, prediction, version)

    return jsonify(prediction)
```

---

## 📚 延伸閱讀

### 官方文檔
- [Flask 文檔](https://flask.palletsprojects.com/)
- [scikit-learn 模型持久化](https://scikit-learn.org/stable/model_persistence.html)
- [Gunicorn 文檔](https://docs.gunicorn.org/)

### 推薦資源
- **書籍**: "Building Machine Learning Powered Applications" by Emmanuel Ameisen
- **課程**: "Deploying Machine Learning Models in Production" (Coursera)
- **工具**: MLflow, Kubeflow, BentoML

### 進階主題
- Model Serving Frameworks (TensorFlow Serving, TorchServe)
- Feature Stores (Feast, Tecton)
- Model Monitoring (Evidently AI, Arize)
- MLOps Platforms (Kubeflow, MLflow)

---

## 📝 總結

本模塊涵蓋了機器學習模型部署的完整流程:

✅ **模型訓練和序列化**: 使用 joblib 保存模型
✅ **REST API 開發**: 使用 Flask 構建生產級 API
✅ **客戶端開發**: 封裝易用的 API 客戶端
✅ **性能測試**: 延遲、吞吐量分析
✅ **生產部署**: Docker、Kubernetes 部署方案
✅ **最佳實踐**: 版本管理、監控、安全性

### 下一步

1. **實踐項目**: 部署您自己的模型
2. **學習 MLOps**: 自動化 ML 工作流
3. **探索工具**: TensorFlow Serving, KFServing
4. **雲端部署**: AWS SageMaker, Azure ML, GCP AI Platform

---

**作者**: MLOps 工程師
**最後更新**: 2025-11-18
**版本**: 1.0.0

如有問題或建議，歡迎提交 Issue 或 Pull Request!
