"""
模型部署 - Flask REST API | Model Deployment - Flask REST API

本教程展示如何將訓練好的機器學習模型部署為 REST API 服務。
This tutorial demonstrates how to deploy a trained machine learning model as a REST API service.

內容包括 | Contents:
1. 訓練並保存模型 | Train and Save Model
2. 創建 Flask API | Create Flask API
3. 實現預測端點 | Implement Prediction Endpoints
4. 錯誤處理和日誌 | Error Handling and Logging
5. API 測試 | API Testing

作者: MLOps 工程師
日期: 2025-11
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 機器學習庫
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Utils
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import RANDOM_STATE, TEST_SIZE

# ============================================================================
# Part 1: 訓練並保存模型 / Train and Save Model
# ============================================================================
print("="*80)
print("Part 1: 訓練並保存模型 | Training and Saving Model")
print("="*80)

def train_and_save_model():
    """
    訓練模型並保存到本地文件系統
    Train model and save to local filesystem
    """
    # 加載數據
    print("\n[1/5] 加載數據...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"數據集形狀: {X.shape}")
    print(f"特徵名稱: {feature_names}")
    print(f"類別名稱: {target_names}")

    # 分割數據
    print(f"\n[2/5] 分割數據 (測試集比例: {TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"訓練集大小: {X_train.shape[0]}")
    print(f"測試集大小: {X_test.shape[0]}")

    # 特徵縮放
    print("\n[3/5] 特徵縮放...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"縮放後特徵範圍: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

    # 訓練模型
    print("\n[4/5] 訓練 Random Forest 模型...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("✓ 模型訓練完成")

    # 評估模型
    print("\n模型評估:")
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_score = accuracy_score(y_train, train_pred)
    test_score = accuracy_score(y_test, test_pred)

    print(f"訓練集準確率: {train_score:.4f}")
    print(f"測試集準確率: {test_score:.4f}")

    print("\n測試集分類報告:")
    print(classification_report(y_test, test_pred, target_names=target_names))

    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特徵重要性:")
    print(feature_importance.to_string(index=False))

    # 保存模型和預處理器
    print("\n[5/5] 保存模型...")
    model_dir = Path(__file__).parent / 'saved_models'
    model_dir.mkdir(exist_ok=True)

    # 保存模型文件
    joblib.dump(model, model_dir / 'iris_model.pkl')
    joblib.dump(scaler, model_dir / 'iris_scaler.pkl')

    # 保存模型元數據
    metadata = {
        'model_type': 'RandomForestClassifier',
        'model_version': '1.0.0',
        'trained_date': datetime.now().isoformat(),
        'features': list(feature_names),
        'target_names': list(target_names),
        'n_features': len(feature_names),
        'n_classes': len(target_names),
        'train_score': float(train_score),
        'test_score': float(test_score),
        'feature_importance': feature_importance.to_dict('records'),
        'model_params': {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'random_state': RANDOM_STATE
        },
        'data_stats': {
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'feature_mean': X_train.mean(axis=0).tolist(),
            'feature_std': X_train.std(axis=0).tolist()
        }
    }

    with open(model_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 模型已保存到: {model_dir}")
    print(f"  - iris_model.pkl (模型文件)")
    print(f"  - iris_scaler.pkl (縮放器)")
    print(f"  - metadata.json (元數據)")

    return model, scaler, metadata

# 訓練模型（只在直接運行此腳本時執行）
if __name__ == '__main__':
    model_dir = Path(__file__).parent / 'saved_models'
    if not (model_dir / 'iris_model.pkl').exists():
        train_and_save_model()
    else:
        print("\n模型文件已存在，跳過訓練步驟")
        print(f"如需重新訓練，請刪除: {model_dir}")

# ============================================================================
# Part 2: 創建 Flask API / Create Flask API
# ============================================================================
print("\n" + "="*80)
print("Part 2: 創建 Flask API | Creating Flask API")
print("="*80)

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 創建 Flask 應用
app = Flask(__name__)

# 全局變量存儲模型
MODEL = None
SCALER = None
METADATA = None

def load_model():
    """
    加載已保存的模型和元數據
    Load saved model and metadata
    """
    global MODEL, SCALER, METADATA

    try:
        MODEL_DIR = Path(__file__).parent / 'saved_models'

        # 檢查模型文件是否存在
        if not MODEL_DIR.exists():
            logger.error(f"模型目錄不存在: {MODEL_DIR}")
            return False

        model_path = MODEL_DIR / 'iris_model.pkl'
        scaler_path = MODEL_DIR / 'iris_scaler.pkl'
        metadata_path = MODEL_DIR / 'metadata.json'

        if not all([model_path.exists(), scaler_path.exists(), metadata_path.exists()]):
            logger.error("模型文件不完整")
            return False

        # 加載模型
        MODEL = joblib.load(model_path)
        SCALER = joblib.load(scaler_path)

        with open(metadata_path, 'r', encoding='utf-8') as f:
            METADATA = json.load(f)

        logger.info("模型加載成功")
        logger.info(f"模型類型: {METADATA['model_type']}")
        logger.info(f"模型版本: {METADATA['model_version']}")
        logger.info(f"訓練日期: {METADATA['trained_date']}")

        return True

    except Exception as e:
        logger.error(f"加載模型失敗: {str(e)}")
        return False

# 在應用啟動時加載模型
with app.app_context():
    if not load_model():
        logger.warning("模型未加載，某些端點將不可用")

# ============================================================================
# Part 3: API 端點定義 / API Endpoints
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """
    API 首頁 / API Home

    Returns:
        API 基本信息和可用端點列表
    """
    return jsonify({
        'service': 'Iris Classification API',
        'version': '1.0.0',
        'description': '基於 Random Forest 的鳶尾花分類 API',
        'model_loaded': MODEL is not None,
        'endpoints': {
            'GET /': 'API 信息',
            'GET /health': '健康檢查',
            'GET /model_info': '模型詳細信息',
            'POST /predict': '單個樣本預測',
            'POST /predict_batch': '批量樣本預測',
            'POST /predict_proba': '預測概率分布'
        },
        'documentation': 'https://github.com/your-repo/docs'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康檢查端點 / Health Check Endpoint

    用於監控和負載均衡器檢查服務狀態
    Used for monitoring and load balancer health checks

    Returns:
        服務健康狀態
    """
    is_healthy = MODEL is not None and SCALER is not None and METADATA is not None

    response = {
        'status': 'healthy' if is_healthy else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'model': MODEL is not None,
            'scaler': SCALER is not None,
            'metadata': METADATA is not None
        }
    }

    status_code = 200 if is_healthy else 503
    return jsonify(response), status_code

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    獲取模型詳細信息 / Get Model Information

    Returns:
        模型元數據和性能指標
    """
    if METADATA is None:
        return jsonify({'error': '模型未加載'}), 503

    return jsonify(METADATA)

@app.route('/predict', methods=['POST'])
def predict():
    """
    單個樣本預測 / Single Sample Prediction

    輸入格式 / Input Format:
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }

    輸出格式 / Output Format:
    {
        "prediction": 0,
        "prediction_label": "setosa",
        "probabilities": [0.98, 0.01, 0.01],
        "confidence": 0.98,
        "timestamp": "2025-11-18T10:00:00"
    }
    """
    try:
        # 檢查模型是否加載
        if MODEL is None or SCALER is None:
            return jsonify({'error': '模型未加載'}), 503

        # 獲取輸入數據
        data = request.get_json()

        if not data:
            return jsonify({'error': '請求體不能為空'}), 400

        if 'features' not in data:
            return jsonify({'error': '缺少 features 字段'}), 400

        # 驗證特徵
        features = np.array(data['features'])

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 驗證特徵數量
        if features.shape[1] != METADATA['n_features']:
            return jsonify({
                'error': f'期望 {METADATA["n_features"]} 個特徵，但收到 {features.shape[1]} 個'
            }), 400

        # 驗證特徵值
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return jsonify({'error': '特徵包含無效值 (NaN 或 Inf)'}), 400

        # 預處理和預測
        features_scaled = SCALER.transform(features)
        prediction = MODEL.predict(features_scaled)[0]
        probabilities = MODEL.predict_proba(features_scaled)[0]

        # 構建響應
        response = {
            'prediction': int(prediction),
            'prediction_label': METADATA['target_names'][prediction],
            'probabilities': {
                METADATA['target_names'][i]: float(prob)
                for i, prob in enumerate(probabilities)
            },
            'confidence': float(probabilities.max()),
            'timestamp': datetime.now().isoformat(),
            'model_version': METADATA['model_version']
        }

        logger.info(f"預測成功 - 輸入: {data['features']}, 輸出: {response['prediction_label']}")
        return jsonify(response)

    except ValueError as e:
        logger.error(f"數據驗證錯誤: {str(e)}")
        return jsonify({'error': f'數據驗證錯誤: {str(e)}'}), 400

    except Exception as e:
        logger.error(f"預測錯誤: {str(e)}", exc_info=True)
        return jsonify({'error': f'服務器內部錯誤: {str(e)}'}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    批量預測 / Batch Prediction

    輸入格式 / Input Format:
    {
        "samples": [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 2.9, 4.3, 1.3],
            [7.3, 2.9, 6.3, 1.8]
        ]
    }

    輸出格式 / Output Format:
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
    """
    try:
        # 檢查模型是否加載
        if MODEL is None or SCALER is None:
            return jsonify({'error': '模型未加載'}), 503

        # 獲取輸入數據
        data = request.get_json()

        if not data:
            return jsonify({'error': '請求體不能為空'}), 400

        if 'samples' not in data:
            return jsonify({'error': '缺少 samples 字段'}), 400

        # 驗證批量大小
        samples = np.array(data['samples'])

        if samples.ndim != 2:
            return jsonify({'error': 'samples 必須是二維數組'}), 400

        if len(samples) == 0:
            return jsonify({'error': 'samples 不能為空'}), 400

        # 限制批量大小
        MAX_BATCH_SIZE = 1000
        if len(samples) > MAX_BATCH_SIZE:
            return jsonify({
                'error': f'批量大小超過限制 (最大: {MAX_BATCH_SIZE})'
            }), 400

        # 驗證特徵數量
        if samples.shape[1] != METADATA['n_features']:
            return jsonify({
                'error': f'期望 {METADATA["n_features"]} 個特徵，但收到 {samples.shape[1]} 個'
            }), 400

        # 預處理和預測
        samples_scaled = SCALER.transform(samples)
        predictions = MODEL.predict(samples_scaled)
        probabilities = MODEL.predict_proba(samples_scaled)

        # 構建響應
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            results.append({
                'sample_id': i,
                'prediction': int(pred),
                'prediction_label': METADATA['target_names'][pred],
                'confidence': float(proba.max()),
                'probabilities': {
                    METADATA['target_names'][j]: float(p)
                    for j, p in enumerate(proba)
                }
            })

        response = {
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat(),
            'model_version': METADATA['model_version']
        }

        logger.info(f"批量預測成功 - 樣本數: {len(samples)}")
        return jsonify(response)

    except ValueError as e:
        logger.error(f"數據驗證錯誤: {str(e)}")
        return jsonify({'error': f'數據驗證錯誤: {str(e)}'}), 400

    except Exception as e:
        logger.error(f"批量預測錯誤: {str(e)}", exc_info=True)
        return jsonify({'error': f'服務器內部錯誤: {str(e)}'}), 500

@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    """
    預測概率分布 / Predict Probability Distribution

    返回所有類別的完整概率分布
    Returns full probability distribution for all classes

    輸入格式同 /predict
    """
    try:
        if MODEL is None or SCALER is None:
            return jsonify({'error': '模型未加載'}), 503

        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({'error': '缺少 features 字段'}), 400

        features = np.array(data['features']).reshape(1, -1)

        if features.shape[1] != METADATA['n_features']:
            return jsonify({
                'error': f'期望 {METADATA["n_features"]} 個特徵，但收到 {features.shape[1]} 個'
            }), 400

        # 預測概率
        features_scaled = SCALER.transform(features)
        probabilities = MODEL.predict_proba(features_scaled)[0]

        # 構建詳細響應
        prob_details = []
        for i, (class_name, prob) in enumerate(zip(METADATA['target_names'], probabilities)):
            prob_details.append({
                'class_id': i,
                'class_name': class_name,
                'probability': float(prob),
                'percentage': f"{prob * 100:.2f}%"
            })

        # 按概率排序
        prob_details.sort(key=lambda x: x['probability'], reverse=True)

        response = {
            'probabilities': prob_details,
            'most_likely': prob_details[0]['class_name'],
            'confidence': prob_details[0]['probability'],
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"概率預測錯誤: {str(e)}", exc_info=True)
        return jsonify({'error': f'服務器內部錯誤: {str(e)}'}), 500

# ============================================================================
# 錯誤處理 / Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """處理 404 錯誤"""
    return jsonify({
        'error': 'Not Found',
        'message': '請求的端點不存在',
        'available_endpoints': list(app.url_map.iter_rules())
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """處理 405 錯誤"""
    return jsonify({
        'error': 'Method Not Allowed',
        'message': '請使用正確的 HTTP 方法'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """處理 500 錯誤"""
    logger.error(f"內部錯誤: {str(error)}", exc_info=True)
    return jsonify({
        'error': 'Internal Server Error',
        'message': '服務器內部錯誤，請稍後重試'
    }), 500

# ============================================================================
# 運行服務器 / Run Server
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("啟動 Flask 服務器 | Starting Flask Server")
    print("="*80)

    # 確保模型已訓練
    model_dir = Path(__file__).parent / 'saved_models'
    if not (model_dir / 'iris_model.pkl').exists():
        print("\n模型文件不存在，開始訓練...")
        train_and_save_model()
        print("\n重新加載模型...")
        load_model()

    print("\n服務器配置:")
    print(f"  - 地址: http://localhost:5000")
    print(f"  - 調試模式: 開啟")
    print(f"  - 日誌文件: {Path(__file__).parent / 'api.log'}")
    print("\n可用端點:")
    print("  - GET  /              API 首頁")
    print("  - GET  /health        健康檢查")
    print("  - GET  /model_info    模型信息")
    print("  - POST /predict       單個預測")
    print("  - POST /predict_batch 批量預測")
    print("  - POST /predict_proba 概率預測")
    print("\n按 Ctrl+C 停止服務器\n")

    # 啟動服務器
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n\n服務器已停止")
        logger.info("服務器正常關閉")
