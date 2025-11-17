"""
模型持久化：保存和加載模型
Model Persistence: Save and Load Models

學會如何保存訓練好的模型，並在生產環境中使用
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import pickle
import os
import json
from datetime import datetime

print("=" * 80)
print("模型持久化完整指南".center(80))
print("=" * 80)

# 創建模型保存目錄
os.makedirs('saved_models', exist_ok=True)

# ============================================================================
# 1. 基本概念
# ============================================================================
print("\n【1】模型持久化基本概念")
print("-" * 80)
print("""
為什麼需要模型持久化？
• 訓練好的模型可以重複使用
• 避免每次都重新訓練
• 部署到生產環境
• 模型版本管理
• 共享模型

常用方法：
1. joblib (推薦，scikit-learn 官方推薦)
   • 對大型 numpy 數組效率高
   • 文件較小
   • 速度快

2. pickle (Python 標準庫)
   • Python 通用序列化
   • 兼容性好
   • 文件較大

3. 模型特定格式
   • TensorFlow/Keras: .h5, SavedModel
   • XGBoost: .json, .bin
   • LightGBM: .txt

保存內容：
• 訓練好的模型
• 預處理器（Scaler, Encoder等）
• 特徵名稱
• 超參數
• 訓練時間、版本等元數據
""")

# ============================================================================
# 2. 準備訓練數據和模型
# ============================================================================
print("\n【2】準備訓練數據和模型")
print("-" * 80)

# 加載數據
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 訓練模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 評估
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"訓練準確率：{train_score:.4f}")
print(f"測試準確率：{test_score:.4f}")

# ============================================================================
# 3. 方法1：使用 joblib 保存/加載
# ============================================================================
print("\n【3】方法1：使用 joblib（推薦）")
print("-" * 80)

# 保存模型
model_path_joblib = 'saved_models/iris_model.joblib'
joblib.dump(model, model_path_joblib)
print(f"✓ 模型已保存到：{model_path_joblib}")

# 獲取文件大小
file_size_joblib = os.path.getsize(model_path_joblib) / 1024  # KB
print(f"  文件大小：{file_size_joblib:.2f} KB")

# 加載模型
loaded_model_joblib = joblib.load(model_path_joblib)
print(f"✓ 模型已從 {model_path_joblib} 加載")

# 驗證
loaded_score = loaded_model_joblib.score(X_test, y_test)
print(f"  加載模型的測試準確率：{loaded_score:.4f}")
print(f"  與原模型一致：{loaded_score == test_score}")

# ============================================================================
# 4. 方法2：使用 pickle 保存/加載
# ============================================================================
print("\n【4】方法2：使用 pickle")
print("-" * 80)

# 保存模型
model_path_pickle = 'saved_models/iris_model.pkl'
with open(model_path_pickle, 'wb') as f:
    pickle.dump(model, f)
print(f"✓ 模型已保存到：{model_path_pickle}")

# 獲取文件大小
file_size_pickle = os.path.getsize(model_path_pickle) / 1024  # KB
print(f"  文件大小：{file_size_pickle:.2f} KB")

# 加載模型
with open(model_path_pickle, 'rb') as f:
    loaded_model_pickle = pickle.load(f)
print(f"✓ 模型已從 {model_path_pickle} 加載")

# 驗證
loaded_score_pkl = loaded_model_pickle.score(X_test, y_test)
print(f"  加載模型的測試準確率：{loaded_score_pkl:.4f}")

# ============================================================================
# 5. 保存完整的 Pipeline
# ============================================================================
print("\n【5】保存完整的 Pipeline（預處理+模型）")
print("-" * 80)

# 創建包含預處理的 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 訓練 Pipeline
pipeline.fit(X_train, y_train)
pipeline_score = pipeline.score(X_test, y_test)
print(f"Pipeline 測試準確率：{pipeline_score:.4f}")

# 保存 Pipeline
pipeline_path = 'saved_models/iris_pipeline.joblib'
joblib.dump(pipeline, pipeline_path)
print(f"✓ Pipeline 已保存到：{pipeline_path}")

# 加載 Pipeline
loaded_pipeline = joblib.load(pipeline_path)
print(f"✓ Pipeline 已加載")

# 使用加載的 Pipeline 進行預測
sample = X_test[0:1]
prediction = loaded_pipeline.predict(sample)
prediction_proba = loaded_pipeline.predict_proba(sample)

print(f"\n預測示例：")
print(f"  輸入：{sample[0]}")
print(f"  預測類別：{iris.target_names[prediction[0]]}")
print(f"  預測概率：{prediction_proba[0]}")

# ============================================================================
# 6. 保存模型元數據
# ============================================================================
print("\n【6】保存模型元數據")
print("-" * 80)

# 創建元數據
metadata = {
    'model_type': 'RandomForestClassifier',
    'n_estimators': 100,
    'train_score': float(train_score),
    'test_score': float(test_score),
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist(),
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'sklearn_version': '1.3.0',  # 實際應該從 sklearn 獲取
}

# 保存元數據為 JSON
metadata_path = 'saved_models/iris_model_metadata.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✓ 元數據已保存到：{metadata_path}")
print(f"\n元數據內容：")
for key, value in metadata.items():
    print(f"  {key}: {value}")

# 加載元數據
with open(metadata_path, 'r', encoding='utf-8') as f:
    loaded_metadata = json.load(f)
print(f"\n✓ 元數據已加載")

# ============================================================================
# 7. 模型版本管理
# ============================================================================
print("\n【7】模型版本管理")
print("-" * 80)

def save_model_with_version(model, model_name, version, metadata=None):
    """保存帶版本號的模型"""
    # 創建版本目錄
    version_dir = f'saved_models/{model_name}/v{version}'
    os.makedirs(version_dir, exist_ok=True)

    # 保存模型
    model_path = f'{version_dir}/model.joblib'
    joblib.dump(model, model_path)

    # 保存元數據
    if metadata:
        metadata['version'] = version
        metadata_path = f'{version_dir}/metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ 模型版本 v{version} 已保存到：{version_dir}")
    return version_dir

# 保存不同版本
v1_dir = save_model_with_version(model, 'iris_classifier', '1.0', metadata)

# 訓練一個不同參數的模型作為 v2
model_v2 = RandomForestClassifier(n_estimators=200, random_state=42)
model_v2.fit(X_train, y_train)

metadata_v2 = metadata.copy()
metadata_v2['n_estimators'] = 200
metadata_v2['test_score'] = float(model_v2.score(X_test, y_test))

v2_dir = save_model_with_version(model_v2, 'iris_classifier', '2.0', metadata_v2)

# ============================================================================
# 8. 加載最新版本的模型
# ============================================================================
print("\n【8】加載特定版本的模型")
print("-" * 80)

def load_model_version(model_name, version):
    """加載指定版本的模型"""
    version_dir = f'saved_models/{model_name}/v{version}'
    model_path = f'{version_dir}/model.joblib'
    metadata_path = f'{version_dir}/metadata.json'

    # 加載模型
    model = joblib.load(model_path)

    # 加載元數據
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"✓ 已加載模型版本 v{version}")
    print(f"  測試準確率：{metadata['test_score']:.4f}")

    return model, metadata

# 加載 v1
model_v1_loaded, metadata_v1 = load_model_version('iris_classifier', '1.0')

# 加載 v2
model_v2_loaded, metadata_v2_loaded = load_model_version('iris_classifier', '2.0')

# ============================================================================
# 9. 生產環境使用示例
# ============================================================================
print("\n【9】生產環境使用示例")
print("-" * 80)

class ModelPredictor:
    """生產環境模型預測器"""

    def __init__(self, model_path, metadata_path=None):
        """初始化預測器"""
        self.model = joblib.load(model_path)

        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        print(f"✓ 預測器已初始化")
        if 'created_at' in self.metadata:
            print(f"  模型訓練時間：{self.metadata['created_at']}")

    def predict(self, X):
        """進行預測"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """預測概率"""
        return self.model.predict_proba(X)

    def get_info(self):
        """獲取模型信息"""
        return self.metadata

# 使用預測器
predictor = ModelPredictor(
    'saved_models/iris_model.joblib',
    'saved_models/iris_model_metadata.json'
)

# 進行預測
test_sample = X_test[:5]
predictions = predictor.predict(test_sample)
probabilities = predictor.predict_proba(test_sample)

print(f"\n批量預測示例：")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"  樣本{i+1}: 預測={iris.target_names[pred]}, "
          f"概率={prob.max():.3f}")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("模型持久化要點總結")
print("=" * 80)
print(f"""
1. 保存方法選擇：
   • joblib：scikit-learn 模型（推薦）
   • pickle：通用 Python 對象
   • 模型特定格式：深度學習模型

2. 最佳實踐：
   ✓ 保存完整的 Pipeline（預處理+模型）
   ✓ 保存模型元數據
   ✓ 使用版本管理
   ✓ 記錄訓練時間、性能指標
   ✓ 保存特徵名稱和順序

3. 文件大小比較（本例）：
   • joblib: {file_size_joblib:.2f} KB
   • pickle: {file_size_pickle:.2f} KB

4. 注意事項：
   • 確保加載環境與訓練環境一致
   • 注意 scikit-learn 版本兼容性
   • 敏感模型加密存儲
   • 定期備份重要模型

5. 生產環境部署：
   • 創建預測器類封裝模型
   • 添加輸入驗證
   • 記錄預測日誌
   • 監控模型性能
   • 定期重新訓練

6. 模型文件組織：
   saved_models/
   ├── model_name/
   │   ├── v1.0/
   │   │   ├── model.joblib
   │   │   └── metadata.json
   │   ├── v2.0/
   │   │   ├── model.joblib
   │   │   └── metadata.json
   └── ...

7. 示例代碼：
   # 保存
   joblib.dump(model, 'model.joblib')

   # 加載
   model = joblib.load('model.joblib')

   # 預測
   predictions = model.predict(X_new)
""")

print("\n✓ 所有示例已完成！")
print(f"✓ 保存的文件位於：saved_models/ 目錄")
