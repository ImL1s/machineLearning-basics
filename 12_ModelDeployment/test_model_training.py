"""
測試模型訓練功能
只測試模型訓練和保存部分，不需要 Flask
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json
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

print("="*80)
print("測試模型訓練和保存功能")
print("="*80)

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
    }
}

with open(model_dir / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n✓ 模型已保存到: {model_dir}")
print(f"  - iris_model.pkl (模型文件)")
print(f"  - iris_scaler.pkl (縮放器)")
print(f"  - metadata.json (元數據)")

# 測試加載
print("\n[驗證] 測試模型加載...")
loaded_model = joblib.load(model_dir / 'iris_model.pkl')
loaded_scaler = joblib.load(model_dir / 'iris_scaler.pkl')

# 測試預測
test_sample = X_test[0:1]
test_scaled = loaded_scaler.transform(test_sample)
prediction = loaded_model.predict(test_scaled)[0]
probabilities = loaded_model.predict_proba(test_scaled)[0]

print(f"\n測試預測:")
print(f"  輸入特徵: {test_sample[0]}")
print(f"  真實標籤: {target_names[y_test[0]]}")
print(f"  預測標籤: {target_names[prediction]}")
print(f"  置信度: {probabilities.max():.4f}")

print("\n" + "="*80)
print("✓ 所有測試通過！模型訓練和保存功能正常")
print("="*80)
