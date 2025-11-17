"""
線性回歸（Linear Regression）
最基礎的回歸算法

原理：找到最佳擬合直線 y = wx + b
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("線性回歸教程".center(80))
print("=" * 80)

# 加載糖尿病數據集
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print(f"數據形狀：{X.shape}")
print(f"特徵：{diabetes.feature_names}")

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. 標準線性回歸
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print(f"\n【線性回歸】")
print(f"R² Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_lr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.4f}")

# 2. Ridge 回歸（L2 正則化）
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print(f"\n【Ridge 回歸】")
print(f"R² Score: {r2_score(y_test, y_pred_ridge):.4f}")

# 3. Lasso 回歸（L1 正則化）
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print(f"\n【Lasso 回歸】")
print(f"R² Score: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"非零特徵數：{np.sum(lasso.coef_ != 0)}")

# 4. ElasticNet（L1 + L2）
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)

print(f"\n【ElasticNet 回歸】")
print(f"R² Score: {r2_score(y_test, y_pred_elastic):.4f}")

# 可視化比較
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 預測 vs 真實值
axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('True Values')
axes[0, 0].set_ylabel('Predictions')
axes[0, 0].set_title('Linear Regression: Predictions vs True Values')
axes[0, 0].grid(True, alpha=0.3)

# 係數比較
models = ['Linear', 'Ridge', 'Lasso', 'ElasticNet']
coefs = [lr.coef_, ridge.coef_, lasso.coef_, elastic.coef_]

for i, (model, coef) in enumerate(zip(models, coefs)):
    axes[0, 1].plot(coef, marker='o', label=model, alpha=0.7)

axes[0, 1].set_xlabel('Feature Index')
axes[0, 1].set_ylabel('Coefficient Value')
axes[0, 1].set_title('Model Coefficients Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# R² 分數比較
r2_scores = [
    r2_score(y_test, y_pred_lr),
    r2_score(y_test, y_pred_ridge),
    r2_score(y_test, y_pred_lasso),
    r2_score(y_test, y_pred_elastic)
]

axes[1, 0].bar(models, r2_scores, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].set_title('Model Performance Comparison')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 殘差圖
residuals = y_test - y_pred_lr
axes[1, 1].scatter(y_pred_lr, residuals, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residual Plot')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_SupervisedLearning/Regression/linear_regression_results.png', dpi=150)
print("\n已保存結果圖表")

print("\n" + "=" * 80)
print("回歸評估指標說明：")
print("• R²：決定係數，越接近1越好")
print("• MSE：均方誤差，越小越好")
print("• RMSE：均方根誤差，與目標變量同單位")
print("• MAE：平均絕對誤差，對異常值不敏感")
print("=" * 80)
