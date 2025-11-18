"""
非線性回歸（Nonlinear Regression）
處理非線性關係的回歸算法

包含：
- 多項式回歸（Polynomial Regression）
- 支持向量回歸（SVR - Support Vector Regression）
- 樣條回歸（Spline Regression）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_diabetes, make_regression
from scipy.interpolate import make_interp_spline, BSpline, UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

# 導入工具模塊 / Import utility modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import RANDOM_STATE, TEST_SIZE, setup_chinese_fonts, create_subplots, get_output_path, save_figure

# 設置中文字體 / Setup Chinese fonts
setup_chinese_fonts()

print("=" * 80)
print("非線性回歸（Nonlinear Regression）教程".center(80))
print("=" * 80)

# ============================================================================
# 第一部分：非線性回歸概述
# Part 1: Nonlinear Regression Overview
# ============================================================================
print("\n【第一部分】非線性回歸概述")
print("-" * 80)
print("""
什麼是非線性回歸？
What is Nonlinear Regression?

當因變量和自變量之間不是線性關係時，需要使用非線性回歸。
When the relationship between dependent and independent variables is not linear,
we need nonlinear regression.

與線性回歸的區別：
Difference from Linear Regression:
• 線性回歸：y = β₀ + β₁x₁ + β₂x₂ + ... (線性組合)
• 非線性回歸：y = f(x) (可以是任何非線性函數)

常見非線性回歸方法：
Common Nonlinear Regression Methods:
1. 多項式回歸（Polynomial Regression）- 使用多項式特徵
2. 支持向量回歸（SVR）- 使用核技巧映射到高維空間
3. 樣條回歸（Spline Regression）- 分段多項式擬合
4. 決策樹回歸（下一個教程）
5. 神經網絡回歸（深度學習章節）
""")

# ============================================================================
# 評估函數 / Evaluation Functions
# ============================================================================
def evaluate_regression(y_true, y_pred, model_name='Model'):
    """
    計算回歸評估指標
    Calculate regression evaluation metrics

    Args:
        y_true: 真實值 / True values
        y_pred: 預測值 / Predicted values
        model_name: 模型名稱 / Model name

    Returns:
        dict: 包含 R², RMSE, MAE 的字典
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n{model_name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    return {'r2': r2, 'rmse': rmse, 'mae': mae}

# ============================================================================
# 第二部分：多項式回歸（Polynomial Regression）
# Part 2: Polynomial Regression
# ============================================================================
print("\n【第二部分】多項式回歸（Polynomial Regression）")
print("-" * 80)
print("""
多項式回歸原理：
Polynomial Regression Principle:

將特徵進行多項式轉換，然後使用線性回歸擬合。
Transform features into polynomial features, then fit with linear regression.

例如，對於單變量 x：
For example, for single variable x:
• 1次（線性）: y = β₀ + β₁x
• 2次（二次）: y = β₀ + β₁x + β₂x²
• 3次（三次）: y = β₀ + β₁x + β₂x² + β₃x³
• n次：y = β₀ + β₁x + β₂x² + ... + βₙxⁿ

PolynomialFeatures 使用：
• degree: 多項式階數
• include_bias: 是否包含截距項
• interaction_only: 是否只包含交互項

注意事項：
Cautions:
⚠ 階數過高會導致過擬合
⚠ 需要特徵縮放（StandardScaler）
⚠ 特徵數量隨階數指數增長
""")

# 生成非線性數據 / Generate nonlinear data
np.random.seed(RANDOM_STATE)
n_samples = 100

# 創建具有非線性關係的數據
X_poly = np.linspace(-3, 3, n_samples).reshape(-1, 1)
y_poly = 0.5 * X_poly.ravel()**3 - 2 * X_poly.ravel()**2 + X_poly.ravel() + np.random.randn(n_samples) * 2

# 數據分割
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y_poly, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"\n訓練集大小：{X_train.shape}")
print(f"測試集大小：{X_test.shape}")

# ============================================================================
# 2.1 不同階數的多項式對比
# ============================================================================
print("\n【2.1】不同階數的多項式對比")
print("-" * 80)

# 測試不同階數的多項式
degrees = [1, 2, 3, 5, 10]
polynomial_results = {}

for degree in degrees:
    # 多項式特徵轉換
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # 訓練線性回歸模型
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # 預測
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # 評估
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    polynomial_results[degree] = {
        'model': model,
        'poly_features': poly_features,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

    print(f"\n多項式階數 {degree}:")
    print(f"  訓練集 R²: {train_r2:.4f}")
    print(f"  測試集 R²: {test_r2:.4f}")
    print(f"  測試集 RMSE: {test_rmse:.4f}")
    print(f"  特徵數量: {X_train_poly.shape[1]}")

    if abs(train_r2 - test_r2) > 0.1:
        print(f"  ⚠ 警告：可能存在過擬合（訓練集和測試集R²差異: {abs(train_r2 - test_r2):.4f}）")

# ============================================================================
# 2.2 可視化：多項式回歸拟合曲線（2x3子圖）
# ============================================================================
print("\n【2.2】生成可視化：多項式回歸拟合曲線")

fig, axes = create_subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# 用於繪製平滑曲線的點
X_plot = np.linspace(X_poly.min(), X_poly.max(), 300).reshape(-1, 1)

for idx, degree in enumerate(degrees):
    result = polynomial_results[degree]

    # 轉換繪圖數據
    X_plot_poly = result['poly_features'].transform(X_plot)
    y_plot = result['model'].predict(X_plot_poly)

    # 繪製
    axes[idx].scatter(X_train, y_train, alpha=0.5, s=30, label='訓練數據', color='blue')
    axes[idx].scatter(X_test, y_test, alpha=0.5, s=30, label='測試數據', color='green')
    axes[idx].plot(X_plot, y_plot, 'r-', linewidth=2, label=f'多項式擬合 (degree={degree})')

    axes[idx].set_xlabel('X', fontsize=11)
    axes[idx].set_ylabel('y', fontsize=11)
    axes[idx].set_title(f'多項式回歸 - 階數 {degree}\nTrain R²={result["train_r2"]:.3f}, Test R²={result["test_r2"]:.3f}',
                       fontsize=12, fontweight='bold')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

# 最後一個子圖：對比所有模型
for degree in degrees:
    result = polynomial_results[degree]
    X_plot_poly = result['poly_features'].transform(X_plot)
    y_plot = result['model'].predict(X_plot_poly)
    axes[5].plot(X_plot, y_plot, linewidth=2, label=f'Degree {degree}', alpha=0.7)

axes[5].scatter(X_train, y_train, alpha=0.3, s=20, color='gray', label='數據點')
axes[5].set_xlabel('X', fontsize=11)
axes[5].set_ylabel('y', fontsize=11)
axes[5].set_title('所有多項式階數對比\nAll Polynomial Degrees Comparison', fontsize=12, fontweight='bold')
axes[5].legend(fontsize=9)
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('polynomial_regression_comparison.png', 'Regression'))

# ============================================================================
# 2.3 過擬合識別
# ============================================================================
print("\n【2.3】過擬合識別")
print("-" * 80)

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# 左圖：訓練誤差 vs 測試誤差
train_r2_scores = [polynomial_results[d]['train_r2'] for d in degrees]
test_r2_scores = [polynomial_results[d]['test_r2'] for d in degrees]

axes[0].plot(degrees, train_r2_scores, 'o-', linewidth=2, markersize=8, label='訓練集 R²', color='blue')
axes[0].plot(degrees, test_r2_scores, 's-', linewidth=2, markersize=8, label='測試集 R²', color='red')
axes[0].set_xlabel('多項式階數 / Polynomial Degree', fontsize=12)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('訓練誤差 vs 測試誤差\nTraining vs Testing Error', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=3, color='green', linestyle='--', alpha=0.5, label='最佳階數')

# 右圖：RMSE
rmse_scores = [polynomial_results[d]['rmse'] for d in degrees]
axes[1].bar(degrees, rmse_scores, color='steelblue', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('多項式階數 / Polynomial Degree', fontsize=12)
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_title('測試集 RMSE vs 多項式階數\nTest RMSE vs Polynomial Degree', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# 標註最小值
min_idx = np.argmin(rmse_scores)
axes[1].bar(degrees[min_idx], rmse_scores[min_idx], color='green', alpha=0.7, edgecolor='black')

plt.tight_layout()
save_figure(fig, get_output_path('polynomial_overfitting_analysis.png', 'Regression'))

print("\n觀察結果：")
print("• 階數太低（1次）：欠擬合，訓練和測試誤差都高")
print("• 階數適中（2-3次）：最佳，測試誤差最小")
print("• 階數太高（10次）：過擬合，訓練誤差很低但測試誤差高")

# ============================================================================
# 第三部分：支持向量回歸（SVR - Support Vector Regression）
# Part 3: Support Vector Regression
# ============================================================================
print("\n【第三部分】支持向量回歸（SVR）")
print("-" * 80)
print("""
SVR 原理：
SVR Principle:

支持向量回歸是 SVM 在回歸問題上的應用。
Support Vector Regression is the application of SVM to regression problems.

核心概念：
Core Concepts:
1. ε-不敏感損失函數（ε-insensitive loss）
   - 只有當誤差 > ε 時才計算損失
   - 在 ε 範圍內的點不貢獻損失

2. 核技巧（Kernel Trick）
   - 將數據映射到高維空間
   - 在高維空間中進行線性回歸

常用核函數：
Common Kernel Functions:
• Linear: K(x, x') = x^T x'
• Polynomial: K(x, x') = (γx^T x' + r)^d
• RBF (高斯): K(x, x') = exp(-γ||x - x'||²)
• Sigmoid: K(x, x') = tanh(γx^T x' + r)

關鍵參數：
Key Parameters:
• C: 懲罰參數，控制對誤差的容忍度
• epsilon (ε): 不敏感區域的寬度
• gamma (γ): RBF、Poly、Sigmoid 核的參數

優點：
✓ 可處理非線性關係
✓ 對高維數據有效
✓ 泛化能力強

缺點：
✗ 訓練時間長（大數據集）
✗ 需要選擇合適的核函數和參數
✗ 需要特徵縮放
""")

# 加載糖尿病數據集 / Load diabetes dataset
diabetes = load_diabetes()
X_diabetes, y_diabetes = diabetes.data, diabetes.target

# 數據標準化（SVR 需要）
scaler = StandardScaler()
X_train_scaled, X_test_scaled, y_train_svr, y_test_svr = train_test_split(
    X_diabetes, y_diabetes, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_train_scaled = scaler.fit_transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)

print(f"\n數據集：{diabetes.DESCR.split('Diabetes dataset')[0]}")
print(f"特徵數量：{X_diabetes.shape[1]}")
print(f"樣本數量：{X_diabetes.shape[0]}")

# ============================================================================
# 3.1 不同核函數對比
# ============================================================================
print("\n【3.1】不同核函數對比")
print("-" * 80)

kernels = {
    'Linear': SVR(kernel='linear', C=1.0),
    'Polynomial (d=2)': SVR(kernel='poly', degree=2, C=1.0),
    'Polynomial (d=3)': SVR(kernel='poly', degree=3, C=1.0),
    'RBF': SVR(kernel='rbf', C=1.0, gamma='scale'),
    'Sigmoid': SVR(kernel='sigmoid', C=1.0, gamma='scale')
}

svr_results = {}

for name, model in kernels.items():
    # 訓練
    model.fit(X_train_scaled, y_train_svr)

    # 預測
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # 評估
    results = evaluate_regression(y_test_svr, y_test_pred, f"SVR ({name})")
    results['train_r2'] = r2_score(y_train_svr, y_train_pred)
    results['y_pred'] = y_test_pred

    svr_results[name] = results

# ============================================================================
# 3.2 可視化：SVR 不同核函數對比（2x3子圖）
# ============================================================================
print("\n【3.2】生成可視化：SVR 核函數對比")

fig, axes = create_subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (name, results) in enumerate(svr_results.items()):
    # 預測 vs 真實值散點圖
    axes[idx].scatter(y_test_svr, results['y_pred'], alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[idx].plot([y_test_svr.min(), y_test_svr.max()],
                   [y_test_svr.min(), y_test_svr.max()],
                   'r--', lw=2, label='完美預測線')

    axes[idx].set_xlabel('真實值 / True Values', fontsize=11)
    axes[idx].set_ylabel('預測值 / Predictions', fontsize=11)
    axes[idx].set_title(f'SVR - {name}\nR²={results["r2"]:.3f}, RMSE={results["rmse"]:.2f}',
                       fontsize=12, fontweight='bold')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

# 最後一個子圖：性能對比柱狀圖
kernel_names = list(svr_results.keys())
r2_scores = [svr_results[k]['r2'] for k in kernel_names]
rmse_scores = [svr_results[k]['rmse'] for k in kernel_names]

x_pos = np.arange(len(kernel_names))
width = 0.35

axes[5].bar(x_pos - width/2, r2_scores, width, label='R² Score', alpha=0.8, color='steelblue')
axes[5].bar(x_pos + width/2, [r/100 for r in rmse_scores], width, label='RMSE/100', alpha=0.8, color='coral')
axes[5].set_xlabel('核函數類型 / Kernel Type', fontsize=11)
axes[5].set_ylabel('分數 / Score', fontsize=11)
axes[5].set_title('SVR 核函數性能對比\nSVR Kernel Performance Comparison', fontsize=12, fontweight='bold')
axes[5].set_xticks(x_pos)
axes[5].set_xticklabels(kernel_names, rotation=15, ha='right', fontsize=9)
axes[5].legend(fontsize=10)
axes[5].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_figure(fig, get_output_path('svr_kernel_comparison.png', 'Regression'))

# ============================================================================
# 3.3 SVR 參數調優 - C 參數影響
# ============================================================================
print("\n【3.3】SVR 參數調優 - C 參數影響")
print("-" * 80)

C_values = [0.1, 1, 10, 100, 1000]
c_results = {}

for C in C_values:
    model = SVR(kernel='rbf', C=C, gamma='scale')
    model.fit(X_train_scaled, y_train_svr)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test_svr, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_svr, y_pred))

    c_results[C] = {'r2': r2, 'rmse': rmse}
    print(f"C={C:6}: R²={r2:.4f}, RMSE={rmse:.2f}")

# ============================================================================
# 3.4 可視化：參數影響分析
# ============================================================================
print("\n【3.4】生成可視化：SVR 參數影響")

fig, axes = create_subplots(1, 3, figsize=(18, 5))

# C 參數影響
C_list = list(c_results.keys())
C_r2 = [c_results[c]['r2'] for c in C_list]
C_rmse = [c_results[c]['rmse'] for c in C_list]

axes[0].semilogx(C_list, C_r2, 'o-', linewidth=2, markersize=8, color='blue')
axes[0].set_xlabel('C (懲罰參數)', fontsize=12)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('C 參數對 R² 的影響\nEffect of C on R²', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# epsilon 參數影響
epsilon_values = [0.01, 0.1, 0.5, 1.0, 2.0]
epsilon_r2 = []

for eps in epsilon_values:
    model = SVR(kernel='rbf', C=1.0, epsilon=eps, gamma='scale')
    model.fit(X_train_scaled, y_train_svr)
    y_pred = model.predict(X_test_scaled)
    epsilon_r2.append(r2_score(y_test_svr, y_pred))

axes[1].plot(epsilon_values, epsilon_r2, 'o-', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('Epsilon (ε)', fontsize=12)
axes[1].set_ylabel('R² Score', fontsize=12)
axes[1].set_title('Epsilon 參數對 R² 的影響\nEffect of Epsilon on R²', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# gamma 參數影響（RBF kernel）
gamma_values = [0.001, 0.01, 0.1, 1, 10]
gamma_r2 = []

for gamma in gamma_values:
    model = SVR(kernel='rbf', C=1.0, gamma=gamma)
    model.fit(X_train_scaled, y_train_svr)
    y_pred = model.predict(X_test_scaled)
    gamma_r2.append(r2_score(y_test_svr, y_pred))

axes[2].semilogx(gamma_values, gamma_r2, 'o-', linewidth=2, markersize=8, color='red')
axes[2].set_xlabel('Gamma (γ)', fontsize=12)
axes[2].set_ylabel('R² Score', fontsize=12)
axes[2].set_title('Gamma 參數對 R² 的影響 (RBF)\nEffect of Gamma on R²', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('svr_parameter_tuning.png', 'Regression'))

# ============================================================================
# 第四部分：樣條回歸（Spline Regression）
# Part 4: Spline Regression
# ============================================================================
print("\n【第四部分】樣條回歸（Spline Regression）")
print("-" * 80)
print("""
樣條回歸原理：
Spline Regression Principle:

樣條回歸使用分段多項式進行擬合，在連接點保持平滑。
Spline regression uses piecewise polynomials that are smooth at connection points.

主要類型：
Main Types:
1. 線性樣條（Linear Spline）- 分段線性
2. 二次樣條（Quadratic Spline）- 分段二次多項式
3. 三次樣條（Cubic Spline）- 分段三次多項式（最常用）
4. B樣條（B-Spline）- 基礎樣條
5. 平滑樣條（Smoothing Spline）- 帶平滑懲罰

優點：
✓ 比高階多項式更穩定
✓ 局部擬合，不會出現龍格現象
✓ 靈活性高

缺點：
✗ 需要選擇節點位置
✗ 邊界處可能不穩定
""")

# 使用之前的非線性數據
X_spline = X_poly
y_spline = y_poly

# 排序（樣條插值需要）
sort_idx = X_spline.ravel().argsort()
X_spline_sorted = X_spline.ravel()[sort_idx]
y_spline_sorted = y_spline[sort_idx]

# ============================================================================
# 4.1 不同樣條方法
# ============================================================================
print("\n【4.1】不同樣條方法對比")

# 創建密集點用於繪製平滑曲線
X_dense = np.linspace(X_spline.min(), X_spline.max(), 300)

# 1. B-Spline (cubic)
spl_cubic = make_interp_spline(X_spline_sorted, y_spline_sorted, k=3)
y_cubic = spl_cubic(X_dense)

# 2. Univariate Spline (smoothing spline)
spl_smooth = UnivariateSpline(X_spline_sorted, y_spline_sorted, s=50)
y_smooth = spl_smooth(X_dense)

# 3. 多項式回歸（3次）作為對比
poly_3 = PolynomialFeatures(degree=3)
X_poly_3 = poly_3.fit_transform(X_spline_sorted.reshape(-1, 1))
lr_poly = LinearRegression()
lr_poly.fit(X_poly_3, y_spline_sorted)
X_dense_poly = poly_3.transform(X_dense.reshape(-1, 1))
y_poly_3 = lr_poly.predict(X_dense_poly)

# ============================================================================
# 4.2 可視化：樣條回歸對比
# ============================================================================
print("\n【4.2】生成可視化：樣條回歸")

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# 左圖：不同樣條方法對比
axes[0].scatter(X_spline, y_spline, alpha=0.5, s=30, label='原始數據', color='gray')
axes[0].plot(X_dense, y_cubic, linewidth=2, label='三次 B-Spline', color='blue')
axes[0].plot(X_dense, y_smooth, linewidth=2, label='平滑樣條 (s=50)', color='green')
axes[0].plot(X_dense, y_poly_3, linewidth=2, label='3次多項式', color='red', linestyle='--')
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('y', fontsize=12)
axes[0].set_title('樣條回歸方法對比\nSpline Regression Methods Comparison', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 右圖：不同平滑參數的影響
smoothness_params = [10, 50, 100, 500]
for s in smoothness_params:
    spl = UnivariateSpline(X_spline_sorted, y_spline_sorted, s=s)
    y_spl = spl(X_dense)
    axes[1].plot(X_dense, y_spl, linewidth=2, label=f's={s}', alpha=0.7)

axes[1].scatter(X_spline, y_spline, alpha=0.3, s=20, color='gray', label='數據點')
axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('y', fontsize=12)
axes[1].set_title('平滑參數 s 的影響\nEffect of Smoothing Parameter s', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('spline_regression.png', 'Regression'))

# ============================================================================
# 第五部分：綜合對比
# Part 5: Comprehensive Comparison
# ============================================================================
print("\n【第五部分】綜合對比 - 所有非線性方法")
print("-" * 80)

# 創建不同類型的測試數據
np.random.seed(RANDOM_STATE)
n_test = 100

# 數據集 1：簡單非線性（單峰）
X_simple = np.linspace(-3, 3, n_test).reshape(-1, 1)
y_simple = X_simple.ravel()**2 + np.random.randn(n_test) * 2

# 數據集 2：複雜非線性（多峰）
X_complex = np.linspace(0, 4*np.pi, n_test).reshape(-1, 1)
y_complex = np.sin(X_complex.ravel()) * X_complex.ravel() + np.random.randn(n_test) * 0.5

# 數據集 3：噪聲數據
X_noisy = np.linspace(-3, 3, n_test).reshape(-1, 1)
y_noisy = 0.5 * X_noisy.ravel()**3 + np.random.randn(n_test) * 5

datasets = {
    '簡單非線性 (單峰)': (X_simple, y_simple),
    '複雜非線性 (多峰)': (X_complex, y_complex),
    '高噪聲數據': (X_noisy, y_noisy)
}

# ============================================================================
# 5.1 在不同數據集上測試所有方法
# ============================================================================
print("\n【5.1】在不同數據集上測試所有方法")

comparison_results = {}

for dataset_name, (X_data, y_data) in datasets.items():
    print(f"\n數據集：{dataset_name}")
    print("-" * 40)

    # 數據分割
    X_tr, X_te, y_tr, y_te = train_test_split(X_data, y_data, test_size=0.3, random_state=RANDOM_STATE)

    results = {}

    # 1. 多項式回歸 (degree=3)
    poly = PolynomialFeatures(degree=3)
    X_tr_poly = poly.fit_transform(X_tr)
    X_te_poly = poly.transform(X_te)
    lr = LinearRegression()
    lr.fit(X_tr_poly, y_tr)
    y_pred = lr.predict(X_te_poly)
    results['多項式回歸 (d=3)'] = r2_score(y_te, y_pred)
    print(f"  多項式回歸 (d=3): R²={results['多項式回歸 (d=3)']:.4f}")

    # 2. SVR (RBF)
    scaler_temp = StandardScaler()
    X_tr_scaled = scaler_temp.fit_transform(X_tr)
    X_te_scaled = scaler_temp.transform(X_te)
    svr = SVR(kernel='rbf', C=10, gamma='scale')
    svr.fit(X_tr_scaled, y_tr)
    y_pred = svr.predict(X_te_scaled)
    results['SVR (RBF)'] = r2_score(y_te, y_pred)
    print(f"  SVR (RBF): R²={results['SVR (RBF)']:.4f}")

    # 3. Ridge回歸（多項式特徵）
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_poly, y_tr)
    y_pred = ridge.predict(X_te_poly)
    results['Ridge回歸 (poly d=3)'] = r2_score(y_te, y_pred)
    print(f"  Ridge回歸 (poly d=3): R²={results['Ridge回歸 (poly d=3)']:.4f}")

    comparison_results[dataset_name] = results

# ============================================================================
# 5.2 可視化：性能對比表格
# ============================================================================
print("\n【5.2】生成可視化：綜合性能對比")

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# 創建對比表格數據
methods = ['多項式回歸 (d=3)', 'SVR (RBF)', 'Ridge回歸 (poly d=3)']
dataset_names = list(comparison_results.keys())

table_data = []
for method in methods:
    row = [comparison_results[ds][method] for ds in dataset_names]
    table_data.append(row)

table_data = np.array(table_data)

# 熱力圖
im = axes[0].imshow(table_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
axes[0].set_xticks(np.arange(len(dataset_names)))
axes[0].set_yticks(np.arange(len(methods)))
axes[0].set_xticklabels(dataset_names, fontsize=10)
axes[0].set_yticklabels(methods, fontsize=10)
axes[0].set_title('不同數據集上的 R² 分數\nR² Scores on Different Datasets', fontsize=13, fontweight='bold')

# 添加數值標註
for i in range(len(methods)):
    for j in range(len(dataset_names)):
        text = axes[0].text(j, i, f'{table_data[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=11, fontweight='bold')

fig.colorbar(im, ax=axes[0], label='R² Score')

# 柱狀圖對比
x = np.arange(len(dataset_names))
width = 0.25

for i, method in enumerate(methods):
    offset = width * (i - 1)
    values = [comparison_results[ds][method] for ds in dataset_names]
    axes[1].bar(x + offset, values, width, label=method, alpha=0.8)

axes[1].set_xlabel('數據集類型', fontsize=12)
axes[1].set_ylabel('R² Score', fontsize=12)
axes[1].set_title('模型在不同數據集上的性能對比\nModel Performance on Different Datasets',
                  fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(dataset_names, fontsize=10)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_figure(fig, get_output_path('comprehensive_comparison.png', 'Regression'))

# ============================================================================
# 第六部分：實際應用案例
# Part 6: Practical Applications
# ============================================================================
print("\n【第六部分】實際應用案例 - 糖尿病數據預測")
print("-" * 80)

# 使用糖尿病數據集進行完整的建模流程
X_app, y_app = diabetes.data, diabetes.target

# 數據分割
X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(
    X_app, y_app, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# 特徵縮放
scaler_app = StandardScaler()
X_train_app_scaled = scaler_app.fit_transform(X_train_app)
X_test_app_scaled = scaler_app.transform(X_test_app)

# 測試多個模型
models_app = {
    '線性回歸': LinearRegression(),
    '多項式回歸 (d=2)': None,  # 需要特殊處理
    'Ridge回歸': Ridge(alpha=1.0),
    'SVR (RBF)': SVR(kernel='rbf', C=10, gamma='scale'),
    'SVR (Poly)': SVR(kernel='poly', degree=2, C=10)
}

app_results = {}

# 線性回歸、Ridge、SVR
for name, model in models_app.items():
    if model is not None:
        if 'SVR' in name:
            model.fit(X_train_app_scaled, y_train_app)
            y_pred = model.predict(X_test_app_scaled)
        else:
            model.fit(X_train_app, y_train_app)
            y_pred = model.predict(X_test_app)

        app_results[name] = {
            'r2': r2_score(y_test_app, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_app, y_pred)),
            'mae': mean_absolute_error(y_test_app, y_pred),
            'y_pred': y_pred
        }

        print(f"\n{name}:")
        print(f"  R²: {app_results[name]['r2']:.4f}")
        print(f"  RMSE: {app_results[name]['rmse']:.2f}")
        print(f"  MAE: {app_results[name]['mae']:.2f}")

# 多項式回歸
poly_app = PolynomialFeatures(degree=2)
X_train_poly_app = poly_app.fit_transform(X_train_app)
X_test_poly_app = poly_app.transform(X_test_app)
lr_poly_app = LinearRegression()
lr_poly_app.fit(X_train_poly_app, y_train_app)
y_pred_poly = lr_poly_app.predict(X_test_poly_app)

app_results['多項式回歸 (d=2)'] = {
    'r2': r2_score(y_test_app, y_pred_poly),
    'rmse': np.sqrt(mean_squared_error(y_test_app, y_pred_poly)),
    'mae': mean_absolute_error(y_test_app, y_pred_poly),
    'y_pred': y_pred_poly
}

print(f"\n多項式回歸 (d=2):")
print(f"  R²: {app_results['多項式回歸 (d=2)']['r2']:.4f}")
print(f"  RMSE: {app_results['多項式回歸 (d=2)']['rmse']:.2f}")
print(f"  MAE: {app_results['多項式回歸 (d=2)']['mae']:.2f}")

# ============================================================================
# 6.1 可視化：實際應用結果
# ============================================================================
print("\n【6.1】生成可視化：實際應用案例結果")

fig, axes = create_subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# 為每個模型繪製預測vs真實值
for idx, (name, results) in enumerate(app_results.items()):
    axes[idx].scatter(y_test_app, results['y_pred'], alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[idx].plot([y_test_app.min(), y_test_app.max()],
                   [y_test_app.min(), y_test_app.max()],
                   'r--', lw=2, label='完美預測')

    axes[idx].set_xlabel('真實值 / True Values', fontsize=11)
    axes[idx].set_ylabel('預測值 / Predictions', fontsize=11)
    axes[idx].set_title(f'{name}\nR²={results["r2"]:.3f}, RMSE={results["rmse"]:.2f}',
                       fontsize=12, fontweight='bold')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

# 最後一個子圖：所有模型性能對比
model_names = list(app_results.keys())
r2_values = [app_results[m]['r2'] for m in model_names]

axes[5].barh(model_names, r2_values, color='steelblue', alpha=0.7, edgecolor='black')
axes[5].set_xlabel('R² Score', fontsize=12)
axes[5].set_title('所有模型 R² 對比\nR² Comparison of All Models', fontsize=13, fontweight='bold')
axes[5].grid(True, alpha=0.3, axis='x')
axes[5].set_xlim(0, max(r2_values) * 1.1)

# 標註數值
for i, (name, value) in enumerate(zip(model_names, r2_values)):
    axes[5].text(value + 0.01, i, f'{value:.4f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
save_figure(fig, get_output_path('application_case_results.png', 'Regression'))

# ============================================================================
# 總結報告
# Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("非線性回歸教程總結".center(80))
print("=" * 80)

print("""
📊 本教程涵蓋的內容：

1. 多項式回歸（Polynomial Regression）
   ✓ 不同階數的影響（1-10次）
   ✓ 過擬合與欠擬合識別
   ✓ 最佳階數選擇

2. 支持向量回歸（SVR）
   ✓ 不同核函數（Linear, Polynomial, RBF, Sigmoid）
   ✓ 參數調優（C, epsilon, gamma）
   ✓ 核函數選擇策略

3. 樣條回歸（Spline Regression）
   ✓ B-Spline
   ✓ 平滑樣條
   ✓ 平滑參數影響

4. 綜合對比
   ✓ 在不同數據集上的表現
   ✓ 方法選擇建議

5. 實際應用
   ✓ 糖尿病數據預測
   ✓ 完整建模流程

📈 生成的可視化圖表：
• polynomial_regression_comparison.png - 多項式回歸階數對比
• polynomial_overfitting_analysis.png - 過擬合分析
• svr_kernel_comparison.png - SVR核函數對比
• svr_parameter_tuning.png - SVR參數調優
• spline_regression.png - 樣條回歸
• comprehensive_comparison.png - 綜合對比
• application_case_results.png - 實際應用案例

💡 關鍵要點：
1. 多項式回歸簡單但容易過擬合，需要謹慎選擇階數
2. SVR適合小到中等規模數據，需要特徵縮放
3. RBF核是SVR的好選擇，參數需要調優
4. 樣條回歸在局部擬合上表現優秀
5. 不同方法適用於不同類型的數據

🎯 下一步：
• 學習基於樹的回歸方法（Decision Tree, Random Forest, XGBoost）
• 探索深度學習回歸（神經網絡）
• 實踐更多真實案例
""")

print("=" * 80)
print("教程結束！所有圖表已保存到 output/Regression/ 目錄".center(80))
print("=" * 80)
