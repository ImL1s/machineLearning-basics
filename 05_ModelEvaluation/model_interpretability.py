"""
模型解釋性完整指南
Model Interpretability Guide with SHAP and LIME

深入理解機器學習模型的決策過程
Understanding the decision-making process of machine learning models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import time

# 導入工具模塊 / Import utility modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import RANDOM_STATE, TEST_SIZE, setup_chinese_fonts, create_subplots

warnings.filterwarnings('ignore')
setup_chinese_fonts()

# ============================================================================
# 依賴檢查 / Dependency Check
# ============================================================================
print("=" * 80)
print("模型解釋性完整指南".center(76))
print("Model Interpretability Guide".center(76))
print("=" * 80)

print("\n【依賴檢查 / Dependency Check】")
print("-" * 80)

# 檢查 SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    print("✓ SHAP 已安裝 (SHAP installed)")
    print(f"  版本 (Version): {shap.__version__}")
except ImportError:
    SHAP_AVAILABLE = False
    print("✗ SHAP 未安裝 (SHAP not installed)")
    print("  安裝方法 (Install): pip install shap")
    print("  文檔 (Docs): https://shap.readthedocs.io/")

# 檢查 LIME
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
    print("✓ LIME 已安裝 (LIME installed)")
    print(f"  版本 (Version): {lime.__version__}")
except ImportError:
    LIME_AVAILABLE = False
    print("✗ LIME 未安裝 (LIME not installed)")
    print("  安裝方法 (Install): pip install lime")
    print("  文檔 (Docs): https://lime-ml.readthedocs.io/")

# 檢查 XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✓ XGBoost 已安裝 (XGBoost installed)")
except ImportError:
    XGB_AVAILABLE = False
    print("✗ XGBoost 未安裝 (XGBoost not installed)")
    print("  安裝方法 (Install): pip install xgboost")

if not SHAP_AVAILABLE and not LIME_AVAILABLE:
    print("\n⚠️  警告: 需要安裝 SHAP 或 LIME 才能運行此示例")
    print("⚠️  Warning: Need to install SHAP or LIME to run this example")
    sys.exit(1)

# ============================================================================
# Part 1: 模型解釋性簡介
# Introduction to Model Interpretability
# ============================================================================
print("\n" + "=" * 80)
print("【Part 1】模型解釋性簡介".center(76))
print("Introduction to Model Interpretability".center(76))
print("=" * 80)

print("""
1. 什麼是模型解釋性？(What is Model Interpretability?)
   - 模型解釋性是指理解和解釋機器學習模型決策的能力
   - 它幫助我們理解「為什麼」模型做出某個預測
   - Interpretability is the ability to understand and explain ML model decisions
   - It helps us understand "why" a model makes certain predictions

2. 為什麼重要？(Why is it Important?)
   ✓ 監管要求 (Regulatory Requirements): GDPR、金融法規要求可解釋性
   ✓ 建立信任 (Building Trust): 讓用戶信任模型決策
   ✓ 調試模型 (Debugging): 發現模型錯誤和偏差
   ✓ 改進模型 (Model Improvement): 指導特徵工程和模型優化
   ✓ 風險管理 (Risk Management): 識別潛在問題和異常

3. 解釋類型 (Types of Explanations)
   A. 全局解釋 (Global Interpretation)
      - 理解模型的整體行為
      - 哪些特徵對模型最重要？
      - Understanding overall model behavior
      - Which features are most important to the model?

   B. 局部解釋 (Local Interpretation)
      - 解釋單個預測
      - 為什麼模型對這個樣本做出這個預測？
      - Explaining individual predictions
      - Why did the model make this prediction for this sample?

4. 解釋方法分類 (Classification of Explanation Methods)
   A. 模型特定方法 (Model-Specific Methods)
      - 僅適用於特定類型的模型
      - 例如: 決策樹的特徵重要性
      - Only work for specific model types

   B. 模型無關方法 (Model-Agnostic Methods)
      - 可以應用於任何機器學習模型
      - 例如: LIME、SHAP
      - Can be applied to any ML model

5. SHAP vs LIME 簡介

   SHAP (SHapley Additive exPlanations)
   ✓ 基於博弈論的 Shapley 值
   ✓ 保證一致性和局部準確性
   ✓ 既可以全局解釋也可以局部解釋
   ✓ 計算成本較高，但結果更可靠

   LIME (Local Interpretable Model-agnostic Explanations)
   ✓ 基於局部線性近似
   ✓ 模型無關，適用於任何黑盒模型
   ✓ 主要用於局部解釋
   ✓ 計算速度快，但可能不夠穩定
""")

# ============================================================================
# Part 2: SHAP (SHapley Additive exPlanations)
# ============================================================================
if SHAP_AVAILABLE:
    print("\n" + "=" * 80)
    print("【Part 2】SHAP (SHapley Additive exPlanations)".center(76))
    print("=" * 80)

    print("""
SHAP 核心概念 (Core Concepts):

1. Shapley 值原理 (Shapley Value from Game Theory)
   - 來自合作博弈論，用於公平分配貢獻
   - 每個特徵被視為「玩家」，預測結果是「總獎勵」
   - 計算每個特徵對預測的平均邊際貢獻

2. SHAP 的優勢 (Advantages of SHAP)
   ✓ 理論保證: 局部準確性、一致性、缺失性
   ✓ 統一框架: 整合多種解釋方法
   ✓ 豐富可視化: 提供多種圖表類型

3. SHAP Explainer 類型
   - TreeExplainer: 針對樹模型優化（快速）
   - LinearExplainer: 針對線性模型
   - KernelExplainer: 模型無關（較慢）
   - DeepExplainer: 針對深度學習模型
    """)

    # ========================================================================
    # 2.1 分類任務示例 - Breast Cancer
    # Classification Task Example
    # ========================================================================
    print("\n" + "-" * 80)
    print("【2.1】SHAP 分類任務示例 - 乳腺癌診斷")
    print("SHAP Classification Example - Breast Cancer Diagnosis")
    print("-" * 80)

    # 載入數據
    cancer = load_breast_cancer()
    X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y_cancer = cancer.target

    print(f"\n數據集信息:")
    print(f"  樣本數: {X_cancer.shape[0]}")
    print(f"  特徵數: {X_cancer.shape[1]}")
    print(f"  類別: {cancer.target_names}")

    # 訓練測試分割
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cancer, y_cancer, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cancer
    )

    # 訓練 Random Forest 分類器
    print("\n訓練 Random Forest 分類器...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_clf.fit(X_train_c, y_train_c)

    train_score = rf_clf.score(X_train_c, y_train_c)
    test_score = rf_clf.score(X_test_c, y_test_c)
    print(f"✓ 訓練集準確率: {train_score:.4f}")
    print(f"✓ 測試集準確率: {test_score:.4f}")

    # 計算 SHAP 值
    print("\n計算 SHAP 值 (使用 TreeExplainer)...")
    start_time = time.time()

    # 使用 TreeExplainer（針對樹模型優化）
    explainer_tree = shap.TreeExplainer(rf_clf)
    shap_values_tree = explainer_tree.shap_values(X_test_c)

    elapsed_time = time.time() - start_time
    print(f"✓ 計算完成，耗時: {elapsed_time:.2f} 秒")

    # 處理不同版本的 SHAP 返回格式
    if isinstance(shap_values_tree, list):
        # 舊版本：返回列表
        shap_values_class1 = shap_values_tree[1]
        print(f"  SHAP values shape (list): {len(shap_values_tree)} classes")
    else:
        # 新版本：返回數組
        if len(shap_values_tree.shape) == 3:
            shap_values_class1 = shap_values_tree[:, :, 1]  # 選擇正類
        else:
            shap_values_class1 = shap_values_tree  # 回歸或單一輸出

    print(f"  使用的 SHAP values shape: {shap_values_class1.shape}")

    # ========================================================================
    # SHAP 可視化 1: Summary Plot (全局特徵重要性)
    # ========================================================================
    print("\n生成可視化圖表...")

    fig1, axes1 = create_subplots(1, 2, figsize=(18, 6))

    # 對於二分類，我們通常只看正類（惡性）的 SHAP 值
    plt.sca(axes1[0])
    shap.summary_plot(
        shap_values_class1,
        X_test_c,
        plot_type="bar",
        show=False,
        max_display=10
    )
    axes1[0].set_title('SHAP Summary Plot - 特徵重要性排名\n(Feature Importance Ranking)',
                      fontsize=14, pad=15)

    plt.sca(axes1[1])
    shap.summary_plot(
        shap_values_class1,
        X_test_c,
        show=False,
        max_display=10
    )
    axes1[1].set_title('SHAP Beeswarm Plot - 特徵影響分布\n(Feature Impact Distribution)',
                      fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig('/tmp/shap_summary_plots.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 1: SHAP Summary Plots 已保存")
    plt.close()

    # ========================================================================
    # SHAP 可視化 2: Waterfall Plot (單個預測解釋)
    # ========================================================================
    fig2, axes2 = create_subplots(1, 2, figsize=(18, 8))

    # 選擇一個被正確分類為惡性的樣本
    malignant_idx = np.where((y_test_c == 1) & (rf_clf.predict(X_test_c) == 1))[0][0]

    # 獲取 expected_value
    if isinstance(explainer_tree.expected_value, (list, np.ndarray)) and len(np.array(explainer_tree.expected_value).shape) > 0:
        expected_val = explainer_tree.expected_value[1] if len(explainer_tree.expected_value) > 1 else explainer_tree.expected_value[0]
    else:
        expected_val = explainer_tree.expected_value

    # 創建 Explanation 對象
    explanation_malignant = shap.Explanation(
        values=shap_values_class1[malignant_idx],
        base_values=expected_val,
        data=X_test_c.iloc[malignant_idx].values,
        feature_names=X_test_c.columns.tolist()
    )

    plt.sca(axes2[0])
    shap.plots.waterfall(explanation_malignant, max_display=10, show=False)
    axes2[0].set_title(f'Waterfall Plot - 惡性樣本 #{malignant_idx}\n(Malignant Sample Explanation)',
                      fontsize=14, pad=15)

    # 選擇一個被正確分類為良性的樣本
    benign_idx = np.where((y_test_c == 0) & (rf_clf.predict(X_test_c) == 0))[0][0]

    explanation_benign = shap.Explanation(
        values=shap_values_class1[benign_idx],
        base_values=expected_val,
        data=X_test_c.iloc[benign_idx].values,
        feature_names=X_test_c.columns.tolist()
    )

    plt.sca(axes2[1])
    shap.plots.waterfall(explanation_benign, max_display=10, show=False)
    axes2[1].set_title(f'Waterfall Plot - 良性樣本 #{benign_idx}\n(Benign Sample Explanation)',
                      fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig('/tmp/shap_waterfall_plots.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 2: SHAP Waterfall Plots 已保存")
    plt.close()

    # ========================================================================
    # SHAP 可視化 3: Force Plot (單個樣本詳細視圖)
    # ========================================================================
    fig3 = plt.figure(figsize=(18, 5))

    # Force plot 顯示推動預測向正類或負類的特徵
    shap.force_plot(
        expected_val,
        shap_values_class1[malignant_idx],
        X_test_c.iloc[malignant_idx],
        matplotlib=True,
        show=False,
        text_rotation=15
    )
    plt.title(f'Force Plot - 惡性樣本 #{malignant_idx} 的特徵貢獻\n(Feature Contributions for Malignant Sample)',
              fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig('/tmp/shap_force_plot.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 3: SHAP Force Plot 已保存")
    plt.close()

    # ========================================================================
    # SHAP 可視化 4: Dependence Plot (特徵交互)
    # ========================================================================
    fig4, axes4 = create_subplots(2, 2, figsize=(18, 14))

    # 選擇最重要的 4 個特徵
    feature_importance = np.abs(shap_values_class1).mean(axis=0)
    top_features_idx = np.argsort(feature_importance)[-4:]

    for idx, feat_idx in enumerate(top_features_idx):
        ax = axes4[idx // 2, idx % 2]
        plt.sca(ax)

        shap.dependence_plot(
            feat_idx,
            shap_values_class1,
            X_test_c,
            show=False,
            ax=ax
        )
        ax.set_title(f'Dependence Plot - {X_test_c.columns[feat_idx]}',
                    fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig('/tmp/shap_dependence_plots.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 4: SHAP Dependence Plots 已保存")
    plt.close()

    # ========================================================================
    # SHAP 可視化 5: Decision Plot (多樣本對比)
    # ========================================================================
    fig5 = plt.figure(figsize=(18, 10))

    # 選擇 20 個樣本進行對比
    sample_indices = np.random.choice(len(X_test_c), size=20, replace=False)

    shap.decision_plot(
        expected_val,
        shap_values_class1[sample_indices],
        X_test_c.iloc[sample_indices],
        show=False,
        feature_display_range=slice(-1, -11, -1)  # 顯示前 10 個特徵
    )
    plt.title('Decision Plot - 多樣本預測路徑對比\n(Multiple Sample Prediction Paths)',
              fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig('/tmp/shap_decision_plot.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 5: SHAP Decision Plot 已保存")
    plt.close()

    # ========================================================================
    # 2.2 回歸任務示例 - Diabetes
    # Regression Task Example
    # ========================================================================
    print("\n" + "-" * 80)
    print("【2.2】SHAP 回歸任務示例 - 糖尿病進展預測")
    print("SHAP Regression Example - Diabetes Progression")
    print("-" * 80)

    # 載入數據
    diabetes = load_diabetes()
    X_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y_diabetes = diabetes.target

    print(f"\n數據集信息:")
    print(f"  樣本數: {X_diabetes.shape[0]}")
    print(f"  特徵數: {X_diabetes.shape[1]}")
    print(f"  目標變量: 糖尿病進展指標")

    # 訓練測試分割
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_diabetes, y_diabetes, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 訓練模型
    if XGB_AVAILABLE:
        print("\n訓練 XGBoost 回歸器...")
        reg_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
    else:
        print("\n訓練 Gradient Boosting 回歸器...")
        reg_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )

    reg_model.fit(X_train_r, y_train_r)

    train_r2 = reg_model.score(X_train_r, y_train_r)
    test_r2 = reg_model.score(X_test_r, y_test_r)
    print(f"✓ 訓練集 R² 分數: {train_r2:.4f}")
    print(f"✓ 測試集 R² 分數: {test_r2:.4f}")

    # 計算 SHAP 值
    print("\n計算 SHAP 值...")
    start_time = time.time()

    explainer_reg = shap.TreeExplainer(reg_model)
    shap_values_reg = explainer_reg.shap_values(X_test_r)

    elapsed_time = time.time() - start_time
    print(f"✓ 計算完成，耗時: {elapsed_time:.2f} 秒")

    # ========================================================================
    # SHAP 可視化 6: 回歸任務特徵重要性對比
    # ========================================================================
    fig6, axes6 = create_subplots(1, 2, figsize=(18, 6))

    # 傳統特徵重要性
    feature_imp_trad = reg_model.feature_importances_
    sorted_idx = np.argsort(feature_imp_trad)[-10:]

    axes6[0].barh(range(len(sorted_idx)), feature_imp_trad[sorted_idx])
    axes6[0].set_yticks(range(len(sorted_idx)))
    axes6[0].set_yticklabels(X_diabetes.columns[sorted_idx])
    axes6[0].set_xlabel('特徵重要性 (Feature Importance)', fontsize=12)
    axes6[0].set_title('傳統特徵重要性\n(Traditional Feature Importance)', fontsize=14)
    axes6[0].grid(axis='x', alpha=0.3)

    # SHAP 特徵重要性
    plt.sca(axes6[1])
    shap.summary_plot(
        shap_values_reg,
        X_test_r,
        plot_type="bar",
        show=False,
        max_display=10
    )
    axes6[1].set_title('SHAP 特徵重要性\n(SHAP Feature Importance)', fontsize=14)

    plt.tight_layout()
    plt.savefig('/tmp/shap_regression_feature_importance.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 6: 特徵重要性對比已保存")
    plt.close()

    # ========================================================================
    # SHAP 可視化 7: 回歸任務 Beeswarm Plot
    # ========================================================================
    fig7 = plt.figure(figsize=(18, 8))

    shap.summary_plot(
        shap_values_reg,
        X_test_r,
        show=False,
        max_display=10
    )
    plt.title('SHAP Beeswarm Plot - 回歸任務特徵影響\n(Regression Feature Impact)',
              fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig('/tmp/shap_regression_beeswarm.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 7: 回歸 Beeswarm Plot 已保存")
    plt.close()

    # ========================================================================
    # SHAP 可視化 8: 特徵交互熱力圖
    # ========================================================================
    print("\n計算特徵交互強度...")

    # 使用較小的樣本計算交互
    sample_size = min(100, len(X_test_r))
    X_sample = X_test_r.iloc[:sample_size]

    shap_interaction_values = explainer_reg.shap_interaction_values(X_sample)

    # 計算交互強度
    interaction_strength = np.abs(shap_interaction_values).mean(0)

    fig8, ax8 = create_subplots(1, 1, figsize=(14, 12))

    sns.heatmap(
        interaction_strength,
        xticklabels=X_diabetes.columns,
        yticklabels=X_diabetes.columns,
        cmap='RdYlBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        ax=ax8,
        cbar_kws={'label': '交互強度 (Interaction Strength)'}
    )
    ax8.set_title('SHAP 特徵交互熱力圖\n(SHAP Feature Interaction Heatmap)',
                 fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig('/tmp/shap_interaction_heatmap.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 8: 特徵交互熱力圖已保存")
    plt.close()

    print("\n" + "=" * 80)
    print("SHAP 示例完成！".center(76))
    print("=" * 80)

# ============================================================================
# Part 3: LIME (Local Interpretable Model-agnostic Explanations)
# ============================================================================
if LIME_AVAILABLE:
    print("\n" + "=" * 80)
    print("【Part 3】LIME (Local Interpretable Model-agnostic Explanations)".center(76))
    print("=" * 80)

    print("""
LIME 核心概念 (Core Concepts):

1. LIME 原理 (LIME Principle)
   - 在樣本附近生成擾動數據
   - 用簡單模型（線性模型）局部近似複雜模型
   - 解釋簡單模型來理解原模型的決策

2. 模型無關性 (Model-Agnostic)
   ✓ 可以解釋任何黑盒模型
   ✓ 只需要能夠獲得模型預測即可
   ✓ 適用於表格、文本、圖像等多種數據類型

3. 適用場景 (Use Cases)
   - 需要快速解釋單個預測
   - 模型是完全的黑盒（無法訪問內部結構）
   - 需要向非技術人員解釋預測
    """)

    # ========================================================================
    # 3.1 表格數據解釋 - Iris 數據集
    # Tabular Data Explanation
    # ========================================================================
    print("\n" + "-" * 80)
    print("【3.1】LIME 表格數據解釋 - 鳶尾花分類")
    print("LIME Tabular Data Explanation - Iris Classification")
    print("-" * 80)

    # 載入數據
    iris = load_iris()
    X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_iris = iris.target

    print(f"\n數據集信息:")
    print(f"  樣本數: {X_iris.shape[0]}")
    print(f"  特徵數: {X_iris.shape[1]}")
    print(f"  類別數: {len(iris.target_names)}")
    print(f"  類別: {iris.target_names}")

    # 訓練測試分割
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_iris, y_iris, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_iris
    )

    # ========================================================================
    # 3.1.1 黑盒模型 1: SVM
    # ========================================================================
    print("\n訓練 SVM 分類器（黑盒模型）...")

    # 標準化數據（SVM 需要）
    scaler = StandardScaler()
    X_train_i_scaled = scaler.fit_transform(X_train_i)
    X_test_i_scaled = scaler.transform(X_test_i)

    svm_model = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    svm_model.fit(X_train_i_scaled, y_train_i)

    svm_score = svm_model.score(X_test_i_scaled, y_test_i)
    print(f"✓ SVM 測試集準確率: {svm_score:.4f}")

    # 創建 LIME 解釋器
    print("\n創建 LIME 解釋器...")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_i_scaled,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        mode='classification',
        random_state=RANDOM_STATE
    )

    # ========================================================================
    # LIME 可視化 9: 多個樣本解釋對比
    # ========================================================================
    print("\n生成 LIME 解釋...")

    fig9, axes9 = create_subplots(2, 2, figsize=(18, 14))

    # 為每個類別選擇一個樣本
    lime_explanations = []

    for class_idx in range(3):
        # 選擇一個該類別的樣本
        class_samples = np.where(y_test_i == class_idx)[0]
        if len(class_samples) > 0:
            sample_idx = class_samples[0]

            # 生成解釋
            exp = lime_explainer.explain_instance(
                X_test_i_scaled[sample_idx],
                svm_model.predict_proba,
                num_features=4,
                top_labels=1
            )

            lime_explanations.append(exp)

            # 繪製
            ax = axes9[class_idx // 2, class_idx % 2]

            # 獲取預測類別
            pred_class = svm_model.predict(X_test_i_scaled[sample_idx].reshape(1, -1))[0]
            pred_proba = svm_model.predict_proba(X_test_i_scaled[sample_idx].reshape(1, -1))[0]

            # 獲取特徵貢獻
            exp_list = exp.as_list(label=pred_class)
            features = [item[0] for item in exp_list]
            values = [item[1] for item in exp_list]

            # 繪製條形圖
            colors = ['green' if v > 0 else 'red' for v in values]
            ax.barh(range(len(features)), values, color=colors, alpha=0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('特徵貢獻 (Feature Contribution)', fontsize=11)
            ax.set_title(
                f'LIME 解釋 - 樣本 {sample_idx}\n'
                f'真實: {iris.target_names[y_test_i[sample_idx]]}, '
                f'預測: {iris.target_names[pred_class]} ({pred_proba[pred_class]:.3f})',
                fontsize=12, pad=10
            )
            ax.grid(axis='x', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # 在第四個子圖中顯示圖例說明
    axes9[1, 1].axis('off')
    legend_text = """
    LIME 解釋說明:

    • 綠色條: 特徵增加該類別的預測概率
    • 紅色條: 特徵降低該類別的預測概率
    • 條的長度: 表示特徵的影響程度

    LIME Explanation Legend:

    • Green bar: Feature increases prediction probability
    • Red bar: Feature decreases prediction probability
    • Bar length: Magnitude of feature impact
    """
    axes9[1, 1].text(0.1, 0.5, legend_text, fontsize=11,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('/tmp/lime_explanations_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 9: LIME 多樣本解釋對比已保存")
    plt.close()

    # ========================================================================
    # 3.1.2 黑盒模型 2: Neural Network
    # ========================================================================
    print("\n" + "-" * 80)
    print("訓練神經網絡（深度黑盒模型）...")

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(20, 10),
        max_iter=1000,
        random_state=RANDOM_STATE,
        early_stopping=True
    )
    mlp_model.fit(X_train_i_scaled, y_train_i)

    mlp_score = mlp_model.score(X_test_i_scaled, y_test_i)
    print(f"✓ MLP 測試集準確率: {mlp_score:.4f}")

    # 創建 LIME 解釋器（同樣的數據）
    lime_explainer_mlp = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_i_scaled,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        mode='classification',
        random_state=RANDOM_STATE
    )

    # ========================================================================
    # LIME 可視化 10: 單個預測的詳細解釋
    # ========================================================================
    fig10, axes10 = create_subplots(1, 2, figsize=(18, 6))

    # 選擇一個樣本
    sample_idx = 0
    sample = X_test_i_scaled[sample_idx]

    # SVM 解釋
    exp_svm = lime_explainer.explain_instance(
        sample,
        svm_model.predict_proba,
        num_features=4
    )

    # MLP 解釋
    exp_mlp = lime_explainer_mlp.explain_instance(
        sample,
        mlp_model.predict_proba,
        num_features=4
    )

    # 繪製 SVM 解釋
    pred_class_svm = svm_model.predict(sample.reshape(1, -1))[0]
    exp_list_svm = exp_svm.as_list(label=pred_class_svm)
    features_svm = [item[0] for item in exp_list_svm]
    values_svm = [item[1] for item in exp_list_svm]
    colors_svm = ['green' if v > 0 else 'red' for v in values_svm]

    axes10[0].barh(range(len(features_svm)), values_svm, color=colors_svm, alpha=0.7)
    axes10[0].set_yticks(range(len(features_svm)))
    axes10[0].set_yticklabels(features_svm)
    axes10[0].set_xlabel('特徵貢獻', fontsize=12)
    axes10[0].set_title(f'LIME 解釋 - SVM 模型\n預測類別: {iris.target_names[pred_class_svm]}',
                       fontsize=14, pad=10)
    axes10[0].grid(axis='x', alpha=0.3)
    axes10[0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # 繪製 MLP 解釋
    pred_class_mlp = mlp_model.predict(sample.reshape(1, -1))[0]
    exp_list_mlp = exp_mlp.as_list(label=pred_class_mlp)
    features_mlp = [item[0] for item in exp_list_mlp]
    values_mlp = [item[1] for item in exp_list_mlp]
    colors_mlp = ['green' if v > 0 else 'red' for v in values_mlp]

    axes10[1].barh(range(len(features_mlp)), values_mlp, color=colors_mlp, alpha=0.7)
    axes10[1].set_yticks(range(len(features_mlp)))
    axes10[1].set_yticklabels(features_mlp)
    axes10[1].set_xlabel('特徵貢獻', fontsize=12)
    axes10[1].set_title(f'LIME 解釋 - MLP 模型\n預測類別: {iris.target_names[pred_class_mlp]}',
                       fontsize=14, pad=10)
    axes10[1].grid(axis='x', alpha=0.3)
    axes10[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    plt.savefig('/tmp/lime_model_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 10: LIME 不同模型解釋對比已保存")
    plt.close()

    print("\n" + "=" * 80)
    print("LIME 示例完成！".center(76))
    print("=" * 80)

# ============================================================================
# Part 4: SHAP vs LIME 對比
# SHAP vs LIME Comparison
# ============================================================================
if SHAP_AVAILABLE and LIME_AVAILABLE:
    print("\n" + "=" * 80)
    print("【Part 4】SHAP vs LIME 詳細對比".center(76))
    print("Detailed Comparison: SHAP vs LIME".center(76))
    print("=" * 80)

    print("""
對比實驗設置 (Experiment Setup):
- 數據集: Iris (鳶尾花分類)
- 模型: Random Forest Classifier
- 任務: 解釋相同的預測結果
- 對比維度: 解釋一致性、運行時間、可視化效果
    """)

    # ========================================================================
    # 4.1 實驗設置
    # ========================================================================
    print("\n【4.1】訓練對比模型")
    print("-" * 80)

    # 使用 Iris 數據集
    X_comp = X_iris.values
    y_comp = y_iris

    X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
        X_comp, y_comp, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_comp
    )

    # 訓練 Random Forest
    rf_comp = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE
    )
    rf_comp.fit(X_train_comp, y_train_comp)

    comp_score = rf_comp.score(X_test_comp, y_test_comp)
    print(f"✓ Random Forest 準確率: {comp_score:.4f}")

    # ========================================================================
    # 4.2 運行時間對比
    # ========================================================================
    print("\n【4.2】運行時間對比")
    print("-" * 80)

    # SHAP 計算時間
    print("\n計算 SHAP 值...")
    start_shap = time.time()
    explainer_comp_shap = shap.TreeExplainer(rf_comp)
    shap_values_comp = explainer_comp_shap.shap_values(X_test_comp)
    time_shap = time.time() - start_shap
    print(f"✓ SHAP 總耗時: {time_shap:.3f} 秒 ({len(X_test_comp)} 個樣本)")
    print(f"  平均每樣本: {time_shap/len(X_test_comp)*1000:.2f} 毫秒")

    # 處理 SHAP 值格式（與前面一致）
    if isinstance(shap_values_comp, list):
        shap_values_comp_array = np.array(shap_values_comp)
    else:
        shap_values_comp_array = shap_values_comp

    # LIME 計算時間（解釋單個樣本）
    print("\n計算 LIME 值...")
    lime_explainer_comp = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_comp,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        mode='classification',
        random_state=RANDOM_STATE
    )

    # 測試 10 個樣本的平均時間
    lime_times = []
    for i in range(min(10, len(X_test_comp))):
        start_lime = time.time()
        exp = lime_explainer_comp.explain_instance(
            X_test_comp[i],
            rf_comp.predict_proba,
            num_features=4
        )
        lime_times.append(time.time() - start_lime)

    time_lime_avg = np.mean(lime_times)
    time_lime_total = time_lime_avg * len(X_test_comp)

    print(f"✓ LIME 平均每樣本: {time_lime_avg*1000:.2f} 毫秒")
    print(f"  預估 {len(X_test_comp)} 個樣本總耗時: {time_lime_total:.3f} 秒")

    # ========================================================================
    # LIME & SHAP 可視化 11: 運行時間對比
    # ========================================================================
    fig11, axes11 = create_subplots(1, 2, figsize=(18, 6))

    # 時間對比條形圖
    methods = ['SHAP\n(批量計算)', 'LIME\n(逐個計算)']
    times = [time_shap, time_lime_total]
    colors_time = ['#2ecc71', '#e74c3c']

    bars = axes11[0].bar(methods, times, color=colors_time, alpha=0.7, edgecolor='black')
    axes11[0].set_ylabel('耗時（秒）', fontsize=12)
    axes11[0].set_title(f'運行時間對比 ({len(X_test_comp)} 個樣本)\n(Runtime Comparison)',
                       fontsize=14, pad=10)
    axes11[0].grid(axis='y', alpha=0.3)

    # 在條形上添加數值標籤
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        axes11[0].text(bar.get_x() + bar.get_width()/2., height,
                      f'{time_val:.3f}s',
                      ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 單樣本時間對比
    times_per_sample = [time_shap/len(X_test_comp)*1000, time_lime_avg*1000]

    bars2 = axes11[1].bar(methods, times_per_sample, color=colors_time, alpha=0.7, edgecolor='black')
    axes11[1].set_ylabel('耗時（毫秒）', fontsize=12)
    axes11[1].set_title('單樣本平均耗時\n(Per-Sample Average Time)', fontsize=14, pad=10)
    axes11[1].grid(axis='y', alpha=0.3)

    for bar, time_val in zip(bars2, times_per_sample):
        height = bar.get_height()
        axes11[1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{time_val:.2f}ms',
                      ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/tmp/shap_lime_time_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 11: 運行時間對比已保存")
    plt.close()

    # ========================================================================
    # 4.3 解釋一致性對比
    # ========================================================================
    print("\n【4.3】解釋一致性對比")
    print("-" * 80)

    # 選擇幾個樣本進行詳細對比
    sample_indices_comp = [0, 5, 10]

    # ========================================================================
    # LIME & SHAP 可視化 12: 同一預測的不同解釋
    # ========================================================================
    fig12, axes12 = create_subplots(3, 2, figsize=(18, 16))

    for row_idx, sample_idx in enumerate(sample_indices_comp):
        sample = X_test_comp[sample_idx]
        true_class = y_test_comp[sample_idx]
        pred_class = rf_comp.predict([sample])[0]

        # SHAP 解釋
        if isinstance(shap_values_comp, list):
            shap_vals = shap_values_comp[pred_class][sample_idx]
        else:
            # 新版本格式: (n_classes, n_samples, n_features)
            shap_vals = shap_values_comp[pred_class, sample_idx, :]

        # 獲取 SHAP 特徵重要性
        shap_importance = list(zip(iris.feature_names, shap_vals))
        shap_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        features_shap = [item[0] for item in shap_importance]
        values_shap = [item[1] for item in shap_importance]
        colors_shap = ['green' if v > 0 else 'red' for v in values_shap]

        # 繪製 SHAP
        ax_shap = axes12[row_idx, 0]
        ax_shap.barh(range(len(features_shap)), values_shap, color=colors_shap, alpha=0.7)
        ax_shap.set_yticks(range(len(features_shap)))
        ax_shap.set_yticklabels(features_shap)
        ax_shap.set_xlabel('SHAP 值', fontsize=11)
        ax_shap.set_title(
            f'SHAP - 樣本 {sample_idx}\n真實: {iris.target_names[true_class]}, '
            f'預測: {iris.target_names[pred_class]}',
            fontsize=12, pad=10
        )
        ax_shap.grid(axis='x', alpha=0.3)
        ax_shap.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

        # LIME 解釋
        exp_lime = lime_explainer_comp.explain_instance(
            sample,
            rf_comp.predict_proba,
            num_features=4
        )

        exp_list = exp_lime.as_list(label=pred_class)
        features_lime = [item[0] for item in exp_list]
        values_lime = [item[1] for item in exp_list]
        colors_lime = ['green' if v > 0 else 'red' for v in values_lime]

        # 繪製 LIME
        ax_lime = axes12[row_idx, 1]
        ax_lime.barh(range(len(features_lime)), values_lime, color=colors_lime, alpha=0.7)
        ax_lime.set_yticks(range(len(features_lime)))
        ax_lime.set_yticklabels(features_lime)
        ax_lime.set_xlabel('LIME 貢獻值', fontsize=11)
        ax_lime.set_title(
            f'LIME - 樣本 {sample_idx}\n真實: {iris.target_names[true_class]}, '
            f'預測: {iris.target_names[pred_class]}',
            fontsize=12, pad=10
        )
        ax_lime.grid(axis='x', alpha=0.3)
        ax_lime.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    plt.savefig('/tmp/shap_lime_explanation_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 12: SHAP vs LIME 解釋對比已保存")
    plt.close()

    # ========================================================================
    # LIME & SHAP 可視化 13: 特徵重要性排名對比
    # ========================================================================
    print("\n【4.4】全局特徵重要性對比")
    print("-" * 80)

    fig13, axes13 = create_subplots(1, 3, figsize=(18, 6))

    # 1. 傳統特徵重要性
    feature_imp_rf = rf_comp.feature_importances_

    axes13[0].barh(range(len(iris.feature_names)), feature_imp_rf, color='skyblue', alpha=0.7)
    axes13[0].set_yticks(range(len(iris.feature_names)))
    axes13[0].set_yticklabels(iris.feature_names)
    axes13[0].set_xlabel('重要性分數', fontsize=11)
    axes13[0].set_title('Random Forest\n特徵重要性', fontsize=13, pad=10)
    axes13[0].grid(axis='x', alpha=0.3)

    # 2. SHAP 全局特徵重要性
    # 對所有類別的 SHAP 值取絕對值平均
    shap_global_importance = np.abs(shap_values_comp_array).mean(axis=0).mean(axis=0)

    axes13[1].barh(range(len(iris.feature_names)), shap_global_importance,
                  color='green', alpha=0.7)
    axes13[1].set_yticks(range(len(iris.feature_names)))
    axes13[1].set_yticklabels(iris.feature_names)
    axes13[1].set_xlabel('平均 |SHAP 值|', fontsize=11)
    axes13[1].set_title('SHAP\n全局特徵重要性', fontsize=13, pad=10)
    axes13[1].grid(axis='x', alpha=0.3)

    # 3. 對比歸一化
    # 歸一化到 0-1
    rf_norm = feature_imp_rf / feature_imp_rf.max()
    shap_norm = shap_global_importance / shap_global_importance.max()

    x = np.arange(len(iris.feature_names))
    width = 0.35

    axes13[2].barh(x - width/2, rf_norm, width, label='Random Forest',
                  color='skyblue', alpha=0.7)
    axes13[2].barh(x + width/2, shap_norm, width, label='SHAP',
                  color='green', alpha=0.7)
    axes13[2].set_yticks(x)
    axes13[2].set_yticklabels(iris.feature_names)
    axes13[2].set_xlabel('歸一化重要性', fontsize=11)
    axes13[2].set_title('特徵重要性對比\n(歸一化)', fontsize=13, pad=10)
    axes13[2].legend(fontsize=10)
    axes13[2].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ 圖表 13: 特徵重要性排名對比已保存")
    plt.close()

    # ========================================================================
    # 4.5 對比總結
    # ========================================================================
    print("\n【4.5】SHAP vs LIME 對比總結")
    print("-" * 80)

    comparison_table = pd.DataFrame({
        '對比維度': [
            '理論基礎',
            '模型無關性',
            '計算速度',
            '批量計算',
            '解釋類型',
            '一致性保證',
            '適用模型',
            '可視化豐富度',
            '學習曲線',
            '推薦場景'
        ],
        'SHAP': [
            '博弈論 Shapley 值',
            '是（但樹模型有優化）',
            '中等（TreeExplainer 快）',
            '✓ 支持批量',
            '全局 + 局部',
            '✓ 理論保證',
            '所有模型（樹模型最優）',
            '非常豐富',
            '較陡',
            '需要深度分析、模型調試'
        ],
        'LIME': [
            '局部線性近似',
            '✓ 完全模型無關',
            '快（單樣本）',
            '✗ 需逐個計算',
            '主要是局部',
            '✗ 無理論保證',
            '任何黑盒模型',
            '中等',
            '平緩',
            '快速解釋、用戶演示'
        ]
    })

    print("\n" + "=" * 80)
    print(comparison_table.to_string(index=False))
    print("=" * 80)

# ============================================================================
# Part 5: 最佳實踐和建議
# Best Practices and Recommendations
# ============================================================================
print("\n" + "=" * 80)
print("【Part 5】最佳實踐和建議".center(76))
print("Best Practices and Recommendations".center(76))
print("=" * 80)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 何時使用 SHAP？(When to use SHAP?)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ✓ 使用樹模型（Random Forest、XGBoost、LightGBM）
     → TreeExplainer 速度很快

   ✓ 需要理論保證和一致性
     → SHAP 基於 Shapley 值，有數學保證

   ✓ 需要全局和局部解釋
     → SHAP 提供豐富的全局可視化

   ✓ 模型調試和特徵工程
     → Dependence plots 可以發現特徵交互

   ✓ 高風險應用（金融、醫療）
     → 需要可靠、一致的解釋

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. 何時使用 LIME？(When to use LIME?)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ✓ 完全黑盒模型
     → 只能訪問預測接口，無法訪問內部結構

   ✓ 需要快速解釋少量預測
     → LIME 單個解釋速度快

   ✓ 向非技術用戶演示
     → LIME 解釋直觀易懂

   ✓ 圖像和文本數據
     → LIME 對這些類型有專門支持

   ✓ 原型階段
     → 快速獲得初步解釋

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. 解釋性與性能的權衡 (Interpretability vs Performance Trade-off)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   • 不要為了解釋性犧牲太多性能
     → 優先選擇性能好的模型，然後用 SHAP/LIME 解釋

   • 考慮使用可解釋的模型
     → 如果性能差距不大，優先選擇決策樹、線性模型

   • 分階段使用解釋工具
     → 開發階段: 深度分析（SHAP）
     → 生產階段: 按需解釋（LIME）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. 常見陷阱 (Common Pitfalls)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ✗ 過度解讀 LIME 的結果
     → LIME 是近似方法，可能不穩定

   ✗ 忽視特徵相關性
     → 高度相關的特徵會影響解釋

   ✗ 只看單個樣本
     → 應該結合全局和局部解釋

   ✗ 忽視數據質量
     → 垃圾進，垃圾出；解釋也一樣

   ✗ 不驗證解釋的合理性
     → 要用領域知識驗證解釋是否合理

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. 實用建議 (Practical Tips)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   1. 先從簡單可視化開始
      → Summary plot, Feature importance

   2. 逐步深入到細節
      → Dependence plots, Individual explanations

   3. 結合多種解釋方法
      → SHAP + LIME + 傳統方法

   4. 記錄和分享解釋
      → 幫助團隊理解模型行為

   5. 持續監控解釋的變化
      → 模型重訓練後，檢查解釋是否一致

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. 進階主題 (Advanced Topics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   • SHAP 交互值 (Interaction Values)
     → 分析特徵之間的交互效應

   • 對抗性解釋 (Adversarial Explanations)
     → 測試解釋的穩定性

   • 時間序列解釋
     → 使用 Time Series explainers

   • 自然語言解釋生成
     → 將 SHAP/LIME 轉換為文字解釋

   • 解釋的不確定性量化
     → 評估解釋的可信度
""")

# ============================================================================
# 總結
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("總結與資源".center(76))
print("Summary and Resources".center(76))
print("=" * 80)

print("""
【生成的可視化圖表】(Generated Visualizations)
""")

if SHAP_AVAILABLE:
    print("""
SHAP 可視化:
  ✓ 圖表 1: SHAP Summary Plots (全局特徵重要性)
  ✓ 圖表 2: SHAP Waterfall Plots (單個預測解釋)
  ✓ 圖表 3: SHAP Force Plot (特徵貢獻)
  ✓ 圖表 4: SHAP Dependence Plots (特徵交互)
  ✓ 圖表 5: SHAP Decision Plot (多樣本對比)
  ✓ 圖表 6: 特徵重要性對比 (傳統 vs SHAP)
  ✓ 圖表 7: SHAP Beeswarm Plot (回歸任務)
  ✓ 圖表 8: SHAP 特徵交互熱力圖
    """)

if LIME_AVAILABLE:
    print("""
LIME 可視化:
  ✓ 圖表 9: LIME 多樣本解釋對比
  ✓ 圖表 10: LIME 不同模型解釋對比
    """)

if SHAP_AVAILABLE and LIME_AVAILABLE:
    print("""
SHAP vs LIME 對比可視化:
  ✓ 圖表 11: 運行時間對比
  ✓ 圖表 12: SHAP vs LIME 解釋對比
  ✓ 圖表 13: 特徵重要性排名對比
    """)

print("""
【學習資源】(Learning Resources)

SHAP:
  • 官方文檔: https://shap.readthedocs.io/
  • 論文: "A Unified Approach to Interpreting Model Predictions"
  • GitHub: https://github.com/slundberg/shap

LIME:
  • 官方文檔: https://lime-ml.readthedocs.io/
  • 論文: "Why Should I Trust You?"
  • GitHub: https://github.com/marcotcr/lime

其他資源:
  • Interpretable ML Book: https://christophm.github.io/interpretable-ml-book/
  • What-If Tool: https://pair-code.github.io/what-if-tool/
  • AI Explainability 360: https://aix360.mybluemix.net/

【關鍵要點】(Key Takeaways)

1. 模型解釋性對於建立信任、調試模型、滿足監管要求至關重要
2. SHAP 提供理論保證和豐富的可視化，適合深度分析
3. LIME 是模型無關的，適合快速解釋和演示
4. 結合使用多種解釋方法可以獲得更全面的理解
5. 解釋性不應該以犧牲模型性能為代價
""")

print("\n" + "=" * 80)
print("模型解釋性指南完成！".center(76))
print("Model Interpretability Guide Complete!".center(76))
print("=" * 80)
