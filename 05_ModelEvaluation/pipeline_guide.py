"""
Pipeline 完整使用指南
Scikit-learn Pipeline Masterclass

Pipeline 是生產環境中的最佳實踐
將數據預處理和模型訓練整合在一起
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("=" * 80)
print("Pipeline 完整使用指南".center(80))
print("=" * 80)

# ============================================================================
# 1. 為什麼使用 Pipeline？
# ============================================================================
print("\n【1】為什麼使用 Pipeline？")
print("-" * 80)
print("""
沒有 Pipeline 的問題：
❌ 預處理步驟容易遺漏
❌ 訓練集和測試集處理不一致
❌ 數據洩漏風險（用測試集fit）
❌ 代碼重複，難以維護
❌ 交叉驗證困難

使用 Pipeline 的優點：
✅ 自動化整個工作流程
✅ 避免數據洩漏
✅ 代碼簡潔易讀
✅ 方便交叉驗證和網格搜索
✅ 易於部署到生產環境
✅ 保證訓練和預測一致性

Pipeline 的本質：
將多個步驟（transformer + estimator）串聯起來
每個步驟自動 fit_transform，最後一個步驟 fit
""")

# 準備數據
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n數據集：{cancer.feature_names[0]} 等 {X.shape[1]} 個特徵")
print(f"訓練集：{X_train.shape}")
print(f"測試集：{X_test.shape}")

# ============================================================================
# 2. 基礎 Pipeline
# ============================================================================
print("\n【2】基礎 Pipeline：預處理 + 模型")
print("-" * 80)

# 方法1：使用 Pipeline 類
pipe_basic = Pipeline([
    ('scaler', StandardScaler()),          # 步驟1：標準化
    ('classifier', LogisticRegression())   # 步驟2：分類器
])

# 訓練（自動執行所有步驟）
pipe_basic.fit(X_train, y_train)

# 預測（自動應用所有步驟）
y_pred_basic = pipe_basic.predict(X_test)
accuracy_basic = accuracy_score(y_test, y_pred_basic)

print(f"基礎 Pipeline 準確率：{accuracy_basic:.4f}")

# 方法2：使用 make_pipeline（自動命名）
pipe_make = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

pipe_make.fit(X_train, y_train)
accuracy_make = pipe_make.score(X_test, y_test)

print(f"make_pipeline 準確率：{accuracy_make:.4f}")

# ============================================================================
# 3. 複雜 Pipeline：多步驟處理
# ============================================================================
print("\n【3】複雜 Pipeline：特徵選擇 + 降維 + 模型")
print("-" * 80)

pipe_complex = Pipeline([
    ('scaler', StandardScaler()),                    # 1. 標準化
    ('feature_selection', SelectKBest(f_classif, k=20)),  # 2. 特徵選擇（前20個）
    ('pca', PCA(n_components=10)),                   # 3. PCA降維到10維
    ('classifier', SVC(kernel='rbf'))                # 4. SVM分類器
])

pipe_complex.fit(X_train, y_train)
accuracy_complex = pipe_complex.score(X_test, y_test)

print(f"複雜 Pipeline 準確率：{accuracy_complex:.4f}")
print(f"\nPipeline 步驟：")
for name, step in pipe_complex.named_steps.items():
    print(f"  {name}: {type(step).__name__}")

# ============================================================================
# 4. Pipeline + 網格搜索
# ============================================================================
print("\n【4】Pipeline + GridSearchCV（超參數調優）")
print("-" * 80)

# 定義參數網格（使用 step_name__parameter 語法）
param_grid = {
    'feature_selection__k': [10, 15, 20],          # 特徵選擇數量
    'pca__n_components': [5, 10, 15],              # PCA 維度
    'classifier__C': [0.1, 1, 10],                 # SVM C 參數
    'classifier__gamma': ['scale', 0.001, 0.01]    # SVM gamma 參數
}

# 創建 Pipeline
pipe_grid = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif)),
    ('pca', PCA()),
    ('classifier', SVC(kernel='rbf'))
])

# 網格搜索
grid_search = GridSearchCV(
    pipe_grid,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

print("正在進行網格搜索...")
grid_search.fit(X_train, y_train)

print(f"最佳參數：{grid_search.best_params_}")
print(f"最佳交叉驗證分數：{grid_search.best_score_:.4f}")
print(f"測試集分數：{grid_search.score(X_test, y_test):.4f}")

# ============================================================================
# 5. Pipeline + 交叉驗證
# ============================================================================
print("\n【5】Pipeline + 交叉驗證")
print("-" * 80)

pipe_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 交叉驗證
cv_scores = cross_val_score(pipe_cv, X, y, cv=5, scoring='accuracy')

print(f"5折交叉驗證分數：{cv_scores}")
print(f"平均準確率：{cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# 6. 訪問 Pipeline 中的步驟
# ============================================================================
print("\n【6】訪問和使用 Pipeline 中的步驟")
print("-" * 80)

pipe_access = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipe_access.fit(X_train, y_train)

# 訪問特定步驟
scaler = pipe_access.named_steps['scaler']
pca = pipe_access.named_steps['pca']
classifier = pipe_access.named_steps['classifier']

print(f"標準化器均值：{scaler.mean_[:3]}...")
print(f"PCA 解釋方差比：{pca.explained_variance_ratio_[:5]}...")
print(f"隨機森林特徵重要性：{classifier.feature_importances_[:5]}...")

# 獲取中間結果
X_train_transformed = pipe_access[:-1].transform(X_train)  # 除了最後一步
print(f"\n經過預處理後的特徵形狀：{X_train_transformed.shape}")

# ============================================================================
# 7. Pipeline 保存和加載
# ============================================================================
print("\n【7】Pipeline 保存和加載")
print("-" * 80)

# 保存完整 Pipeline
pipe_save = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipe_save.fit(X_train, y_train)

# 保存
import os
os.makedirs('saved_models', exist_ok=True)
joblib.dump(pipe_save, 'saved_models/complete_pipeline.joblib')
print("✓ Pipeline 已保存到 saved_models/complete_pipeline.joblib")

# 加載
pipe_loaded = joblib.load('saved_models/complete_pipeline.joblib')
print("✓ Pipeline 已加載")

# 驗證
accuracy_loaded = pipe_loaded.score(X_test, y_test)
print(f"加載的 Pipeline 準確率：{accuracy_loaded:.4f}")

# ============================================================================
# 8. ColumnTransformer：處理不同類型特徵
# ============================================================================
print("\n【8】ColumnTransformer：處理混合類型特徵")
print("-" * 80)

# 創建示例數據（數值 + 類別特徵）
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = iris.target

# 假設有數值和類別特徵
numeric_features = ['sepal length (cm)', 'sepal width (cm)']
categorical_features = ['species']

# 定義不同的預處理器
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 組合預處理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 創建完整 Pipeline
pipe_column = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# 準備數據
X_iris = df_iris[numeric_features + categorical_features]
y_iris = (df_iris['species'] > 0).astype(int)  # 二分類

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

pipe_column.fit(X_train_iris, y_train_iris)
accuracy_column = pipe_column.score(X_test_iris, y_test_iris)

print(f"ColumnTransformer Pipeline 準確率：{accuracy_column:.4f}")
print("\n處理流程：")
print("  數值特徵 → StandardScaler")
print("  類別特徵 → OneHotEncoder")
print("  合併 → LogisticRegression")

# ============================================================================
# 9. 實戰示例：完整的生產級 Pipeline
# ============================================================================
print("\n【9】生產級 Pipeline 示例")
print("-" * 80)

# 創建生產級 Pipeline
production_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('numeric', Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ]), slice(0, 2)),  # 前2個特徵
        ('passthrough', 'passthrough', slice(2, None))  # 其餘特徵保持不變
    ])),
    ('feature_selection', SelectKBest(f_classif, k=15)),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ))
])

production_pipeline.fit(X_train, y_train)
prod_accuracy = production_pipeline.score(X_test, y_test)

print(f"生產級 Pipeline 準確率：{prod_accuracy:.4f}")

# 保存生產模型
joblib.dump(production_pipeline, 'saved_models/production_model.joblib')
print("✓ 生產模型已保存")

# ============================================================================
# 10. Pipeline 最佳實踐
# ============================================================================
print("\n" + "=" * 80)
print("Pipeline 最佳實踐")
print("=" * 80)
print("""
1. 基本原則：
   ✓ 所有預處理步驟都放入 Pipeline
   ✓ 永遠不要手動 fit 測試集
   ✓ 保存完整的 Pipeline，不只是模型

2. 命名規範：
   • 使用有意義的步驟名稱
   • 參數調優時使用 step_name__param_name
   • 使用 named_steps 訪問特定步驟

3. 常見模式：
   # 模式1：簡單 Pipeline
   pipe = Pipeline([
       ('scaler', StandardScaler()),
       ('model', LogisticRegression())
   ])

   # 模式2：複雜 Pipeline
   pipe = Pipeline([
       ('preprocessing', ColumnTransformer([...])),
       ('feature_selection', SelectKBest()),
       ('model', RandomForestClassifier())
   ])

   # 模式3：Pipeline + GridSearch
   grid = GridSearchCV(pipe, param_grid, cv=5)
   grid.fit(X_train, y_train)

4. 調試技巧：
   # 查看 Pipeline 結構
   pipe.named_steps

   # 獲取中間結果
   X_transformed = pipe[:-1].transform(X)

   # 設置 verbose
   pipe = Pipeline([...], verbose=True)

5. 部署建議：
   # 保存 Pipeline
   joblib.dump(pipe, 'model.joblib')

   # 加載和預測
   pipe = joblib.load('model.joblib')
   predictions = pipe.predict(X_new)

6. 常見錯誤：
   ❌ 在 Pipeline 外部預處理數據
   ❌ 用測試集 fit 預處理器
   ❌ 忘記保存預處理器
   ❌ Pipeline 中使用有狀態的對象

7. 性能優化：
   • 使用 n_jobs=-1 並行處理
   • 緩存中間結果（memory 參數）
   • 選擇合適的預處理方法

8. 示例代碼模板：
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier

   pipe = Pipeline([
       ('scaler', StandardScaler()),
       ('classifier', RandomForestClassifier())
   ])

   pipe.fit(X_train, y_train)
   predictions = pipe.predict(X_test)
   joblib.dump(pipe, 'model.joblib')
""")

# 可視化比較
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 不同 Pipeline 性能比較
ax1 = axes[0, 0]
pipelines = ['Basic', 'Complex', 'GridSearch', 'Production']
accuracies = [accuracy_basic, accuracy_complex,
             grid_search.score(X_test, y_test), prod_accuracy]
bars = ax1.bar(pipelines, accuracies, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
ax1.set_ylabel('Accuracy')
ax1.set_title('Pipeline Performance Comparison', fontweight='bold')
ax1.set_ylim([0.9, 1.0])
ax1.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.3f}', ha='center', va='bottom')

# 2. Pipeline 流程圖（文本）
ax2 = axes[0, 1]
ax2.axis('off')
pipeline_flow = """
Pipeline 工作流程：

訓練階段：
X_train → fit_transform(Step1)
       → fit_transform(Step2)
       → fit_transform(Step3)
       → fit(Model)

預測階段：
X_test → transform(Step1)
      → transform(Step2)
      → transform(Step3)
      → predict(Model)

優點：
✓ 自動化
✓ 防止數據洩漏
✓ 易於部署
"""
ax2.text(0.1, 0.9, pipeline_flow, fontsize=10, family='monospace',
        verticalalignment='top')
ax2.set_title('Pipeline Workflow', fontweight='bold', fontsize=12)

# 3. 交叉驗證分數
ax3 = axes[1, 0]
ax3.bar(range(1, 6), cv_scores, color='steelblue', alpha=0.7, edgecolor='black')
ax3.axhline(y=cv_scores.mean(), color='r', linestyle='--', linewidth=2,
           label=f'Mean: {cv_scores.mean():.3f}')
ax3.set_xlabel('Fold')
ax3.set_ylabel('Accuracy')
ax3.set_title('Cross-Validation Scores', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. 特徵重要性（生產 Pipeline）
ax4 = axes[1, 1]
if hasattr(production_pipeline.named_steps['classifier'], 'feature_importances_'):
    importances = production_pipeline.named_steps['classifier'].feature_importances_[:10]
    ax4.barh(range(len(importances)), importances, color='purple', alpha=0.7)
    ax4.set_yticks(range(len(importances)))
    ax4.set_yticklabels([f'Feature {i}' for i in range(len(importances))])
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 10 Feature Importances', fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('05_ModelEvaluation/pipeline_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ 已保存 Pipeline 比較圖表")

print("\n" + "=" * 80)
print("Pipeline 教程完成！")
print("=" * 80)
print("""
記住：
• Pipeline 是生產環境的最佳實踐
• 將所有預處理和模型整合在一起
• 避免數據洩漏，保證一致性
• 便於部署和維護
""")
