"""
決策樹分類器範例 - Decision Tree Classifier Example
使用 ID3 演算法（基於信息增益）進行分類
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 導入工具模塊（如果在專案根目錄運行）
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import RANDOM_STATE
    from utils.plotting import setup_chinese_fonts
    USE_UTILS = True
except ImportError:
    # 如果無法導入工具模塊，使用默認值
    RANDOM_STATE = 42
    USE_UTILS = False

print("=" * 60)
print("決策樹分類器範例 - Decision Tree Classifier")
print("=" * 60)

# 讀取 CSV 文件並獲取特徵和標籤
print("\n步驟 1: 讀取數據...")

# 使用絕對路徑（基於文件位置）
data_path = Path(__file__).parent / 'data.csv'

if not data_path.exists():
    raise FileNotFoundError(f"數據文件不存在: {data_path}")

try:
    allElectronicsData = open(data_path, 'r', encoding='utf-8')
    reader = csv.reader(allElectronicsData)
    headers = next(reader)
    print(f"欄位名稱: {headers}")

    featureList = []
    labelList = []

    for row in reader:
        if not row:  # 跳過空行
            continue
        labelList.append(row[len(row) - 1])
        rowDict = {}
        for i in range(1, len(row) - 1):
            rowDict[headers[i]] = row[i]
        featureList.append(rowDict)

    allElectronicsData.close()

    print(f"\n共有 {len(featureList)} 筆數據")
    print(f"特徵範例: {featureList[0]}")
    print(f"標籤範例: {labelList[:5]}")

except FileNotFoundError as e:
    print(f"✗ 錯誤: {e}")
    print("請確保 data.csv 文件存在於 DecisionTree 目錄中")
    exit(1)
except Exception as e:
    print(f"✗ 讀取數據時出錯: {e}")
    exit(1)

# 步驟 2: 特徵向量化 (One-Hot Encoding)
print("\n步驟 2: 特徵向量化...")
vec = DictVectorizer()
X = vec.fit_transform(featureList).toarray()
print(f"特徵矩陣形狀: {X.shape}")
print(f"特徵名稱: {vec.get_feature_names_out()}")

# 步驟 3: 標籤編碼
print("\n步驟 3: 標籤編碼...")
lb = preprocessing.LabelEncoder()
y = lb.fit_transform(labelList)
print(f"標籤類別: {lb.classes_}")
print(f"編碼後的標籤: {y}")

# 步驟 4: 建立決策樹分類器
print("\n步驟 4: 訓練決策樹分類器...")
# 使用 entropy (信息增益) 作為分割標準，相當於 ID3 演算法
clf = tree.DecisionTreeClassifier(
    criterion='entropy',  # 使用信息增益 (ID3)
    max_depth=5,  # 限制樹的深度避免過擬合
    min_samples_split=2,
    random_state=42
)
clf.fit(X, y)

# 計算訓練準確率
train_score = clf.score(X, y)
print(f"訓練準確率: {train_score * 100:.2f}%")

# 顯示特徵重要性
print("\n特徵重要性:")
feature_importance = sorted(
    zip(vec.get_feature_names_out(), clf.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)
for feature, importance in feature_importance:
    if importance > 0:
        print(f"  {feature}: {importance:.4f}")

# 步驟 5: 導出決策樹可視化文件
print("\n步驟 5: 導出決策樹圖形...")
with open("allElectronicInformationGainOri.dot", "w", encoding='utf-8') as f:
    tree.export_graphviz(
        clf,
        feature_names=vec.get_feature_names_out(),
        class_names=lb.classes_,
        out_file=f,
        filled=True,
        rounded=True
    )
print("已生成 .dot 文件，可使用以下命令生成 PDF:")
print("  dot -Tpdf allElectronicInformationGainOri.dot -o output.pdf")

# 步驟 6: 預測測試
print("\n步驟 6: 預測測試...")
# 測試原始數據的第一筆
testRowX = X[0].copy()
print(f"原始測試數據: {featureList[0]}")
print(f"原始預測結果: {lb.classes_[clf.predict([testRowX])[0]]}")

# 修改特徵進行預測
newRowX = testRowX.copy()
# 假設修改某些特徵值
if len(newRowX) > 2:
    newRowX[0] = 1
    newRowX[2] = 0
predictedY = clf.predict([newRowX])
print(f"修改後預測結果: {lb.classes_[predictedY[0]]}")

# 步驟 7: 混淆矩陣
print("\n步驟 7: 模型評估...")
y_pred = clf.predict(X)
print("混淆矩陣:")
print(confusion_matrix(y, y_pred))
print("\n分類報告:")
print(classification_report(y, y_pred, target_names=lb.classes_))

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)

allElectronicsData.close()
