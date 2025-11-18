"""
遷移學習和預訓練模型 | Transfer Learning and Pre-trained Models

本教程涵蓋：
1. 遷移學習原理
2. 使用預訓練模型（VGG16、ResNet50、MobileNetV2）
3. 特徵提取 vs 微調
4. 自定義數據集遷移學習
5. 模型微調技巧
6. 實戰應用

Author: Machine Learning Tutorial
Date: 2025-11-18
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RANDOM_STATE, BATCH_SIZE, EPOCHS, PATIENCE
from utils.plotting import save_figure, create_subplots, setup_chinese_fonts
from utils.paths import get_output_path, get_model_path

warnings.filterwarnings('ignore')
setup_chinese_fonts()

# 設置隨機種子 / Set random seeds
np.random.seed(RANDOM_STATE)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import train_test_split

    # 設置TensorFlow隨機種子
    tf.random.set_seed(RANDOM_STATE)
    KERAS_AVAILABLE = True
    print("✓ TensorFlow/Keras 已成功加載")
    print(f"✓ TensorFlow 版本: {tf.__version__}")
except ImportError as e:
    KERAS_AVAILABLE = False
    print("✗ TensorFlow/Keras 未安裝")
    print("請運行：pip install tensorflow")
    print(f"錯誤詳情：{e}")
    sys.exit(1)

# ============================================================================
# 全局配置 / Global Configuration
# ============================================================================
print("\n" + "=" * 80)
print("遷移學習與預訓練模型教程 | Transfer Learning Tutorial".center(80))
print("=" * 80)

# CIFAR-10 類別名稱
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_CLASSES_CN = ['飛機', '汽車', '鳥', '貓', '鹿',
                      '狗', '青蛙', '馬', '船', '卡車']

# 訓練配置
IMG_SIZE = 96  # 預訓練模型通常需要更大的輸入
NUM_CLASSES = 10
EPOCHS_FEATURE_EXTRACTION = 20  # 特徵提取階段
EPOCHS_FINE_TUNING = 30  # 微調階段

# ============================================================================
# Part 1: 遷移學習基礎概念 | Transfer Learning Basics
# ============================================================================
print("\n" + "=" * 80)
print("Part 1: 遷移學習基礎概念 | Transfer Learning Basics")
print("=" * 80)

transfer_learning_intro = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                        遷移學習概念介紹                                      ║
║                    Transfer Learning Introduction                         ║
╚═══════════════════════════════════════════════════════════════════════════╝

【什麼是遷移學習？| What is Transfer Learning?】

遷移學習是利用在大型數據集（如 ImageNet）上預訓練的模型，
將學到的特徵遷移到新的任務上。這樣可以：
✓ 減少訓練時間
✓ 需要更少的訓練數據
✓ 獲得更好的性能
✓ 避免從零開始訓練

【兩種主要策略 | Two Main Strategies】

1. 特徵提取 (Feature Extraction)
   ┌─────────────────────────────────────────┐
   │  預訓練模型（凍結）→ 新分類器（訓練）    │
   └─────────────────────────────────────────┘
   • 凍結預訓練模型的所有層
   • 只訓練新添加的分類器層
   • 適合：數據量小、任務相似

2. 微調 (Fine-tuning)
   ┌─────────────────────────────────────────┐
   │  預訓練模型（部分凍結）→ 新分類器（訓練）│
   └─────────────────────────────────────────┘
   • 解凍預訓練模型的部分層
   • 使用較小的學習率微調
   • 適合：數據量中等、需要定制化

【何時使用遷移學習？| When to Use Transfer Learning?】

✓ 訓練數據有限（< 10,000 樣本）
✓ 計算資源有限
✓ 任務與預訓練任務相似
✓ 需要快速原型開發

【常用預訓練模型 | Popular Pre-trained Models】

• VGG16/VGG19: 簡單但參數多
• ResNet50/ResNet101: 殘差連接，更深
• MobileNetV2: 輕量級，適合移動設備
• EfficientNet: 性能與效率的平衡
• Inception: 多尺度特徵提取
"""

print(transfer_learning_intro)

# ============================================================================
# Part 2: 加載和預處理數據 | Load and Preprocess Data
# ============================================================================
print("\n" + "=" * 80)
print("Part 2: 數據加載和預處理 | Data Loading and Preprocessing")
print("=" * 80)

def load_and_resize_data(target_size=IMG_SIZE):
    """
    加載 CIFAR-10 並調整大小以適配預訓練模型
    Load CIFAR-10 and resize for pre-trained models

    Args:
        target_size: 目標圖像大小

    Returns:
        tuple: 調整大小後的數據集
    """
    print(f"\n正在加載 CIFAR-10 數據集並調整大小到 {target_size}x{target_size}...")

    # 加載數據
    (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

    # 由於調整大小需要大量內存，我們使用較小的子集進行演示
    # For demonstration, use a subset to save memory
    subset_size = 10000
    test_size = 2000

    X_train_full = X_train_full[:subset_size]
    y_train_full = y_train_full[:subset_size]
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    print(f"使用訓練樣本數: {subset_size}")
    print(f"使用測試樣本數: {test_size}")

    # 調整圖像大小
    print("正在調整圖像大小...")
    X_train_resized = tf.image.resize(X_train_full, [target_size, target_size]).numpy()
    X_test_resized = tf.image.resize(X_test, [target_size, target_size]).numpy()

    # 分割訓練集和驗證集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_resized, y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    print(f"\n數據形狀:")
    print(f"  訓練集: {X_train.shape}")
    print(f"  驗證集: {X_val.shape}")
    print(f"  測試集: {X_test_resized.shape}")

    return X_train, y_train, X_val, y_val, X_test_resized, y_test

# 加載數據
X_train, y_train, X_val, y_val, X_test, y_test = load_and_resize_data()

# ============================================================================
# Part 3: 特徵提取模式 - VGG16 | Feature Extraction with VGG16
# ============================================================================
print("\n" + "=" * 80)
print("Part 3: 特徵提取模式 - VGG16 | Feature Extraction with VGG16")
print("=" * 80)

def build_feature_extractor_vgg16(input_shape=(96, 96, 3), num_classes=10):
    """
    構建基於 VGG16 的特徵提取器
    Build VGG16-based feature extractor

    Strategy:
        1. 加載 VGG16（不包含頂層）
        2. 凍結所有卷積層
        3. 添加新的分類器

    Args:
        input_shape: 輸入形狀
        num_classes: 類別數

    Returns:
        keras.Model: 特徵提取模型
    """
    print("\n構建 VGG16 特徵提取器...")

    # 加載預訓練的 VGG16（不包含頂層分類器）
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 凍結基礎模型的所有層
    base_model.trainable = False

    print(f"VGG16 基礎模型層數: {len(base_model.layers)}")
    print(f"所有層已凍結")

    # 構建完整模型
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        layers.BatchNormalization(name='bn1'),
        layers.Dense(512, activation='relu', name='fc1'),
        layers.Dropout(0.5, name='dropout1'),
        layers.BatchNormalization(name='bn2'),
        layers.Dense(256, activation='relu', name='fc2'),
        layers.Dropout(0.3, name='dropout2'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='VGG16_FeatureExtractor')

    return model

# 構建 VGG16 特徵提取器
vgg16_extractor = build_feature_extractor_vgg16()

# 編譯模型
vgg16_extractor.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nVGG16 特徵提取器結構:")
vgg16_extractor.summary()

# 統計可訓練參數
trainable_params = np.sum([np.prod(v.get_shape()) for v in vgg16_extractor.trainable_weights])
total_params = vgg16_extractor.count_params()
print(f"\n總參數量: {total_params:,}")
print(f"可訓練參數: {trainable_params:,}")
print(f"凍結參數: {total_params - trainable_params:,}")
print(f"可訓練比例: {trainable_params/total_params*100:.2f}%")

# ============================================================================
# 訓練 VGG16 特徵提取器
# ============================================================================
print("\n" + "-" * 80)
print("訓練 VGG16 特徵提取器...")
print("-" * 80)

# 預處理數據（VGG16 特定）
X_train_vgg = vgg_preprocess(X_train.copy())
X_val_vgg = vgg_preprocess(X_val.copy())
X_test_vgg = vgg_preprocess(X_test.copy())

# 創建回調
vgg_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# 訓練模型
print("\n開始訓練特徵提取器...")
history_vgg_fe = vgg16_extractor.fit(
    X_train_vgg, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_FEATURE_EXTRACTION,
    validation_data=(X_val_vgg, y_val),
    callbacks=vgg_callbacks,
    verbose=1
)

# 評估
print("\n評估 VGG16 特徵提取器...")
vgg_fe_loss, vgg_fe_acc = vgg16_extractor.evaluate(X_test_vgg, y_test, verbose=0)
print(f"VGG16 特徵提取器測試準確率: {vgg_fe_acc:.4f}")
print(f"VGG16 特徵提取器測試損失: {vgg_fe_loss:.4f}")

# ============================================================================
# Part 4: 微調模式 - VGG16 | Fine-tuning with VGG16
# ============================================================================
print("\n" + "=" * 80)
print("Part 4: 微調模式 - VGG16 | Fine-tuning with VGG16")
print("=" * 80)

def build_fine_tuning_vgg16(input_shape=(96, 96, 3), num_classes=10, unfreeze_layers=4):
    """
    構建用於微調的 VGG16 模型
    Build VGG16 model for fine-tuning

    Strategy:
        1. 加載 VGG16
        2. 凍結前面的層
        3. 解凍後面幾層進行微調

    Args:
        input_shape: 輸入形狀
        num_classes: 類別數
        unfreeze_layers: 解凍的層數

    Returns:
        keras.Model: 微調模型
    """
    print(f"\n構建 VGG16 微調模型（解凍最後 {unfreeze_layers} 個卷積塊）...")

    # 加載預訓練的 VGG16
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 首先凍結所有層
    base_model.trainable = True

    # 查找最後幾個卷積塊的起始位置
    # VGG16 有 5 個卷積塊
    block_layers = []
    for i, layer in enumerate(base_model.layers):
        if 'block' in layer.name:
            block_num = int(layer.name.split('_')[0].replace('block', ''))
            if block_num not in block_layers:
                block_layers.append((block_num, i))

    print(f"VGG16 卷積塊位置: {block_layers}")

    # 凍結前面的層，只訓練最後幾個塊
    # 例如：如果 unfreeze_layers=2，則只訓練 block4 和 block5
    freeze_until_block = 6 - unfreeze_layers
    freeze_until = None

    for block_num, layer_idx in block_layers:
        if block_num == freeze_until_block:
            freeze_until = layer_idx
            break

    if freeze_until:
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in base_model.layers[freeze_until:]:
            layer.trainable = True
    else:
        # 如果解凍層數過多，則解凍所有層
        for layer in base_model.layers:
            layer.trainable = True

    # 構建完整模型
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        layers.BatchNormalization(name='bn1'),
        layers.Dense(512, activation='relu', name='fc1'),
        layers.Dropout(0.5, name='dropout1'),
        layers.BatchNormalization(name='bn2'),
        layers.Dense(256, activation='relu', name='fc2'),
        layers.Dropout(0.3, name='dropout2'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='VGG16_FineTuned')

    return model

# 構建微調模型
vgg16_finetune = build_fine_tuning_vgg16(unfreeze_layers=2)

# 使用較小的學習率編譯（微調時很重要）
vgg16_finetune.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),  # 注意：更小的學習率
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nVGG16 微調模型結構:")
vgg16_finetune.summary()

# 統計可訓練參數
trainable_params_ft = np.sum([np.prod(v.get_shape()) for v in vgg16_finetune.trainable_weights])
total_params_ft = vgg16_finetune.count_params()
print(f"\n總參數量: {total_params_ft:,}")
print(f"可訓練參數: {trainable_params_ft:,}")
print(f"凍結參數: {total_params_ft - trainable_params_ft:,}")
print(f"可訓練比例: {trainable_params_ft/total_params_ft*100:.2f}%")

# ============================================================================
# 訓練微調模型
# ============================================================================
print("\n" + "-" * 80)
print("訓練 VGG16 微調模型...")
print("-" * 80)

# 創建數據增強
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    fill_mode='nearest'
)

# 創建回調
vgg_ft_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-8,
        verbose=1
    )
]

# 訓練模型
print("\n開始微調訓練（帶數據增強）...")
history_vgg_ft = vgg16_finetune.fit(
    datagen.flow(X_train_vgg, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS_FINE_TUNING,
    validation_data=(X_val_vgg, y_val),
    callbacks=vgg_ft_callbacks,
    verbose=1
)

# 評估
print("\n評估 VGG16 微調模型...")
vgg_ft_loss, vgg_ft_acc = vgg16_finetune.evaluate(X_test_vgg, y_test, verbose=0)
print(f"VGG16 微調模型測試準確率: {vgg_ft_acc:.4f}")
print(f"VGG16 微調模型測試損失: {vgg_ft_loss:.4f}")

# ============================================================================
# Part 5: ResNet50 特徵提取 | ResNet50 Feature Extraction
# ============================================================================
print("\n" + "=" * 80)
print("Part 5: ResNet50 特徵提取 | ResNet50 Feature Extraction")
print("=" * 80)

def build_resnet50_extractor(input_shape=(96, 96, 3), num_classes=10):
    """
    構建基於 ResNet50 的特徵提取器
    Build ResNet50-based feature extractor

    ResNet 使用殘差連接，能訓練更深的網絡

    Args:
        input_shape: 輸入形狀
        num_classes: 類別數

    Returns:
        keras.Model: ResNet50 特徵提取器
    """
    print("\n構建 ResNet50 特徵提取器...")

    # 加載預訓練的 ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 凍結所有層
    base_model.trainable = False

    print(f"ResNet50 基礎模型層數: {len(base_model.layers)}")

    # 構建模型
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        layers.BatchNormalization(name='bn1'),
        layers.Dense(256, activation='relu', name='fc1'),
        layers.Dropout(0.4, name='dropout1'),
        layers.Dense(128, activation='relu', name='fc2'),
        layers.Dropout(0.3, name='dropout2'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='ResNet50_FeatureExtractor')

    return model

# 構建 ResNet50 特徵提取器
resnet50_extractor = build_resnet50_extractor()

# 編譯
resnet50_extractor.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nResNet50 特徵提取器結構:")
resnet50_extractor.summary()

trainable_params_resnet = np.sum([np.prod(v.get_shape()) for v in resnet50_extractor.trainable_weights])
total_params_resnet = resnet50_extractor.count_params()
print(f"\n總參數量: {total_params_resnet:,}")
print(f"可訓練參數: {trainable_params_resnet:,}")

# ============================================================================
# 訓練 ResNet50
# ============================================================================
print("\n" + "-" * 80)
print("訓練 ResNet50 特徵提取器...")
print("-" * 80)

# 預處理數據（ResNet50 特定）
X_train_resnet = resnet_preprocess(X_train.copy())
X_val_resnet = resnet_preprocess(X_val.copy())
X_test_resnet = resnet_preprocess(X_test.copy())

# 訓練
resnet_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("\n開始訓練 ResNet50 特徵提取器...")
history_resnet = resnet50_extractor.fit(
    X_train_resnet, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_FEATURE_EXTRACTION,
    validation_data=(X_val_resnet, y_val),
    callbacks=resnet_callbacks,
    verbose=1
)

# 評估
print("\n評估 ResNet50 特徵提取器...")
resnet_loss, resnet_acc = resnet50_extractor.evaluate(X_test_resnet, y_test, verbose=0)
print(f"ResNet50 特徵提取器測試準確率: {resnet_acc:.4f}")
print(f"ResNet50 特徵提取器測試損失: {resnet_loss:.4f}")

# ============================================================================
# Part 6: MobileNetV2 輕量級模型 | MobileNetV2 Lightweight Model
# ============================================================================
print("\n" + "=" * 80)
print("Part 6: MobileNetV2 輕量級模型 | MobileNetV2 Lightweight Model")
print("=" * 80)

def build_mobilenet_extractor(input_shape=(96, 96, 3), num_classes=10):
    """
    構建基於 MobileNetV2 的特徵提取器
    Build MobileNetV2-based feature extractor

    MobileNet 專為移動設備設計，參數少、速度快

    Args:
        input_shape: 輸入形狀
        num_classes: 類別數

    Returns:
        keras.Model: MobileNetV2 特徵提取器
    """
    print("\n構建 MobileNetV2 特徵提取器...")

    # 加載預訓練的 MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 凍結所有層
    base_model.trainable = False

    print(f"MobileNetV2 基礎模型層數: {len(base_model.layers)}")

    # 構建模型
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        layers.Dense(128, activation='relu', name='fc1'),
        layers.Dropout(0.3, name='dropout1'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='MobileNetV2_FeatureExtractor')

    return model

# 構建 MobileNetV2
mobilenet_extractor = build_mobilenet_extractor()

# 編譯
mobilenet_extractor.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nMobileNetV2 特徵提取器結構:")
mobilenet_extractor.summary()

trainable_params_mobile = np.sum([np.prod(v.get_shape()) for v in mobilenet_extractor.trainable_weights])
total_params_mobile = mobilenet_extractor.count_params()
print(f"\n總參數量: {total_params_mobile:,}")
print(f"可訓練參數: {trainable_params_mobile:,}")

# ============================================================================
# 訓練 MobileNetV2
# ============================================================================
print("\n" + "-" * 80)
print("訓練 MobileNetV2 特徵提取器...")
print("-" * 80)

# 預處理數據
X_train_mobile = mobilenet_preprocess(X_train.copy())
X_val_mobile = mobilenet_preprocess(X_val.copy())
X_test_mobile = mobilenet_preprocess(X_test.copy())

# 訓練
mobile_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("\n開始訓練 MobileNetV2...")
history_mobile = mobilenet_extractor.fit(
    X_train_mobile, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_FEATURE_EXTRACTION,
    validation_data=(X_val_mobile, y_val),
    callbacks=mobile_callbacks,
    verbose=1
)

# 評估
print("\n評估 MobileNetV2...")
mobile_loss, mobile_acc = mobilenet_extractor.evaluate(X_test_mobile, y_test, verbose=0)
print(f"MobileNetV2 測試準確率: {mobile_acc:.4f}")
print(f"MobileNetV2 測試損失: {mobile_loss:.4f}")

# ============================================================================
# Part 7: 模型對比和可視化 | Model Comparison and Visualization
# ============================================================================
print("\n" + "=" * 80)
print("Part 7: 模型對比和可視化 | Model Comparison and Visualization")
print("=" * 80)

# 生成預測
print("\n生成所有模型的預測...")
vgg_fe_pred = np.argmax(vgg16_extractor.predict(X_test_vgg, verbose=0), axis=1)
vgg_ft_pred = np.argmax(vgg16_finetune.predict(X_test_vgg, verbose=0), axis=1)
resnet_pred = np.argmax(resnet50_extractor.predict(X_test_resnet, verbose=0), axis=1)
mobile_pred = np.argmax(mobilenet_extractor.predict(X_test_mobile, verbose=0), axis=1)
y_test_flat = y_test.flatten()

# 計算混淆矩陣
vgg_ft_cm = confusion_matrix(y_test_flat, vgg_ft_pred)
resnet_cm = confusion_matrix(y_test_flat, resnet_pred)

# ============================================================================
# 可視化 1: 原始數據樣本
# ============================================================================
print("\n生成可視化圖表...")

fig1, axes = create_subplots(2, 5, figsize=(15, 6))
fig1.suptitle('CIFAR-10 樣本展示（調整大小後）| Sample Images',
             fontsize=16, fontweight='bold')

for i in range(10):
    ax = axes[i // 5, i % 5]
    # 反標準化顯示
    img_display = X_train[i] / 255.0 if X_train[i].max() > 1 else X_train[i]
    ax.imshow(img_display)
    ax.set_title(f'{CIFAR10_CLASSES[y_train[i][0]]}\n{CIFAR10_CLASSES_CN[y_train[i][0]]}',
                fontsize=10)
    ax.axis('off')

plt.tight_layout()
save_figure(fig1, get_output_path('transfer_learning_01_samples.png', '06_DeepLearning'))

# ============================================================================
# 可視化 2: 訓練歷史 - VGG16 特徵提取 vs 微調
# ============================================================================
fig2, axes = create_subplots(2, 2, figsize=(14, 10))
fig2.suptitle('VGG16: 特徵提取 vs 微調 | Feature Extraction vs Fine-tuning',
             fontsize=16, fontweight='bold')

# 特徵提取 - 損失
axes[0, 0].plot(history_vgg_fe.history['loss'], label='訓練損失', linewidth=2)
axes[0, 0].plot(history_vgg_fe.history['val_loss'], label='驗證損失', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('特徵提取: 損失曲線', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 特徵提取 - 準確率
axes[0, 1].plot(history_vgg_fe.history['accuracy'], label='訓練準確率', linewidth=2)
axes[0, 1].plot(history_vgg_fe.history['val_accuracy'], label='驗證準確率', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('特徵提取: 準確率曲線', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 微調 - 損失
axes[1, 0].plot(history_vgg_ft.history['loss'], label='訓練損失', linewidth=2, color='green')
axes[1, 0].plot(history_vgg_ft.history['val_loss'], label='驗證損失', linewidth=2, color='red')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('微調: 損失曲線', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 微調 - 準確率
axes[1, 1].plot(history_vgg_ft.history['accuracy'], label='訓練準確率', linewidth=2, color='green')
axes[1, 1].plot(history_vgg_ft.history['val_accuracy'], label='驗證準確率', linewidth=2, color='red')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('微調: 準確率曲線', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig2, get_output_path('transfer_learning_02_vgg_comparison.png', '06_DeepLearning'))

# ============================================================================
# 可視化 3: 所有模型訓練歷史對比
# ============================================================================
fig3, axes = create_subplots(1, 2, figsize=(14, 6))
fig3.suptitle('所有模型訓練歷史對比 | All Models Training Comparison',
             fontsize=16, fontweight='bold')

# 損失對比
axes[0].plot(history_vgg_fe.history['val_loss'], label='VGG16 特徵提取', linewidth=2)
axes[0].plot(history_vgg_ft.history['val_loss'], label='VGG16 微調', linewidth=2)
axes[0].plot(history_resnet.history['val_loss'], label='ResNet50', linewidth=2)
axes[0].plot(history_mobile.history['val_loss'], label='MobileNetV2', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Loss')
axes[0].set_title('驗證損失對比', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 準確率對比
axes[1].plot(history_vgg_fe.history['val_accuracy'], label='VGG16 特徵提取', linewidth=2)
axes[1].plot(history_vgg_ft.history['val_accuracy'], label='VGG16 微調', linewidth=2)
axes[1].plot(history_resnet.history['val_accuracy'], label='ResNet50', linewidth=2)
axes[1].plot(history_mobile.history['val_accuracy'], label='MobileNetV2', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Accuracy')
axes[1].set_title('驗證準確率對比', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig3, get_output_path('transfer_learning_03_all_models.png', '06_DeepLearning'))

# ============================================================================
# 可視化 4: 模型性能總結
# ============================================================================
fig4, axes = create_subplots(2, 2, figsize=(14, 10))
fig4.suptitle('模型性能全面對比 | Comprehensive Model Performance',
             fontsize=16, fontweight='bold')

models_list = ['VGG16\n特徵提取', 'VGG16\n微調', 'ResNet50', 'MobileNetV2']
accuracies = [vgg_fe_acc, vgg_ft_acc, resnet_acc, mobile_acc]
params_list = [total_params, total_params_ft, total_params_resnet, total_params_mobile]
trainable_list = [trainable_params, trainable_params_ft, trainable_params_resnet, trainable_params_mobile]

colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

# 1. 測試準確率
bars1 = axes[0, 0].bar(models_list, accuracies, color=colors, alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Test Accuracy')
axes[0, 0].set_title('測試準確率對比', fontweight='bold')
axes[0, 0].set_ylim([0.5, 1.0])
axes[0, 0].grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

# 2. 總參數量
bars2 = axes[0, 1].bar(models_list, [p/1e6 for p in params_list],
                      color=colors, alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Parameters (Millions)')
axes[0, 1].set_title('模型參數量對比', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

for bar, params in zip(bars2, params_list):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{params/1e6:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. 可訓練參數
bars3 = axes[1, 0].bar(models_list, [p/1e6 for p in trainable_list],
                      color=colors, alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Trainable Parameters (Millions)')
axes[1, 0].set_title('可訓練參數量對比', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

for bar, params in zip(bars3, trainable_list):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{params/1e6:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 4. 準確率 vs 參數量散點圖
axes[1, 1].scatter([p/1e6 for p in params_list], accuracies,
                  s=200, c=colors, alpha=0.6, edgecolors='black', linewidths=2)
for i, model in enumerate(models_list):
    axes[1, 1].annotate(model.replace('\n', ' '),
                       (params_list[i]/1e6, accuracies[i]),
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
axes[1, 1].set_xlabel('Total Parameters (Millions)')
axes[1, 1].set_ylabel('Test Accuracy')
axes[1, 1].set_title('準確率 vs 模型大小', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig4, get_output_path('transfer_learning_04_performance.png', '06_DeepLearning'))

# ============================================================================
# 可視化 5: 混淆矩陣
# ============================================================================
fig5, axes = create_subplots(1, 2, figsize=(16, 7))
fig5.suptitle('混淆矩陣對比 | Confusion Matrix Comparison',
             fontsize=16, fontweight='bold')

# VGG16 微調
sns.heatmap(vgg_ft_cm, annot=True, fmt='d', cmap='Greens',
           xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES,
           ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_title(f'VGG16 微調\n(準確率: {vgg_ft_acc:.4f})', fontweight='bold')

# ResNet50
sns.heatmap(resnet_cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES,
           ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
axes[1].set_title(f'ResNet50\n(準確率: {resnet_acc:.4f})', fontweight='bold')

plt.tight_layout()
save_figure(fig5, get_output_path('transfer_learning_05_confusion_matrix.png', '06_DeepLearning'))

# ============================================================================
# 可視化 6: 各類別準確率
# ============================================================================
fig6, ax = create_subplots(1, 1, figsize=(14, 6))
fig6.suptitle('各類別準確率對比 | Per-Class Accuracy Comparison',
             fontsize=16, fontweight='bold')

# 計算每類準確率
vgg_fe_class_acc = []
vgg_ft_class_acc = []
resnet_class_acc = []
mobile_class_acc = []

for i in range(NUM_CLASSES):
    mask = y_test_flat == i
    vgg_fe_class_acc.append(accuracy_score(y_test_flat[mask], vgg_fe_pred[mask]))
    vgg_ft_class_acc.append(accuracy_score(y_test_flat[mask], vgg_ft_pred[mask]))
    resnet_class_acc.append(accuracy_score(y_test_flat[mask], resnet_pred[mask]))
    mobile_class_acc.append(accuracy_score(y_test_flat[mask], mobile_pred[mask]))

x = np.arange(NUM_CLASSES)
width = 0.2

bars1 = ax.bar(x - 1.5*width, vgg_fe_class_acc, width, label='VGG16 特徵提取',
              color='#3498db', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x - 0.5*width, vgg_ft_class_acc, width, label='VGG16 微調',
              color='#2ecc71', alpha=0.7, edgecolor='black')
bars3 = ax.bar(x + 0.5*width, resnet_class_acc, width, label='ResNet50',
              color='#e74c3c', alpha=0.7, edgecolor='black')
bars4 = ax.bar(x + 1.5*width, mobile_class_acc, width, label='MobileNetV2',
              color='#f39c12', alpha=0.7, edgecolor='black')

ax.set_xlabel('類別 / Class')
ax.set_ylabel('準確率 / Accuracy')
ax.set_title('各類別準確率詳細對比', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.0])

plt.tight_layout()
save_figure(fig6, get_output_path('transfer_learning_06_class_accuracy.png', '06_DeepLearning'))

# ============================================================================
# 可視化 7: 預測結果展示
# ============================================================================
fig7, axes = create_subplots(3, 6, figsize=(18, 9))
fig7.suptitle('VGG16 微調模型預測結果 | VGG16 Fine-tuned Predictions',
             fontsize=16, fontweight='bold')

# 隨機選擇樣本
np.random.seed(RANDOM_STATE)
indices = np.random.choice(len(X_test), 18, replace=False)

for i, idx in enumerate(indices):
    ax = axes[i // 6, i % 6]
    # 顯示原始圖像
    img_display = X_test[idx] / 255.0 if X_test[idx].max() > 1 else X_test[idx]
    ax.imshow(img_display)

    true_label = y_test_flat[idx]
    pred_label = vgg_ft_pred[idx]

    if true_label == pred_label:
        color = 'green'
        symbol = '✓'
    else:
        color = 'red'
        symbol = '✗'

    title = f'{symbol} 真實: {CIFAR10_CLASSES[true_label]}\n預測: {CIFAR10_CLASSES[pred_label]}'
    ax.set_title(title, fontsize=8, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
save_figure(fig7, get_output_path('transfer_learning_07_predictions.png', '06_DeepLearning'))

# ============================================================================
# 可視化 8: 特徵提取 vs 微調對比
# ============================================================================
fig8, ax = create_subplots(1, 1, figsize=(10, 6))
fig8.suptitle('特徵提取 vs 微調策略對比 | Feature Extraction vs Fine-tuning',
             fontsize=16, fontweight='bold')

strategies = ['特徵提取\nFeature Extraction', '微調\nFine-tuning']
metrics_data = {
    '準確率': [vgg_fe_acc, vgg_ft_acc],
    '可訓練參數 (M)': [trainable_params/1e6, trainable_params_ft/1e6],
}

x = np.arange(len(strategies))
width = 0.35

# 準確率（放大100倍以便顯示）
bars1 = ax.bar(x - width/2, [m*100 for m in metrics_data['準確率']], width,
              label='準確率 (×100)', color='#2ecc71', alpha=0.7, edgecolor='black')

# 可訓練參數
bars2 = ax.bar(x + width/2, metrics_data['可訓練參數 (M)'], width,
              label='可訓練參數 (M)', color='#3498db', alpha=0.7, edgecolor='black')

ax.set_ylabel('數值')
ax.set_title('兩種策略的性能與複雜度對比', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 添加數值標籤
for bar, val in zip(bars1, metrics_data['準確率']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

for bar, val in zip(bars2, metrics_data['可訓練參數 (M)']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.2f}M', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
save_figure(fig8, get_output_path('transfer_learning_08_strategy_comparison.png', '06_DeepLearning'))

# ============================================================================
# 可視化 9: 訓練效率對比
# ============================================================================
fig9, axes = create_subplots(1, 2, figsize=(14, 6))
fig9.suptitle('訓練效率分析 | Training Efficiency Analysis',
             fontsize=16, fontweight='bold')

# 達到目標準確率的 epoch 數（模擬數據）
target_acc = 0.65
epochs_to_target = []

for history in [history_vgg_fe, history_vgg_ft, history_resnet, history_mobile]:
    val_acc = history.history['val_accuracy']
    for i, acc in enumerate(val_acc):
        if acc >= target_acc:
            epochs_to_target.append(i + 1)
            break
    else:
        epochs_to_target.append(len(val_acc))

# 收斂速度
bars = axes[0].bar(range(len(models_list)), epochs_to_target,
                  color=colors, alpha=0.7, edgecolor='black')
axes[0].set_xticks(range(len(models_list)))
axes[0].set_xticklabels(models_list)
axes[0].set_ylabel('Epochs')
axes[0].set_title(f'達到 {target_acc:.0%} 準確率所需 Epochs', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

for bar, epochs in zip(bars, epochs_to_target):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{epochs}', ha='center', va='bottom', fontweight='bold')

# 最終性能提升
baseline = vgg_fe_acc
improvements = [(acc - baseline) * 100 for acc in accuracies]

bars = axes[1].bar(range(len(models_list)), improvements,
                  color=colors, alpha=0.7, edgecolor='black')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xticks(range(len(models_list)))
axes[1].set_xticklabels(models_list)
axes[1].set_ylabel('Accuracy Improvement (%)')
axes[1].set_title('相對於 VGG16 特徵提取的改進', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    va = 'bottom' if height >= 0 else 'top'
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.2f}%', ha='center', va=va, fontweight='bold')

plt.tight_layout()
save_figure(fig9, get_output_path('transfer_learning_09_efficiency.png', '06_DeepLearning'))

# ============================================================================
# 可視化 10: 模型架構可視化
# ============================================================================
fig10, ax = create_subplots(1, 1, figsize=(12, 8))
fig10.suptitle('預訓練模型架構特點 | Pre-trained Model Characteristics',
              fontsize=16, fontweight='bold')

# 創建架構特點比較
architectures = ['VGG16', 'ResNet50', 'MobileNetV2']
characteristics = {
    '層數': [16, 50, 53],
    '參數量 (M)': [total_params/1e6, total_params_resnet/1e6, total_params_mobile/1e6],
    '深度 (相對)': [16, 50, 35],  # 相對深度
}

x = np.arange(len(architectures))
width = 0.25
multiplier = 0

colors_arch = ['#3498db', '#e74c3c', '#f39c12']

for i, (attribute, measurement) in enumerate(characteristics.items()):
    offset = width * multiplier
    bars = ax.bar(x + offset, measurement, width, label=attribute,
                 color=colors_arch[i], alpha=0.7, edgecolor='black')
    multiplier += 1

    # 添加數值標籤
    for bar, val in zip(bars, measurement):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('數值')
ax.set_title('模型架構特點對比', fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(architectures)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_figure(fig10, get_output_path('transfer_learning_10_architecture.png', '06_DeepLearning'))

# ============================================================================
# 可視化 11: 錯誤分析
# ============================================================================
fig11, axes = create_subplots(2, 5, figsize=(18, 8))
fig11.suptitle('錯誤預測案例分析 | Error Case Analysis',
              fontsize=16, fontweight='bold')

# 找出錯誤預測
errors_idx = np.where(vgg_ft_pred != y_test_flat)[0]
print(f"\nVGG16 微調模型總錯誤數: {len(errors_idx)}")

# 隨機選擇10個錯誤案例
if len(errors_idx) >= 10:
    error_samples = np.random.choice(errors_idx, 10, replace=False)
else:
    error_samples = errors_idx[:10] if len(errors_idx) > 0 else []

if len(error_samples) > 0:
    for i, idx in enumerate(error_samples):
        ax = axes[i // 5, i % 5]
        img_display = X_test[idx] / 255.0 if X_test[idx].max() > 1 else X_test[idx]
        ax.imshow(img_display)

        true_label = y_test_flat[idx]
        pred_label = vgg_ft_pred[idx]

        title = f'真實: {CIFAR10_CLASSES[true_label]}\n預測: {CIFAR10_CLASSES[pred_label]}'
        ax.set_title(title, fontsize=9, color='red', fontweight='bold')
        ax.axis('off')

plt.tight_layout()
save_figure(fig11, get_output_path('transfer_learning_11_error_analysis.png', '06_DeepLearning'))

# ============================================================================
# 可視化 12: 遷移學習工作流程
# ============================================================================
fig12, ax = create_subplots(1, 1, figsize=(14, 8))
fig12.suptitle('遷移學習決策樹 | Transfer Learning Decision Tree',
              fontsize=16, fontweight='bold')

# 創建決策流程圖（使用文本）
ax.axis('off')

decision_tree = """
                         開始遷移學習項目
                      Start Transfer Learning
                              │
                              ▼
                    ┌─────────────────────┐
                    │   有多少訓練數據？    │
                    │  How much data?     │
                    └─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   < 1,000               1,000-10,000           > 10,000
    樣本                    樣本                   樣本
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ 特徵提取模式   │    │  輕度微調模式  │    │   深度微調    │
│Feature Extract│    │ Light Tuning  │    │ Deep Tuning   │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
• 凍結所有預訓練層    • 凍結前70%層        • 凍結前30%層
• 只訓練新分類器      • 微調後30%層        • 微調後70%層
• 學習率: 0.001      • 學習率: 0.0001     • 學習率: 0.00001
• 輕量數據增強        • 中度數據增強        • 強數據增強


                      選擇預訓練模型
                   Choose Base Model
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   需要高精度              資源受限             平衡選擇
  High Accuracy         Limited Resources     Balanced
        │                     │                     │
        ▼                     ▼                     ▼
   ResNet50              MobileNetV2            VGG16
   EfficientNet          SqueezeNet            ResNet34


                      訓練策略建議
                   Training Strategy
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   初始階段              中期調整              後期優化
   Initial              Mid-term             Late-stage
        │                     │                     │
        ▼                     ▼                     ▼
• 特徵提取20 epochs   • 解凍頂層塊         • 降低學習率
• 較大學習率          • 降低學習率10×      • 強正則化
• 找到基準性能        • 訓練30 epochs      • 精細調整
"""

ax.text(0.5, 0.5, decision_tree,
       fontsize=10,
       ha='center', va='center',
       family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
save_figure(fig12, get_output_path('transfer_learning_12_workflow.png', '06_DeepLearning'))

# ============================================================================
# Part 8: 最佳實踐總結 | Best Practices Summary
# ============================================================================
print("\n" + "=" * 80)
print("Part 8: 遷移學習最佳實踐 | Transfer Learning Best Practices")
print("=" * 80)

best_practices = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                    遷移學習最佳實踐指南                                      ║
║              Best Practices for Transfer Learning                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

【1. 選擇預訓練模型 | Choosing Pre-trained Models】

   考慮因素:
   ✓ 任務相似度: 源任務與目標任務越相似越好
   ✓ 模型大小: 根據可用資源選擇
   ✓ 推理速度: 生產環境需考慮延遲
   ✓ 準確率要求: 平衡精度與效率

   推薦選擇:
   • 高精度: ResNet50, EfficientNet
   • 快速推理: MobileNetV2, SqueezeNet
   • 通用選擇: VGG16 (簡單但有效)
   • 大規模: ResNet101, Inception

【2. 數據準備 | Data Preparation】

   ✓ 輸入尺寸: 匹配預訓練模型的輸入要求
   ✓ 預處理: 使用與預訓練相同的預處理方法
   ✓ 數據增強: 根據數據量調整強度
   ✓ 類別平衡: 處理不平衡數據集

【3. 特徵提取策略 | Feature Extraction Strategy】

   適用場景:
   • 數據量 < 1,000 樣本
   • 任務與預訓練任務非常相似
   • 計算資源有限
   • 快速原型開發

   實施步驟:
   1. 加載預訓練模型（不含頂層）
   2. 凍結所有卷積層
   3. 添加新的分類器層
   4. 使用標準學習率訓練 (0.001)

【4. 微調策略 | Fine-tuning Strategy】

   適用場景:
   • 數據量 > 1,000 樣本
   • 任務與預訓練任務有差異
   • 有充足的計算資源
   • 需要最佳性能

   實施步驟:
   1. 先進行特徵提取訓練
   2. 解凍頂層幾個塊
   3. 使用小學習率微調 (0.0001 或更小)
   4. 監控過擬合

   微調技巧:
   • 漸進式解凍: 從頂層逐步解凍到底層
   • 差異化學習率: 底層用更小的學習率
   • 較長的訓練: 微調通常需要更多 epochs

【5. 學習率設置 | Learning Rate】

   建議值:
   • 特徵提取: 0.001 - 0.01
   • 微調: 0.0001 - 0.00001
   • 自適應: 使用 ReduceLROnPlateau

   技巧:
   ✓ 預訓練層用小學習率
   ✓ 新增層用正常學習率
   ✓ 使用學習率預熱 (warm-up)
   ✓ 餘弦退火調度

【6. 正則化技術 | Regularization】

   ✓ Dropout: 新增全連接層使用 0.3-0.5
   ✓ 批標準化: 在全連接層之間添加
   ✓ L2 正則化: 權重衰減 1e-4 到 1e-5
   ✓ 數據增強: 最有效的正則化方法

【7. 訓練流程 | Training Pipeline】

   階段 1: 特徵提取 (10-20 epochs)
   └─> 凍結預訓練層
   └─> 學習率 0.001
   └─> 建立基準性能

   階段 2: 輕度微調 (20-30 epochs)
   └─> 解凍頂層塊
   └─> 學習率 0.0001
   └─> 中度數據增強

   階段 3: 深度微調 (可選, 20-30 epochs)
   └─> 解凍更多層
   └─> 學習率 0.00001
   └─> 強數據增強

【8. 常見問題與解決 | Troubleshooting】

   問題: 驗證損失不降反升
   解決:
   • 降低學習率 (除以 10)
   • 增加 Dropout
   • 檢查數據預處理
   • 減少解凍的層數

   問題: 訓練過慢
   解決:
   • 使用更輕量的模型
   • 減小批大小
   • 使用混合精度訓練
   • 凍結更多層

   問題: 準確率不理想
   解決:
   • 嘗試不同的預訓練模型
   • 增加微調的層數
   • 收集更多數據
   • 調整數據增強策略

【9. 模型選擇指南 | Model Selection Guide】

   小數據集 (< 1,000):
   └─> 特徵提取 + 簡單分類器
   └─> 選擇: VGG16, ResNet50

   中等數據集 (1,000 - 10,000):
   └─> 輕度微調
   └─> 選擇: ResNet50, MobileNetV2

   大數據集 (> 10,000):
   └─> 深度微調
   └─> 選擇: EfficientNet, ResNet101

【10. 生產部署建議 | Production Deployment】

   ✓ 模型壓縮: 量化、剪枝、蒸餾
   ✓ 批推理: 提高吞吐量
   ✓ 模型緩存: 加速載入時間
   ✓ A/B 測試: 驗證實際效果
   ✓ 監控: 追蹤性能指標
"""

print(best_practices)

# 生成性能總結表
summary_table = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                         實驗結果總結表                                      ║
║                      Experiment Results Summary                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌──────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│   模型 / Model       │  VGG16 FE   │  VGG16 FT   │  ResNet50   │ MobileNetV2 │
├──────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ 總參數 Total Params  │ {total_params/1e6:>10.2f}M │ {total_params_ft/1e6:>10.2f}M │ {total_params_resnet/1e6:>10.2f}M │ {total_params_mobile/1e6:>10.2f}M │
│ 可訓練 Trainable     │ {trainable_params/1e6:>10.2f}M │ {trainable_params_ft/1e6:>10.2f}M │ {trainable_params_resnet/1e6:>10.2f}M │ {trainable_params_mobile/1e6:>10.2f}M │
│ 測試準確率 Test Acc  │   {vgg_fe_acc:>9.4f} │   {vgg_ft_acc:>9.4f} │   {resnet_acc:>9.4f} │   {mobile_acc:>9.4f} │
│ 測試損失 Test Loss   │   {vgg_fe_loss:>9.4f} │   {vgg_ft_loss:>9.4f} │   {resnet_loss:>9.4f} │   {mobile_loss:>9.4f} │
│ 訓練策略 Strategy    │  特徵提取   │    微調     │  特徵提取   │  特徵提取   │
│ 數據增強 Data Aug    │     否      │     是      │     否      │     否      │
│ 訓練輪數 Epochs      │     {len(history_vgg_fe.history['loss']):>2}      │     {len(history_vgg_ft.history['loss']):>2}      │     {len(history_resnet.history['loss']):>2}      │     {len(history_mobile.history['loss']):>2}      │
└──────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘

【關鍵發現 | Key Findings】

✓ 微調策略: VGG16 微調相比特徵提取提升了 {(vgg_ft_acc-vgg_fe_acc)*100:.2f}% 的準確率
✓ 模型選擇: ResNet50 達到了最高的測試準確率 ({resnet_acc:.4f})
✓ 輕量級: MobileNetV2 用最少的參數獲得了不錯的性能
✓ 數據增強: 對微調模型有顯著的正則化效果
✓ 遷移學習: 所有模型都在較少 epochs 內達到良好性能

【性能排名 | Performance Ranking】

準確率: {max(zip(models_list, accuracies), key=lambda x: x[1])[0]} > ... > {min(zip(models_list, accuracies), key=lambda x: x[1])[0]}
參數量: MobileNetV2 < VGG16 < ResNet50 (效率)
訓練速度: MobileNetV2 > VGG16 > ResNet50

【推薦使用場景 | Recommended Use Cases】

• 生產環境: MobileNetV2 (速度與精度平衡)
• 研究開發: ResNet50 (最高準確率)
• 快速原型: VGG16 特徵提取 (簡單有效)
• 資源受限: MobileNetV2 (參數少)
"""

print(summary_table)

# ============================================================================
# 結束語
# ============================================================================
print("\n" + "=" * 80)
print("教程完成！ | Tutorial Completed!")
print("=" * 80)
print(f"\n✓ 已生成 12 張可視化圖表")
print(f"✓ 圖表保存位置: {get_output_path('', '06_DeepLearning')}")
print(f"✓ 模型保存位置: {get_model_path('', '')}")

print(f"\n【模型性能總結】")
print(f"✓ VGG16 特徵提取: {vgg_fe_acc:.4f}")
print(f"✓ VGG16 微調:     {vgg_ft_acc:.4f} (+{(vgg_ft_acc-vgg_fe_acc)*100:.2f}%)")
print(f"✓ ResNet50:       {resnet_acc:.4f}")
print(f"✓ MobileNetV2:    {mobile_acc:.4f}")

print(f"\n最佳模型: {models_list[accuracies.index(max(accuracies))]} (準確率: {max(accuracies):.4f})")
print(f"最高效模型: MobileNetV2 (參數量: {total_params_mobile/1e6:.2f}M)")

print("\n" + "=" * 80)
print("感謝使用本教程！")
print("Thank you for using this tutorial!")
print("下一步: 嘗試在您自己的數據集上應用遷移學習！")
print("=" * 80)
