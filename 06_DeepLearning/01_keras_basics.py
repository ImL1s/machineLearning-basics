"""
深度學習入門 - Keras/TensorFlow 基礎
Deep Learning Basics with Keras

從零開始學習神經網絡
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
    from keras.utils import to_categorical
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("警告：TensorFlow/Keras 未安裝")
    print("請運行：pip install tensorflow")

if KERAS_AVAILABLE:
    print("=" * 80)
    print("深度學習入門 - Keras 基礎".center(80))
    print("=" * 80)

    # ====================================================================
    # 1. 全連接神經網絡（MLP）
    # ====================================================================
    print("\n【1】全連接神經網絡（Multi-Layer Perceptron）")
    print("-" * 80)

    # 加載手寫數字數據
    digits = load_digits()
    X, y = digits.data, digits.target

    # 數據預處理
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # One-hot 編碼標籤
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    print(f"訓練集形狀：{X_train.shape}")
    print(f"測試集形狀：{X_test.shape}")

    # 建立模型
    model_mlp = Sequential([
        Dense(128, activation='relu', input_shape=(64,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])

    # 編譯模型
    model_mlp.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n模型結構：")
    model_mlp.summary()

    # 訓練模型
    print("\n訓練中...")
    history_mlp = model_mlp.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # 評估模型
    test_loss, test_acc = model_mlp.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n測試準確率：{test_acc:.4f}")
    print(f"測試損失：{test_loss:.4f}")

    # ====================================================================
    # 2. 卷積神經網絡（CNN）示例
    # ====================================================================
    print("\n【2】卷積神經網絡（Convolutional Neural Network）")
    print("-" * 80)

    # 重塑數據為圖像格式
    X_train_img = X_train.reshape(-1, 8, 8, 1)
    X_test_img = X_test.reshape(-1, 8, 8, 1)

    # 建立 CNN 模型
    model_cnn = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    model_cnn.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nCNN 模型結構：")
    model_cnn.summary()

    # 訓練 CNN
    print("\n訓練 CNN...")
    history_cnn = model_cnn.fit(
        X_train_img, y_train_cat,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # 評估 CNN
    test_loss_cnn, test_acc_cnn = model_cnn.evaluate(X_test_img, y_test_cat, verbose=0)
    print(f"\nCNN 測試準確率：{test_acc_cnn:.4f}")

    # ====================================================================
    # 可視化
    # ====================================================================

    fig = plt.figure(figsize=(16, 10))

    # 1. MLP 訓練歷史 - 損失
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history_mlp.history['loss'], label='Training Loss')
    ax1.plot(history_mlp.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('MLP: Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. MLP 訓練歷史 - 準確率
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history_mlp.history['accuracy'], label='Training Accuracy')
    ax2.plot(history_mlp.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('MLP: Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. CNN 訓練歷史 - 準確率
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(history_cnn.history['accuracy'], label='Training Accuracy')
    ax3.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('CNN: Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 預測示例
    ax4 = plt.subplot(2, 3, 4)
    sample_idx = 0
    sample_image = X_test[sample_idx].reshape(8, 8)
    sample_pred = model_mlp.predict(X_test[sample_idx:sample_idx+1], verbose=0)
    predicted_class = np.argmax(sample_pred)
    true_class = y_test[sample_idx]

    ax4.imshow(sample_image, cmap='gray')
    ax4.set_title(f'Prediction: {predicted_class}, True: {true_class}',
                 fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 5. 預測概率分布
    ax5 = plt.subplot(2, 3, 5)
    ax5.bar(range(10), sample_pred[0], alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Digit')
    ax5.set_ylabel('Probability')
    ax5.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. 模型比較
    ax6 = plt.subplot(2, 3, 6)
    models = ['MLP', 'CNN']
    accuracies = [test_acc, test_acc_cnn]
    bars = ax6.bar(models, accuracies, color=['blue', 'green'], alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Test Accuracy')
    ax6.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax6.set_ylim([0.9, 1.0])
    ax6.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('06_DeepLearning/keras_basics_results.png', dpi=150)
    print("\n已保存結果圖表")

    print("\n" + "=" * 80)
    print("深度學習要點總結")
    print("=" * 80)
    print("""
    1. 神經網絡基礎：
       • 輸入層：接收數據
       • 隱藏層：特徵提取和轉換
       • 輸出層：最終預測

    2. 常用層類型：
       • Dense（全連接層）：標準神經網絡層
       • Conv2D（卷積層）：處理圖像數據
       • MaxPooling2D（池化層）：降低維度
       • Dropout：防止過擬合
       • Flatten：展平數據

    3. 激活函數：
       • ReLU：隱藏層常用，解決梯度消失
       • Sigmoid：二分類輸出
       • Softmax：多分類輸出
       • Tanh：範圍 [-1, 1]

    4. 優化器：
       • SGD：隨機梯度下降
       • Adam：自適應學習率（最常用）
       • RMSprop：適合 RNN
       • Adagrad：稀疏數據

    5. 損失函數：
       • binary_crossentropy：二分類
       • categorical_crossentropy：多分類
       • mse：回歸問題
       • sparse_categorical_crossentropy：整數標籤多分類

    6. 防止過擬合：
       • Dropout：隨機丟棄神經元
       • Early Stopping：提前停止訓練
       • L1/L2 正則化：懲罰大權重
       • Data Augmentation：數據增強
       • Batch Normalization：批量標準化

    7. 實踐建議：
       • 從簡單模型開始
       • 使用預訓練模型（遷移學習）
       • 調整學習率
       • 監控訓練和驗證損失
       • 使用 GPU 加速訓練
       • 保存最佳模型
    """)

else:
    print("\n請安裝 TensorFlow 以運行深度學習示例：")
    print("pip install tensorflow")
