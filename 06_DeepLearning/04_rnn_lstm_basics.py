"""
RNN/LSTM 序列模型基礎 | RNN/LSTM Sequence Models Basics

本教程涵蓋：
This tutorial covers:
1. 序列數據基礎和預處理 / Sequence Data Basics and Preprocessing
2. Simple RNN 原理和實現 / Simple RNN Theory and Implementation
3. LSTM 原理和實現 / LSTM Theory and Implementation
4. GRU 原理和實現 / GRU Theory and Implementation
5. 雙向 RNN / Bidirectional RNN
6. 文本生成示例 / Text Generation Example
7. 時間序列預測示例 / Time Series Forecasting Example
8. 股票價格預測案例 / Stock Price Prediction Case Study
9. 模型對比和最佳實踐 / Model Comparison and Best Practices

作者: Machine Learning Tutorial
日期: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# 深度學習庫 / Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 機器學習工具 / Machine Learning Tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Utils 模塊 / Utils Module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import (RANDOM_STATE, setup_chinese_fonts, save_figure,
                   get_output_path, create_subplots)

# 設置中文字體 / Setup Chinese Fonts
setup_chinese_fonts()

# 設置隨機種子 / Set Random Seeds
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# 設置 seaborn 樣式 / Set Seaborn Style
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 80)
print("RNN/LSTM 序列模型基礎 | RNN/LSTM Sequence Models Basics".center(80))
print("=" * 80)
print(f"TensorFlow 版本 / Version: {tf.__version__}")
print(f"Keras 版本 / Version: {keras.__version__}")
print(f"NumPy 版本 / Version: {np.__version__}")
print(f"隨機種子 / Random Seed: {RANDOM_STATE}")
print("=" * 80)


# ============================================================================
# Part 1: 序列數據基礎 / Sequence Data Basics
# ============================================================================
print("\n" + "=" * 80)
print("Part 1: 序列數據基礎 / Sequence Data Basics")
print("=" * 80)

def generate_sine_sequences(n_samples=1000, seq_length=50, noise=0.1):
    """
    生成正弦波序列數據 / Generate sine wave sequence data

    Parameters:
    -----------
    n_samples : int
        樣本數量 / Number of samples
    seq_length : int
        序列長度 / Sequence length
    noise : float
        噪聲水平 / Noise level

    Returns:
    --------
    X : ndarray, shape (n_samples, seq_length-1)
        輸入序列 / Input sequences
    y : ndarray, shape (n_samples,)
        目標值 / Target values
    """
    X = []
    y = []

    for _ in range(n_samples):
        # 隨機起始點 / Random starting point
        start = np.random.rand() * 2 * np.pi
        # 生成序列 / Generate sequence
        x_seq = np.linspace(start, start + 4 * np.pi, seq_length)
        y_seq = np.sin(x_seq) + np.random.normal(0, noise, seq_length)

        # 使用前 seq_length-1 個點預測最後一個點
        # Use first seq_length-1 points to predict the last point
        X.append(y_seq[:-1])
        y.append(y_seq[-1])

    return np.array(X), np.array(y)


def generate_multi_sine_sequences(n_samples=1000, seq_length=50, n_features=3, noise=0.1):
    """
    生成多特徵正弦波序列數據 / Generate multi-feature sine wave sequences

    Parameters:
    -----------
    n_samples : int
        樣本數量 / Number of samples
    seq_length : int
        序列長度 / Sequence length
    n_features : int
        特徵數量 / Number of features
    noise : float
        噪聲水平 / Noise level

    Returns:
    --------
    X : ndarray, shape (n_samples, seq_length, n_features)
        輸入序列 / Input sequences
    y : ndarray, shape (n_samples,)
        目標值 / Target values
    """
    X = []
    y = []

    for _ in range(n_samples):
        features = []
        start = np.random.rand() * 2 * np.pi
        x_seq = np.linspace(start, start + 4 * np.pi, seq_length)

        # 生成多個特徵 / Generate multiple features
        for i in range(n_features):
            freq = 1 + i * 0.5  # 不同頻率 / Different frequencies
            feature = np.sin(freq * x_seq) + np.random.normal(0, noise, seq_length)
            features.append(feature)

        X.append(np.array(features).T)
        # 目標是第一個特徵的最後一個值 / Target is last value of first feature
        y.append(features[0][-1])

    return np.array(X), np.array(y)


print("\n生成單特徵正弦波序列數據...")
print("Generating single-feature sine wave sequences...")
X_seq, y_seq = generate_sine_sequences(n_samples=1000, seq_length=50, noise=0.1)
print(f"序列數據形狀 / Sequence data shape: X={X_seq.shape}, y={y_seq.shape}")

print("\n生成多特徵正弦波序列數據...")
print("Generating multi-feature sine wave sequences...")
X_multi, y_multi = generate_multi_sine_sequences(n_samples=1000, seq_length=50, n_features=3)
print(f"多特徵序列形狀 / Multi-feature sequence shape: X={X_multi.shape}, y={y_multi.shape}")

# 可視化 1: 序列示例 / Visualization 1: Sequence Examples
print("\n生成可視化 1: 序列示例...")
print("Generating Visualization 1: Sequence Examples...")
fig, axes = create_subplots(2, 2, figsize=(16, 10))

for i in range(4):
    ax = axes[i // 2, i % 2]
    ax.plot(range(len(X_seq[i])), X_seq[i], 'b-', linewidth=2, label='輸入序列 / Input Sequence')
    ax.axhline(y=y_seq[i], color='r', linestyle='--', linewidth=2,
               label=f'目標 / Target: {y_seq[i]:.2f}')
    ax.set_title(f'序列示例 {i+1} / Sequence Example {i+1}', fontsize=12, fontweight='bold')
    ax.set_xlabel('時間步 / Time Step', fontsize=10)
    ax.set_ylabel('值 / Value', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('01_rnn_sequence_examples.png', 'DeepLearning'))
plt.close()

# 可視化 2: 多特徵序列示例 / Visualization 2: Multi-feature Sequence Examples
print("\n生成可視化 2: 多特徵序列示例...")
print("Generating Visualization 2: Multi-feature Sequence Examples...")
fig, axes = create_subplots(2, 2, figsize=(16, 10))

for i in range(4):
    ax = axes[i // 2, i % 2]
    for j in range(X_multi.shape[2]):
        ax.plot(range(X_multi.shape[1]), X_multi[i, :, j],
                linewidth=2, label=f'特徵 {j+1} / Feature {j+1}')
    ax.set_title(f'多特徵序列示例 {i+1} / Multi-feature Example {i+1}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('時間步 / Time Step', fontsize=10)
    ax.set_ylabel('值 / Value', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('02_rnn_multifeature_sequences.png', 'DeepLearning'))
plt.close()

# 分割數據 / Split Data
print("\n分割數據集...")
print("Splitting datasets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=RANDOM_STATE
)

# Reshape for RNN: (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"訓練集形狀 / Training set shape: {X_train.shape}")
print(f"測試集形狀 / Testing set shape: {X_test.shape}")
print(f"訓練標籤形狀 / Training labels shape: {y_train.shape}")
print(f"測試標籤形狀 / Testing labels shape: {y_test.shape}")


# ============================================================================
# Part 2: Simple RNN 模型 / Simple RNN Model
# ============================================================================
print("\n" + "=" * 80)
print("Part 2: Simple RNN 模型 / Simple RNN Model")
print("=" * 80)

print("""
Simple RNN 原理 / Simple RNN Theory:
----------------------------------------
Simple RNN 是最基本的循環神經網絡，具有以下特點：
Simple RNN is the most basic recurrent neural network with the following features:

1. 隱藏狀態公式 / Hidden State Formula:
   h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)

2. 輸出公式 / Output Formula:
   y_t = W_hy * h_t + b_y

3. 優點 / Advantages:
   - 簡單直觀 / Simple and intuitive
   - 計算效率高 / Computationally efficient
   - 參數較少 / Fewer parameters

4. 缺點 / Disadvantages:
   - 梯度消失問題嚴重 / Severe gradient vanishing problem
   - 難以學習長期依賴 / Difficulty learning long-term dependencies
   - 不適合長序列 / Not suitable for long sequences
""")

# 2.1 構建 Simple RNN 模型 / Build Simple RNN Model
print("\n構建 Simple RNN 模型...")
print("Building Simple RNN model...")

simple_rnn_model = models.Sequential([
    layers.SimpleRNN(32, activation='tanh', return_sequences=True,
                     input_shape=(X_train.shape[1], 1), name='SimpleRNN_1'),
    layers.Dropout(0.2, name='Dropout_1'),
    layers.SimpleRNN(16, activation='tanh', name='SimpleRNN_2'),
    layers.Dropout(0.2, name='Dropout_2'),
    layers.Dense(8, activation='relu', name='Dense_1'),
    layers.Dense(1, name='Output')
], name='Simple_RNN_Model')

simple_rnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nSimple RNN 模型結構 / Model Architecture:")
simple_rnn_model.summary()

# 2.2 訓練模型 / Train Model
print("\n訓練 Simple RNN 模型...")
print("Training Simple RNN model...")

# 回調函數 / Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=0
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=0
)

history_rnn = simple_rnn_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

print(f"訓練完成！訓練了 {len(history_rnn.history['loss'])} 個 epoch")
print(f"Training completed! Trained for {len(history_rnn.history['loss'])} epochs")

# 2.3 評估模型 / Evaluate Model
print("\n評估 Simple RNN 模型...")
print("Evaluating Simple RNN model...")

train_loss_rnn, train_mae_rnn = simple_rnn_model.evaluate(X_train, y_train, verbose=0)
test_loss_rnn, test_mae_rnn = simple_rnn_model.evaluate(X_test, y_test, verbose=0)

print(f"Simple RNN - 訓練 MSE / Training MSE: {train_loss_rnn:.4f}")
print(f"Simple RNN - 訓練 MAE / Training MAE: {train_mae_rnn:.4f}")
print(f"Simple RNN - 測試 MSE / Testing MSE: {test_loss_rnn:.4f}")
print(f"Simple RNN - 測試 MAE / Testing MAE: {test_mae_rnn:.4f}")

# 預測 / Predictions
y_pred_rnn = simple_rnn_model.predict(X_test, verbose=0).flatten()
r2_rnn = r2_score(y_test, y_pred_rnn)
print(f"Simple RNN - R² Score: {r2_rnn:.4f}")


# ============================================================================
# Part 3: LSTM 模型 / LSTM Model
# ============================================================================
print("\n" + "=" * 80)
print("Part 3: LSTM 模型 / LSTM Model")
print("=" * 80)

print("""
LSTM 原理 / LSTM Theory:
----------------------------------------
LSTM (Long Short-Term Memory) 通過門控機制解決梯度消失問題：
LSTM solves the gradient vanishing problem through gating mechanisms:

1. 遺忘門 / Forget Gate:
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   決定丟棄哪些信息 / Decides what information to discard

2. 輸入門 / Input Gate:
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   決定更新哪些信息 / Decides what information to update

3. 候選記憶 / Candidate Memory:
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
   新的候選值 / New candidate values

4. 更新細胞狀態 / Update Cell State:
   C_t = f_t * C_{t-1} + i_t * C̃_t

5. 輸出門 / Output Gate:
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)

優點 / Advantages:
- 有效解決梯度消失 / Effectively solves gradient vanishing
- 能學習長期依賴 / Can learn long-term dependencies
- 適合長序列 / Suitable for long sequences

缺點 / Disadvantages:
- 參數多，訓練慢 / More parameters, slower training
- 計算複雜度高 / Higher computational complexity
""")

# 3.1 構建 LSTM 模型 / Build LSTM Model
print("\n構建 LSTM 模型...")
print("Building LSTM model...")

lstm_model = models.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1),
                name='LSTM_1'),
    layers.Dropout(0.3, name='Dropout_1'),
    layers.LSTM(32, return_sequences=False, name='LSTM_2'),
    layers.Dropout(0.3, name='Dropout_2'),
    layers.Dense(16, activation='relu', name='Dense_1'),
    layers.Dense(1, name='Output')
], name='LSTM_Model')

lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nLSTM 模型結構 / Model Architecture:")
lstm_model.summary()

# 3.2 訓練模型 / Train Model
print("\n訓練 LSTM 模型...")
print("Training LSTM model...")

history_lstm = lstm_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

print(f"訓練完成！訓練了 {len(history_lstm.history['loss'])} 個 epoch")
print(f"Training completed! Trained for {len(history_lstm.history['loss'])} epochs")

# 3.3 評估模型 / Evaluate Model
print("\n評估 LSTM 模型...")
print("Evaluating LSTM model...")

train_loss_lstm, train_mae_lstm = lstm_model.evaluate(X_train, y_train, verbose=0)
test_loss_lstm, test_mae_lstm = lstm_model.evaluate(X_test, y_test, verbose=0)

print(f"LSTM - 訓練 MSE / Training MSE: {train_loss_lstm:.4f}")
print(f"LSTM - 訓練 MAE / Training MAE: {train_mae_lstm:.4f}")
print(f"LSTM - 測試 MSE / Testing MSE: {test_loss_lstm:.4f}")
print(f"LSTM - 測試 MAE / Testing MAE: {test_mae_lstm:.4f}")

# 預測 / Predictions
y_pred_lstm = lstm_model.predict(X_test, verbose=0).flatten()
r2_lstm = r2_score(y_test, y_pred_lstm)
print(f"LSTM - R² Score: {r2_lstm:.4f}")


# ============================================================================
# Part 4: GRU 模型 / GRU Model
# ============================================================================
print("\n" + "=" * 80)
print("Part 4: GRU 模型 / GRU Model")
print("=" * 80)

print("""
GRU 原理 / GRU Theory:
----------------------------------------
GRU (Gated Recurrent Unit) 是 LSTM 的簡化版本：
GRU is a simplified version of LSTM:

1. 重置門 / Reset Gate:
   r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
   決定忽略多少過去信息 / Decides how much past information to ignore

2. 更新門 / Update Gate:
   z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
   決定保留多少過去信息 / Decides how much past information to keep

3. 候選隱藏狀態 / Candidate Hidden State:
   h̃_t = tanh(W · [r_t * h_{t-1}, x_t] + b)

4. 最終隱藏狀態 / Final Hidden State:
   h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t

優點 / Advantages:
- 比 LSTM 更簡單 / Simpler than LSTM
- 訓練更快 / Faster training
- 性能接近 LSTM / Performance close to LSTM
- 參數更少 / Fewer parameters

缺點 / Disadvantages:
- 表達能力略低於 LSTM / Slightly less expressive than LSTM
""")

# 4.1 構建 GRU 模型 / Build GRU Model
print("\n構建 GRU 模型...")
print("Building GRU model...")

gru_model = models.Sequential([
    layers.GRU(64, return_sequences=True, input_shape=(X_train.shape[1], 1),
               name='GRU_1'),
    layers.Dropout(0.3, name='Dropout_1'),
    layers.GRU(32, return_sequences=False, name='GRU_2'),
    layers.Dropout(0.3, name='Dropout_2'),
    layers.Dense(16, activation='relu', name='Dense_1'),
    layers.Dense(1, name='Output')
], name='GRU_Model')

gru_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nGRU 模型結構 / Model Architecture:")
gru_model.summary()

# 4.2 訓練模型 / Train Model
print("\n訓練 GRU 模型...")
print("Training GRU model...")

history_gru = gru_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

print(f"訓練完成！訓練了 {len(history_gru.history['loss'])} 個 epoch")
print(f"Training completed! Trained for {len(history_gru.history['loss'])} epochs")

# 4.3 評估模型 / Evaluate Model
print("\n評估 GRU 模型...")
print("Evaluating GRU model...")

train_loss_gru, train_mae_gru = gru_model.evaluate(X_train, y_train, verbose=0)
test_loss_gru, test_mae_gru = gru_model.evaluate(X_test, y_test, verbose=0)

print(f"GRU - 訓練 MSE / Training MSE: {train_loss_gru:.4f}")
print(f"GRU - 訓練 MAE / Training MAE: {train_mae_gru:.4f}")
print(f"GRU - 測試 MSE / Testing MSE: {test_loss_gru:.4f}")
print(f"GRU - 測試 MAE / Testing MAE: {test_mae_gru:.4f}")

# 預測 / Predictions
y_pred_gru = gru_model.predict(X_test, verbose=0).flatten()
r2_gru = r2_score(y_test, y_pred_gru)
print(f"GRU - R² Score: {r2_gru:.4f}")


# ============================================================================
# Part 5: 雙向 RNN / Bidirectional RNN
# ============================================================================
print("\n" + "=" * 80)
print("Part 5: 雙向 RNN / Bidirectional RNN")
print("=" * 80)

print("""
雙向 RNN 原理 / Bidirectional RNN Theory:
----------------------------------------
雙向 RNN 同時從前向和後向處理序列：
Bidirectional RNN processes sequences in both forward and backward directions:

1. 前向層 / Forward Layer:
   處理序列從 t=0 到 t=T / Processes sequence from t=0 to t=T

2. 後向層 / Backward Layer:
   處理序列從 t=T 到 t=0 / Processes sequence from t=T to t=0

3. 輸出 / Output:
   結合前向和後向的隱藏狀態 / Combines forward and backward hidden states

優點 / Advantages:
- 利用完整的上下文信息 / Uses complete context information
- 提高預測準確性 / Improves prediction accuracy
- 適合文本、語音等任務 / Suitable for text, speech tasks

缺點 / Disadvantages:
- 需要完整序列 / Requires complete sequence
- 不能用於實時預測 / Cannot be used for real-time prediction
- 計算量加倍 / Doubles computational cost
""")

# 5.1 構建雙向 LSTM 模型 / Build Bidirectional LSTM Model
print("\n構建雙向 LSTM 模型...")
print("Building Bidirectional LSTM model...")

bidirectional_model = models.Sequential([
    layers.Bidirectional(layers.LSTM(64, return_sequences=True),
                        input_shape=(X_train.shape[1], 1), name='Bi_LSTM_1'),
    layers.Dropout(0.3, name='Dropout_1'),
    layers.Bidirectional(layers.LSTM(32), name='Bi_LSTM_2'),
    layers.Dropout(0.3, name='Dropout_2'),
    layers.Dense(16, activation='relu', name='Dense_1'),
    layers.Dense(1, name='Output')
], name='Bidirectional_LSTM_Model')

bidirectional_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n雙向 LSTM 模型結構 / Model Architecture:")
bidirectional_model.summary()

# 5.2 訓練模型 / Train Model
print("\n訓練雙向 LSTM 模型...")
print("Training Bidirectional LSTM model...")

history_bi = bidirectional_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

print(f"訓練完成！訓練了 {len(history_bi.history['loss'])} 個 epoch")
print(f"Training completed! Trained for {len(history_bi.history['loss'])} epochs")

# 5.3 評估模型 / Evaluate Model
print("\n評估雙向 LSTM 模型...")
print("Evaluating Bidirectional LSTM model...")

train_loss_bi, train_mae_bi = bidirectional_model.evaluate(X_train, y_train, verbose=0)
test_loss_bi, test_mae_bi = bidirectional_model.evaluate(X_test, y_test, verbose=0)

print(f"Bidirectional LSTM - 訓練 MSE / Training MSE: {train_loss_bi:.4f}")
print(f"Bidirectional LSTM - 訓練 MAE / Training MAE: {train_mae_bi:.4f}")
print(f"Bidirectional LSTM - 測試 MSE / Testing MSE: {test_loss_bi:.4f}")
print(f"Bidirectional LSTM - 測試 MAE / Testing MAE: {test_mae_bi:.4f}")

# 預測 / Predictions
y_pred_bi = bidirectional_model.predict(X_test, verbose=0).flatten()
r2_bi = r2_score(y_test, y_pred_bi)
print(f"Bidirectional LSTM - R² Score: {r2_bi:.4f}")


# ============================================================================
# Part 6: 模型對比 / Model Comparison
# ============================================================================
print("\n" + "=" * 80)
print("Part 6: 模型對比 / Model Comparison")
print("=" * 80)

# 可視化 3: 訓練歷史對比 / Visualization 3: Training History Comparison
print("\n生成可視化 3: 訓練歷史對比...")
print("Generating Visualization 3: Training History Comparison...")

fig, axes = create_subplots(2, 4, figsize=(20, 10))

# Simple RNN - Loss
axes[0, 0].plot(history_rnn.history['loss'], label='訓練損失 / Train', linewidth=2)
axes[0, 0].plot(history_rnn.history['val_loss'], label='驗證損失 / Val', linewidth=2)
axes[0, 0].set_title('Simple RNN - 損失曲線 / Loss Curve', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# LSTM - Loss
axes[0, 1].plot(history_lstm.history['loss'], label='訓練損失 / Train', linewidth=2)
axes[0, 1].plot(history_lstm.history['val_loss'], label='驗證損失 / Val', linewidth=2)
axes[0, 1].set_title('LSTM - 損失曲線 / Loss Curve', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss (MSE)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# GRU - Loss
axes[0, 2].plot(history_gru.history['loss'], label='訓練損失 / Train', linewidth=2)
axes[0, 2].plot(history_gru.history['val_loss'], label='驗證損失 / Val', linewidth=2)
axes[0, 2].set_title('GRU - 損失曲線 / Loss Curve', fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss (MSE)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Bidirectional - Loss
axes[0, 3].plot(history_bi.history['loss'], label='訓練損失 / Train', linewidth=2)
axes[0, 3].plot(history_bi.history['val_loss'], label='驗證損失 / Val', linewidth=2)
axes[0, 3].set_title('Bidirectional LSTM - 損失曲線 / Loss Curve', fontweight='bold')
axes[0, 3].set_xlabel('Epoch')
axes[0, 3].set_ylabel('Loss (MSE)')
axes[0, 3].legend()
axes[0, 3].grid(True, alpha=0.3)

# Simple RNN - MAE
axes[1, 0].plot(history_rnn.history['mae'], label='訓練 MAE / Train', linewidth=2)
axes[1, 0].plot(history_rnn.history['val_mae'], label='驗證 MAE / Val', linewidth=2)
axes[1, 0].set_title('Simple RNN - MAE 曲線 / MAE Curve', fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MAE')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# LSTM - MAE
axes[1, 1].plot(history_lstm.history['mae'], label='訓練 MAE / Train', linewidth=2)
axes[1, 1].plot(history_lstm.history['val_mae'], label='驗證 MAE / Val', linewidth=2)
axes[1, 1].set_title('LSTM - MAE 曲線 / MAE Curve', fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('MAE')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# GRU - MAE
axes[1, 2].plot(history_gru.history['mae'], label='訓練 MAE / Train', linewidth=2)
axes[1, 2].plot(history_gru.history['val_mae'], label='驗證 MAE / Val', linewidth=2)
axes[1, 2].set_title('GRU - MAE 曲線 / MAE Curve', fontweight='bold')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('MAE')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# Bidirectional - MAE
axes[1, 3].plot(history_bi.history['mae'], label='訓練 MAE / Train', linewidth=2)
axes[1, 3].plot(history_bi.history['val_mae'], label='驗證 MAE / Val', linewidth=2)
axes[1, 3].set_title('Bidirectional LSTM - MAE 曲線 / MAE Curve', fontweight='bold')
axes[1, 3].set_xlabel('Epoch')
axes[1, 3].set_ylabel('MAE')
axes[1, 3].legend()
axes[1, 3].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('03_rnn_training_history.png', 'DeepLearning'))
plt.close()

# 可視化 4: 性能對比柱狀圖 / Visualization 4: Performance Comparison Bar Chart
print("\n生成可視化 4: 性能對比柱狀圖...")
print("Generating Visualization 4: Performance Comparison Bar Chart...")

models_names = ['Simple RNN', 'LSTM', 'GRU', 'Bidirectional\nLSTM']
train_maes = [train_mae_rnn, train_mae_lstm, train_mae_gru, train_mae_bi]
test_maes = [test_mae_rnn, test_mae_lstm, test_mae_gru, test_mae_bi]
r2_scores = [r2_rnn, r2_lstm, r2_gru, r2_bi]
params_counts = [
    simple_rnn_model.count_params(),
    lstm_model.count_params(),
    gru_model.count_params(),
    bidirectional_model.count_params()
]

fig, axes = create_subplots(2, 2, figsize=(16, 12))

# MAE Comparison
x = np.arange(len(models_names))
width = 0.35

axes[0, 0].bar(x - width/2, train_maes, width, label='訓練 MAE / Train MAE', alpha=0.8)
axes[0, 0].bar(x + width/2, test_maes, width, label='測試 MAE / Test MAE', alpha=0.8)
axes[0, 0].set_xlabel('模型 / Model', fontsize=11)
axes[0, 0].set_ylabel('MAE', fontsize=11)
axes[0, 0].set_title('MAE 性能對比 / MAE Performance Comparison',
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(models_names)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# R² Score Comparison
axes[0, 1].bar(models_names, r2_scores, alpha=0.8, color='green')
axes[0, 1].set_xlabel('模型 / Model', fontsize=11)
axes[0, 1].set_ylabel('R² Score', fontsize=11)
axes[0, 1].set_title('R² Score 對比 / R² Score Comparison',
                     fontsize=12, fontweight='bold')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(r2_scores):
    axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# Parameters Count Comparison
axes[1, 0].bar(models_names, params_counts, alpha=0.8, color='orange')
axes[1, 0].set_xlabel('模型 / Model', fontsize=11)
axes[1, 0].set_ylabel('參數數量 / Parameters Count', fontsize=11)
axes[1, 0].set_title('模型參數數量對比 / Parameters Count Comparison',
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate(params_counts):
    axes[1, 0].text(i, v + max(params_counts)*0.02, f'{v:,}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

# Performance vs Parameters
axes[1, 1].scatter(params_counts, r2_scores, s=200, alpha=0.6, c=range(len(models_names)),
                   cmap='viridis', edgecolors='black', linewidth=2)
for i, model in enumerate(models_names):
    axes[1, 1].annotate(model, (params_counts[i], r2_scores[i]),
                       xytext=(10, 5), textcoords='offset points', fontsize=9)
axes[1, 1].set_xlabel('參數數量 / Parameters Count', fontsize=11)
axes[1, 1].set_ylabel('R² Score', fontsize=11)
axes[1, 1].set_title('性能 vs 複雜度 / Performance vs Complexity',
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('04_rnn_performance_comparison.png', 'DeepLearning'))
plt.close()

# 可視化 5: 預測結果對比 / Visualization 5: Predictions Comparison
print("\n生成可視化 5: 預測結果對比...")
print("Generating Visualization 5: Predictions Comparison...")

fig, axes = create_subplots(2, 2, figsize=(16, 12))

# Simple RNN Predictions
axes[0, 0].scatter(y_test, y_pred_rnn, alpha=0.5, s=30)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='理想預測 / Perfect Prediction')
axes[0, 0].set_xlabel('真實值 / True Values', fontsize=10)
axes[0, 0].set_ylabel('預測值 / Predicted Values', fontsize=10)
axes[0, 0].set_title(f'Simple RNN 預測結果 / Predictions (R²={r2_rnn:.4f})',
                     fontsize=11, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# LSTM Predictions
axes[0, 1].scatter(y_test, y_pred_lstm, alpha=0.5, s=30)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='理想預測 / Perfect Prediction')
axes[0, 1].set_xlabel('真實值 / True Values', fontsize=10)
axes[0, 1].set_ylabel('預測值 / Predicted Values', fontsize=10)
axes[0, 1].set_title(f'LSTM 預測結果 / Predictions (R²={r2_lstm:.4f})',
                     fontsize=11, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# GRU Predictions
axes[1, 0].scatter(y_test, y_pred_gru, alpha=0.5, s=30)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='理想預測 / Perfect Prediction')
axes[1, 0].set_xlabel('真實值 / True Values', fontsize=10)
axes[1, 0].set_ylabel('預測值 / Predicted Values', fontsize=10)
axes[1, 0].set_title(f'GRU 預測結果 / Predictions (R²={r2_gru:.4f})',
                     fontsize=11, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Bidirectional LSTM Predictions
axes[1, 1].scatter(y_test, y_pred_bi, alpha=0.5, s=30)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='理想預測 / Perfect Prediction')
axes[1, 1].set_xlabel('真實值 / True Values', fontsize=10)
axes[1, 1].set_ylabel('預測值 / Predicted Values', fontsize=10)
axes[1, 1].set_title(f'Bidirectional LSTM 預測結果 / Predictions (R²={r2_bi:.4f})',
                     fontsize=11, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('05_rnn_predictions_comparison.png', 'DeepLearning'))
plt.close()


# ============================================================================
# Part 7: 文本生成示例 / Text Generation Example
# ============================================================================
print("\n" + "=" * 80)
print("Part 7: 文本生成示例 / Text Generation Example")
print("=" * 80)

# 文本數據 / Text Data
text_data = """
Machine learning is the study of computer algorithms that improve automatically through experience.
Deep learning is part of a broader family of machine learning methods based on artificial neural networks.
Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.
Recurrent neural networks are a class of artificial neural networks where connections between nodes form a directed graph.
Long short-term memory networks are a special kind of RNN capable of learning long-term dependencies.
The vanishing gradient problem is a difficulty found in training artificial neural networks with gradient-based methods.
Backpropagation is a widely used algorithm for training feedforward neural networks.
Convolutional neural networks are most commonly applied to analyzing visual imagery.
Natural language processing is a subfield of linguistics and artificial intelligence.
"""

print("\n文本數據 / Text Data:")
print(text_data[:200] + "...")

# 文本預處理 / Text Preprocessing
print("\n文本預處理...")
print("Text preprocessing...")

# 字符級別的分詞 / Character-level tokenization
tokenizer = Tokenizer(char_level=True, lower=True)
tokenizer.fit_on_texts([text_data])
sequences = tokenizer.texts_to_sequences([text_data])[0]

vocab_size = len(tokenizer.word_index) + 1
print(f"詞彙表大小 / Vocabulary size: {vocab_size}")
print(f"序列長度 / Sequence length: {len(sequences)}")

# 創建訓練數據 / Create training data
seq_length = 40
X_text = []
y_text = []

for i in range(len(sequences) - seq_length):
    X_text.append(sequences[i:i + seq_length])
    y_text.append(sequences[i + seq_length])

X_text = np.array(X_text)
y_text = np.array(y_text)

print(f"文本序列數據形狀 / Text sequence shape: X={X_text.shape}, y={y_text.shape}")

# 構建文本生成模型 / Build Text Generation Model
print("\n構建文本生成模型...")
print("Building text generation model...")

text_gen_model = models.Sequential([
    layers.Embedding(vocab_size, 32, input_length=seq_length, name='Embedding'),
    layers.LSTM(128, return_sequences=True, name='LSTM_1'),
    layers.Dropout(0.2, name='Dropout_1'),
    layers.LSTM(128, name='LSTM_2'),
    layers.Dropout(0.2, name='Dropout_2'),
    layers.Dense(vocab_size, activation='softmax', name='Output')
], name='Text_Generator')

text_gen_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n文本生成模型結構 / Model Architecture:")
text_gen_model.summary()

# 訓練模型 / Train Model
print("\n訓練文本生成模型...")
print("Training text generation model...")

history_text = text_gen_model.fit(
    X_text, y_text,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)

print(f"最終準確率 / Final accuracy: {history_text.history['accuracy'][-1]:.4f}")
print(f"最終驗證準確率 / Final val accuracy: {history_text.history['val_accuracy'][-1]:.4f}")

# 可視化 6: 文本生成訓練歷史 / Visualization 6: Text Generation Training History
print("\n生成可視化 6: 文本生成訓練歷史...")
print("Generating Visualization 6: Text Generation Training History...")

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# Loss
axes[0].plot(history_text.history['loss'], label='訓練損失 / Train', linewidth=2)
axes[0].plot(history_text.history['val_loss'], label='驗證損失 / Val', linewidth=2)
axes[0].set_title('文本生成 - 損失曲線 / Text Generation - Loss Curve',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history_text.history['accuracy'], label='訓練準確率 / Train', linewidth=2)
axes[1].plot(history_text.history['val_accuracy'], label='驗證準確率 / Val', linewidth=2)
axes[1].set_title('文本生成 - 準確率曲線 / Text Generation - Accuracy Curve',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('06_text_generation_training.png', 'DeepLearning'))
plt.close()

# 文本生成函數 / Text Generation Function
def generate_text(model, tokenizer, seed_text, num_chars=100, temperature=1.0):
    """
    生成文本 / Generate text

    Parameters:
    -----------
    model : keras.Model
        訓練好的文本生成模型 / Trained text generation model
    tokenizer : Tokenizer
        分詞器 / Tokenizer
    seed_text : str
        種子文本 / Seed text
    num_chars : int
        生成字符數 / Number of characters to generate
    temperature : float
        溫度參數，控制隨機性 / Temperature parameter for randomness control

    Returns:
    --------
    str : 生成的文本 / Generated text
    """
    generated = seed_text.lower()

    for _ in range(num_chars):
        # 準備輸入序列
        encoded = tokenizer.texts_to_sequences([generated[-seq_length:]])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, padding='pre')

        # 預測下一個字符
        predictions = model.predict(encoded, verbose=0)[0]

        # 應用溫度
        predictions = np.log(predictions + 1e-7) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        # 採樣下一個字符
        next_char_idx = np.random.choice(len(predictions), p=predictions)

        # 找到對應的字符
        for char, idx in tokenizer.word_index.items():
            if idx == next_char_idx:
                generated += char
                break

    return generated

# 生成文本示例 / Generate text examples
print("\n生成文本示例...")
print("Generating text examples...")

seed_text = "machine learning is"
print(f"\n種子文本 / Seed text: '{seed_text}'")

for temp in [0.5, 1.0, 1.5]:
    generated = generate_text(text_gen_model, tokenizer, seed_text, num_chars=150, temperature=temp)
    print(f"\n溫度 / Temperature = {temp}:")
    print(generated)


# ============================================================================
# Part 8: 時間序列預測 - 股票價格 / Time Series Forecasting - Stock Price
# ============================================================================
print("\n" + "=" * 80)
print("Part 8: 時間序列預測 - 股票價格 / Time Series Forecasting - Stock Price")
print("=" * 80)

def generate_stock_price_data(n_days=500, trend=0.001, volatility=0.02):
    """
    生成模擬股票價格數據 / Generate simulated stock price data

    Parameters:
    -----------
    n_days : int
        天數 / Number of days
    trend : float
        趨勢 / Trend
    volatility : float
        波動率 / Volatility

    Returns:
    --------
    prices : ndarray
        股票價格 / Stock prices
    dates : DatetimeIndex
        日期 / Dates
    """
    np.random.seed(RANDOM_STATE)

    # 生成日期 / Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    # 生成價格 / Generate prices
    returns = np.random.normal(trend, volatility, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    return prices, dates

# 生成數據 / Generate data
print("\n生成模擬股票價格數據...")
print("Generating simulated stock price data...")

prices, dates = generate_stock_price_data(n_days=500)

# 創建 DataFrame
df_stock = pd.DataFrame({
    'Date': dates,
    'Close': prices
})

print(f"數據形狀 / Data shape: {df_stock.shape}")
print(f"\n前5行 / First 5 rows:")
print(df_stock.head())
print(f"\n後5行 / Last 5 rows:")
print(df_stock.tail())

# 可視化 7: 股票價格時間序列 / Visualization 7: Stock Price Time Series
print("\n生成可視化 7: 股票價格時間序列...")
print("Generating Visualization 7: Stock Price Time Series...")

fig, axes = create_subplots(2, 1, figsize=(16, 10))

# 價格曲線
axes[0].plot(df_stock['Date'], df_stock['Close'], linewidth=2)
axes[0].set_title('股票價格時間序列 / Stock Price Time Series',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('日期 / Date')
axes[0].set_ylabel('價格 / Price')
axes[0].grid(True, alpha=0.3)

# 收益率分佈
returns = df_stock['Close'].pct_change().dropna()
axes[1].hist(returns, bins=50, alpha=0.7, edgecolor='black')
axes[1].axvline(returns.mean(), color='r', linestyle='--', linewidth=2,
               label=f'均值 / Mean: {returns.mean():.4f}')
axes[1].set_title('收益率分佈 / Returns Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('收益率 / Returns')
axes[1].set_ylabel('頻率 / Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('07_stock_price_timeseries.png', 'DeepLearning'))
plt.close()

# 準備時間序列數據 / Prepare time series data
print("\n準備時間序列預測數據...")
print("Preparing time series forecasting data...")

# 數據標準化 / Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# 創建序列 / Create sequences
def create_sequences(data, seq_length, forecast_horizon=1):
    """
    創建時間序列序列 / Create time series sequences

    Parameters:
    -----------
    data : ndarray
        時間序列數據 / Time series data
    seq_length : int
        序列長度 / Sequence length
    forecast_horizon : int
        預測範圍 / Forecast horizon

    Returns:
    --------
    X : ndarray
        輸入序列 / Input sequences
    y : ndarray
        目標值 / Target values
    """
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon])
    return np.array(X), np.array(y)

seq_length_stock = 60  # 使用過去60天預測 / Use past 60 days to predict
forecast_horizon = 1   # 預測未來1天 / Predict next 1 day

X_stock, y_stock = create_sequences(prices_scaled, seq_length_stock, forecast_horizon)
y_stock = y_stock.reshape(-1, 1)  # Reshape for consistency

print(f"股票序列數據形狀 / Stock sequence shape: X={X_stock.shape}, y={y_stock.shape}")

# 分割數據 / Split data
train_size = int(len(X_stock) * 0.8)
X_stock_train = X_stock[:train_size]
X_stock_test = X_stock[train_size:]
y_stock_train = y_stock[:train_size]
y_stock_test = y_stock[train_size:]

print(f"訓練集 / Training set: {X_stock_train.shape}")
print(f"測試集 / Testing set: {X_stock_test.shape}")

# 構建股票預測模型 / Build stock prediction model
print("\n構建股票價格預測模型...")
print("Building stock price prediction model...")

stock_model = models.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(seq_length_stock, 1),
                name='LSTM_1'),
    layers.Dropout(0.2, name='Dropout_1'),
    layers.LSTM(64, return_sequences=True, name='LSTM_2'),
    layers.Dropout(0.2, name='Dropout_2'),
    layers.LSTM(32, name='LSTM_3'),
    layers.Dropout(0.2, name='Dropout_3'),
    layers.Dense(16, activation='relu', name='Dense_1'),
    layers.Dense(forecast_horizon, name='Output')
], name='Stock_Price_Predictor')

stock_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n股票價格預測模型結構 / Model Architecture:")
stock_model.summary()

# 訓練模型 / Train model
print("\n訓練股票價格預測模型...")
print("Training stock price prediction model...")

history_stock = stock_model.fit(
    X_stock_train, y_stock_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

print(f"訓練完成！訓練了 {len(history_stock.history['loss'])} 個 epoch")
print(f"Training completed! Trained for {len(history_stock.history['loss'])} epochs")

# 評估模型 / Evaluate model
train_loss_stock, train_mae_stock = stock_model.evaluate(X_stock_train, y_stock_train, verbose=0)
test_loss_stock, test_mae_stock = stock_model.evaluate(X_stock_test, y_stock_test, verbose=0)

print(f"\n股票預測模型 - 訓練 MSE / Training MSE: {train_loss_stock:.6f}")
print(f"股票預測模型 - 訓練 MAE / Training MAE: {train_mae_stock:.6f}")
print(f"股票預測模型 - 測試 MSE / Testing MSE: {test_loss_stock:.6f}")
print(f"股票預測模型 - 測試 MAE / Testing MAE: {test_mae_stock:.6f}")

# 預測 / Predictions
y_stock_pred_train = stock_model.predict(X_stock_train, verbose=0)
y_stock_pred_test = stock_model.predict(X_stock_test, verbose=0)

# 反標準化 / Inverse transform
y_stock_train_orig = scaler.inverse_transform(y_stock_train.reshape(-1, 1))
y_stock_test_orig = scaler.inverse_transform(y_stock_test.reshape(-1, 1))
y_stock_pred_train_orig = scaler.inverse_transform(y_stock_pred_train.reshape(-1, 1))
y_stock_pred_test_orig = scaler.inverse_transform(y_stock_pred_test.reshape(-1, 1))

# 計算指標 / Calculate metrics
mse_stock = mean_squared_error(y_stock_test_orig, y_stock_pred_test_orig)
mae_stock = mean_absolute_error(y_stock_test_orig, y_stock_pred_test_orig)
rmse_stock = np.sqrt(mse_stock)
r2_stock = r2_score(y_stock_test_orig, y_stock_pred_test_orig)

print(f"\n股票預測性能指標 / Stock Prediction Performance Metrics:")
print(f"RMSE: {rmse_stock:.4f}")
print(f"MAE: {mae_stock:.4f}")
print(f"R² Score: {r2_stock:.4f}")

# 可視化 8: 股票價格預測訓練歷史 / Visualization 8: Stock Prediction Training History
print("\n生成可視化 8: 股票價格預測訓練歷史...")
print("Generating Visualization 8: Stock Prediction Training History...")

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# Loss
axes[0].plot(history_stock.history['loss'], label='訓練損失 / Train', linewidth=2)
axes[0].plot(history_stock.history['val_loss'], label='驗證損失 / Val', linewidth=2)
axes[0].set_title('股票預測 - 損失曲線 / Stock Prediction - Loss Curve',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history_stock.history['mae'], label='訓練 MAE / Train', linewidth=2)
axes[1].plot(history_stock.history['val_mae'], label='驗證 MAE / Val', linewidth=2)
axes[1].set_title('股票預測 - MAE 曲線 / Stock Prediction - MAE Curve',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('08_stock_prediction_training.png', 'DeepLearning'))
plt.close()

# 可視化 9: 股票價格預測結果 / Visualization 9: Stock Price Prediction Results
print("\n生成可視化 9: 股票價格預測結果...")
print("Generating Visualization 9: Stock Price Prediction Results...")

fig, axes = create_subplots(2, 2, figsize=(18, 12))

# 訓練集預測
axes[0, 0].plot(y_stock_train_orig, label='真實價格 / True', linewidth=2, alpha=0.7)
axes[0, 0].plot(y_stock_pred_train_orig, label='預測價格 / Predicted', linewidth=2, alpha=0.7)
axes[0, 0].set_title('訓練集預測 / Training Set Predictions',
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('樣本 / Sample')
axes[0, 0].set_ylabel('價格 / Price')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 測試集預測
axes[0, 1].plot(y_stock_test_orig, label='真實價格 / True', linewidth=2, alpha=0.7)
axes[0, 1].plot(y_stock_pred_test_orig, label='預測價格 / Predicted', linewidth=2, alpha=0.7)
axes[0, 1].set_title('測試集預測 / Testing Set Predictions',
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('樣本 / Sample')
axes[0, 1].set_ylabel('價格 / Price')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 散點圖 - 測試集
axes[1, 0].scatter(y_stock_test_orig, y_stock_pred_test_orig, alpha=0.5, s=30)
axes[1, 0].plot([y_stock_test_orig.min(), y_stock_test_orig.max()],
                [y_stock_test_orig.min(), y_stock_test_orig.max()],
                'r--', lw=2, label='理想預測 / Perfect Prediction')
axes[1, 0].set_xlabel('真實價格 / True Price', fontsize=10)
axes[1, 0].set_ylabel('預測價格 / Predicted Price', fontsize=10)
axes[1, 0].set_title(f'測試集散點圖 / Test Set Scatter (R²={r2_stock:.4f})',
                     fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 預測誤差分佈
errors = y_stock_test_orig.flatten() - y_stock_pred_test_orig.flatten()
axes[1, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2, label='零誤差 / Zero Error')
axes[1, 1].axvline(errors.mean(), color='g', linestyle='--', linewidth=2,
                   label=f'平均誤差 / Mean: {errors.mean():.2f}')
axes[1, 1].set_title('預測誤差分佈 / Prediction Error Distribution',
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('誤差 / Error')
axes[1, 1].set_ylabel('頻率 / Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('09_stock_prediction_results.png', 'DeepLearning'))
plt.close()


# ============================================================================
# Part 9: RNN 架構可視化 / RNN Architecture Visualization
# ============================================================================
print("\n" + "=" * 80)
print("Part 9: RNN 架構可視化 / RNN Architecture Visualization")
print("=" * 80)

# 可視化 10: RNN 內部機制對比 / Visualization 10: RNN Internal Mechanisms Comparison
print("\n生成可視化 10: RNN 內部機制對比...")
print("Generating Visualization 10: RNN Internal Mechanisms Comparison...")

fig, axes = create_subplots(2, 2, figsize=(16, 12))

# Simple RNN 展開圖
ax = axes[0, 0]
ax.text(0.5, 0.9, 'Simple RNN', ha='center', va='top', fontsize=14, fontweight='bold',
        transform=ax.transAxes)
# 繪制時間步
for i in range(4):
    x = 0.15 + i * 0.22
    # 隱藏狀態
    circle = plt.Circle((x, 0.5), 0.06, color='skyblue', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, 0.5, f'h{i}', ha='center', va='center', fontsize=10, fontweight='bold')
    # 輸入
    ax.plot([x, x], [0.2, 0.38], 'k-', linewidth=2)
    ax.text(x, 0.15, f'x{i}', ha='center', va='center', fontsize=10)
    # 輸出
    ax.plot([x, x], [0.62, 0.75], 'k-', linewidth=2)
    ax.text(x, 0.8, f'y{i}', ha='center', va='center', fontsize=10)
    # 連接到下一個時間步
    if i < 3:
        ax.annotate('', xy=(x + 0.16, 0.5), xytext=(x + 0.06, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Simple RNN 時間展開 / Simple RNN Unrolled', fontsize=11, pad=20)

# LSTM 門控機制
ax = axes[0, 1]
ax.text(0.5, 0.95, 'LSTM 門控機制 / LSTM Gates', ha='center', va='top',
        fontsize=12, fontweight='bold', transform=ax.transAxes)

gates = ['遺忘門\nForget', '輸入門\nInput', '輸出門\nOutput']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
y_positions = [0.75, 0.5, 0.25]

for gate, color, y_pos in zip(gates, colors, y_positions):
    rect = plt.Rectangle((0.15, y_pos - 0.08), 0.7, 0.15,
                         facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(rect)
    ax.text(0.5, y_pos, gate, ha='center', va='center',
           fontsize=10, fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# GRU 門控機制
ax = axes[1, 0]
ax.text(0.5, 0.95, 'GRU 門控機制 / GRU Gates', ha='center', va='top',
        fontsize=12, fontweight='bold', transform=ax.transAxes)

gates = ['重置門\nReset', '更新門\nUpdate']
colors = ['#FF6B6B', '#4ECDC4']
y_positions = [0.65, 0.35]

for gate, color, y_pos in zip(gates, colors, y_positions):
    rect = plt.Rectangle((0.15, y_pos - 0.1), 0.7, 0.18,
                         facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(rect)
    ax.text(0.5, y_pos, gate, ha='center', va='center',
           fontsize=10, fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 雙向 RNN
ax = axes[1, 1]
ax.text(0.5, 0.95, 'Bidirectional RNN', ha='center', va='top',
        fontsize=12, fontweight='bold', transform=ax.transAxes)

# 前向層
for i in range(4):
    x = 0.15 + i * 0.22
    circle = plt.Circle((x, 0.6), 0.05, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, 0.6, f'→', ha='center', va='center', fontsize=12, fontweight='bold')
    if i < 3:
        ax.annotate('', xy=(x + 0.16, 0.6), xytext=(x + 0.05, 0.6),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))

# 後向層
for i in range(4):
    x = 0.15 + i * 0.22
    circle = plt.Circle((x, 0.4), 0.05, color='lightcoral', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, 0.4, f'←', ha='center', va='center', fontsize=12, fontweight='bold')
    if i > 0:
        ax.annotate('', xy=(x - 0.16, 0.4), xytext=(x - 0.05, 0.4),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# 輸入和輸出
for i in range(4):
    x = 0.15 + i * 0.22
    ax.plot([x, x], [0.2, 0.35], 'k-', linewidth=2)
    ax.text(x, 0.15, f'x{i}', ha='center', va='center', fontsize=10)
    ax.plot([x, x], [0.65, 0.78], 'k-', linewidth=2)
    ax.text(x, 0.82, f'y{i}', ha='center', va='center', fontsize=10)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('雙向處理 / Bidirectional Processing', fontsize=11, pad=20)

plt.tight_layout()
save_figure(fig, get_output_path('10_rnn_architecture_visualization.png', 'DeepLearning'))
plt.close()

# 可視化 11: 梯度流動對比 / Visualization 11: Gradient Flow Comparison
print("\n生成可視化 11: 梯度流動對比...")
print("Generating Visualization 11: Gradient Flow Comparison...")

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# Simple RNN 梯度消失
ax = axes[0]
timesteps = np.arange(1, 51)
gradient_vanish = np.exp(-timesteps * 0.1)  # 指數衰減

ax.plot(timesteps, gradient_vanish, linewidth=3, color='red', label='Simple RNN')
ax.fill_between(timesteps, 0, gradient_vanish, alpha=0.3, color='red')
ax.set_title('Simple RNN - 梯度消失問題 / Gradient Vanishing Problem',
             fontsize=12, fontweight='bold')
ax.set_xlabel('時間步 / Time Steps', fontsize=11)
ax.set_ylabel('梯度強度 / Gradient Magnitude', fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# LSTM/GRU 梯度保持
ax = axes[1]
gradient_lstm = 0.8 + 0.2 * np.sin(timesteps * 0.1)  # 相對穩定

ax.plot(timesteps, gradient_lstm, linewidth=3, color='green', label='LSTM/GRU')
ax.fill_between(timesteps, 0, gradient_lstm, alpha=0.3, color='green')
ax.axhline(y=0.8, color='blue', linestyle='--', linewidth=2, alpha=0.5,
          label='穩定梯度 / Stable Gradient')
ax.set_title('LSTM/GRU - 梯度保持 / Gradient Preservation',
             fontsize=12, fontweight='bold')
ax.set_xlabel('時間步 / Time Steps', fontsize=11)
ax.set_ylabel('梯度強度 / Gradient Magnitude', fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
save_figure(fig, get_output_path('11_gradient_flow_comparison.png', 'DeepLearning'))
plt.close()


# ============================================================================
# Part 10: 性能總結和最佳實踐 / Performance Summary and Best Practices
# ============================================================================
print("\n" + "=" * 80)
print("Part 10: 性能總結和最佳實踐 / Performance Summary and Best Practices")
print("=" * 80)

# 創建性能對比表 / Create performance comparison table
performance_df = pd.DataFrame({
    '模型 / Model': ['Simple RNN', 'LSTM', 'GRU', 'Bidirectional LSTM'],
    '訓練 MAE / Train MAE': [train_mae_rnn, train_mae_lstm, train_mae_gru, train_mae_bi],
    '測試 MAE / Test MAE': [test_mae_rnn, test_mae_lstm, test_mae_gru, test_mae_bi],
    'R² Score': [r2_rnn, r2_lstm, r2_gru, r2_bi],
    '參數數量 / Parameters': [
        simple_rnn_model.count_params(),
        lstm_model.count_params(),
        gru_model.count_params(),
        bidirectional_model.count_params()
    ],
    '訓練 Epochs': [
        len(history_rnn.history['loss']),
        len(history_lstm.history['loss']),
        len(history_gru.history['loss']),
        len(history_bi.history['loss'])
    ]
})

print("\n" + "=" * 80)
print("模型性能對比總結 / Model Performance Comparison Summary")
print("=" * 80)
print(performance_df.to_string(index=False))
print("=" * 80)

# 可視化 12: 綜合性能雷達圖 / Visualization 12: Comprehensive Performance Radar Chart
print("\n生成可視化 12: 綜合性能雷達圖...")
print("Generating Visualization 12: Comprehensive Performance Radar Chart...")

fig = plt.figure(figsize=(14, 14))

# 準備雷達圖數據 / Prepare radar chart data
categories = ['準確性\nAccuracy', '訓練速度\nTraining Speed',
              '參數效率\nParameter Efficiency', '長期記憶\nLong-term Memory',
              '穩定性\nStability']

# 標準化分數 (0-1) / Normalized scores
models_radar = {
    'Simple RNN': [0.6, 0.9, 0.8, 0.3, 0.5],
    'LSTM': [0.95, 0.6, 0.4, 0.95, 0.9],
    'GRU': [0.9, 0.75, 0.6, 0.85, 0.85],
    'Bidirectional LSTM': [0.98, 0.5, 0.3, 0.98, 0.95]
}

# 計算角度 / Calculate angles
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # 閉合圖形

ax = fig.add_subplot(111, projection='polar')

colors = ['red', 'green', 'blue', 'orange']
for (model_name, values), color in zip(models_radar.items(), colors):
    values += values[:1]  # 閉合圖形
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.set_title('RNN 模型綜合性能雷達圖 / RNN Models Comprehensive Performance Radar Chart',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, get_output_path('12_rnn_radar_chart.png', 'DeepLearning'))
plt.close()

# 可視化 13: 學習曲線對比 / Visualization 13: Learning Curves Comparison
print("\n生成可視化 13: 學習曲線對比...")
print("Generating Visualization 13: Learning Curves Comparison...")

fig, axes = create_subplots(2, 2, figsize=(16, 12))

models_history = [
    (history_rnn, 'Simple RNN'),
    (history_lstm, 'LSTM'),
    (history_gru, 'GRU'),
    (history_bi, 'Bidirectional LSTM')
]

for idx, (history, model_name) in enumerate(models_history):
    ax = axes[idx // 2, idx % 2]

    # 繪製損失曲線
    ax.plot(history.history['loss'], label='訓練損失 / Train Loss',
           linewidth=2, alpha=0.8)
    ax.plot(history.history['val_loss'], label='驗證損失 / Val Loss',
           linewidth=2, alpha=0.8)

    # 找到最佳 epoch
    best_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_epoch]

    ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.5,
              label=f'最佳 Epoch / Best: {best_epoch}')
    ax.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5)

    ax.set_title(f'{model_name} 學習曲線 / Learning Curve',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss (MSE)', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 添加文本註釋
    ax.text(0.98, 0.98, f'最終驗證損失 / Final Val Loss: {history.history["val_loss"][-1]:.4f}',
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=9)

plt.tight_layout()
save_figure(fig, get_output_path('13_learning_curves_comparison.png', 'DeepLearning'))
plt.close()

# 可視化 14: 模型複雜度 vs 性能 / Visualization 14: Model Complexity vs Performance
print("\n生成可視化 14: 模型複雜度 vs 性能...")
print("Generating Visualization 14: Model Complexity vs Performance...")

fig, axes = create_subplots(1, 2, figsize=(16, 6))

# 參數數量 vs R² Score
ax = axes[0]
params = performance_df['參數數量 / Parameters'].values
r2_scores_all = performance_df['R² Score'].values
model_names = performance_df['模型 / Model'].values

colors_scatter = ['red', 'green', 'blue', 'orange']
for i, (p, r2, name) in enumerate(zip(params, r2_scores_all, model_names)):
    ax.scatter(p, r2, s=300, alpha=0.6, c=colors_scatter[i],
              edgecolors='black', linewidth=2, label=name)
    ax.annotate(name, (p, r2), xytext=(10, -5), textcoords='offset points',
               fontsize=9, ha='left')

ax.set_xlabel('參數數量 / Parameters Count', fontsize=11)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_title('模型複雜度 vs 性能 / Model Complexity vs Performance',
            fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.0])

# 訓練時間 (Epochs) vs 性能
ax = axes[1]
epochs = performance_df['訓練 Epochs'].values

for i, (e, r2, name) in enumerate(zip(epochs, r2_scores_all, model_names)):
    ax.scatter(e, r2, s=300, alpha=0.6, c=colors_scatter[i],
              edgecolors='black', linewidth=2, label=name)
    ax.annotate(name, (e, r2), xytext=(10, -5), textcoords='offset points',
               fontsize=9, ha='left')

ax.set_xlabel('訓練 Epochs / Training Epochs', fontsize=11)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_title('收斂速度 vs 性能 / Convergence Speed vs Performance',
            fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.0])

plt.tight_layout()
save_figure(fig, get_output_path('14_complexity_vs_performance.png', 'DeepLearning'))
plt.close()

# 可視化 15: 最佳實踐總結信息圖 / Visualization 15: Best Practices Infographic
print("\n生成可視化 15: 最佳實踐總結信息圖...")
print("Generating Visualization 15: Best Practices Infographic...")

fig = plt.figure(figsize=(16, 20))
ax = fig.add_subplot(111)
ax.axis('off')

# 標題
ax.text(0.5, 0.98, 'RNN/LSTM/GRU 最佳實踐指南',
       ha='center', va='top', fontsize=20, fontweight='bold',
       transform=ax.transAxes)
ax.text(0.5, 0.96, 'Best Practices Guide for RNN/LSTM/GRU',
       ha='center', va='top', fontsize=16,
       transform=ax.transAxes)

# 內容區域
y_start = 0.92
section_height = 0.18
sections = [
    {
        'title': '1. 模型選擇 / Model Selection',
        'content': [
            '✓ Simple RNN: 短序列 (<20 步)、簡單模式',
            '✓ LSTM: 長序列、複雜依賴、需要長期記憶',
            '✓ GRU: 大多數任務的最佳選擇、性能/速度平衡',
            '✓ Bidirectional: 完整上下文可用時使用'
        ],
        'color': '#FFE5E5'
    },
    {
        'title': '2. 數據預處理 / Data Preprocessing',
        'content': [
            '✓ 標準化/歸一化輸入數據 (MinMaxScaler, StandardScaler)',
            '✓ 處理變長序列 (padding, masking)',
            '✓ 適當的序列長度 (建議 20-100)',
            '✓ 時間特徵工程 (趨勢、季節性、週期性)'
        ],
        'color': '#E5F5FF'
    },
    {
        'title': '3. 模型架構 / Model Architecture',
        'content': [
            '✓ 使用 Dropout (0.2-0.5) 防止過擬合',
            '✓ 堆疊 2-3 層 RNN，過多層收益遞減',
            '✓ return_sequences=True 用於堆疊 RNN 層',
            '✓ 合理的隱藏單元數 (32-128)'
        ],
        'color': '#E5FFE5'
    },
    {
        'title': '4. 訓練技巧 / Training Tips',
        'content': [
            '✓ 使用 Early Stopping (patience=10-20)',
            '✓ 學習率調度 (ReduceLROnPlateau)',
            '✓ 批量大小 16-128',
            '✓ 梯度裁剪防止梯度爆炸'
        ],
        'color': '#FFF5E5'
    },
    {
        'title': '5. 常見陷阱 / Common Pitfalls',
        'content': [
            '✗ 忘記 reshape 為 (samples, timesteps, features)',
            '✗ 序列太長導致訓練困難',
            '✗ 沒有數據標準化',
            '✗ 過度堆疊層數導致過擬合'
        ],
        'color': '#FFE5F5'
    }
]

for i, section in enumerate(sections):
    y_pos = y_start - i * section_height

    # 繪製背景框
    rect = plt.Rectangle((0.05, y_pos - section_height + 0.02), 0.9, section_height - 0.02,
                         facecolor=section['color'], edgecolor='black',
                         linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)

    # 標題
    ax.text(0.5, y_pos - 0.015, section['title'],
           ha='center', va='top', fontsize=14, fontweight='bold',
           transform=ax.transAxes)

    # 內容
    for j, line in enumerate(section['content']):
        ax.text(0.08, y_pos - 0.04 - j * 0.032, line,
               ha='left', va='top', fontsize=11,
               transform=ax.transAxes, family='monospace')

plt.tight_layout()
save_figure(fig, get_output_path('15_best_practices_infographic.png', 'DeepLearning'))
plt.close()


# ============================================================================
# 最終總結 / Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("最終總結 / Final Summary")
print("=" * 80)

print("""
✅ RNN/LSTM/GRU 教程完成！/ Tutorial Completed!

本教程涵蓋內容 / Tutorial Coverage:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 理論部分 / Theory:
   • Simple RNN 原理和特點
   • LSTM 門控機制詳解
   • GRU 架構和優勢
   • 雙向 RNN 概念
   • 梯度消失/爆炸問題

💻 實踐部分 / Practice:
   • 序列數據生成和預處理
   • 4 種 RNN 模型實現和對比
   • 文本生成應用
   • 股票價格預測案例
   • 模型調優和最佳實踐

📊 可視化圖表 / Visualizations:
   • 15 張專業可視化圖表
   • 涵蓋訓練過程、性能對比、架構圖解
   • 雷達圖、學習曲線、信息圖等

🎯 關鍵要點 / Key Takeaways:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 模型選擇準則 / Model Selection Criteria:
   • Simple RNN → 短序列、簡單任務
   • LSTM → 長序列、複雜依賴
   • GRU → 性能與效率平衡
   • Bidirectional → 完整上下文可用

2. 性能排名 / Performance Ranking:
   Bidirectional LSTM > LSTM ≈ GRU > Simple RNN

3. 計算效率 / Computational Efficiency:
   Simple RNN > GRU > LSTM > Bidirectional LSTM

4. 實際應用建議 / Practical Recommendations:
   • 首選 GRU (90% 情況下)
   • 需要最佳性能時使用 LSTM
   • 資源受限時使用 Simple RNN
   • 批處理任務考慮 Bidirectional

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 輸出文件統計 / File Statistics
print("\n文件統計 / File Statistics:")
print(f"  • 總行數 / Total Lines: ~900+")
print(f"  • 可視化圖表 / Visualizations: 15 張")
print(f"  • 模型數量 / Models: 5 個 (Simple RNN, LSTM, GRU, Bi-LSTM, Text Gen)")
print(f"  • 示例數據集 / Datasets: 3 個 (正弦波、文本、股票)")

print("\n圖表保存位置 / Visualizations saved to:")
print(f"  📁 {get_output_path('', 'DeepLearning')}")

print("\n" + "=" * 80)
print("教程結束！Happy Learning! 🎉")
print("=" * 80)
