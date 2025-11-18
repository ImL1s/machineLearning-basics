"""
模型服務示例 - 客戶端和性能測試 | Model Serving Example - Client and Performance Testing

本教程展示如何調用已部署的模型 API，並進行性能測試。
This tutorial demonstrates how to call the deployed model API and perform performance testing.

內容包括 | Contents:
1. API 客戶端封裝 | API Client Wrapper
2. 基本調用示例 | Basic Usage Examples
3. 批量預測示例 | Batch Prediction Examples
4. 性能測試 | Performance Testing
5. 錯誤處理測試 | Error Handling Testing

作者: MLOps 工程師
日期: 2025-11

使用前提:
請先運行 01_flask_api_deployment.py 啟動 API 服務器
Please run 01_flask_api_deployment.py first to start the API server
"""

import requests
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from typing import List, Dict, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns

# Utils
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import setup_chinese_fonts, save_figure

# 設置中文字體
setup_chinese_fonts()

# ============================================================================
# Part 1: API 客戶端封裝 / API Client Wrapper
# ============================================================================
print("="*80)
print("Part 1: API 客戶端封裝 | API Client Wrapper")
print("="*80)

class IrisAPIClient:
    """
    鳶尾花分類 API 客戶端
    Iris Classification API Client

    提供簡單易用的接口來調用模型 API
    Provides easy-to-use interface for calling the model API
    """

    def __init__(self, base_url: str = 'http://localhost:5000', timeout: int = 30):
        """
        初始化客戶端

        Parameters:
        -----------
        base_url : str
            API 服務器地址
        timeout : int
            請求超時時間（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

        print(f"✓ 客戶端已初始化: {self.base_url}")

    def health_check(self) -> Dict:
        """
        檢查 API 健康狀態
        Check API health status
        """
        try:
            response = self.session.get(
                f'{self.base_url}/health',
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'status': 'error', 'message': str(e)}

    def get_model_info(self) -> Dict:
        """
        獲取模型信息
        Get model information
        """
        try:
            response = self.session.get(
                f'{self.base_url}/model_info',
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}

    def predict(self, features: List[float]) -> Dict:
        """
        單個樣本預測
        Single sample prediction

        Parameters:
        -----------
        features : List[float]
            特徵值列表，例如 [5.1, 3.5, 1.4, 0.2]

        Returns:
        --------
        Dict
            預測結果
        """
        try:
            response = self.session.post(
                f'{self.base_url}/predict',
                json={'features': features},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}

    def predict_batch(self, samples: List[List[float]]) -> Dict:
        """
        批量預測
        Batch prediction

        Parameters:
        -----------
        samples : List[List[float]]
            多個樣本的特徵值

        Returns:
        --------
        Dict
            批量預測結果
        """
        try:
            response = self.session.post(
                f'{self.base_url}/predict_batch',
                json={'samples': samples},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}

    def predict_proba(self, features: List[float]) -> Dict:
        """
        預測概率分布
        Predict probability distribution

        Parameters:
        -----------
        features : List[float]
            特徵值列表

        Returns:
        --------
        Dict
            概率分布結果
        """
        try:
            response = self.session.post(
                f'{self.base_url}/predict_proba',
                json={'features': features},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}

    def close(self):
        """關閉會話"""
        self.session.close()

# ============================================================================
# Part 2: 基本調用示例 / Basic Usage Examples
# ============================================================================
print("\n" + "="*80)
print("Part 2: 基本調用示例 | Basic Usage Examples")
print("="*80)

def test_basic_usage():
    """
    測試基本 API 調用
    Test basic API calls
    """
    # 創建客戶端
    client = IrisAPIClient()

    # 1. 健康檢查
    print("\n[1] 健康檢查:")
    health = client.health_check()
    print(json.dumps(health, indent=2, ensure_ascii=False))

    if health.get('status') != 'healthy':
        print("\n⚠️ API 服務器未就緒，請先啟動 01_flask_api_deployment.py")
        return

    # 2. 獲取模型信息
    print("\n[2] 模型信息:")
    model_info = client.get_model_info()
    if 'error' not in model_info:
        print(f"模型類型: {model_info['model_type']}")
        print(f"模型版本: {model_info['model_version']}")
        print(f"訓練日期: {model_info['trained_date']}")
        print(f"測試集準確率: {model_info['test_score']:.4f}")
        print(f"特徵數量: {model_info['n_features']}")
        print(f"類別數量: {model_info['n_classes']}")

    # 3. 單個預測 - Setosa (花萼短小)
    print("\n[3] 單個預測 - Setosa 樣本:")
    features_setosa = [5.1, 3.5, 1.4, 0.2]
    result = client.predict(features_setosa)
    if 'error' not in result:
        print(f"輸入特徵: {features_setosa}")
        print(f"預測類別: {result['prediction_label']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"概率分布: {json.dumps(result['probabilities'], indent=2, ensure_ascii=False)}")

    # 4. 單個預測 - Versicolor (中等大小)
    print("\n[4] 單個預測 - Versicolor 樣本:")
    features_versicolor = [6.2, 2.9, 4.3, 1.3]
    result = client.predict(features_versicolor)
    if 'error' not in result:
        print(f"輸入特徵: {features_versicolor}")
        print(f"預測類別: {result['prediction_label']}")
        print(f"置信度: {result['confidence']:.4f}")

    # 5. 單個預測 - Virginica (花萼大)
    print("\n[5] 單個預測 - Virginica 樣本:")
    features_virginica = [7.3, 2.9, 6.3, 1.8]
    result = client.predict(features_virginica)
    if 'error' not in result:
        print(f"輸入特徵: {features_virginica}")
        print(f"預測類別: {result['prediction_label']}")
        print(f"置信度: {result['confidence']:.4f}")

    # 6. 概率預測
    print("\n[6] 詳細概率預測:")
    result = client.predict_proba([5.8, 2.7, 5.1, 1.9])
    if 'error' not in result:
        print(f"最可能類別: {result['most_likely']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("\n完整概率分布:")
        for prob_info in result['probabilities']:
            print(f"  {prob_info['class_name']:15s}: {prob_info['percentage']:>7s} (prob={prob_info['probability']:.4f})")

    client.close()

# ============================================================================
# Part 3: 批量預測示例 / Batch Prediction Examples
# ============================================================================
print("\n" + "="*80)
print("Part 3: 批量預測示例 | Batch Prediction Examples")
print("="*80)

def test_batch_prediction():
    """
    測試批量預測功能
    Test batch prediction functionality
    """
    client = IrisAPIClient()

    # 準備測試樣本
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.2, 2.9, 4.3, 1.3],  # Versicolor
        [7.3, 2.9, 6.3, 1.8],  # Virginica
        [5.0, 3.6, 1.4, 0.2],  # Setosa
        [6.7, 3.1, 4.7, 1.5],  # Versicolor
        [7.7, 3.0, 6.1, 2.3],  # Virginica
    ]

    print(f"\n批量預測 {len(test_samples)} 個樣本:")

    # 批量預測
    start_time = time.time()
    result = client.predict_batch(test_samples)
    elapsed_time = time.time() - start_time

    if 'error' not in result:
        print(f"\n預測完成，耗時: {elapsed_time*1000:.2f} ms")
        print(f"總樣本數: {result['count']}")
        print(f"平均每樣本耗時: {elapsed_time/result['count']*1000:.2f} ms")

        # 顯示結果
        print("\n詳細結果:")
        results_df = pd.DataFrame(result['results'])
        print(results_df.to_string(index=False))

        # 統計各類別數量
        print("\n預測統計:")
        prediction_counts = results_df['prediction_label'].value_counts()
        for label, count in prediction_counts.items():
            print(f"  {label}: {count} 個")

    client.close()

# ============================================================================
# Part 4: 性能測試 / Performance Testing
# ============================================================================
print("\n" + "="*80)
print("Part 4: 性能測試 | Performance Testing")
print("="*80)

def performance_test(n_requests: int = 100):
    """
    性能測試
    Performance testing

    Parameters:
    -----------
    n_requests : int
        測試請求數量
    """
    client = IrisAPIClient()

    # 檢查服務器是否就緒
    health = client.health_check()
    if health.get('status') != 'healthy':
        print("⚠️ API 服務器未就緒")
        return

    print(f"\n執行性能測試 (請求數: {n_requests})...")

    # 準備測試數據
    test_features = [5.1, 3.5, 1.4, 0.2]

    # 測試單個預測
    print("\n[1] 單個預測性能測試:")
    latencies = []
    errors = 0

    for i in range(n_requests):
        start_time = time.time()
        result = client.predict(test_features)
        latency = (time.time() - start_time) * 1000  # ms

        if 'error' in result:
            errors += 1
        else:
            latencies.append(latency)

        if (i + 1) % 20 == 0:
            print(f"  進度: {i+1}/{n_requests}")

    # 統計結果
    latencies = np.array(latencies)
    print(f"\n性能統計:")
    print(f"  總請求數: {n_requests}")
    print(f"  成功請求: {len(latencies)}")
    print(f"  失敗請求: {errors}")
    print(f"  平均延遲: {latencies.mean():.2f} ms")
    print(f"  中位數延遲: {np.median(latencies):.2f} ms")
    print(f"  最小延遲: {latencies.min():.2f} ms")
    print(f"  最大延遲: {latencies.max():.2f} ms")
    print(f"  P95 延遲: {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99 延遲: {np.percentile(latencies, 99):.2f} ms")
    print(f"  標準差: {latencies.std():.2f} ms")
    print(f"  吞吐量: {len(latencies) / (latencies.sum() / 1000):.2f} req/s")

    # 測試批量預測性能
    print("\n[2] 批量預測性能測試:")
    batch_sizes = [1, 5, 10, 20, 50, 100]
    batch_results = []

    for batch_size in batch_sizes:
        samples = [test_features] * batch_size

        start_time = time.time()
        result = client.predict_batch(samples)
        elapsed_time = (time.time() - start_time) * 1000  # ms

        if 'error' not in result:
            avg_latency = elapsed_time / batch_size
            throughput = batch_size / (elapsed_time / 1000)

            batch_results.append({
                'batch_size': batch_size,
                'total_time_ms': elapsed_time,
                'avg_latency_ms': avg_latency,
                'throughput': throughput
            })

            print(f"  批量大小 {batch_size:3d}: 總耗時 {elapsed_time:6.2f} ms, "
                  f"平均延遲 {avg_latency:6.2f} ms, 吞吐量 {throughput:6.2f} req/s")

    # 可視化性能結果
    visualize_performance(latencies, batch_results)

    client.close()

def visualize_performance(latencies: np.ndarray, batch_results: List[Dict]):
    """
    可視化性能測試結果
    Visualize performance test results
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 延遲分布直方圖
    ax1 = axes[0, 0]
    ax1.hist(latencies, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(latencies.mean(), color='red', linestyle='--', linewidth=2, label=f'平均值: {latencies.mean():.2f} ms')
    ax1.axvline(np.median(latencies), color='green', linestyle='--', linewidth=2, label=f'中位數: {np.median(latencies):.2f} ms')
    ax1.set_xlabel('延遲 (ms)', fontsize=12)
    ax1.set_ylabel('頻率', fontsize=12)
    ax1.set_title('單個請求延遲分布', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. 延遲箱線圖
    ax2 = axes[0, 1]
    box = ax2.boxplot([latencies], labels=['API 延遲'], patch_artist=True)
    box['boxes'][0].set_facecolor('lightblue')
    ax2.set_ylabel('延遲 (ms)', fontsize=12)
    ax2.set_title('延遲箱線圖', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加統計信息
    stats_text = f'均值: {latencies.mean():.2f} ms\n'
    stats_text += f'中位數: {np.median(latencies):.2f} ms\n'
    stats_text += f'P95: {np.percentile(latencies, 95):.2f} ms\n'
    stats_text += f'P99: {np.percentile(latencies, 99):.2f} ms'
    ax2.text(1.15, latencies.mean(), stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. 批量大小 vs 平均延遲
    if batch_results:
        batch_df = pd.DataFrame(batch_results)

        ax3 = axes[1, 0]
        ax3.plot(batch_df['batch_size'], batch_df['avg_latency_ms'],
                marker='o', linewidth=2, markersize=8, color='steelblue')
        ax3.set_xlabel('批量大小', fontsize=12)
        ax3.set_ylabel('平均延遲 (ms)', fontsize=12)
        ax3.set_title('批量大小 vs 平均延遲', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. 批量大小 vs 吞吐量
        ax4 = axes[1, 1]
        ax4.plot(batch_df['batch_size'], batch_df['throughput'],
                marker='s', linewidth=2, markersize=8, color='green')
        ax4.set_xlabel('批量大小', fontsize=12)
        ax4.set_ylabel('吞吐量 (req/s)', fontsize=12)
        ax4.set_title('批量大小 vs 吞吐量', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure('model_deployment_performance')
    print(f"\n✓ 性能測試圖表已保存")

# ============================================================================
# Part 5: 錯誤處理測試 / Error Handling Testing
# ============================================================================
print("\n" + "="*80)
print("Part 5: 錯誤處理測試 | Error Handling Testing")
print("="*80)

def test_error_handling():
    """
    測試 API 的錯誤處理能力
    Test API error handling capabilities
    """
    client = IrisAPIClient()

    print("\n測試各種錯誤場景:")

    # 1. 特徵數量錯誤
    print("\n[1] 測試特徵數量錯誤:")
    result = client.predict([5.1, 3.5, 1.4])  # 只有3個特徵
    if 'error' in result:
        print(f"  ✓ 正確捕獲錯誤: {result['error']}")

    # 2. 特徵數量過多
    print("\n[2] 測試特徵數量過多:")
    result = client.predict([5.1, 3.5, 1.4, 0.2, 1.0])  # 5個特徵
    if 'error' in result:
        print(f"  ✓ 正確捕獲錯誤: {result['error']}")

    # 3. 空請求
    print("\n[3] 測試空樣本列表:")
    result = client.predict_batch([])
    if 'error' in result:
        print(f"  ✓ 正確捕獲錯誤: {result['error']}")

    # 4. 批量大小超限
    print("\n[4] 測試批量大小超限:")
    large_batch = [[5.1, 3.5, 1.4, 0.2]] * 1001  # 超過1000限制
    result = client.predict_batch(large_batch)
    if 'error' in result:
        print(f"  ✓ 正確捕獲錯誤: {result['error']}")

    # 5. 無效端點
    print("\n[5] 測試無效端點:")
    try:
        response = client.session.get(f'{client.base_url}/invalid_endpoint')
        print(f"  HTTP 狀態碼: {response.status_code}")
        if response.status_code == 404:
            print(f"  ✓ 正確返回 404 錯誤")
    except Exception as e:
        print(f"  錯誤: {e}")

    client.close()

# ============================================================================
# 主程序 / Main Program
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("模型服務客戶端測試程序")
    print("Model Serving Client Test Program")
    print("="*80)

    # 檢查服務器是否運行
    print("\n檢查 API 服務器狀態...")
    client = IrisAPIClient()
    health = client.health_check()
    client.close()

    if health.get('status') != 'healthy':
        print("\n⚠️ API 服務器未運行!")
        print("\n請先在另一個終端運行:")
        print("  python 01_flask_api_deployment.py")
        print("\n然後再運行此腳本。")
        sys.exit(1)

    print("✓ API 服務器運行正常\n")

    # 運行所有測試
    try:
        # 1. 基本使用示例
        test_basic_usage()

        # 2. 批量預測示例
        test_batch_prediction()

        # 3. 性能測試
        performance_test(n_requests=100)

        # 4. 錯誤處理測試
        test_error_handling()

        print("\n" + "="*80)
        print("所有測試完成!")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n測試被用戶中斷")
    except Exception as e:
        print(f"\n測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
