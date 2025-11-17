"""
路徑管理工具 / Path Management Utilities

提供統一的路徑管理，避免硬編碼相對路徑
Provides unified path management, avoiding hardcoded relative paths
"""

from pathlib import Path
import os

# ============================================================================
# 項目根目錄 / Project Root Directory
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# ============================================================================
# 主要目錄 / Main Directories
# ============================================================================
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = PROJECT_ROOT / 'saved_models'

# 確保目錄存在 / Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# 路徑獲取函數 / Path Retrieval Functions
# ============================================================================

def get_data_path(filename: str) -> Path:
    """
    獲取數據文件路徑
    Get data file path

    Args:
        filename: 數據文件名 / Data filename

    Returns:
        Path: 數據文件的完整路徑 / Full path to data file

    Example:
        >>> path = get_data_path('iris.csv')
        >>> print(path)
        /home/user/machineLearning-basics/data/iris.csv
    """
    return DATA_DIR / filename


def get_output_path(filename: str, subfolder: str = None) -> Path:
    """
    獲取輸出文件路徑
    Get output file path

    Args:
        filename: 輸出文件名 / Output filename
        subfolder: 可選的子文件夾 / Optional subfolder

    Returns:
        Path: 輸出文件的完整路徑 / Full path to output file

    Example:
        >>> path = get_output_path('model_results.png', 'classification')
        >>> print(path)
        /home/user/machineLearning-basics/output/classification/model_results.png
    """
    if subfolder:
        output_path = OUTPUT_DIR / subfolder
        output_path.mkdir(exist_ok=True, parents=True)
        return output_path / filename
    return OUTPUT_DIR / filename


def get_model_path(model_name: str, version: str = None) -> Path:
    """
    獲取模型保存路徑
    Get model save path

    Args:
        model_name: 模型名稱 / Model name
        version: 可選的版本號 / Optional version number

    Returns:
        Path: 模型文件的完整路徑 / Full path to model file

    Example:
        >>> path = get_model_path('random_forest', '1.0')
        >>> print(path)
        /home/user/machineLearning-basics/saved_models/random_forest_v1.0.joblib
    """
    if version:
        filename = f"{model_name}_v{version}.joblib"
    else:
        filename = f"{model_name}.joblib"
    return MODELS_DIR / filename


def get_figure_path(figure_name: str, module: str = None) -> Path:
    """
    獲取圖表保存路徑
    Get figure save path

    Args:
        figure_name: 圖表文件名 / Figure filename
        module: 可選的模塊名稱 / Optional module name

    Returns:
        Path: 圖表文件的完整路徑 / Full path to figure file

    Example:
        >>> path = get_figure_path('confusion_matrix.png', '02_SupervisedLearning')
        >>> print(path)
        /home/user/machineLearning-basics/output/02_SupervisedLearning/confusion_matrix.png
    """
    return get_output_path(figure_name, subfolder=module)
