"""
機器學習基礎 - 工具模塊
Machine Learning Basics - Utility Modules

提供路徑管理、配置管理和繪圖工具等共享功能
Provides path management, configuration, and plotting utilities
"""

from .config import *
from .paths import *
from .plotting import *

__all__ = ['RANDOM_STATE', 'TEST_SIZE', 'FIGURE_SIZE', 'DPI',
           'PROJECT_ROOT', 'DATA_DIR', 'OUTPUT_DIR', 'MODELS_DIR',
           'get_data_path', 'get_output_path', 'get_model_path',
           'setup_chinese_fonts', 'save_figure']
