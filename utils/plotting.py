"""
繪圖工具 / Plotting Utilities

提供統一的繪圖配置和輔助函數
Provides unified plotting configuration and helper functions
"""

import matplotlib.pyplot as plt
from pathlib import Path
from .config import DPI


def setup_chinese_fonts():
    """
    設置中文字體支持
    Setup Chinese font support

    這個函數配置 matplotlib 以正確顯示中文字符
    This function configures matplotlib to properly display Chinese characters

    Example:
        >>> from utils import setup_chinese_fonts
        >>> setup_chinese_fonts()
        >>> plt.title('中文標題')  # 現在可以正常顯示中文
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def save_figure(fig, filepath: Path, dpi: int = DPI, bbox_inches: str = 'tight'):
    """
    安全地保存圖表
    Save figure safely

    Args:
        fig: matplotlib 圖表對象 / matplotlib figure object
        filepath: 保存路徑 / Save path
        dpi: 圖像分辨率 / Image resolution
        bbox_inches: 邊界框設置 / Bounding box setting

    Returns:
        bool: 是否成功保存 / Whether saved successfully

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> from utils import save_figure, get_output_path
        >>> save_figure(fig, get_output_path('plot.png'))
        ✓ 圖表已保存: /path/to/output/plot.png
        True
    """
    try:
        # 確保目錄存在
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)

        # 保存圖表
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"✓ 圖表已保存: {filepath}")
        return True

    except Exception as e:
        print(f"✗ 保存圖表失敗: {e}")
        return False


def create_subplots(nrows: int = 1, ncols: int = 1, figsize: tuple = None,
                   **kwargs):
    """
    創建子圖並設置中文字體
    Create subplots with Chinese font support

    Args:
        nrows: 行數 / Number of rows
        ncols: 列數 / Number of columns
        figsize: 圖表大小 / Figure size
        **kwargs: 其他傳遞給 plt.subplots 的參數 / Other arguments for plt.subplots

    Returns:
        fig, axes: 圖表和軸對象 / Figure and axes objects

    Example:
        >>> from utils import create_subplots
        >>> fig, axes = create_subplots(2, 2, figsize=(12, 10))
        >>> axes[0, 0].plot([1, 2, 3])
    """
    setup_chinese_fonts()

    if figsize is None:
        # 根據子圖數量自動調整大小
        figsize = (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes
