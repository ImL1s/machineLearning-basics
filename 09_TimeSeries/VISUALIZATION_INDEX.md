# 时间序列分析可视化图表索引
# Time Series Analysis Visualization Index

## 总览 (Overview)

本模块共生成 **29 张高质量可视化图表**，涵盖时间序列分析的各个方面。

Total: **29 high-quality visualization charts** covering all aspects of time series analysis.

---

## 📊 图表列表 (Chart List)

### 🎯 01_time_series_basics.py (8 张图表)

| # | 文件名 | 中文标题 | 英文标题 | 描述 |
|---|--------|---------|---------|------|
| 1 | `01_time_series_components.png` | 时间序列组成部分 | Time Series Components | 展示趋势、季节性、周期性、噪声的分解 |
| 2 | `02_acf_pacf_plots.png` | ACF 和 PACF 图 | ACF and PACF Plots | 自相关和偏自相关函数图 |
| 3 | `03_differencing_comparison.png` | 差分前后对比 | Differencing Comparison | 原始数据与各阶差分的对比 |
| 4 | `04_additive_decomposition.png` | 加法模型分解 | Additive Model Decomposition | 使用加法模型分解时间序列 |
| 5 | `05_lag_features_correlation.png` | 滞后特征相关性 | Lag Features Correlation | 不同滞后期与当前值的相关性散点图（6个子图）|
| 6 | `06_rolling_statistics.png` | 滚动统计 | Rolling Statistics | 滚动均值、标准差、最大/最小值 |
| 7 | `07_time_features_distribution.png` | 时间特征分布 | Time Feature Distribution | 月度、星期、季度等时间特征的统计分布（6个子图）|
| 8 | `08_feature_importance.png` | 特征重要性 | Feature Importance | Top 20 特征的相关性排序 |

**关键学习点：**
- ✅ 理解时间序列的组成部分
- ✅ 掌握平稳性检验方法
- ✅ 学习特征工程技巧
- ✅ 识别 ACF/PACF 模式

---

### 📈 02_arima.py (9 张图表)

| # | 文件名 | 中文标题 | 英文标题 | 描述 |
|---|--------|---------|---------|------|
| 9 | `09_co2_data_exploration.png` | CO2 数据探索 | CO2 Data Exploration | 原始数据、分布、年度变化、月度模式（4个子图）|
| 10 | `10_acf_pacf_original.png` | 原始数据 ACF/PACF | Original Data ACF/PACF | 用于确定初始 ARIMA 参数 |
| 11 | `11_acf_pacf_differenced.png` | 差分后 ACF/PACF | Differenced ACF/PACF | 差分后的相关性分析 |
| 12 | `12_aic_bic_comparison.png` | AIC/BIC 参数选择 | AIC/BIC Parameter Selection | Top 10 模型的 AIC 和 BIC 值对比 |
| 13 | `13_residual_diagnostics.png` | 残差诊断 | Residual Diagnostics | 残差时序图、分布、Q-Q 图、ACF（4个子图）|
| 14 | `14_arima_forecast.png` | ARIMA 预测结果 | ARIMA Forecast Results | 训练数据、测试数据、预测值、置信区间 |
| 15 | `15_models_comparison.png` | 多模型对比 | Multiple Models Comparison | ARIMA vs Naive vs SMA 的预测和指标对比 |
| 16 | `16_sarima_vs_arima.png` | SARIMA vs ARIMA | SARIMA vs ARIMA | 季节性模型与非季节性模型的对比 |
| 17 | `17_performance_summary.png` | 性能总结 | Performance Summary | 所有模型的 MAE、RMSE、MAPE 综合对比 |

**关键学习点：**
- ✅ ARIMA 参数选择方法
- ✅ 使用 AIC/BIC 准则
- ✅ 残差诊断技巧
- ✅ SARIMA 处理季节性数据

---

### 🚀 03_forecasting_methods.py (12 张图表)

| # | 文件名 | 中文标题 | 英文标题 | 描述 |
|---|--------|---------|---------|------|
| 18 | `18_baseline_models_comparison.png` | 基线模型对比 | Baseline Models Comparison | Naive, SMA, WMA, SES 的预测结果（4个子图）|
| 19 | `19_exponential_smoothing_comparison.png` | 指数平滑方法对比 | Exponential Smoothing Comparison | SES, DES, TES 三种指数平滑方法 |
| 20 | `20_arima_vs_sarima.png` | ARIMA vs SARIMA | ARIMA vs SARIMA | 统计模型的预测能力对比 |
| 21 | `21_feature_importance_comparison.png` | 特征重要性对比 | Feature Importance Comparison | Random Forest vs XGBoost 的 Top 15 特征 |
| 22 | `22_ml_methods_comparison.png` | 机器学习方法对比 | ML Methods Comparison | Random Forest vs XGBoost 预测结果 |
| 23 | `23_all_models_performance.png` | 所有模型性能 | All Models Performance | MAE、RMSE、MAPE、R² 四个指标的 Top 10（4个子图）|
| 24 | `24_top5_models_forecast.png` | Top 5 模型预测 | Top 5 Models Forecast | 最佳 5 个模型的预测结果叠加对比 |
| 25 | `25_error_distribution.png` | 误差分布 | Error Distribution | 6 个代表性模型的预测误差直方图 |
| 26 | `26_training_time_comparison.png` | 训练时间对比 | Training Time Comparison | 所有模型的训练时间柱状图 |
| 27 | `27_accuracy_vs_speed.png` | 精度 vs 速度 | Accuracy vs Speed | 模型的精度-速度权衡散点图 |
| 28 | `28_sales_forecasting_example.png` | 销售预测案例 | Sales Forecasting Example | 实际应用案例：模拟销售数据预测 |
| 29 | `29_method_selection_guide.png` | 方法选择指南 | Method Selection Guide | 决策树流程图和快速选择建议 |

**关键学习点：**
- ✅ 多种预测方法的对比
- ✅ 如何选择合适的模型
- ✅ 精度与效率的权衡
- ✅ 实际应用案例

---

## 📁 文件组织 (File Organization)

```
09_TimeSeries/
├── 01_time_series_basics.py      # 717 行代码
├── 02_arima.py                    # 825 行代码
├── 03_forecasting_methods.py     # 1164 行代码
├── README.md                      # 详细文档
└── VISUALIZATION_INDEX.md         # 本文件

output/                            # 所有图表保存在这里
├── 01_time_series_components.png
├── 02_acf_pacf_plots.png
├── ...
└── 29_method_selection_guide.png
```

---

## 🎨 图表类型统计 (Chart Type Statistics)

| 图表类型 | 数量 | 百分比 |
|---------|------|--------|
| 时序图 (Line Charts) | 12 | 41.4% |
| 柱状图 (Bar Charts) | 8 | 27.6% |
| 散点图 (Scatter Plots) | 4 | 13.8% |
| 直方图 (Histograms) | 3 | 10.3% |
| 箱线图 (Box Plots) | 1 | 3.4% |
| 文本图 (Text Diagrams) | 1 | 3.4% |
| **总计** | **29** | **100%** |

---

## 🎯 图表用途分类 (Chart Purpose Classification)

### 📌 探索性分析 (Exploratory Analysis) - 9 张
- 数据探索和分布
- 组成部分分解
- 时间特征分布
- 相关性分析

### 📌 模型诊断 (Model Diagnostics) - 6 张
- ACF/PACF 图
- 残差诊断
- 平稳性检验
- 参数选择

### 📌 预测结果 (Forecast Results) - 8 张
- 各种模型的预测
- 置信区间
- 误差分析
- 实际案例

### 📌 模型对比 (Model Comparison) - 6 张
- 性能指标对比
- 训练时间对比
- 精度-速度权衡
- 方法选择指南

---

## 🔍 图表质量特点 (Chart Quality Features)

### ✅ 统一的视觉风格
- 使用专业配色方案（#2E86AB, #A23B72, #F18F01 等）
- 中英文双语标题和标签
- 清晰的图例和网格线
- 适当的透明度和标记大小

### ✅ 信息丰富
- 每张图都包含关键统计信息
- 文本框显示评估指标
- 数值标签标注重要数据点
- 参考线和置信区间

### ✅ 高分辨率
- 默认 DPI = 150
- 适合打印和演示
- 清晰的字体和线条

### ✅ 教学导向
- 每张图都服务于特定学习目标
- 循序渐进的难度设计
- 理论与实践相结合

---

## 📖 使用建议 (Usage Recommendations)

### 🎓 学习路径建议

**初学者：**
1. 从 01-08 号图表开始，理解基础概念
2. 重点关注组成部分分解和特征工程
3. 学习如何读懂 ACF/PACF 图

**中级用户：**
1. 研究 09-17 号图表，掌握 ARIMA 建模
2. 理解参数选择和残差诊断
3. 学习模型评估方法

**高级用户：**
1. 分析 18-29 号图表，对比多种方法
2. 理解精度-效率权衡
3. 根据业务场景选择最优模型

### 📊 演示和报告建议

**技术演示：**
- 使用 13、17、23 号图表展示建模流程
- 使用 27 号图表讨论模型选择

**业务报告：**
- 使用 14、16、24 号图表展示预测结果
- 使用 28 号图表展示实际应用
- 使用 29 号图表指导决策

**学术论文：**
- 使用 4、10、13 号图表展示方法论
- 使用 17、23 号图表展示实验结果

---

## 🛠️ 自定义图表 (Customizing Charts)

### 修改颜色方案
```python
# 在代码中找到颜色定义
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# 替换为你的配色
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
```

### 调整图表大小
```python
# 在 create_subplots 调用中修改
fig, axes = create_subplots(2, 2, figsize=(20, 16))  # 增大尺寸
```

### 修改 DPI
```python
# 在 save_figure 调用中修改
save_figure(fig, get_output_path('chart.png'), dpi=300)  # 更高分辨率
```

---

## 📚 扩展阅读 (Further Reading)

### 推荐可视化资源
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/
- **Seaborn Tutorial**: https://seaborn.pydata.org/tutorial.html
- **Plotly Time Series**: https://plotly.com/python/time-series/

### 时间序列可视化最佳实践
1. 始终包含时间轴标签
2. 使用适当的日期格式
3. 标注重要事件和转折点
4. 提供置信区间和不确定性信息
5. 使用颜色区分不同序列
6. 添加参考线（均值、中位数等）

---

## 📞 技术支持 (Technical Support)

### 常见可视化问题

**Q: 图表中文显示为方块？**
```python
# 确保调用了中文字体设置
from utils import setup_chinese_fonts
setup_chinese_fonts()
```

**Q: 图表太拥挤？**
```python
# 增加图表尺寸或减少数据点
fig, ax = create_subplots(1, 1, figsize=(20, 10))
# 或者
data_sampled = data[::2]  # 每隔一个点取样
```

**Q: 保存失败？**
```python
# 确保输出目录存在
from utils import get_output_path
path = get_output_path('chart.png')
# 这会自动创建必要的目录
```

---

## 📊 图表检查清单 (Chart Checklist)

在生成图表后，检查以下项目：

- [ ] 标题清晰且有意义（中英文）
- [ ] 坐标轴有标签和单位
- [ ] 图例位置合适且易读
- [ ] 颜色对比度足够
- [ ] 网格线不会喧宾夺主
- [ ] 数据点/线条清晰可见
- [ ] 文本大小适中
- [ ] 没有数据重叠或遮挡
- [ ] 保存路径正确
- [ ] 分辨率满足需求

---

**最后更新**: 2025-11-18

**总图表数**: 29 张

**总代码行数**: 2,706 行

**文档完整性**: ✅ 完整
