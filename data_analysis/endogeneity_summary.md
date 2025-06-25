# 内生性检验功能说明

## 添加的功能概述

为 `data_analysis/main.py` 添加了 `perform_endogeneity_test()` 函数，用于检验极性mean和std的内生性问题。

## 主要检验方法

### 1. 工具变量设计

创建了以下工具变量来识别内生性：

- **iv_polarity_mean_yearly**: 同年度其他电影的平均极性（排除自身）
- **iv_polarity_std_type**: 同类型其他电影的平均极性标准差
- **iv_comment_count_sq**: 评论数量的平方（非线性工具变量）
- **iv_year_drama**: 年份与戏剧类型的交互项
- **iv_year_comedy**: 年份与喜剧类型的交互项

### 2. 内生性检验步骤

#### 2.1 Hausman检验
- 比较OLS和2SLS回归结果
- 检验内生变量系数的差异
- 如果差异显著，说明存在内生性

#### 2.2 第一阶段F检验（弱工具变量检验）
- 对每个内生变量进行第一阶段回归
- 检验工具变量的联合显著性
- F统计量 > 10 表示工具变量强度充足

#### 2.3 过度识别检验（Sargan检验）
- 当工具变量数量超过内生变量数量时进行
- 检验工具变量的外生性假设
- 确保工具变量不直接影响因变量

#### 2.4 系统GMM方法
- 使用广义矩方法估计
- 利用更多的矩条件信息
- 提供更稳健的内生性检验

### 3. 检验输出

#### 3.1 统计结果
- 各工具变量与内生变量的相关性
- 工具变量与因变量的直接相关性
- 第一阶段回归的F统计量
- 2SLS与OLS系数的比较

#### 3.2 可视化图表
生成 `data_analysis/figs/endogeneity_test.png`，包含：
- 第一阶段F统计量条形图
- 工具变量-内生变量相关性热力图
- OLS vs 2SLS系数比较图
- 残差诊断图
- 工具变量与残差相关性图
- 因变量分布图

## 实际测试结果

通过模拟数据测试显示：

```
=== 工具变量相关性检验 ===

polarity_mean:
  与 iv_polarity_mean_yearly: 0.1169
  与 iv_polarity_std_type: 0.1307
  与 iv_comment_count_sq: -0.0310
  与 iv_year_drama: 0.0435
  与 iv_year_comedy: -0.0113

polarity_std:
  与 iv_polarity_mean_yearly: 0.0457
  与 iv_polarity_std_type: -0.0953
  与 iv_comment_count_sq: -0.0384
  与 iv_year_drama: 0.0499
  与 iv_year_comedy: 0.0311

工具变量与因变量(log_box_off)的直接相关性:
  iv_polarity_mean_yearly: 0.0178
  iv_polarity_std_type: 0.0442
  iv_comment_count_sq: -0.0221
  iv_year_drama: 0.0314
  iv_year_comedy: -0.0577
```

## 使用方法

在主函数中，内生性检验会在回归分析后自动运行：

```python
# 2. 回归分析
regression_results = perform_regression_analysis(df)

# 2.5. 内生性检验
endogeneity_results = perform_endogeneity_test(df)
```

## 解释建议

### 当检验结果显示内生性存在时：
1. 优先使用2SLS或GMM估计结果
2. 报告内生性检验的统计量
3. 解释工具变量的合理性

### 当检验结果显示无明显内生性时：
1. 可以使用OLS结果
2. 但仍需报告内生性检验过程
3. 增强结果的可信度

## 技术特点

- **稳健性**: 使用多种检验方法交叉验证
- **可视化**: 提供直观的图表展示
- **自动化**: 集成在主分析流程中
- **解释性**: 提供详细的统计指标和建议

## 注意事项

1. 工具变量的有效性依赖于理论假设
2. 样本量需要足够大（建议>100）
3. 需要根据具体研究背景调整工具变量
4. 弱工具变量问题需要特别关注

这样的内生性检验大大增强了研究的严谨性和可信度喵~ ฅ^•ﻌ•^ฅ