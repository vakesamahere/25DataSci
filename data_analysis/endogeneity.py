# 添加内生性检验完整代码
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels import IV2SLS

def load_and_clean_data():
    """加载和清理数据 - 参考main.py的实现"""
    print("=== 数据加载与预处理 ===")
    
    # 加载数据
    df = pd.read_csv('data/data_main.csv')
    print(f"原始数据形状: {df.shape}")
    
    # 基本信息
    print("\n数据概览:")
    print(df.info())
    print("\n描述性统计:")
    print(df.describe())
    
    # 处理缺失值
    print(f"\n缺失值情况:")
    print(df.isnull().sum())
    
    # 删除评论数过少的记录（小于30条评论）
    LEAST_COMS_COUNT = 30
    print(f"\n删除评论数小于{LEAST_COMS_COUNT}的记录前: {len(df)}")
    df = df[df['comment_count'] >= LEAST_COMS_COUNT]
    print(f"删除后: {len(df)}")
    
    # 对票房进行对数转换
    df['log_box_off'] = np.log(df['box_off'] + 1)  # 加1避免log(0)
    
    return df

df = load_and_clean_data()

print("\n=========== 完整内生性检验 ===========")

# 定义变量
endogenous_vars = ['polarity_mean', 'polarity_std']  # 可能的内生变量
exogenous_vars = ['tp_drama', 'tp_comedy', 'tp_action', 'tp_romance', 'comment_count', 'year']  # 控制变量
instrument_vars = [
    'iv_polarity_mean_yearly',
    'iv_polarity_std_type',
    'iv_comment_count_sq',
    'iv_year_drama',
    'iv_year_comedy'
]

# 筛选实际存在的变量
endogenous_vars = [var for var in endogenous_vars if var in df.columns]
exogenous_vars = [var for var in exogenous_vars if var in df.columns]
instrument_vars = [var for var in instrument_vars if var in df.columns and not df[var].isna().all()]

print(f"内生变量: {endogenous_vars}")
print(f"外生变量: {exogenous_vars}")
print(f"工具变量: {instrument_vars}")

# 确保因变量存在
if 'log_box_off' not in df.columns:
    print("错误: 缺少因变量 log_box_off")
else:
    # 准备回归数据
    all_vars = endogenous_vars + exogenous_vars + instrument_vars + ['log_box_off']
    regression_df = df[all_vars].dropna()
    print(f"有效样本数: {len(regression_df)}")
    
    if len(regression_df) > len(endogenous_vars) + len(exogenous_vars):
        print("\n1. OLS回归")
        y = regression_df['log_box_off']
        X = regression_df[endogenous_vars + exogenous_vars]
        X = sm.add_constant(X)
        
        ols_model = sm.OLS(y, X).fit()
        print(ols_model.summary().tables[1])
        
        print("\n2. 第一阶段回归和弱工具变量检验")
        for endvar in endogenous_vars:
            print(f"\n为 {endvar} 进行第一阶段回归:")
            first_stage_X = regression_df[exogenous_vars + instrument_vars]
            first_stage_X = sm.add_constant(first_stage_X)
            first_stage_y = regression_df[endvar]
            
            first_stage_model = sm.OLS(first_stage_y, first_stage_X).fit()
            print(f"R²: {first_stage_model.rsquared:.4f}")
            print(f"调整R²: {first_stage_model.rsquared_adj:.4f}")
            print(f"第一阶段F统计量: {first_stage_model.fvalue:.4f}")
            
            # 弱工具变量检验
            if first_stage_model.fvalue > 10:
                print("结论: 工具变量强度充足 (F > 10)")
            else:
                print("结论: 可能存在弱工具变量问题 (F < 10)")
            
            # 打印工具变量的系数和显著性
            iv_params = first_stage_model.params.filter(regex='^iv_')
            iv_pvalues = first_stage_model.pvalues.filter(regex='^iv_')
            
            print("工具变量系数和显著性:")
            for iv_name in iv_params.index:
                coef = iv_params[iv_name]
                pval = iv_pvalues[iv_name]
                sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
                print(f"  {iv_name}: {coef:.4f} {sig} (p={pval:.4f})")
        
        print("\n3. 2SLS回归")
        dependent = regression_df[['log_box_off']]
        exog = regression_df[exogenous_vars]
        endog = regression_df[endogenous_vars]
        instruments = regression_df[instrument_vars]
        
        try:
            iv_model = IV2SLS(dependent, exog, endog, instruments).fit()
            print(iv_model.summary.tables[1])
            
            # 系数比较
            print("\n4. 内生性检验: OLS vs 2SLS系数比较")
            for var in endogenous_vars:
                ols_coef = ols_model.params[var]
                iv_coef = iv_model.params[var]
                diff = abs(ols_coef - iv_coef)
                diff_percent = diff / abs(ols_coef) * 100 if ols_coef != 0 else float('inf')
                
                print(f"{var}:")
                print(f"  OLS系数: {ols_coef:.6f}")
                print(f"  2SLS系数: {iv_coef:.6f}")
                print(f"  绝对差异: {diff:.6f}")
                print(f"  相对差异: {diff_percent:.2f}%")
                
                if diff_percent > 20:  # 20%作为参考阈值
                    print("  结论: 存在显著差异，可能有内生性问题")
                else:
                    print("  结论: 差异较小，内生性可能不严重")
            
            # Durbin-Wu-Hausman检验
            print("\n5. Durbin-Wu-Hausman检验")
            
            for endvar in endogenous_vars:
                # 第一阶段回归
                fs_X = sm.add_constant(np.column_stack([
                    regression_df[exogenous_vars].values,
                    regression_df[instrument_vars].values
                ]))
                fs_y = regression_df[endvar].values
                fs_model = sm.OLS(fs_y, fs_X).fit()
                residuals = fs_model.resid
                
                # 第二阶段回归
                ss_X = sm.add_constant(np.column_stack([
                    regression_df[exogenous_vars + endogenous_vars].values,
                    residuals
                ]))
                ss_y = regression_df['log_box_off'].values
                ss_model = sm.OLS(ss_y, ss_X).fit()
                
                # 残差系数及其显著性
                residual_idx = ss_model.params.shape[0] - 1  # 最后一个系数
                residual_coef = ss_model.params[residual_idx]
                residual_pval = ss_model.pvalues[residual_idx]
                residual_tval = ss_model.tvalues[residual_idx]
                
                print(f"{endvar}的残差项:")
                print(f"  系数: {residual_coef:.6f}")
                print(f"  t值: {residual_tval:.4f}")
                print(f"  p值: {residual_pval:.6f}")
                
                if residual_pval < 0.05:
                    print(f"  结论: 拒绝原假设，{endvar}可能存在内生性问题")
                else:
                    print(f"  结论: 未拒绝原假设，未检测到{endvar}存在显著内生性问题")
            
            # 过度识别检验 (如果工具变量数量>内生变量数量)
            if len(instrument_vars) > len(endogenous_vars):
                print("\n6. 过度识别检验 (Sargan-Hansen J检验)")
                try:
                    j_stat = iv_model.j_stat
                    print(f"  J统计量: {j_stat.stat:.4f}")
                    print(f"  p值: {j_stat.pval:.6f}")
                    
                    if j_stat.pval < 0.05:
                        print("  结论: 拒绝原假设，工具变量集合可能无效")
                    else:
                        print("  结论: 未拒绝原假设，工具变量集合有效")
                except Exception as e:
                    print(f"  过度识别检验失败: {e}")
            else:
                print("\n注意: 无法进行过度识别检验，因为工具变量数量不超过内生变量数量")
            
            # Hausman检验
            print("\n7. Hausman检验")
            try:
                from linearmodels.iv import compare
                
                hausman_results = compare({"OLS": ols_model, "IV": iv_model})
                print(hausman_results)
                
                # 检验是否支持内生性假设
                if hasattr(hausman_results, 'stat') and hasattr(hausman_results.stat, 'values'):
                    hausman_stat = hausman_results.stat.values[0]
                    hausman_pval = hausman_results.pval.values[0]
                    
                    print(f"  Hausman统计量: {hausman_stat:.4f}")
                    print(f"  p值: {hausman_pval:.6f}")
                    
                    if hausman_pval < 0.05:
                        print("  结论: 拒绝原假设，存在内生性问题，应使用工具变量法")
                    else:
                        print("  结论: 未拒绝原假设，未检测到显著内生性问题，可使用OLS")
                else:
                    print("  Hausman检验结果格式不兼容，无法提取详细统计量")
                
            except Exception as e:
                print(f"  Hausman检验失败: {e}")
                print("  提示: 可能需要调整linearmodels库的版本或检查数据格式")
        
        except Exception as e:
            print(f"2SLS回归失败: {e}")
    else:
        print("样本量不足，无法进行完整的内生性检验")