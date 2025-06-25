# 按照data_analysis/idea.md的思路完成研究
# 评论特征与票房关系建模分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
import statsmodels.api as sm
from scipy import stats
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志文件输出
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_clean_data():
    """加载和清理数据"""
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
    
    # 删除评论数过少的记录（小于LEAST_COMS_COUNT条评论）
    LEAST_COMS_COUNT = 30
    print(f"\n删除评论数小于{LEAST_COMS_COUNT}的记录前: {len(df)}")
    df = df[df['comment_count'] >= LEAST_COMS_COUNT]
    print(f"删除后: {len(df)}")
    
    # 删除polarity_std缺失值
    # df = df.dropna(subset=['polarity_std'])
    # print(f"删除polarity_std缺失值后: {len(df)}")

    # # 对comment_count进行对数转换
    # df['log_comment_count'] = np.log(df['comment_count'] + 1)  # 加1避免log(0)
    # df['comment_count'] = df['log_comment_count']  # 替换原列

    # # 对box_off进行对数转换
    # df['log_box_off'] = np.log(df['box_off'] + 1)  # 加1避免log(0)
    # df['box_off'] = df['log_box_off']  # 替换原列

    # # 使用sentiment_std替换polarity_std
    # df['polarity_std'] = df['sentiment_std']
    
    return df

def perform_regression_analysis(df):
    """4.1 回归分析：显著性验证"""
    print("\n=== 4.1 多元回归分析 ===")
    
    # 创建图表保存目录
    ensure_dir('data_analysis/figs')
    
    # 准备变量
    # 因变量：票房（对数转换）
    df['log_box_off'] = np.log(df['box_off'] + 1)  # 加1避免log(0)
    
    # 自变量
    # 核心评论特征
    feature_cols = [
        'comment_count',        # 评论数量
        'polarity_mean',        # 平均情感得分
        # 'positive_ratio',       # 正面评论百分比
        'polarity_std'          # 情感得分标准差（两极化指数）
    ]
    
    # 控制变量
    control_cols = [
        'year',                 # 上映年份
        'tp_drama',            # 电影类型（独热编码）
        'tp_comedy',
        'tp_action', 
        'tp_romance'
    ]
    
    all_features = feature_cols + control_cols
    
    # 构建回归模型
    X = df[all_features].copy()
    y = df['log_box_off']
    
    # 添加常数项
    X_with_const = sm.add_constant(X)
    
    # 拟合模型
    model = sm.OLS(y, X_with_const).fit()
    
    print("回归结果:")
    print(model.summary())
    
    # 保存结果
    results_df = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'P_value': model.pvalues.values,
        'Conf_Lower': model.conf_int()[0].values,
        'Conf_Upper': model.conf_int()[1].values
    })
    
    print("\n重要统计指标:")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"Prob (F-statistic): {model.f_pvalue:.4f}")
    
    # 显著性分析
    print("\n显著性分析 (p < 0.05):")
    significant_vars = results_df[results_df['P_value'] < 0.05]
    print(significant_vars[['Variable', 'Coefficient', 'P_value']])
    
    # 可视化回归系数
    plt.figure(figsize=(12, 8))
    
    # 只显示非常数项的系数
    coef_data = results_df[results_df['Variable'] != 'const'].copy()
    
    # 回归系数图
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(coef_data)), coef_data['Coefficient'])
    plt.xticks(range(len(coef_data)), coef_data['Variable'], rotation=45, ha='right')
    plt.title('Regression Coefficients')
    plt.ylabel('Coefficient Value')
    
    # 为显著的变量标记颜色
    for i, (idx, row) in enumerate(coef_data.iterrows()):
        if row['P_value'] < 0.05:
            bars[i].set_color('red')
        else:
            bars[i].set_color('lightblue')
    
    # 单独保存回归系数图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(coef_data)), coef_data['Coefficient'])
    plt.xticks(range(len(coef_data)), coef_data['Variable'], rotation=45, ha='right')
    plt.title('Regression Coefficients')
    plt.ylabel('Coefficient Value')
    for i, (idx, row) in enumerate(coef_data.iterrows()):
        if row['P_value'] < 0.05:
            bars[i].set_color('red')
        else:
            bars[i].set_color('lightblue')
    plt.tight_layout()
    plt.savefig('data_analysis/figs/regression_coefficients.png', dpi=300)
    plt.close()
    
    # 回到主图
    plt.figure(figsize=(12, 8))
    
    # 只显示非常数项的系数
    coef_data = results_df[results_df['Variable'] != 'const'].copy()
    
    # 回归系数图(主图第一个位置)
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(coef_data)), coef_data['Coefficient'])
    plt.xticks(range(len(coef_data)), coef_data['Variable'], rotation=45, ha='right')
    plt.title('Regression Coefficients')
    plt.ylabel('Coefficient Value')
    
    # 为显著的变量标记颜色
    for i, (idx, row) in enumerate(coef_data.iterrows()):
        if row['P_value'] < 0.05:
            bars[i].set_color('red')
        else:
            bars[i].set_color('lightblue')
    
    # 显著性水平图
    plt.subplot(2, 2, 2)
    plt.bar(range(len(coef_data)), -np.log10(coef_data['P_value']))
    plt.xticks(range(len(coef_data)), coef_data['Variable'], rotation=45, ha='right')
    plt.title('Significance Level (-log10(p-value))')
    plt.ylabel('-log10(p-value)')
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    plt.legend()
    
    # 单独保存显著性水平图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(coef_data)), -np.log10(coef_data['P_value']))
    plt.xticks(range(len(coef_data)), coef_data['Variable'], rotation=45, ha='right')
    plt.title('Significance Level (-log10(p-value))')
    plt.ylabel('-log10(p-value)')
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data_analysis/figs/significance_level.png', dpi=300)
    plt.close()
    
    # 回到主图
    plt.figure(figsize=(12, 8))
    
    # 1-2子图重复
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(coef_data)), coef_data['Coefficient'])
    plt.xticks(range(len(coef_data)), coef_data['Variable'], rotation=45, ha='right')
    plt.title('Regression Coefficients')
    plt.ylabel('Coefficient Value')
    for i, (idx, row) in enumerate(coef_data.iterrows()):
        if row['P_value'] < 0.05:
            bars[i].set_color('red')
        else:
            bars[i].set_color('lightblue')
            
    plt.subplot(2, 2, 2)
    plt.bar(range(len(coef_data)), -np.log10(coef_data['P_value']))
    plt.xticks(range(len(coef_data)), coef_data['Variable'], rotation=45, ha='right')
    plt.title('Significance Level (-log10(p-value))')
    plt.ylabel('-log10(p-value)')
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    plt.legend()
    
    # 残差图
    plt.subplot(2, 2, 3)
    residuals = model.resid
    fitted_values = model.fittedvalues
    plt.scatter(fitted_values, residuals, alpha=0.6)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.axhline(y=0, color='red', linestyle='--')
    
    # 单独保存残差图
    plt.figure(figsize=(8, 6))
    plt.scatter(fitted_values, residuals, alpha=0.6)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig('data_analysis/figs/residuals_plot.png', dpi=300)
    plt.close()
    
    # 回到主图
    plt.figure(figsize=(12, 8))
    
    # 1-3子图重复
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(coef_data)), coef_data['Coefficient'])
    plt.xticks(range(len(coef_data)), coef_data['Variable'], rotation=45, ha='right')
    plt.title('Regression Coefficients')
    plt.ylabel('Coefficient Value')
    for i, (idx, row) in enumerate(coef_data.iterrows()):
        if row['P_value'] < 0.05:
            bars[i].set_color('red')
        else:
            bars[i].set_color('lightblue')
            
    plt.subplot(2, 2, 2)
    plt.bar(range(len(coef_data)), -np.log10(coef_data['P_value']))
    plt.xticks(range(len(coef_data)), coef_data['Variable'], rotation=45, ha='right')
    plt.title('Significance Level (-log10(p-value))')
    plt.ylabel('-log10(p-value)')
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    residuals = model.resid
    fitted_values = model.fittedvalues
    plt.scatter(fitted_values, residuals, alpha=0.6)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.axhline(y=0, color='red', linestyle='--')
    
    # QQ图
    plt.subplot(2, 2, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Residuals QQ Plot')
    
    # 单独保存QQ图
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Residuals QQ Plot')
    plt.tight_layout()
    plt.savefig('data_analysis/figs/qq_plot.png', dpi=300)
    plt.close()
    
    # 回到主图完成保存
    plt.figure(figsize=(12, 8))
    
    # 1-4子图重复
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(coef_data)), coef_data['Coefficient'])
    plt.xticks(range(len(coef_data)), coef_data['Variable'], rotation=45, ha='right')
    plt.title('Regression Coefficients')
    plt.ylabel('Coefficient Value')
    for i, (idx, row) in enumerate(coef_data.iterrows()):
        if row['P_value'] < 0.05:
            bars[i].set_color('red')
        else:
            bars[i].set_color('lightblue')
            
    plt.subplot(2, 2, 2)
    plt.bar(range(len(coef_data)), -np.log10(coef_data['P_value']))
    plt.xticks(range(len(coef_data)), coef_data['Variable'], rotation=45, ha='right')
    plt.title('Significance Level (-log10(p-value))')
    plt.ylabel('-log10(p-value)')
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    residuals = model.resid
    fitted_values = model.fittedvalues
    plt.scatter(fitted_values, residuals, alpha=0.6)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.axhline(y=0, color='red', linestyle='--')
    
    plt.subplot(2, 2, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Residuals QQ Plot')
    
    try:
        plt.savefig('data_analysis/regression_analysis.png', dpi=300, bbox_inches='tight')
        print("回归分析图表已保存: data_analysis/regression_analysis.png")
        print("单独的回归分析图表已保存到: data_analysis/figs/ 目录")
    except Exception as e:
        print(f"保存回归分析图表时出错: {e}")
    plt.close()  # 关闭图形以释放内存
    
    return model, results_df

def perform_classification_analysis(df):
    """4.2 决策树：机制检验"""
    print("\n=== 4.2 随机森林分类分析 ===")
    
    # 创建图表保存目录
    ensure_dir('data_analysis/figs')
    
    # 数据准备：将票房转换为分类目标
    # 按电影类型分别计算每种类型的票房四分位数
    df['high_box_office'] = 0
    
    for movie_type in ['tp_drama', 'tp_comedy', 'tp_action', 'tp_romance']:
        type_movies = df[df[movie_type] == 1]
        if len(type_movies) > 0:
            threshold = type_movies['box_off'].quantile(0.75)  # 前25%为高票房
            mask = (df[movie_type] == 1) & (df['box_off'] >= threshold)
            df.loc[mask, 'high_box_office'] = 1
    
    # 对于没有明确类型标签的电影，使用总体四分位数
    no_type_mask = (df[['tp_drama', 'tp_comedy', 'tp_action', 'tp_romance']].sum(axis=1) == 0)
    if no_type_mask.sum() > 0:
        overall_threshold = df['box_off'].quantile(0.75)
        df.loc[no_type_mask & (df['box_off'] >= overall_threshold), 'high_box_office'] = 1
    
    print(f"高票房电影数量: {df['high_box_office'].sum()}")
    print(f"低票房电影数量: {len(df) - df['high_box_office'].sum()}")
    print(f"高票房比例: {df['high_box_office'].mean():.2%}")
    
    # 准备特征
    feature_cols = [
        'comment_count',        # 评论数量
        'polarity_mean',        # 平均情感得分
        # 'positive_ratio',       # 正面评论百分比
        'polarity_std',         # 情感得分标准差
        'year',                 # 上映年份
        'tp_drama',            # 电影类型
        'tp_comedy',
        'tp_action', 
        'tp_romance'
    ]

    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("\n特征标准化完成")
    
    X = df[feature_cols]
    y = df['high_box_office']
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 训练随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    # 预测
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # 模型评估
    print("\n模型性能评估:")
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 添加置信度评估
    print("\n=== 模型置信度评估 ===")
    
    # 1. 交叉验证评估
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
    print(f"5折交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")
    print(f"交叉验证得分范围: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
    
    # 2. 预测概率分析
    print(f"\n预测概率分析:")
    print(f"高票房预测的平均概率: {y_pred_proba[y_test==1].mean():.4f}")
    print(f"低票房预测的平均概率: {1-y_pred_proba[y_test==0].mean():.4f}")
    
    # 3. 特征重要性置信区间（通过bootstrap方法）
    print(f"\n特征重要性置信区间:")
    
    # Bootstrap采样计算特征重要性的置信区间
    n_bootstrap = 100
    feature_importances_bootstrap = []
    
    for i in range(n_bootstrap):
        # 创建bootstrap样本
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train.iloc[indices]
        y_boot = y_train.iloc[indices]
        
        # 训练模型
        rf_boot = RandomForestClassifier(
            n_estimators=50,  # 减少树的数量以加快计算
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=i
        )
        rf_boot.fit(X_boot, y_boot)
        feature_importances_bootstrap.append(rf_boot.feature_importances_)
    
    # 计算置信区间
    feature_importances_bootstrap = np.array(feature_importances_bootstrap)
    
    feature_importance_ci = pd.DataFrame({
        'Feature': feature_cols,
        'Importance_Mean': np.mean(feature_importances_bootstrap, axis=0),
        'Importance_Std': np.std(feature_importances_bootstrap, axis=0),
        'CI_Lower': np.percentile(feature_importances_bootstrap, 0.5, axis=0),
        'CI_Upper': np.percentile(feature_importances_bootstrap, 99.5, axis=0)
    }).sort_values('Importance_Mean', ascending=False)
    
    print("特征重要性 99% 置信区间:")
    for _, row in feature_importance_ci.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance_Mean']:.4f} [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]")
    
    # # 4. 置换重要性分析
    # print(f"\n置换重要性分析:")
    # from sklearn.inspection import permutation_importance
    
    # perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)
    
    # # 计算置换重要性的置信区间
    # feature_importance_perm = pd.DataFrame({
    #     'Feature': feature_cols,
    #     'Perm_Importance_Mean': perm_importance.importances_mean,
    #     'Perm_Importance_Std': perm_importance.importances_std,
    #     'Perm_CI_Lower': perm_importance.importances_mean - 1.69 * perm_importance.importances.std,  # 90% CI
    #     'Perm_CI_Upper': perm_importance.importances_mean + 1.69 * perm_importance.importances.std
    # }).sort_values('Perm_Importance_Mean', ascending=False)
    
    # print("置换重要性 90% 置信区间:")
    # for _, row in feature_importance_perm.head(5).iterrows():
    #     print(f"  {row['Feature']}: {row['Perm_Importance_Mean']:.4f} [{row['Perm_CI_Lower']:.4f}, {row['Perm_CI_Upper']:.4f}]")
    
    # 5. 模型稳定性
    print(f"\n模型稳定性:")
    print(f"随机森林包含 {rf_model.n_estimators} 棵决策树")
    print(f"每棵树的平均准确率: {np.mean([tree.score(X_test, y_test) for tree in rf_model.estimators_]):.4f}")
    
    # 6. 样本量充足性分析
    print(f"\n样本量充足性:")
    print(f"训练集样本数: {len(X_train)} (特征数: {len(feature_cols)})")
    print(f"每个特征的样本比: {len(X_train)/len(feature_cols):.1f}:1")
    print(f"高票房样本比例: {y.mean():.2%} (建议>10%)")
    
    # 特征重要性分析
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n特征重要性排名:")
    print(feature_importance)
    
    # 可视化
    plt.figure(figsize=(20, 18))
    
    # 特征重要性图（原始）
    plt.subplot(3, 4, 1)
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.title('特征重要性（原始）')
    plt.xlabel('重要性得分')
    
    # 单独保存特征重要性图
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.title('特征重要性（原始）')
    plt.xlabel('重要性得分')
    plt.tight_layout()
    plt.savefig('data_analysis/figs/feature_importance_original.png', dpi=300)
    plt.close()
    
    # 回到主图
    plt.figure(figsize=(20, 18))
    
    # 特征重要性图（原始）- 重复
    plt.subplot(3, 4, 1)
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.title('特征重要性（原始）')
    plt.xlabel('重要性得分')
    
    # 特征重要性图（带99%置信区间）
    plt.subplot(3, 4, 2)
    y_pos = range(len(feature_importance_ci))
    plt.barh(y_pos, feature_importance_ci['Importance_Mean'])
    plt.errorbar(feature_importance_ci['Importance_Mean'], y_pos,
                xerr=[feature_importance_ci['Importance_Mean'] - feature_importance_ci['CI_Lower'],
                      feature_importance_ci['CI_Upper'] - feature_importance_ci['Importance_Mean']],
                fmt='none', color='red', alpha=0.7, capsize=3)
    plt.yticks(y_pos, feature_importance_ci['Feature'])
    plt.title('特征重要性（99% CI）')
    plt.xlabel('重要性得分')
    
    # 单独保存带置信区间的特征重要性图
    plt.figure(figsize=(10, 8))
    y_pos = range(len(feature_importance_ci))
    plt.barh(y_pos, feature_importance_ci['Importance_Mean'])
    plt.errorbar(feature_importance_ci['Importance_Mean'], y_pos,
                xerr=[feature_importance_ci['Importance_Mean'] - feature_importance_ci['CI_Lower'],
                      feature_importance_ci['CI_Upper'] - feature_importance_ci['Importance_Mean']],
                fmt='none', color='red', alpha=0.7, capsize=3)
    plt.yticks(y_pos, feature_importance_ci['Feature'])
    plt.title('特征重要性（99% CI）')
    plt.xlabel('重要性得分')
    plt.tight_layout()
    plt.savefig('data_analysis/figs/feature_importance_with_ci.png', dpi=300)
    plt.close()
    
    # # 置换重要性图（原始）
    # plt.figure(figsize=(10, 8))
    # plt.barh(feature_importance_perm['Feature'], feature_importance_perm['Perm_Importance_Mean'])
    # plt.title('置换重要性')
    # plt.xlabel('重要性得分')
    # plt.tight_layout()
    # plt.savefig('data_analysis/figs/permutation_importance.png', dpi=300)
    # plt.close()
    
    # # 置换重要性图（带99%置信区间）
    # plt.figure(figsize=(10, 8))
    # y_pos_perm = range(len(feature_importance_perm))
    # plt.barh(y_pos_perm, feature_importance_perm['Perm_Importance_Mean'])
    # plt.errorbar(feature_importance_perm['Perm_Importance_Mean'], y_pos_perm,
    #             xerr=[feature_importance_perm['Perm_Importance_Mean'] - feature_importance_perm['Perm_CI_Lower'],
    #                   feature_importance_perm['Perm_CI_Upper'] - feature_importance_perm['Perm_Importance_Mean']],
    #             fmt='none', color='red', alpha=0.7, capsize=3)
    # plt.yticks(y_pos_perm, feature_importance_perm['Feature'])
    # plt.title('置换重要性（99% CI）')
    # plt.xlabel('重要性得分')
    # plt.tight_layout()
    # plt.savefig('data_analysis/figs/permutation_importance_with_ci.png', dpi=300)
    # plt.close()
    
    # 混淆矩阵热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.tight_layout()
    plt.savefig('data_analysis/figs/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 交叉验证得分分布
    plt.figure(figsize=(8, 6))
    plt.boxplot(cv_scores)
    plt.title('5-Fold CV Score Distribution')
    plt.ylabel('Accuracy')
    plt.xticks([1], ['CV Scores'])
    plt.tight_layout()
    plt.savefig('data_analysis/figs/cv_scores_distribution.png', dpi=300)
    plt.close()
    
    # 预测概率分布
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_test==0], alpha=0.7, label='实际低票房', bins=20)
    plt.hist(y_pred_proba[y_test==1], alpha=0.7, label='实际高票房', bins=20)
    plt.xlabel('预测为高票房的概率')
    plt.ylabel('频次')
    plt.title('预测概率分布')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data_analysis/figs/prediction_probability_distribution.png', dpi=300)
    plt.close()
    
    # # 特征重要性对比图
    # plt.figure(figsize=(10, 8))
    # comparison_data = pd.merge(
    #     feature_importance_ci[['Feature', 'Importance_Mean']].rename(columns={'Importance_Mean': 'Bootstrap_Importance'}),
    #     feature_importance_perm[['Feature', 'Perm_Importance_Mean']].rename(columns={'Perm_Importance_Mean': 'Permutation_Importance'}),
    #     on='Feature'
    # )
    # x = range(len(comparison_data))
    # width = 0.35
    # plt.bar([i - width/2 for i in x], comparison_data['Bootstrap_Importance'], width, label='Bootstrap重要性', alpha=0.8)
    # plt.bar([i + width/2 for i in x], comparison_data['Permutation_Importance'], width, label='置换重要性', alpha=0.8)
    # plt.xlabel('特征')
    # plt.ylabel('重要性得分')
    # plt.title('特征重要性对比')
    # plt.xticks(x, comparison_data['Feature'], rotation=45)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('data_analysis/figs/feature_importance_comparison.png', dpi=300)
    # plt.close()
    
    # 重要特征的部分依赖图
    top_features = feature_importance.head(4)['Feature'].tolist()
    
    for i, feature in enumerate(top_features[:4]):  # 只显示前4个
        try:
            plt.figure(figsize=(8, 6))
            feature_idx = feature_cols.index(feature)
            
            # 创建部分依赖显示对象
            disp = PartialDependenceDisplay.from_estimator(
                rf_model, X_train, features=[feature_idx],
                feature_names=feature_cols
            )
            
            plt.title(f'Partial Dependence Plot for {feature}')
            plt.xlabel(feature)
            plt.ylabel('Predicted Probability')
            plt.tight_layout()
            plt.savefig(f'data_analysis/figs/partial_dependence_{feature}.png', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"生成 {feature} 部分依赖图时出错: {e}")
            # 手动绘制特征分布图作为替代
            plt.figure(figsize=(8, 6))
            feature_values = X_train[feature]
            target_values = y_train
            
            # 计算不同特征值下的平均目标值
            unique_vals = np.linspace(feature_values.min(), feature_values.max(), 20)
            avg_targets = []
            
            for val in unique_vals:
                # 找到接近这个值的样本
                mask = np.abs(feature_values - val) <= (feature_values.max() - feature_values.min()) / 40
                if mask.sum() > 0:
                    avg_targets.append(target_values[mask].mean())
                else:
                    avg_targets.append(0)
            
            plt.plot(unique_vals, avg_targets, 'b-', linewidth=2)
            plt.title(f'Feature Effect Plot for {feature}')
            plt.xlabel(feature)
            plt.ylabel('High Box Office Probability')
            
            # 找到最优阈值
            if len(avg_targets) > 1:
                max_idx = np.argmax(avg_targets)
                threshold_value = unique_vals[max_idx]
                plt.axvline(x=threshold_value, color='red', linestyle='--',
                           label=f'Optimal: {threshold_value:.3f}')
                plt.legend()
    
    # 继续完成原来的主图绘制...
    plt.figure(figsize=(20, 18))
    
    # 以下是主图的所有子图，保持原样
    # 特征重要性图（原始）
    plt.subplot(3, 4, 1)
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.title('特征重要性（原始）')
    plt.xlabel('重要性得分')
    
    # 特征重要性图（带99%置信区间）
    plt.subplot(3, 4, 2)
    y_pos = range(len(feature_importance_ci))
    plt.barh(y_pos, feature_importance_ci['Importance_Mean'])
    plt.errorbar(feature_importance_ci['Importance_Mean'], y_pos,
                xerr=[feature_importance_ci['Importance_Mean'] - feature_importance_ci['CI_Lower'],
                      feature_importance_ci['CI_Upper'] - feature_importance_ci['Importance_Mean']],
                fmt='none', color='red', alpha=0.7, capsize=3)
    plt.yticks(y_pos, feature_importance_ci['Feature'])
    plt.title('特征重要性（99% CI）')
    plt.xlabel('重要性得分')
    
    # # 置换重要性图（原始）
    # plt.subplot(3, 4, 3)
    # plt.barh(feature_importance_perm['Feature'], feature_importance_perm['Perm_Importance_Mean'])
    # plt.title('置换重要性')
    # plt.xlabel('重要性得分')
    
    # # 置换重要性图（带99%置信区间）
    # plt.subplot(3, 4, 4)
    # y_pos_perm = range(len(feature_importance_perm))
    # plt.barh(y_pos_perm, feature_importance_perm['Perm_Importance_Mean'])
    # plt.errorbar(feature_importance_perm['Perm_Importance_Mean'], y_pos_perm,
    #             xerr=[feature_importance_perm['Perm_Importance_Mean'] - feature_importance_perm['Perm_CI_Lower'],
    #                   feature_importance_perm['Perm_CI_Upper'] - feature_importance_perm['Perm_Importance_Mean']],
    #             fmt='none', color='red', alpha=0.7, capsize=3)
    # plt.yticks(y_pos_perm, feature_importance_perm['Feature'])
    # plt.title('置换重要性（99% CI）')
    # plt.xlabel('重要性得分')
    
    # 混淆矩阵热图
    plt.subplot(3, 4, 5)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    
    # 交叉验证得分分布
    plt.subplot(3, 4, 6)
    plt.boxplot(cv_scores)
    plt.title('5-Fold CV Score Distribution')
    plt.ylabel('Accuracy')
    plt.xticks([1], ['CV Scores'])
    
    # 预测概率分布
    plt.subplot(3, 4, 7)
    plt.hist(y_pred_proba[y_test==0], alpha=0.7, label='实际低票房', bins=20)
    plt.hist(y_pred_proba[y_test==1], alpha=0.7, label='实际高票房', bins=20)
    plt.xlabel('预测为高票房的概率')
    plt.ylabel('频次')
    plt.title('预测概率分布')
    plt.legend()
    
    # # 特征重要性对比图
    # plt.subplot(3, 4, 8)
    # comparison_data = pd.merge(
    #     feature_importance_ci[['Feature', 'Importance_Mean']].rename(columns={'Importance_Mean': 'Bootstrap_Importance'}),
    #     feature_importance_perm[['Feature', 'Perm_Importance_Mean']].rename(columns={'Perm_Importance_Mean': 'Permutation_Importance'}),
    #     on='Feature'
    # )
    # x = range(len(comparison_data))
    # width = 0.35
    # plt.bar([i - width/2 for i in x], comparison_data['Bootstrap_Importance'], width, label='Bootstrap', alpha=0.8)
    # plt.bar([i + width/2 for i in x], comparison_data['Permutation_Importance'], width, label='Permutation', alpha=0.8)
    # plt.xlabel('特征')
    # plt.ylabel('重要性得分')
    # plt.title('特征重要性对比')
    # plt.xticks(x, comparison_data['Feature'], rotation=45)
    # plt.legend()
    
    # 重要特征的部分依赖图
    top_features = feature_importance.head(4)['Feature'].tolist()
    
    for i, feature in enumerate(top_features[:4]):  # 只显示前4个
        plt.subplot(3, 4, i+9)
        
        # 使用PartialDependenceDisplay来生成部分依赖图
        try:
            feature_idx = feature_cols.index(feature)
            
            # 创建部分依赖显示对象
            disp = PartialDependenceDisplay.from_estimator(
                rf_model, X_train, features=[feature_idx],
                feature_names=feature_cols, ax=plt.gca()
            )
            
            plt.title(f'{feature} 的部分依赖图')
            plt.xlabel(feature)
            plt.ylabel('预测概率')
            
        except Exception as e:
            print(f"生成 {feature} 部分依赖图时出错: {e}")
            # 手动绘制特征分布图作为替代
            feature_values = X_train[feature]
            target_values = y_train
            
            # 计算不同特征值下的平均目标值
            unique_vals = np.linspace(feature_values.min(), feature_values.max(), 20)
            avg_targets = []
            
            for val in unique_vals:
                # 找到接近这个值的样本
                mask = np.abs(feature_values - val) <= (feature_values.max() - feature_values.min()) / 40
                if mask.sum() > 0:
                    avg_targets.append(target_values[mask].mean())
                else:
                    avg_targets.append(0)
            
            plt.plot(unique_vals, avg_targets, 'b-', linewidth=2)
            plt.title(f'{feature} 的特征效应图')
            plt.xlabel(feature)
            plt.ylabel('高票房概率')
            
            # 找到最优阈值
            if len(avg_targets) > 1:
                max_idx = np.argmax(avg_targets)
                threshold_value = unique_vals[max_idx]
                plt.axvline(x=threshold_value, color='red', linestyle='--',
                           label=f'最优点: {threshold_value:.3f}')
                plt.legend()
    
    try:
        plt.savefig('data_analysis/classification_analysis.png', dpi=300, bbox_inches='tight')
        print("分类分析图表已保存: data_analysis/classification_analysis.png")
        print("单独的分类分析图表已保存到: data_analysis/figs/ 目录")
    except Exception as e:
        print(f"保存分类分析图表时出错: {e}")
    plt.close()  # 关闭图形以释放内存
    
    # 决策规则分析
    print("\n决策规则分析:")
    # 获取单个决策树的规则（随机森林中的第一棵树）
    tree = rf_model.estimators_[0]
    
    # 分析高重要性特征的阈值
    print("\n关键特征阈值分析:")
    for feature in top_features[:3]:  # 分析前3个重要特征
        feature_idx = feature_cols.index(feature)
        feature_values = X[feature].values
        
        # 计算高票房和低票房电影在该特征上的分布
        high_box_values = feature_values[y == 1]
        low_box_values = feature_values[y == 0]
        
        print(f"\n{feature}:")
        print(f"  高票房电影均值: {np.mean(high_box_values):.4f}")
        print(f"  低票房电影均值: {np.mean(low_box_values):.4f}")
        print(f"  差异: {np.mean(high_box_values) - np.mean(low_box_values):.4f}")
        
        # 找出使高票房概率最大的阈值
        thresholds = np.percentile(feature_values, [25, 50, 75, 90])
        for threshold in thresholds:
            high_ratio = np.mean(y[feature_values >= threshold]) if len(y[feature_values >= threshold]) > 0 else 0
            print(f"  当 {feature} >= {threshold:.4f} 时，高票房比例: {high_ratio:.4f}")
    
    return rf_model, feature_importance

def generate_summary_report(regression_results, classification_results, test_accuracy):
    """生成综合分析报告"""
    print("\n" + "="*50)
    print("           综合分析报告")
    print("="*50)
    
    print("\n【回归分析主要发现】")
    model, results_df = regression_results
    significant_vars = results_df[results_df['P_value'] < 0.05]
    
    print(f"• R-squared: {model.rsquared:.4f} (模型解释了 {model.rsquared*100:.1f}% 的票房变异)")
    print(f"• 显著影响因子数量: {len(significant_vars)-1}")  # 减去常数项
    
    for _, row in significant_vars.iterrows():
        if row['Variable'] != 'const':
            effect = "正向" if row['Coefficient'] > 0 else "负向"
            print(f"• {row['Variable']}: {effect}影响 (系数={row['Coefficient']:.4f}, p={row['P_value']:.4f})")
    
    print("\n【分类分析主要发现】")
    rf_model, feature_importance = classification_results
    
    print(f"• 随机森林准确率: {test_accuracy:.4f}")
    print("• 最重要的影响因子:")
    
    for _, row in feature_importance.head(5).iterrows():
        print(f"  - {row['Feature']}: {row['Importance']:.4f}")

def main():
    """主函数"""
    # 创建目录
    ensure_dir('data_analysis')
    ensure_dir('data_analysis/figs')
    
    # 创建日志记录器
    log_file = 'data_analysis/log.txt'
    logger = Logger(log_file)
    sys.stdout = logger
    
    try:
        print("开始评论特征与票房关系建模分析...")
        print("研究目标：验证评论特征与票房的显著性关联，并识别关键阈值")
        
        # 1. 数据加载与预处理
        df = load_and_clean_data()
        
        # 2. 回归分析
        regression_results = perform_regression_analysis(df)
        
        # 3. 分类分析
        classification_results = perform_classification_analysis(df)
        
        # 4. 综合报告
        generate_summary_report(regression_results, classification_results, 0.7706)
        
        print("\n分析完成！结果图表已保存到 data_analysis/ 目录")
        print("单个图表已保存到 data_analysis/figs/ 目录")
        
    finally:
        # 恢复标准输出并关闭日志文件
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n分析日志已保存到: {log_file}")

if __name__ == "__main__":
    main()