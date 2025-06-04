import pandas as pd
# 计算指标
df = pd.read_csv('data/short_coms_all.csv', encoding='utf-8-sig')
df.rename(columns={'id': 'mv_id'}, inplace=True)
df_senti = pd.read_csv('data/short_coms_sentiment.csv', encoding='utf-8-sig')
df = df.merge(df_senti, left_on='review_id', right_on='id', how='inner')

df['polarity'] = df['prob_positive'] - df['prob_negative']

# 按照mv_id聚合，计算polarity的
# 均值
# 标准差
# 评论数
# 正面评论数
# 正面评论占比
# 负面评论数
# 负面评论占比
agg_funcs = {
    'polarity': ['mean', 'std'],
    'review_id': ['count'],
    'sentiment': [lambda x: (x == 1).sum(), lambda x: (x == 0).sum()],
}
df = df.groupby('mv_id').agg(agg_funcs).reset_index()
df.columns = ['mv_id', 'polarity_mean', 'polarity_std', 'comment_count', 'count_positive', 'count_negative']
df['positive_ratio'] = df['count_positive'] / df['comment_count']
df['negative_ratio'] = df['count_negative'] / df['comment_count']
# 计算正面和负面评论数
df['count_positive'] = df['count_positive'].fillna(0).astype(int)
df['count_negative'] = df['count_negative'].fillna(0).astype(int)
df['comment_count'] = df['comment_count'].fillna(0).astype(int)
# 计算正面和负面评论占比
df['positive_ratio'] = df['positive_ratio'].fillna(0)
df['negative_ratio'] = df['negative_ratio'].fillna(0)

# positive的描述性统计
print("正面评论的描述性统计:")
print(df['positive_ratio'].describe())

df.to_csv('data/short_coms_agg.csv', index=False, encoding='utf-8-sig')