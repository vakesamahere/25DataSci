import pandas as pd
import ast
import collections

df = pd.read_csv('data/details.csv', encoding='utf-8-sig')

df_box_office = pd.read_csv('data/maoyan_box_office_all.csv', encoding='utf-8-sig')
df = pd.merge(df, df_box_office, left_on='title', right_on='Title', how='inner')

df_coms = pd.read_csv('data/short_coms_agg.csv', encoding='utf-8-sig')
df = pd.merge(df, df_coms, left_on='id', right_on='mv_id', how='inner')

type_cols = ['tp_drama', 'tp_comedy', 'tp_action', 'tp_romance']
df['tp_drama'] = df['tags'].apply(lambda x: 1 if '剧情' in ast.literal_eval(x) else 0)
df['tp_comedy'] = df['tags'].apply(lambda x: 1 if '喜剧' in ast.literal_eval(x) else 0)
df['tp_action'] = df['tags'].apply(lambda x: 1 if '动作' in ast.literal_eval(x) else 0)
df['tp_romance'] = df['tags'].apply(lambda x: 1 if '爱情' in ast.literal_eval(x) else 0)

# 删除type_cols中全0的记录
df = df[(df[type_cols] != 0).any(axis=1)]

df.rename(columns={
    'Box Office': 'box_off',
    'Average Price': 'avg_price',
    'Average Attendance': 'avg_attend',
}, inplace=True)
# 保留的列
cols = [
  "id", "title", "year",
  "rating_count", "rating_score",
  "tp_drama", "tp_comedy", "tp_action", "tp_romance",
  "polarity_mean", "polarity_std", "comment_count", "positive_ratio", "negative_ratio", "sentiment_std",
  "box_off", "avg_price", "avg_attend",
]

df = df[cols]
# 输出summarize
print("数据摘要:")
print(df.describe(include='all'))
df.to_csv('data/data_main.csv', index=False, encoding='utf-8-sig')
print("数据已保存至 data/data_main.csv")


# # 将标签字符串转换为实际的列表对象
# df['tags_list'] = df['tags'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# # 获取所有唯一的标签
# all_tags = set()
# for tags in df['tags_list']:
#     all_tags.update(tags)
# print(f"共有 {len(all_tags)} 种不同的标签类型")

# # 统计每种标签的出现频率
# tag_counts = collections.Counter()
# for tags in df['tags_list']:
#     for tag in tags:
#         tag_counts[tag] += 1

# # 按频率降序排列并打印结果
# print("\n标签出现频率统计:")
# for tag, count in tag_counts.most_common():
#     print(f"{tag}: {count}")