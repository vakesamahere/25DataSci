import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data/short_coms_all.csv')
# 统计每个id的评论数并做一个直方图
# 仅保留在data/box_office.csv中有对应title的id
# 从details.csv中获取title

# load data/details.csv to get titles
df_details = pd.read_csv('data/details.csv', encoding='utf-8-sig')
df = df.merge(df_details, on='id', how='inner')
# df = df[df['title'].isin(pd.read_csv('box_office.csv')['title'])]
comment_counts = df['id'].value_counts()
bound = 50
# 截取评论大于等于10的id
comment_counts = comment_counts[comment_counts >= bound]

print("Number of unique IDs:", len(comment_counts))

# 查看details的tags列，它是形如['喜剧', '爱情']的字符串，需要统计每一种类型的电影个数
# 首先，将tags列转换为列表
# 然后，统计有多少个不同的类型（'戏剧'这样的单个字符串算一个类型）
# 创建一个集合存储展开的类型
tags_set = set()
for tags in df['tags'].dropna():
    # 将字符串转换为列表
    tags_list = eval(tags) if isinstance(tags, str) else []
    tags_set.update(tags_list)
# 统计每种类型的电影个数（title）和评论（记录）个数
tags_counts = {tag: 0 for tag in tags_set}
for tags in df['tags'].dropna():
    # 将字符串转换为列表
    tags_list = eval(tags) if isinstance(tags, str) else []
    for tag in tags_list:
        if tag in tags_counts:
            tags_counts[tag] += 1
# 将结果转换为DataFrame
tags_df = pd.DataFrame(list(tags_counts.items()), columns=['tag', 'comment_count'])
# 在tags_df中添加一列mv_count，表示每个tag对应的电影数量，在df_detail中匹配tag in tags而且不同的title
tags_df['mv_count'] = 0
for index, row in tags_df.iterrows():
    tag = row['tag']
    # 统计有多少个不同的title包含这个tag
    tags_df.at[index, 'mv_count'] = df[df['tags'].str.contains(tag, na=False)]['title'].nunique()
# 按照count降序排序
tags_df = tags_df.sort_values(by='mv_count', ascending=False)
print("Number of unique tags:", len(tags_df))
# 输出每种类型的电影个数
print(tags_df)


if 0:
    plt.figure(figsize=(10, 6))
    comment_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Comments per ID')

    plt.xlabel('ID')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()