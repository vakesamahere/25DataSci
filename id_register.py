import pandas as pd

def is_new(title):
    # 检查title是否已经存在于id_title.csv中
    try:
        id_title_df = pd.read_csv('data/id_title.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        id_title_df = pd.DataFrame(columns=['id', 'title'])
    if id_title_df.empty:
        return True
    existing_record = id_title_df[id_title_df['title'] == title]
    return existing_record.empty
    
def get_id(title):
    # 如果已经记录过title，则返回对应title的id，否则创建一个record并返回id
    try:
        id_title_df = pd.read_csv('data/id_title.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        id_title_df = pd.DataFrame(columns=['id', 'title'])
    if id_title_df.empty:
        id_title_df = pd.DataFrame(columns=['id', 'title'])
    # 检查title是否已经存在
    existing_record = id_title_df[id_title_df['title'] == title]
    if not existing_record.empty:
        return existing_record['id'].values[0]
    else:
        # 如果不存在，则创建一个新的id
        new_id = len(id_title_df) + 1
        new_record = pd.DataFrame({'id': [new_id], 'title': [title]})
        id_title_df = pd.concat([id_title_df, new_record], ignore_index=True)
        id_title_df.to_csv('data/id_title.csv', index=False, encoding='utf-8-sig')
        return new_id
    
def get_title(id):
    # 如果已经记录过id，则返回对应id的title，否则返回None
    try:
        id_title_df = pd.read_csv('data/id_title.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        return None
    if id_title_df.empty:
        return None
    existing_record = id_title_df[id_title_df['id'] == id]
    if not existing_record.empty:
        return existing_record['title'].values[0]
    else:
        return None