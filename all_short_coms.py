import requests
import re
# from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

# User-agent pool
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 10; Pixel 3 XL Build/QP1A.190711.020) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
    "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 OPR/77.0.4054.203",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 10; SM-G980F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Vivaldi/4.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 9; SM-G960F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Brave/1.26.77",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/91.0.4472.124 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-N975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; Pixel 4 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 OPR/77.0.4054.203",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Linux; Android 10; SM-G970F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Vivaldi/4.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 9; SM-G965F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Brave/1.26.77",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/91.0.4472.124 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-N960F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; Pixel 4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 OPR/77.0.4054.203",
    "Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Vivaldi/4.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 9; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
]

def get_random_user_agent():
    """
    随机选择一个User-Agent
    """
    return random.choice(user_agents)

def get_short_comments(media_id, page_size=20, sort=0, cursor=None):
    """
    获取B站番剧/影视的短评
    
    参数:
        media_id (int): 媒体ID
        page_size (int): 每页评论数量
        sort (int): 排序方式
        
    返回:
        dict: 返回的JSON数据
    """
    url = f"https://api.bilibili.com/pgc/review/short/list"
    
    params = {
        "media_id": media_id,
        "ps": page_size,
        "sort": sort,
        "web_location": "666.19"
    }

    if cursor:
        params["cursor"] = cursor
    
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7",
        "origin": "https://www.bilibili.com",
        "priority": "u=1, i",
        "referer": "https://www.bilibili.com/",
        "sec-ch-ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": get_random_user_agent()
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None
    
def fetch_multiple_pages(media_id, num_pages=10, page_size=20, sort=0):
    """
    批量获取多页短评数据
    
    参数:
        media_id (int): 媒体ID
        num_pages (int): 要获取的页数
        page_size (int): 每页评论数量
        sort (int): 排序方式
        
    返回:
        pandas.DataFrame: 包含所有页数据的DataFrame
    """
    if num_pages == -1:
        num_pages = 1000
    all_comments_df = pd.DataFrame()
    cursor = None
    
    for page in range(num_pages):
        
        # 获取当前页数据
        result = get_short_comments(media_id, page_size, sort, cursor)
        
        if not result or result.get("code") != 0:
            print(f"获取第 {page+1} 页数据失败")
            return
            
        # 处理数据
        page_df = process_comments_to_dataframe(result)
        print(f"\t\tMedia-ID:{media_id}, 获取第 {page+1} 页数据... 共 {len(page_df)} 条评论")
        
        if page_df.empty:
            print(f"第 {page+1} 页没有数据")
            break
            
        # 合并数据
        all_comments_df = pd.concat([all_comments_df, page_df], ignore_index=True)
        
        # 获取下一页的cursor
        cursor = result.get("data", {}).get("next")
        
        if not cursor or cursor == "0":
            print("没有更多数据")
            break
        
        if len(all_comments_df) >= 2000:
            print("已获取超过2000条评论，停止获取")
            break
            
        # 间隔一段时间，避免请求过于频繁
        # time.sleep(1)
    
    print(f"\t共获取 {len(all_comments_df)} 条评论")
    return all_comments_df

def process_comments_to_dataframe(json_data):
    """
    处理获取的短评JSON数据，提取关键信息到DataFrame
    
    参数:
        json_data (dict): 原始JSON数据
        
    返回:
        pandas.DataFrame: 包含关键信息的DataFrame
    """
    if not json_data or json_data.get("code") != 0:
        print("数据无效或API返回错误")
        return pd.DataFrame()
    
    comments_list = json_data.get("data", {}).get("list", [])
    if not comments_list:
        print("未找到评论数据")
        return pd.DataFrame()
    
    # 准备存储提取的数据
    extracted_data = []
    
    for comment in comments_list:
        author = comment.get("author", {})
        
        # 判断会员情况
        vip = author.get("vip", {}).get("vipStatus", 0)
        
        # 清理评论内容的特殊字符
        content = comment.get("content", "")
        if content:
            # 将换行符替换为空格
            content = content.replace("\n", " ")
            # 移除其他可能的特殊字符
            content = re.sub(r'[^\w\s,.?!，。？！：:;；""\'\'()]', ' ', content)
        
        # 提取所需数据
        comment_data = {
            "user_id": author.get("mid", ""),
            "username": author.get("uname", ""),
            "user_level": author.get("level", ""),
            "user_vip": vip,
            "review_id": comment.get("review_id", ""),
            "review_content": content,
            "review_push_time": comment.get("push_time_str", ""),
            "review_likes": comment.get("stat", {}).get("likes", 0),
            "score_in_review": comment.get("score", "")
        }
        
        extracted_data.append(comment_data)
    
    # 创建DataFrame
    df = pd.DataFrame(extracted_data)
    
    return df

def get_short_comments_from_file():
    global fail_time
    try:
        df = pd.read_csv("data/mv_lst.csv", encoding='utf-8-sig')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return pd.DataFrame()
    
    # 创建目录
    os.makedirs("data/short_coms", exist_ok=True)
    os.makedirs("data/short_coms/temp", exist_ok=True)  # 为临时文件创建目录
    
    # 创建一个记录已处理ID的集合
    processed_ids = set()
    
    # 尝试读取已处理记录
    try:
        processed_df = pd.read_csv('data/processed_ids.csv')
        processed_ids = set(processed_df['id'].tolist())
        print(f"已加载{len(processed_ids)}个处理过的ID")
    except:
        print("没有找到处理记录文件，将创建新文件")
    
    # 每个批次的ID数量和当前批次信息
    batch_size = 20
    current_batch_ids = []
    current_batch_df = pd.DataFrame()
    
    # 确定当前批次编号
    existing_files = [f for f in os.listdir("data/short_coms") if f.startswith("short_coms_batch_")]
    next_batch_number = len(existing_files) + 1
    
    # 检查是否有未完成的批次文件
    temp_files = [f for f in os.listdir("data/short_coms/temp") if f.startswith("temp_")]
    temp_ids = set()
    
    # 如果有临时文件，加载它们
    if temp_files:
        print(f"找到{len(temp_files)}个临时文件，尝试加载...")
        
        # 获取最后一个批次的编号
        if existing_files:
            last_batch_file = sorted(existing_files)[-1]
            try:
                batch_number = int(last_batch_file.split('_')[-1].split('.')[0])
                next_batch_number = batch_number + 1
            except:
                pass
        
        # 加载临时文件的ID列表
        for temp_file in temp_files:
            try:
                temp_id = int(temp_file.split('_')[1].split('.')[0])
                temp_ids.add(temp_id)
                processed_ids.add(temp_id)  # 将临时文件ID加入已处理列表
                
                # 加载临时数据
                temp_df = pd.read_csv(f"data/short_coms/temp/{temp_file}", encoding='utf-8-sig')
                current_batch_df = pd.concat([current_batch_df, temp_df], ignore_index=True)
                current_batch_ids.append(temp_id)
                
                print(f"已加载临时文件: {temp_file}")
            except Exception as e:
                print(f"加载临时文件 {temp_file} 失败: {e}")
        
        print(f"成功加载{len(current_batch_ids)}个ID的临时数据")
    
    # 处理每个ID
    for index, row in df.iterrows():
        id = row.get("id")
        media_id = row.get("media_id")
        
        # 跳过已处理的ID
        if id in processed_ids and id not in temp_ids:
            print(f"跳过ID {id}，已处理过")
            continue
        
        # 如果ID在临时文件中已存在，也跳过
        if id in temp_ids:
            print(f"ID {id} 已在临时文件中，跳过重复处理")
            continue
        
        try:
            temp_df = fetch_multiple_pages(media_id, num_pages=-1)
            
            if temp_df is None or temp_df.empty:
                print(f"ID {id} 没有评论或获取失败")
                continue
                
            # 添加ID列
            temp_df['id'] = id
            
            # 将当前ID的数据保存为临时文件
            temp_file = f"data/short_coms/temp/temp_{id}.csv"
            temp_df.to_csv(temp_file, index=False, encoding='utf-8-sig', 
                          escapechar='\\', quotechar='"', quoting=1)
            
            # 添加到当前批次
            current_batch_df = pd.concat([current_batch_df, temp_df], ignore_index=True)
            current_batch_ids.append(id)
            processed_ids.add(id)
            
            # 更新已处理ID记录
            update_processed_ids(processed_ids)
            
            fail_time = 0  # 重置失败次数
            print(f"处理进度 {index + 1}/{len(df)}: ID {id} - Media ID: {media_id}")
            print(f"ID {id} 数据已保存至临时文件")
            
            # 当批次达到指定大小时合并保存并清理临时文件
            if len(current_batch_ids) >= batch_size:
                save_batch(current_batch_df, current_batch_ids, next_batch_number)
                
                # 清理此批次的临时文件
                for batch_id in current_batch_ids:
                    temp_path = f"data/short_coms/temp/temp_{batch_id}.csv"
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # 重置批次数据并递增批次编号
                current_batch_df = pd.DataFrame()
                current_batch_ids = []
                next_batch_number += 1
                
        except Exception as e:
            print(f"处理ID {id}时出错: {e}")
            fail_time += 1
            raise Exception("FAIL")
    
    # 保存最后一批数据(如果有)
    if len(current_batch_ids) > 0:
        save_batch(current_batch_df, current_batch_ids, next_batch_number)
        
        # 清理此批次的临时文件
        for batch_id in current_batch_ids:
            temp_path = f"data/short_coms/temp/temp_{batch_id}.csv"
            if os.path.exists(temp_path):
                os.remove(temp_path)

def save_batch(batch_df, batch_ids, batch_number):
    """保存批次数据到单独的CSV文件"""
    if batch_df.empty:
        return
        
    # 文件名格式: short_coms_batch_1.csv, short_coms_batch_2.csv, ...
    filename = f"data/short_coms/short_coms_batch_{batch_number}.csv"
    
    batch_df.to_csv(filename, index=False, encoding='utf-8-sig', 
                  escapechar='\\', quotechar='"', quoting=1)
    
    # 创建批次记录文件，记录此批次包含的ID
    batch_info = pd.DataFrame({'id': batch_ids, 'batch': batch_number})
    batch_info.to_csv(f"data/short_coms/batch_{batch_number}_info.csv", 
                     index=False)
    
    print(f"成功保存批次 {batch_number}，包含 {len(batch_ids)} 个ID，共 {len(batch_df)} 条评论")

def update_processed_ids(processed_ids):
    """更新已处理ID记录"""
    pd.DataFrame({'id': list(processed_ids)}).to_csv('data/processed_ids.csv', 
                                                  index=False)
    print(f"已更新处理记录，总计处理 {len(processed_ids)} 个ID")

def combine_all_batches():
    """合并所有批次文件为一个完整数据集(用于分析)"""
    batch_files = [f for f in os.listdir("data/short_coms") if f.startswith("short_coms_batch_")]
    
    if not batch_files:
        print("没有找到批次文件")
        return
        
    all_data = pd.DataFrame()
    
    for batch_file in sorted(batch_files):
        try:
            batch_df = pd.read_csv(f"data/short_coms/{batch_file}", encoding='utf-8-sig')
            all_data = pd.concat([all_data, batch_df], ignore_index=True)
            print(f"已加载 {batch_file}，{len(batch_df)} 条数据")
        except Exception as e:
            print(f"读取 {batch_file} 失败: {e}")
    
    # 保存合并后的完整数据集
    all_data.to_csv("data/short_coms_all.csv", index=False, encoding='utf-8-sig', 
                  escapechar='\\', quotechar='"', quoting=1)
    print(f"合并完成，共 {len(all_data)} 条评论数据")
    return all_data

# 示例使用
if __name__ == "__main__":
    # media_id = 28339083
    
    # # 获取5页数据（每页20条评论，共100条）
    # all_comments = fetch_multiple_pages(media_id, num_pages=10)
    
    # print(f"共获取 {len(all_comments)} 条评论")
    # print("\n数据预览:")
    # print(all_comments.head())
    
    # # 可选：保存到CSV
    # # all_comments.to_csv(f"bili_comments_{media_id}_multiple_pages.csv", index=False, encoding="utf-8-sig")
    fail_time = 0
    waiting_time_unit = 45  # 初始等待时间为45秒
    while True:
        # 被封禁自动等待更长时间，每次翻倍，成功后重置为45s
        try:
            get_short_comments_from_file()
            break
        except Exception as e:
            fail_time += 1
            print(f"发生错误: {e}")
            waiting_time = 30 * (2 ** (fail_time - 1)) if fail_time > 0 else waiting_time_unit
            print(f"等待{waiting_time}秒后重试...")
            time.sleep(waiting_time)
