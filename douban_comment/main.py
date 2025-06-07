import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import time
import tqdm
import random


# 发送请求
df = pd.DataFrame(
    columns=[
        "user_name", "user_id", "time", "rating", "location", "content", "like"
    ]
)

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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Brave/1.26.77",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/91.0.4472.124 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-N970F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 OPR/77.0.4054.203",
    "Mozilla/5.0 (X11; Debian; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Linux; Android 10; SM-G975U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Vivaldi/4.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 9; SM-G965U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
]

def get_random_user_agent():
    return random.choice(user_agents)

def get_comments(mv_id=35595615, start=0)-> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            "user_name", "user_id", "time", "rating", "location", "content", "like"
        ]
    )
    url = f"https://movie.douban.com/subject/{mv_id}/comments"

    # 请求参数
    params = {
        "percent_type": "",
        "start": start,
        "limit": 20,
        "status": "P",
        "sort": "new_score",
        "comments_only": 1
    }

    # 请求头
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7",
        "priority": "u=1, i",
        "referer": f"https://movie.douban.com/subject/{mv_id}/comments?start={start}&limit=20&status=P&sort=new_score",
        "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    }

    # Cookie
    cookies = {
        "ll": '"108296"',
        "bid": "YpcQiOepiFU",
        "ap_v": "0,6.0"
    }


    response = requests.get(url, params=params, headers=headers, cookies=cookies)
    # 检查响应
    if response.status_code == 200:
        # 解析JSON响应
        data: dict = response.json()
        data: str = data.get("html", {})
        # print(data)
            
        # # 将完整响应保存到文件
        # with open("temp/test.html", "w", encoding="utf-8") as f:
        #     f.write(data)
        soup = BeautifulSoup(data, "html.parser")
        comments = soup.select("div.comment")
        for com in comments:
            like = com.select_one("h3 > span.comment-vote > span.votes.vote-count").text.strip()
            user_name = com.select_one("h3 > span.comment-info > a").text.strip()
            user_id = com.select_one("h3 > span.comment-info > a")["href"].split("/")[-2]
            time = com.select_one("h3 > span.comment-info > span.comment-time")["title"].strip()
            try:
                rating_classes = com.select_one("h3 > span.comment-info > span.rating")["class"]
                if rating_classes: rating_class = rating_classes[0]
                else: rating_class = "-1"
                rating = rating_class[-2:]
            except:
                rating = "-1"  # 如果没有评分，则设置为-1
            location = com.select_one("h3 > span.comment-info > span.comment-location").text.strip() if com.select_one("h3 > span.comment-info > span.comment-location") else ""
            content = com.select_one("p.comment-content > span.short").text.strip().replace("\n", " ")
            df_temp = pd.DataFrame([{
                "user_name": user_name,
                "user_id": user_id,
                "time": time,
                "rating": rating,
                "location": location,
                "content": content,
                "like": like
            }])
            df = pd.concat([df, df_temp], ignore_index=True)
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(response.text)
        raise Exception(f"获取评论失败，状态码: {response.status_code}")
    return df

def get_all_comments(mv_id=35595615, start=0, limit=100) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            "user_name", "user_id", "time", "rating", "location", "content", "like"
        ]
    )
    for i in range(start, limit, 20):
        # print(f"正在获取评论 {i} 到 {i + 20}")
        df_temp = get_comments(mv_id, i)
        df = pd.concat([df, df_temp], ignore_index=True)
    return df

def get_movies_by_year(start=0, count=20, year="2025") -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            "id", "title", "rating", "year", "url"
        ]
    )
    
    url = "https://m.douban.com/rexxar/api/v2/movie/recommend"
    
    params = {
        "refresh": 0,
        "start": start,
        "count": count,
        "selected_categories": '{"类型":""}',
        "uncollect": "false",
        "score_range": "0,10",
        "tags": str(year)
    }
    
    # 请求头
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7",
        "priority": "u=1, i",
        "referer": "https://movie.douban.com/subject/36053256/comments?start=20&limit=20&status=P&sort=new_score",
        "sec-ch-ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": get_random_user_agent()
    }

    # Cookie
    cookies = {
        "ll": "\"108296\"",
        "bid": "YpcQiOepiFU",
        "dbcl2": "\"289178155:GdwD5dUhMqU\"",
        "ck": "YUys",
        "push_noty_num": "0",
        "push_doumail_num": "0",
        "frodotk_db": "\"c78898b3f5516c9f5699ca1bdff445dd\""
    }
    
    response = requests.get(url, params=params, headers=headers, cookies=cookies)
    
    if response.status_code == 200:
        data = response.json()
        items = data.get("items", [])
        
        for item in items:
            movie = item
            df_temp = pd.DataFrame([{
                "id": movie.get("id"),
                "title": movie.get("title"),
                "rating": movie.get("rating", {}).get("value", 0),
                "year": year,
                "url": movie.get("url")
            }])
            df = pd.concat([df, df_temp], ignore_index=True)
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(response.text)
    
    return df

def get_all_mvs(start=0):
    df_all = None
    for year in range(2010, 2026):
        print(f"正在获取 {year} 年的电影数据...")
        time.sleep(1)
        df = get_movies_by_year(start=start, count=1200, year=year)
        print(f"获取到 {len(df)} 部电影")
        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)
        df_all.to_csv(f"new_data/movies.csv", index=False, encoding='utf-8-sig')

def check_if_boxoffice():
    df = pd.read_csv("new_data/movies.csv", encoding='utf-8-sig')
    df_boxoffice = pd.read_csv("new_data/maoyan_box_office_all.csv", encoding='utf-8-sig')
    df = df[df['title'].isin(df_boxoffice['Title'])]
    df = df.merge(df_boxoffice, left_on='title', right_on='Title', how='inner')
    df.to_csv("new_data/movies_with_boxoffice.csv", index=False, encoding='utf-8-sig')

def get_details(cooldown=0.5):
    df = pd.read_csv("new_data/movies_with_boxoffice.csv", encoding='utf-8-sig')
    df_res = pd.DataFrame(
        columns=[
            "id", "title", "year", "rating_count", "rating_score",
            "tags", "box_off"
        ]
    )
    # 尝试加载已存在的详情数据
    try:
        df_res = pd.read_csv("new_data/details.csv", encoding='utf-8-sig')
        print(f"已加载 {len(df_res)} 条电影详情数据")
        # 从df中删除已存在的电影
        df = df[~df['id'].isin(df_res['id'])]
    except FileNotFoundError:
        print("未找到已有的电影详情数据，将从头开始获取")
    # 使用进度条
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="获取电影详情"):
        # print(f"正在获取电影 {row['title']} 的详情...")
        time.sleep(cooldown)
        df_temp = get_detail_by_id(row['id'])
        if df_temp.empty:
            df_res.to_csv("new_data/details.csv", index=False, encoding='utf-8-sig')
            raise Exception("获取电影 {row['title']} 的详情失败，可能是网络问题")
        df_temp['box_off'] = row['Box Office']
        df_res = pd.concat([df_res, df_temp], ignore_index=True)
        # 每隔10条数据保存一次
        if (index + 1) % 10 == 0:
            df_res.to_csv("new_data/details.csv", index=False, encoding='utf-8-sig')
            # print(f"已保存 {len(df_res)} 条电影详情数据")
    df_res.to_csv("new_data/details.csv", index=False, encoding='utf-8-sig')

def get_detail_by_id(id=36282639)-> pd.DataFrame:
    "to get meta data of movies"
    df = pd.DataFrame(
        columns=[
            "id", "title", "year", "rating_count", "rating_score",
            "tags",
        ]
    )
    # df = pd.read_csv("new_data/movies_with_boxoffice.csv", encoding='utf-8-sig')
    url = f"https://movie.douban.com/subject/{id}/"
    
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7",
        "cache-control": "max-age=0",
        "priority": "u=0, i",
        "referer": "https://movie.douban.com/explore?support_type=movie&is_all=false&category=%E7%83%AD%E9%97%A8&type=%E5%85%A8%E9%83%A8",
        "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    }
    
    cookies = {
        "ll": '"108296"',
        "bid": "YpcQiOepiFU",
        "ap_v": "0,6.0"
    }
    
    res = requests.get(url, headers=headers, cookies=cookies)
    # print(f"获取页面 {url} 的状态码: {res.status_code}")
    # with open("temp/test.html", "w", encoding="utf-8") as f:
    #     f.write(res.text)
    # print("已将页面内容保存到 temp/test.html")
    try:
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            data = soup.select_one("script[type='application/ld+json']").text.strip()
            # 处理可能的非法JSON字符
            data = data.replace("\n", "").replace("\r", "")
            data = json.loads(data)
            rating_data = data.get("aggregateRating", {})
            rating_count = rating_data.get("ratingCount", 0)
            if rating_count == "":
                rating_count = 0
            rating_value = rating_data.get("ratingValue", 0)
            if rating_value == "":
                rating_value = 0
            tags = soup.select("span[property='v:genre']")
            tags = [tag.text for tag in tags]
            df = pd.DataFrame([{
                "id": id,
                "title": data.get("name", ""),
                "year": data.get("datePublished", "").split("-")[0],
                "rating_count": rating_count,
                "rating_score": rating_value,
                "tags": str(tags)
            }])
            return df
        else:
            print(f"请求失败，状态码: {res.status_code}")
            return df
    except Exception as e:
        print(f"解析数据时出错: {e}")
        print(f"id={id}")
        return df
    
def get_all_movies_coms(cooldown=0.5):
    df = pd.read_csv("new_data/movies_with_boxoffice.csv", encoding='utf-8-sig')
    df_res = pd.DataFrame(
        columns=[
            "user_name", "user_id", "time", "rating", "location", "content", "like",
            "mv_id"
        ]
    )
    # 尝试加载已存在的评论数据
    try:
        df_res = pd.read_csv("new_data/movies_coms.csv", encoding='utf-8-sig')
        print(f"已加载 {len(df_res)} 条电影评论数据")
        # 从df中删除已存在的电影
        df = df[~df['id'].isin(df_res['mv_id'])]
    except FileNotFoundError:
        print("未找到已有的电影评论数据，将从头开始获取")
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="获取电影评论"):
        mv_id = row['id']
        # print(f"正在获取电影 {row['title']} 的评论...")
        time.sleep(cooldown)
        df_temp = get_all_comments(mv_id=mv_id, start=0, limit=100)
        if df_temp.empty:
            print(f"电影 {row['title']} 没有评论")
            continue
        df_temp['mv_id'] = mv_id
        df_res = pd.concat([df_res, df_temp], ignore_index=True)
        # 每隔10条数据保存一次
        if (index + 1) % 10 == 0:
            df_res.to_csv("new_data/movies_coms.csv", index=False, encoding='utf-8-sig')
            # print(f"已保存 {len(df_res)} 条电影评论数据")
    
if __name__ == "__main__":
    # mv_id = 35595615  # 替换为你想要获取评论的电影ID
    # year = 2018  # 替换为你想要获取的年份
    # start = 0  # 起始索引
    # get_all_mvs()
    # check_if_boxoffice()
    cooldown = 2.5
    while True:
        try:
            # get_details(cooldown=cooldown)
            get_all_movies_coms(cooldown=cooldown)
            break
        except Exception as e:
            print(f"发生错误: {e}")
            cooldown += 1
            print("正在重试...")
            time.sleep(300)
    # print(get_detail_by_id(4271894))