import requests
import json

url = "https://movie.douban.com/subject/35595615/comments"

# 请求参数
params = {
    "percent_type": "",
    "start": 20,
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
    "referer": "https://movie.douban.com/subject/35595615/comments?start=20&limit=20&status=P&sort=new_score",
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

# 发送请求
response = requests.get(url, params=params, headers=headers, cookies=cookies)

# 检查响应
if response.status_code == 200:
    # 解析JSON响应
    data = response.json()
    print("成功获取数据")
    print(f"评论数量: {len(data.get('comments', []))}")
    
    # 打印部分评论内容示例
    for i, comment in enumerate(data.get("comments", [])[:3]):
        print(f"\n评论 {i+1}:")
        print(f"用户: {comment.get('user', {}).get('name', '未知')}")
        print(f"评分: {comment.get('rating', {}).get('value', '无评分')}")
        print(f"内容: {comment.get('content', '无内容')}")
        
    # 将完整响应保存到文件
    with open("temp/douban_comments.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        print("\n完整数据已保存到 douban_comments.json")
else:
    print(f"请求失败，状态码: {response.status_code}")
    print(response.text)