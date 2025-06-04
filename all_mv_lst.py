import requests
import pandas as pd
import id_register as ir

TEST = 0

url = "https://api.bilibili.com/pgc/season/index/result"
# area = 1: 大陆
# area = '6,7': 港台
params = {
    'style_id': -1,
    'area': '6,7',
    'release_date': -1,
    'season_status': -1,
    'order': 2,
    'sort': 0,
    'page': 1,
    'season_type': 2,
    'pagesize': 20,
    'type': 1
}
# https://api.bilibili.com/pgc/season/index/result?st=2&style_id=-1&area=6%2C7&release_date=-1&season_status=-1&order=2&sort=0&page=1&season_type=2&pagesize=20&type=1
headers = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7",
    "origin": "https://www.bilibili.com",
    "priority": "u=1, i",
    "referer": "https://www.bilibili.com/movie/index/?from_spmid=666.7.index.1",
    "sec-ch-ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
}

# try to open data/mv_lst.csv and load df
try:
    df = pd.read_csv('data/mv_lst.csv', encoding='utf-8-sig')
except FileNotFoundError:
    print("data/mv_lst.csv not found, creating a new one.")
    df = pd.DataFrame(columns=['id', 'title', 'link', 'media_id'])
# if df is empty, create a new one
if df.empty:
    df = pd.DataFrame(columns=['id', 'title', 'link', 'media_id'])

page = 1
while True:
    # Update the page number in params
    params['page'] = page
    # Make the request to get the data
    response = requests.get(url, params=params, headers=headers)
    print(f"Fetching page {page}: {response.status_code}")
    if response.status_code != 200:
        print(f"Error fetching page {page}: {response.status_code}")
        break
    lst = response.json().get('data', {}).get('list', [])
    print(f"Number of items found on page {page}: {len(lst)}")
    if lst is None or len(lst) == 0:
        print(f"No more data found on page {page}.")
        break  # Exit the loop if no more data is found

    for item in lst:
        link = item.get('link', '')
        media_id = item.get('media_id', '')
        title = item.get('title', '')
        if TEST:
            print(f"TEST mode: Skipping item {title} (ID: {media_id})")
            continue
        if TEST or not ir.is_new(title):
            continue
        id = ir.get_id(title)
        # Create a temporary DataFrame for the current item
        temp_df = pd.DataFrame({
            'id': [id],
            'title': [title],
            'link': [link],
            'media_id': [media_id]
        })
        # Append the temporary DataFrame to the main DataFrame
        print(f"Processing item: {title} (ID: {id})")
        df = pd.concat([df, temp_df], ignore_index=True)      
    # logging
    print(f"Page {page} processed. Total records: {len(df)}")
    
    has_next = response.json().get('data', {}).get('has_next', False)
    if not has_next:
        print(f"No more pages to fetch after page {page}.")
        break  # Exit the loop if no more pages are available
    page += 1   

if not TEST:
    print("Saving data to data/mv_lst.csv")
    df.to_csv('data/mv_lst.csv', index=False, encoding='utf-8-sig')
