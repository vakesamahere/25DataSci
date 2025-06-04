import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_media_details(media_id):
    url = "https://api.bilibili.com/pgc/review/user"
    params = {
        'media_id': f'{media_id}',
        'ts': '1748588342515'
    }
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7",
        "origin": "https://www.bilibili.com",
        "priority": "u=1, i",
        "referer": "https://www.bilibili.com/bangumi/play/ep673045?theme=movie&from_spmid=666.7.feed.0",
        "sec-ch-ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    try:
        data = response.json()
        return data
    except:
        return None
    
def get_detail2(media_id):
    url = f"https://www.bilibili.com/bangumi/media/md{media_id}"
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7",
        "cache-control": "max-age=0",
        "priority": "u=0, i",
        "sec-ch-ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    }
    # get the page
    response = requests.get(url, headers=headers)
    # select div.media-info-r
    soup = BeautifulSoup(response.text, 'html.parser')
    # select the first div with class media-info-r
    media_info_div = soup.find('div', class_='media-info-r')
    # interface:
    # tags: list, play_count: str, fans: str, danmu: str, review_times:str, 
    if media_info_div:
        # select all tags: div.media-tag
        tags = media_info_div.select('span.media-tag')
        if tags:
            # extract the text from each tag
            tag_texts = [tag.get_text(strip=True) for tag in tags]
        else:
            # print(f"No tags found for media_id: {media_id}")
            tag_texts = ['无标签']
        # period
        period = media_info_div.find('div', class_='media-info-time')
        # 其中的最后一个span就是时长
        if period:
            duration = period.find_all('span')[-1].get_text(strip=True)
        else:
            print(f"No period found for media_id: {media_id}")
            return False
        # return a dict
        return {
          # 'media_id': [media_id],
          'tags': tag_texts,
          'duration': duration
        }
    else:
        print(f"No media info found for media_id: {media_id}")
        with open('snapshot.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        time.sleep(60)
        return False


df = pd.read_csv('data/mv_lst.csv', encoding='utf-8-sig')
try:
    media_details_df = pd.read_csv('data/details.csv', encoding='utf-8-sig')
except:
    print("data/details.csv not found, creating a new one.")
    media_details_df = pd.DataFrame(columns=['id', 'url', 'title', 'areas_ids', 'areas_names',
                                           'time', 'rating_count', 'rating_score',
                                           'season_id', 'ep_id', 'media_id', 'type_name', 'tags', 'duration'])
# if media_details_df is empty, create a new one
if media_details_df.empty:
    media_details_df = pd.DataFrame(columns=['id', 'url', 'title', 'areas_ids', 'areas_names',
                                           'time', 'rating_count', 'rating_score',
                                           'season_id', 'ep_id', 'media_id', 'type_name', 'tags', 'duration'])

# for each url in urls
for index, row in df.iterrows():
    id = row['id']
    url = row['link']
    title = row['title']
    media_id = row['media_id']

    if media_details_df['id'].eq(id).any():
        print(f"Skipping {title} as it has already been processed.")
        continue

    # get detail
    detail = get_media_details(media_id)
    detail2 = get_detail2(media_id)
    if detail and detail2:
        print(f"Processed {index + 1}/{len(df)}: {title} - Media ID: {media_id}")
        # create a df and concat to media_details_df
        result = detail.get('result', {}).get('media', {})
        if not result:
            print(f"No result found for {title}")
            continue
        areas = result.get('areas', [])
        area_ids = [area['id'] for area in areas]
        area_names = [area['name'] for area in areas]

        time_released = result.get('new_ep', {}).get('index_show', '未知时间')

        rating = result.get('rating', {})
        rating_count = rating.get('count', None)
        rating_score = rating.get('score', None)

        season_id = result.get('season_id', None)
        ep_id = result.get('new_ep', {}).get('id', None)

        title = result.get('title', '未知标题')
        type_name = result.get('type_name', '未知类型')

        key_map = {
            'id': id,
            'url': url,
            'title': title,

            'areas_ids': area_ids,
            'areas_names': area_names,

            'time': time_released,

            'rating_count': rating_count,
            'rating_score': rating_score,

            'season_id': season_id,
            'ep_id': ep_id,
            'media_id': media_id,

            'type_name': type_name,
            **detail2
        }
        temp_df = pd.DataFrame([key_map])
        # concat to media_details_df
        media_details_df = pd.concat([media_details_df, temp_df], ignore_index=True)
        # print newest media_details_df with beautiful format
        string = f"{id:<5} | {str(title)[:30]:<30} | {media_id:<10} | {', '.join(area_names or ['未知'])[:15]:<15} | {str(time_released or '未知')[:15]:<15} | {rating_count or 0:<10} | {rating_score or 0:<5} | {str(type_name or '未知')[:10]:<10} || {str(detail2.get('duration', '未知'))[:10]:<10} | {', '.join(detail2.get('tags', ['无标签']))}"
        print(string)

        media_details_df.to_csv('data/details.csv', index=False, encoding='utf-8-sig', escapechar='\\', quotechar='"', quoting=1)
    else:
        print(f"Failed to retrieve media ID for {title}")
    # sleep for 5 seconds to avoid being blocked
    time.sleep(1)