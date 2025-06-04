import requests
import json
import pandas as pd
import time
from tqdm import tqdm
import os
import re

def get_headers(year, tab):
    """根据年份和标签生成请求头"""
    # headers = {
    #     "accept": "*/*",
    #     "accept-language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7",
    #     "priority": "u=1, i",
    #     "referer": "https://piaofang.maoyan.com/rankings/year",
    #     "sec-ch-ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
    #     "sec-ch-ua-mobile": "?0",
    #     "sec-ch-ua-platform": "\"Windows\"",
    #     "sec-fetch-dest": "empty",
    #     "sec-fetch-mode": "cors",
    #     "sec-fetch-site": "same-origin",
    #     "uid": "d92bff268212272496c8f61c38f5eeb1cb93aa02",
    #     "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    #     "x-requested-with": "XMLHttpRequest"
    # }
    
    # 添加mygsig - 每个年份和标签对应不同的值
    mygsig_map = {
        2025: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"548f50f323c290ba65a45724cf45bf61","ts":1749029087654,"ts1":1749029082286},
        2024: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"e300ada573295861f02ebb9c104f3434","ts":1749029091401,"ts1":1749029082286},
        2023: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"22cbccbbc2a2411bd86d7cadd0905c4c","ts":1749029094520,"ts1":1749029082286},
        2022: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"a392c8e2e6265f9864860c6b4a557d9b","ts":1749029096867,"ts1":1749029082286},
        2021: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"6fd68c527487ad4cb6b7af8d0bb18654","ts":1749029098618,"ts1":1749029082286},
        2020: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"7cc1023556471748b32595dd484f6b0b","ts":1749029100970,"ts1":1749029082286},
        2019: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"84e3aeddcd03e6e672139e3849d42994","ts":1749029103181,"ts1":1749029082286},
        2018: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"5d7286f6d086fd2f5ea9506dc935bae6","ts":1749029105400,"ts1":1749029082286},
        2017: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"7e99089c89ed59094dcc710ed28faff2","ts":1749029107311,"ts1":1749029082286},
        2016: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"7e9d9e038b0c36863b7afebe6f7d7164","ts":1749029109345,"ts1":1749029082286},
        2015: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"e5d04452c83c3aa97850000eb3f6182d","ts":1749029112025,"ts1":1749029082286},
        2014: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"d7213d1ca26ed60d739a5c5db564c79b","ts":1749029113939,"ts1":1749029082286},
        2013: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"875582029b490a3908f54296f0734dca","ts":1749029116040,"ts1":1749029082286},
        2012: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"b4cb7328b9179918bbe77d47666d6991","ts":1749029118102,"ts1":1749029082286},
        2011: {"m1":"0.0.2","m2":0,"m3":"0.0.57_tool","ms1":"520e374ad8dc198ba773b3901b18d095","ts":1749029120134,"ts1":1749029082286},
    }
    str_map = {
        2025: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"548f50f323c290ba65a45724cf45bf61\",\"ts\":1749029087654}",
        2024: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"e300ada573295861f02ebb9c104f3434\",\"ts\":1749029091401}",
        2023: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"22cbccbbc2a2411bd86d7cadd0905c4c\",\"ts\":1749029094520}",
        2022: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"a392c8e2e6265f9864860c6b4a557d9b\",\"ts\":1749029096867}",
        2021: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"6fd68c527487ad4cb6b7af8d0bb18654\",\"ts\":1749029098618}",
        2020: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"7cc1023556471748b32595dd484f6b0b\",\"ts\":1749029100970}",
        2019: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"84e3aeddcd03e6e672139e3849d42994\",\"ts\":1749029103181}",
        2018: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"5d7286f6d086fd2f5ea9506dc935bae6\",\"ts\":1749029105400}",
        2017: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"7e99089c89ed59094dcc710ed28faff2\",\"ts\":1749029107311}",
        2016: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"7e9d9e038b0c36863b7afebe6f7d7164\",\"ts\":1749029109345}",
        2015: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"e5d04452c83c3aa97850000eb3f6182d\",\"ts\":1749029112025}",
        2014: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"d7213d1ca26ed60d739a5c5db564c79b\",\"ts\":1749029113939}",
        2013: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"875582029b490a3908f54296f0734dca\",\"ts\":1749029116040}",
        2012: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"b4cb7328b9179918bbe77d47666d6991\",\"ts\":1749029118102}",
        2011: "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"520e374ad8dc198ba773b3901b18d095\",\"ts\":1749029120134}",
    }
    
    # headers["mygsig"] = json.dumps(mygsig_map.get(year, mygsig_map[2024]))
    
    # # 添加cookie
    # cookies = "_lxsdk_cuid=19720920d8ec8-0c2a7764dea9d38-26011f51-384000-19720920d8ec8; _lxsdk=19720920d8ec8-0c2a7764dea9d38-26011f51-384000-19720920d8ec8; _lx_utm=utm_source%3DBaidu%26utm_medium%3Dorganic; _lxsdk_s=1973a41dcc5-562-d16-8c2%7C%7C1"
    # headers["Cookie"] = cookies

    headers = {
        "Uid": "d92bff268212272496c8f61c38f5eeb1cb93aa02",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "mygsig": str_map.get(year, str_map[2024]),
    }
    
    return headers

def fetch_year_box_office(year):
    """获取指定年份的票房数据"""
    # 年份和标签的映射关系
    tab_map = {
        2025: 1, 2024: 2, 2023: 3, 2022: 4, 2021: 5, 2020: 6, 2019: 7, 
        2018: 8, 2017: 9, 2016: 10, 2015: 11, 2014: 12, 2013: 13, 2012: 14, 2011: 15
    }
    
    url = f"https://piaofang.maoyan.com/rankings/year?year={year}&limit=100&tab={tab_map[year]}&WuKongReady=h5"
    headers = get_headers(year, tab_map[year])
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 确保请求成功
        data = response.json()
        
        if data["yearList"]:
            return data["yearList"]
        else:
            print(f"获取{year}年数据失败: {data.get('msg', '未知错误')}")
            return []
    except Exception as e:
        print(f"获取{year}年数据时出错: {str(e)}")
        return []

def process_movie_data(data, year):
    """处理电影数据，返回格式化的列表"""
    df = pd.DataFrame(columns=["Title", "Box Office", "Average Price", "Average Attendance"])
    # data 目前是字符串，需要用正则
    pattren_item = r'<ul class="row" .*?<p class="first-line">(.*?)</p>.*?<li class="col2 tr">([\d\.]+)</li>.*?<li class="col3 tr">([\d\.]+)</li>.*?<li class="col4 tr">(\d+)</li>'
    data = data.replace("\n", " ")
    items = re.findall(pattren_item, data)

    print(f"找到 {len(items)} 条票房数据")
    for item in items:
        title = item[0]
        box_office = item[1]
        average_price = item[2]
        average_attendance = item[3]
        
        # 添加到DataFrame using concat
        temp_serie = pd.Series({
            "Title": title,
            "Box Office": box_office,
            "Average Price": average_price,
            "Average Attendance": average_attendance
        })
        df = pd.concat([df, temp_serie.to_frame().T], ignore_index=True)
    return df

def save_to_csv(data, filename="maoyan_box_office.csv"):
    """保存数据到CSV文件"""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"数据已保存至 {filename}, 共{len(df)}条记录")

def main():
    years = list(range(2011, 2026))  # 2011-2025年
    
    print("开始爬取猫眼电影年度票房数据...")
    
    # 创建保存目录
    if not os.path.exists("data"):
        os.makedirs("data")
    
    df = pd.DataFrame(columns=["Title", "Box Office", "Average Price", "Average Attendance"])
    # 使用tqdm显示进度
    for year in tqdm(years, desc="爬取年份"):
        data = fetch_year_box_office(year)
        if data:
            temp_df = process_movie_data(data, year)
            df = pd.concat([df, temp_df], ignore_index=True)
    
    # 所有年份的数据合并保存
    save_to_csv(df, "data/maoyan_box_office_all.csv")
    print("全部数据爬取完成!")

if __name__ == "__main__":
    main()