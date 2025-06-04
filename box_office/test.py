import requests
import pandas as pd
import json, re

# test 
# 请求URL和参数
url = "https://piaofang.maoyan.com/rankings/year"
params = {
    "year": "2025",
    "limit": "100",
    "tab": "1",
    "WuKongReady": "h5"
}


# 请求头
headers = {
    "Uid": "d92bff268212272496c8f61c38f5eeb1cb93aa02",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    # "mygsig": "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"246948f692261f2d0076fcd80d0fba28\",\"ts\":1749027830613}",
    "mygsig": "{\"m1\":\"0.0.2\",\"m2\":0,\"m3\":\"0.0.57_tool\",\"ms1\":\"548f50f323c290ba65a45724cf45bf61\",\"ts\":1749029087654}",
}
response = requests.get(url, params=params, headers=headers)
try:
    data = response.json().get("yearList", {})
except:
    # 保存到snapshot/error.html
    with open("snapshot/error.html", "w", encoding="utf-8") as f:
        f.write(response.text)

print(data)

'''
data struct:
<ul class="row" data-com="hrefTo,href:'/movie/246063'">
    <li class="col0">1</li>
    <li class="col1">
        <p class="first-line">美人鱼</p>
        <p class="second-line">2016-02-08 上映</p>
    </li>
    <li class="col2 tr">338627</li>
    <li class="col3 tr">36.697346</li>
    <li class="col4 tr">44</li>
</ul>
'''
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
# save to csv
df.to_csv("data/box_office_test.csv", index=False, encoding="utf-8-sig")
# print(data)