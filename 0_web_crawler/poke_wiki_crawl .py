import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import csv


def get_html(url):
    html = ""
    resp = requests.get(url)

    if resp.status_code == 200:
        html = resp.text

    return html


def get_desc(rounded_all):
    rounded = rounded_all[2]('td', class_="rounded")

    if rounded == []:
        rounded = rounded_all[4]('td', class_="rounded")
        if rounded == []:
            rounded = rounded_all[6]('td', class_="rounded")

    desc = ""
    prev_d = ""
    for td in rounded:
        d = td.get_text().strip()
        if prev_d == d or d == '':
            continue
        prev_d = d
        desc += d
    if desc == "":
        print("Desc Null : [{}] {}".format(i, pk_names[i]))

    return desc


def get_type(rounded_all):
    rounded = rounded_all[0]('span', class_="split-cell text-white")
    pk_type = ''
    pk_type_sub = []
    for span in rounded:
        pk_type_sub.append(span.get_text().strip())
#         pk_type += span.get_text().strip()

    pk_type = ','.join(pk_type_sub)

    return pk_type


def get_egggroup(rounded_all):
    rounded = rounded_all[0]('td')
    egg_group = ''
    egg_group_sub = []
    for td in rounded:
        flag = False
        for a in td('a'):
            if flag:
                egg_group_sub.append(a.get_text().strip())
#                 egg_group += a.get_text().strip()
            if a.get_text().strip() == "알그룹":
                flag = True
    egg_group = ','.join(egg_group_sub)
    return egg_group

# 1. url 불러오기
url = "http://ko.pokemon.wikia.com/wiki/%EC%A0%84%EA%B5%AD%EB%8F%84%EA%B0%90/1%EC%84%B8%EB%8C%80"
c = get_html(url)

# 2. url 내의 세부 링크 받아오기
soup = BeautifulSoup(c, "html5lib")

pages = []
table = soup.find_all(class_="bg-white")

pk_names = []
prev_name = ""
for i in range(len(table)):
    pk_name = table[i]('td')[3].a.get('title')
#     pk_name = table[i]('td')[3].a['title']
    if prev_name == pk_name:
        continue
    prev_name = pk_name
    link = table[i]('td')[3].a.get('href')
#     link = table[i]('td')[3].a['href']
    link = "http://ko.pokemon.wikia.com" + link
    pk_name = pk_name.split(' ')[0]
    pk_names.append(pk_name)
    pages.append(link)

# 3. url 내의 필요한 부분 크롤링
i = 0
pk_desc = []
pk_types = []
egg_groups = []
print("Crawling Proceeding..")
for page in pages:
    print("[{}] {}".format(i, pk_names[i]))
    # page = pages[2]
    c = get_html(page)
    soup = BeautifulSoup(c, "html5lib")

    rounded_all = soup.find_all("div", class_="rounded")

    # 도감 설명
    desc = get_desc(rounded_all)
    pk_desc.append(desc)

    # 속성
    pk_type = get_type(rounded_all)
    pk_types.append(pk_type)

    # 알 그룹
    egg_group = get_egggroup(rounded_all)
    egg_groups.append(egg_group)

    i += 1

# 4. data csv로 저장
DATA_PATH = "./data/"

pk_data = pd.DataFrame()
pk_data['name'] = pk_names
pk_data['desc'] = pk_desc
pk_data['type'] = pk_types
pk_data['egg_group'] = egg_groups

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

# pk_data.to_csv(DATA_PATH + "pk_data_g1.csv")
pk_data.to_csv(DATA_PATH + "pk_data_g4.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)


