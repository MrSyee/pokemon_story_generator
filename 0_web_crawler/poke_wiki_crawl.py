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
            if rounded == []:
                rounded = rounded_all[8]('td', class_="rounded")
                if rounded == []:
                    rounded = rounded_all[10]('td', class_="rounded")
                    if rounded == []:
                        rounded = rounded_all[12]('td', class_="rounded")

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
    pk_type_sub1 = ''
    pk_type_sub2 = ''

    for span in rounded:
        if pk_type_sub1 == '':
            pk_type_sub1 = span.get_text().strip()

        else:
            if span.get_text().strip() != '':
                pk_type_sub2 = span.get_text().strip()
            else:
                pk_type_sub2 = None

#     pk_type = ','.join(pk_type_sub)

    return pk_type_sub1, pk_type_sub2



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



def get_life(soup):
    
    for tag in soup.find_all('h2'):
        if tag.get_text().strip() == '생태':
            life_h2_tag = tag
            
    pk_life = ''
    pk_life_sub = []
    try:
        for element in life_h2_tag.next_elements:
            if element.name == 'h2':
                break

            if element.name == 'p':
                pk_life_sub.append(element.get_text().strip())
            elif element.name == 'li':
                    pk_life_sub.append(element.get_text().strip())
    except:
        pass
    
    pk_life = ' '.join(pk_life_sub)
    
#     if pk_life == "":
#         if '생태' in soup.find_all('meta',property="og:description")[0].attrs['content']:
#             pk_life = soup.find_all('meta',property="og:description")[0].attrs['content'].split('생태')[1]
#         else:
#             pk_life = soup.find_all('meta',property="og:description")[0].attrs['content']
                
    if pk_life == "":
        print("Life Null : [{}] {}".format(i, pk_names[i]))

    return pk_life


# 1. url 불러오기
url_idx =1 # 전국도감 1세대 ~ 7세대까지
url_number = 0 # file 저장용
while url_idx < 8:
    
    print("Get different url : {}".format(url_idx))
    url = "http://ko.pokemon.wikia.com/wiki/%EC%A0%84%EA%B5%AD%EB%8F%84%EA%B0%90/{}%EC%84%B8%EB%8C%80".format(url_idx)
    url_number = url_idx
    
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
    pk_type1 = []
    pk_type2 = []
    egg_groups = []
#     pk_life = []
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
        pk_type_sub1, pk_type_sub2 = get_type(rounded_all)
        pk_type1.append(pk_type_sub1)
        pk_type2.append(pk_type_sub2)

        # 알 그룹
        egg_group = get_egggroup(rounded_all)
        egg_groups.append(egg_group)
        
        # 생태
#         desc = desc + " " +get_life(soup)
#         pk_life.append(life)
#         pk_desc.append(desc)

        i += 1

    # 4. data csv로 저장
    DATA_PATH = "./data/"

    pk_data = pd.DataFrame()
    pk_data['name'] = pk_names
    pk_data['desc'] = pk_desc
    pk_data['type1'] = pk_type1
    pk_data['type2'] = pk_type2
    pk_data['egg_group'] = egg_groups
#     pk_data['life'] = pk_life

    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    # pk_data.to_csv(DATA_PATH + "pk_data_g1.csv")
    pk_data.to_csv(DATA_PATH + "pk_data_g{}.csv".format(url_number), index=False, quotechar='"', encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)

    url_idx += 1 # 전국도감 url 변경