import requests
from bs4 import BeautifulSoup

def get_html(url):
    html = ""
    resp = requests.get(url)

    if resp.status_code == 200:
        html = resp.text

    return html

# 1. url 불러오기
url = "https://namu.wiki/w/%EB%B6%84%EB%A5%98:1%EC%84%B8%EB%8C%80%20%ED%8F%AC%EC%BC%93%EB%AA%AC"
c = get_html(url)

# 2. url 내의 세부 링크 받아오기
soup = BeautifulSoup(c, "html5lib")

pages = []
table = soup.find(class_="wiki-category-container")
for li in table('li'):
    for a in li('a'):
        print("name : {}    link : {}".format(a.get_text(), a['href']))
        pages.append("https://namu.wiki" + a['href'])

index = []
text_list = []

def get_description(table):
    # 도감 설명
    ptag = table.find_all("p")
    for pp in ptag:
        if pp.get_text().strip() == "도감설명" or pp.get_text().strip() == "도감 설명":
            for td in table('td'):
                for p in td('p'):
                    text = p.get_text().strip()
                    if len(text) > 10:  # 긴 문장만 크롤링
                        print(text)
                        index.append(i)
                        text_list.append(i)

def get_type(table):
    # tdTag = table.find_all("td")
    ptag = table.find_all("p")
    count = 0
    td_list = []
    for pp in ptag:
        if i == 58:
            continue
        if pp.get_text().strip() == "성비":
            for tr in table('tr'):
                for td in tr('td'):
                    count += 1
                if count > 4:  # 제일 기본 (이름, 도감번호, 성비, 타입)
                    txt = td.get_text().strip()
                    td_list.append(txt)
                    print(td_list[-1])
                count = 0


# 3. url 내의 필요한 부분 크롤링
i = 0
for page in pages:
    i += 1
    c = get_html(page)
    soup = BeautifulSoup(c, "html5lib")

    pk_name = soup.find(class_="wiki-document-title").get_text().strip()
    print("[{}] {}".format(i, pk_name))

    article = soup.find(class_="container-fluid wiki-article")
    tables = article.find_all(class_="wiki-table")

    for table in tables:
        # 도감 설명
        get_description(table)

        # 속성
        get_type(table)

        # 제일 기본 (포켓몬, 분류, 신장, 체중, 알그룹, 포획률)
        # count = 0
        # for td in tdTag:
        #     if td.get_text().strip() == "알 그룹" or td.get_text().strip() == "알그룹":
        #         for tr in table('tr'):
        #             for td in tr('td'):
        #                 count += 1
        #             if count > 5:
        #                 pass


    if i > 2:
        break


