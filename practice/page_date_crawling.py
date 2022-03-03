import requests
from bs4 import BeautifulSoup
from crawling import crawler
from cate_crawling import news_20_crawling
import re
import pandas as pd
import datetime

# data_frame = []
# date = datetime.datetime.now()
# yesterday = date - datetime.timedelta(days=1)
# date = yesterday.strftime('%Y%m%d')
# print(date)
# page = 1
# url = f'https://news.naver.com/main/list.naver?mode=LS2D&mid=shm&sid1=101&sid2=259&date={int(date)}&page={page}'
#
# headers = {
#         'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
#     }
#
# resp = requests.get(url, headers=headers)
# soup = BeautifulSoup(resp.content, 'lxml')
# print(soup)

# news_list_select = soup.select('body > div > table > tr > td > div > div > ul > li > dl > dt:nth-child(1) > a')
# print(news_list_select)
#
# url_list = []
# for i in news_list_select:
#     url = i['href']
#     url = url.replace('amp;','')
#     url_list.append(url)
# print(url_list)
#
# code = re.findall(r'\d+', url_list[0])
# print(code)
#
# code_list = []
# for idx in range(len(url_list)):
#     df = crawler(url_list[idx], data_frame)
#
#     sid1 = re.findall(r'\d+', url_list[idx])[2]
#     sid2 = re.findall(r'\d+', url_list[idx])[4]
#     oid = re.findall(r'\d+', url_list[idx])[5]
#     aid = re.findall(r'\d+', url_list[idx])[6]
#     code_list.append([sid1, sid2, oid, aid])
#
# _code = pd.DataFrame(code_list, columns=['sid1', 'sid2', 'oid', 'aid'])
# df['url'] = url_list
#
# df = pd.concat([_code, df], axis=1)
# df.to_csv('test.csv', index=False, encoding='utf-8')
#
# print(df)

########################################################################################################################

'''
    1. 하루 날짜의 모든 페이지 ==> 컬럼 date, page 추가
    2. 날짜별 / 카테고리별 
    3. 멀티프로세싱 
'''

# page = soup.find('div', attrs={'class':'paging'}).find_all('a', attrs={'class':'nclicks(fls.page)'})
# print(page[-1])

data_frame = []

while True:

    days = 0
    page = 1
    date = datetime.datetime.now()
    yesterday = date - datetime.timedelta(days=days)
    date = yesterday.strftime('%Y%m%d')
    # print(date)
    url = f'https://news.naver.com/main/list.naver?mode=LS2D&mid=shm&sid1=101&sid2=259&date={int(date)}&page={int(page)}'

    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
    }

    resp = requests.get(url, headers=headers)
    print(resp)
    soup = BeautifulSoup(resp.content, 'lxml')

    page = soup.find('div', attrs={'class':'paging'}).find_all('a', attrs={'class':'nclicks(fls.page)'})
    print(page[-1])

    for page in page:
        if len(page) == 10:
            page = int(page.text)
            url = f'https://news.naver.com/main/list.naver?mode=LS2D&mid=shm&sid1=101&sid2=259&date={int(date)}&page={page}'
            news_20_crawling(url, headers)
            if page.text == '다음':
                pass

        else:
            url = f'https://news.naver.com/main/list.naver?mode=LS2D&mid=shm&sid1=101&sid2=259&date={int(date)}&page={page}'
            news_20_crawling(url, headers)
