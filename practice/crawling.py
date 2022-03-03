import requests
import pandas as pd
from bs4 import BeautifulSoup


def crawler(url, data_frame):

    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
    }

    resp = requests.get(url, headers=headers)
    # print(f'resp: {resp}')

    soup = BeautifulSoup(resp.content, 'lxml')

    time = soup.find('span', attrs={'class':'t11'}).get_text()
    media = soup.find('img')['title']
    writer = soup.find('p', attrs={'class':'b_text'}).get_text().strip()
    title = soup.find('title').get_text()

    raw_content = soup.find('div', attrs={'id':'articleBodyContents'})
    if raw_content.select_one('em'):
        raw_content.select_one('em').decompose()

    content = raw_content.text.replace('    ', '').strip()

    print(f'time: {time}')
    print(f'media: {media}')
    print(f'writer: {writer}')
    print(f'title: {title}')
    print(f'content: {content}\n')
    data_frame.append([time, media, writer, title, content])

    df = pd.DataFrame(data_frame, columns=['time', 'media', 'writer', 'title', 'content'])
    # print(df)

    return df

# print(raw_content.find('em', attrs={'class':'img_desc'}))
# em = raw_content.find('em', attrs={'class':'img_desc'}).decompose()
# print(em)
# if content.find('em'):


# print('\n')

# time = soup.find('span', attrs={'class':'t11'}).get_text()
# print(time)
#
#
# print(media)
#main_content > div.article_header > div.press_logo > a > img
# <img src="https://mimgnews.pstatic.net/image/upload/office_logo/001/2017/12/18/logo_001_38_20171218155018.png" height="35" alt="연합뉴스" title="연합뉴스" class="">
# //*[@id="main_content"]/div[1]/div[1]/a/img