# try / except 구문에 쓰일 공통 인자들 함수 생성

import requests
import json
import pandas as pd
import time
import numpy as np
from tqdm import tqdm

class Crawler:
    def to_csv(self, df, load, data_frame, i, idx):

        global dataset

        info_idx = load['result']['place']['list'][idx]

        # 컬럼값 명시
        store = info_idx['name']
        store_x = info_idx['x']
        store_y = info_idx['y']
        store_addr = info_idx['address']
        store_addr_new = info_idx['roadAddress']
        store_tel = info_idx['tel']
        open_hours = info_idx['bizhourInfo']
        n_link = info_idx['id']
        website = info_idx['homePage']

        data_frame.append([int(i+22543), df['store_addr'][i][:2], store, store_x, store_y, store_addr, store_addr_new, store_tel, open_hours, n_link, website])
        dataset = pd.DataFrame(data_frame, columns=['store_id', 'region', 'store_name', 'store_x', 'store_y', 'store_addr', 'store_addr_new', 'store_tel', 'open_hours', 'n_link', 'website'])
        print(store)

    def action_naver_store_info(self, df):

        print('-----store crawling 시작-----')

        data_frame = []
        '''
            json.load를 계속 호출하다 보면 api 사용을 막아버림.
            time sleep을 랜덤으로 설정해주면 오래걸리지만 막히지 않음.
        '''
        for i in tqdm(range(1200,1220)):
            time.sleep(np.random.randint(0, 3))

            '''
                store_name : 망향비빔국수 본점
                store_address: 경기도 연천군 청산면 궁평리 231-2
                - 예를 들어, "망향 비빔국수 경기도 연천군 청산면" 을 입력으로 두면,
                  한 번에 정확한 주소로 들어가짐.
                  그냥 "망향 비빔국수 본점" 으로 입력하게 되면 네이버지도상 가게 리스트가 나와서 한 번의 과정을 더 거쳐야 함.
                  map.naver.com에서 확인해볼 수 있음.
            '''

            df['new_name'] = df['store_name'][i] + ' ' + df['store_addr'][i][:12]
            new_name = df['new_name'][i]

            headers = {
                        'authority': 'map.naver.com',
                        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"',
                        'pragma': 'no-cache',
                        'accept-language': 'ko-KR,ko;q=0.8,en-US;q=0.6,en;q=0.4',
                        'sec-ch-ua-mobile': '?0',
                        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
                        'content-type': 'application/json',
                        'accept': 'application/json, text/plain, */*',
                        'expires': 'Sat, 01 Jan 2000 00:00:00 GMT',
                        'cache-control': 'no-cache',
                        'sec-ch-ua-platform': '"Windows"',
                        'sec-fetch-site': 'same-origin',
                        'sec-fetch-mode': 'cors',
                        'sec-fetch-dest': 'empty',
                        'referer': 'https://map.naver.com/',
                        'cookie': 'NNB=P52ZOJCX3GZGC; nx_ssl=2; nid_inf=19877329; NID_AUT=oQd6mkpDrJs1vkdeycVkS5391jnY0PYll0aYYoBV3gkIN7++K31ixOSTvYkXlAHZ; NID_JKL=evk+1JoX3dwNkQ6VMKWC5Pq3IhlOSyzWRtMt5hjHSYI=; csrf_token=c1dc8dbab511448d72e52b6dfcdb3e700a687ac213fb8fbcbbab741cf8a6b7719939ee141a8706275fea0df0bb8e6bef007849179b7307d26f657c9bc1d7bb1b; page_uid=hO71xdp0J1sssCPoGSRssssssKK-085086; BMR=s=1641402554641&r=https%3A%2F%2Fm.blog.naver.com%2FPostView.naver%3FisHttpsRedirect%3Dtrue%26blogId%3Dlovejhs96%26logNo%3D220447981320&r2=https%3A%2F%2Fwww.google.com%2F; NID_SES=AAABp1aqxXd0QM8Sdl55MC/eNDbEPmUzJtLrlhoKkhjn26qBiMOmFGtth2XNR3llkA2YeXgWw2c38QXzsFtW9IhQDgX3sx3827HACACDO4F3wTMYxafrMwvapARhcErPtki/Hje/YZKtv/UGK44s2cJ7qjdIcTo2MGT35EQaCbUpmUj42FzkiTn9FblJ/3AEx0HLXA/qd6AOLQtxFMUHH00ewt5cNh9wPtPz/3q4LUzyNXsnz46VmkF6R5kG+mmbu4fTl7zBHhjMmvDyuFgu2pwlRQJGuRy6fZgfSJgRKYy+G7G9Rp6m/gOIoB9iyxyOPbwhCG7cIhcA6uUEx5YpZTWCTiDHfgjiIIaNL5jimJxA+ezAwfoSHyvr8jn+0EBOwBw4yg7fkpv/RMjxmabyTgjAnGNYU6+Val0sEZls0UuBPdcmWy+jBuWh1l1Qus5Nd7zZnNxiuK5boH94jPLhvJhYsQxIEQidd+AkJCRbWcFpgJViyiwwwCk12QEjpVZ3tXAjYXL3HtdH/ED1RTQvKRDsQR5yHOlJYZIWlBtBxg9We3dpC8r01FyekFBCNP0X9Jcunw==; page_uid=1e253746-0306-4331-be4d-9684e5aeb471',
}

            params = (
                        ('query', new_name),
                        ('page', '1'),
                        ('displayCount', '20'),
                    )
            response = requests.get('https://map.naver.com/v5/api/search', headers=headers, params=params)
            load = json.loads(response.text)

            try:
                try:
                    try:
                        # storeInfo_1.csv 상의 주소가 siksin에서 크롤링한 주소임
                        # 네이버, 구글상 주소와 다른 곳이 많음
                        # 따라서 '경기도 연천군 연천읍' 까지 1차 슬라이싱

                        csv_addr = df['store_addr'][i][:12]

                        for idx in range(len(load['result']['place']['list'])):
                            store_addr = load['result']['place']['list'][idx]['address'][:12]

                            if store_addr == csv_addr:
                                self.to_csv(df, load, data_frame, i, idx)
                                break
                            # 주소가 가지각색이라 조건문 추가
                            elif store_addr[:15] == csv_addr[:15]:
                                self.to_csv(df, load, data_frame, i, idx)
                                break
                            elif store_addr[:7] == csv_addr[:7]:
                                self.to_csv(df, load, data_frame, i, idx)
                                break
                            elif store_addr[:10] == csv_addr[:10]:
                                self.to_csv(df, load, data_frame, i, idx)
                                break
                    # 식당 이름 + 주소 전체로 입력값
                    except:
                        new_name = df['store_name'][i] + ' ' + df['store_addr'][i]
                        params = (
                            ('query', new_name),
                            ('page', '1'),
                            ('displayCount', '20'),
                        )
                        response = requests.get('https://map.naver.com/v5/api/search', headers=headers, params=params)
                        load = json.loads(response.text)
                        self.to_csv(df, load, data_frame, i, idx)
                # 송학원한방삼계탕누룽지백숙 -> 송하원 누룽지 한방 백숙
                except:
                    new_name = df['store_name'][i][:8]
                    params = (
                        ('query', new_name),
                        ('page', '1'),
                        ('displayCount', '20'),
                    )
                    response = requests.get('https://map.naver.com/v5/api/search', headers=headers, params=params)
                    load = json.loads(response.text)

                    csv_addr = df['store_addr'][i][:12]
                    web_name = df['store_name'][i] + ' ' + df['store_addr'][i]

                    for idx in range(len(load['result']['place']['list'])):
                        store_addr = load['result']['place']['list'][idx]['address'][:12]


                        if store_addr == csv_addr:
                            self.to_csv(df, load, data_frame, i, idx)
                            break
                        elif store_addr[:15] == csv_addr[:15]:
                            self.to_csv(df, load, data_frame, i, idx)
                            break
            except:
                data_frame.append('')
        time.sleep(np.random.randint(0, 3))
        return dataset