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

        data_frame.append([i+22543, df['store_addr'][i][:2], store, store_x, store_y, store_addr, store_addr_new, store_tel, open_hours, n_link, website])
        dataset = pd.DataFrame(data_frame, columns=['store_id', 'region', 'store_name', 'store_x', 'store_y', 'store_addr', 'store_addr_new', 'store_tel', 'open_hours', 'n_link', 'website'])
        print(store)

    def action_naver_store_info(self, df):

        print('-----store crawling 시작-----')

        data_frame = []
        '''
            json.load를 계속 호출하다 보면 api 사용을 막아버림.
            time sleep을 랜덤으로 설정해주면 오래걸리지만 막히지 않음.
        '''
        for i in tqdm(range(1,10)):
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
                       'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36',
                       'cookie': 'NNB=YYF7WTTONT7GA; NID_AUT=6XDDHNp+RcKAoZR4X6WEGoEfGvfj/yinwhkuRt1FzzhwIWk1O2ghLoG+HBCOEkOf; NID_JKL=CcQOovGLIQDJuVb+/+nCI5UhKmVJKR84vJlqN/fIt9o=; _ga=GA1.2.878040348.1627623526; ASID=dc5f3dbd0000017bc9a08e5f0000005a; NV_WETR_LOCATION_RGN_M="MDUyMDA2MjQ="; NV_WETR_LAST_ACCESS_RGN_M="MDUyMDA2MjQ="; m_loc=568796a9a798b031c79ce34c474a916b745a603ef49a3efae027330981851c165c45d208eb010b902a04b8027ba06bef; NFS=2; MM_NEW=1; BMR=s=1635143616649&r=https%3A%2F%2Fm.blog.naver.com%2FPostView.naver%3FisHttpsRedirect%3Dtrue%26blogId%3Dmck0903%26logNo%3D221442957432&r2=https%3A%2F%2Fwww.google.com%2F; page_uid=hUpiasp0JywssLOingosssssthC-510028; NID_SES=AAABpbD2qjHEUIH3pWB4k7ykWkstFCWkjxSXmzGoQA6z5h2HqI5xJKRYWq3L1In+Id8MLlQkomzzt4MoE26CuvOcQ4xaL12x6Xj6jbavekPWZhilepZfZkmlLJDpHI1mxWu3QJPFb4dltRh4ZuFMNpsy149f2enHKaQv5VxOwUaunoEz4A6/VHB8lAlR3US2J1bdWvnhx0YcjtTGPJwYad32ygfxSxNDwLI57+StZkeTh+hXyTQPDadd5fo7FS7g6RwHf3Xo999b8ub9F43Z9u0Ua+5D9+qxdvZReRVdNVRsR45CvAG/poMR2XAw0bj8vSgHh5dx62dFZKB763UKFPTm+l8/APzys2yvohX/nbjbALdv43XO/nQVoSjjpBYrfTni4jG5OvjFkPCqJAFP6kNhsrqGOFnc7BSDlAvSrnutBgLwrnUvDFn4Lnbdou3Vxx3m9KUflIpitWBxqHCYg7c4iAl7rVsn9b7iZm8HRa/GxG8JiXIZ7iJZ+zlBunEM2nQRgGN0kkjWwawfZFynjVqkcGVM02fxavo4jr9Z0AkMcGs+EzYdJmEHTxv9NzE4QHY9gA==; csrf_token=0462579b6db64a53558dd357d5d0f718c775ca6ca9ee0a51f671bfc170779510819d3335dabf0700e10fd6a8a79b8a5ccff05683e8c47095049a248ab522e86a; page_uid=f332993b-0836-4b3c-86a2-821516a4cc7b',
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