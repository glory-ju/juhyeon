import pandas as pd
import requests
import json
import time
import numpy as np
import math
import datetime


def action_naver_review_crawler(df):
    data_frame = []

    for idx in range(len(df)):

        # 네이버와 구글에서 api 사용을 위해선 time.sleep을 랜덤으로 줘야 함.
        # 경험상 0초에서 3초로 지정해줘야 막히지 않음.(2초도 막힘)
        # store가 바뀔 때마다 , display를 100개씩 crawling할 때마다 time.sleep을 지정해줘야 함(display는 아래에 나옴)
        time.sleep(np.random.randint(0, 3))

        headers = {
                'authority': 'pcmap-api.place.naver.com',
                'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"',
                'accept': '*/*',
                'content-type': 'application/json',
                'accept-language': 'ko',
                'sec-ch-ua-mobile': '?0',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
                'sec-ch-ua-platform': '"Windows"',
                'origin': 'https://pcmap.place.naver.com',
                'sec-fetch-site': 'same-site',
                'sec-fetch-mode': 'cors',
                'sec-fetch-dest': 'empty',
                'referer': 'https://pcmap.place.naver.com/restaurant/11718339/review/visitor',
                'cookie': 'NNB=P52ZOJCX3GZGC; nx_ssl=2; nid_inf=19877329; NID_AUT=oQd6mkpDrJs1vkdeycVkS5391jnY0PYll0aYYoBV3gkIN7++K31ixOSTvYkXlAHZ; NID_JKL=evk+1JoX3dwNkQ6VMKWC5Pq3IhlOSyzWRtMt5hjHSYI=; page_uid=hO71xdp0J1sssCPoGSRssssssKK-085086; BMR=s=1641402554641&r=https%3A%2F%2Fm.blog.naver.com%2FPostView.naver%3FisHttpsRedirect%3Dtrue%26blogId%3Dlovejhs96%26logNo%3D220447981320&r2=https%3A%2F%2Fwww.google.com%2F; NID_SES=AAABp1aqxXd0QM8Sdl55MC/eNDbEPmUzJtLrlhoKkhjn26qBiMOmFGtth2XNR3llkA2YeXgWw2c38QXzsFtW9IhQDgX3sx3827HACACDO4F3wTMYxafrMwvapARhcErPtki/Hje/YZKtv/UGK44s2cJ7qjdIcTo2MGT35EQaCbUpmUj42FzkiTn9FblJ/3AEx0HLXA/qd6AOLQtxFMUHH00ewt5cNh9wPtPz/3q4LUzyNXsnz46VmkF6R5kG+mmbu4fTl7zBHhjMmvDyuFgu2pwlRQJGuRy6fZgfSJgRKYy+G7G9Rp6m/gOIoB9iyxyOPbwhCG7cIhcA6uUEx5YpZTWCTiDHfgjiIIaNL5jimJxA+ezAwfoSHyvr8jn+0EBOwBw4yg7fkpv/RMjxmabyTgjAnGNYU6+Val0sEZls0UuBPdcmWy+jBuWh1l1Qus5Nd7zZnNxiuK5boH94jPLhvJhYsQxIEQidd+AkJCRbWcFpgJViyiwwwCk12QEjpVZ3tXAjYXL3HtdH/ED1RTQvKRDsQR5yHOlJYZIWlBtBxg9We3dpC8r01FyekFBCNP0X9Jcunw==',
}

        nan_value = float(df['n_link'][idx])

        # n_link 컬럼을 float으로 형변환해준 후 그 값이 nan값인지 아닌지 확인
        if math.isnan(nan_value) == False:
            businessid = str(int(df['n_link'][idx]))
        else:
            continue

        data = '[{"operationName":"getVisitorReviews","variables":{"input":{"businessId":"' + businessid + '","businessType":"restaurant","item":"0","bookingBusinessId":null,"page":1,"display":100,"isPhotoUsed":false,"includeContent":true,"getAuthorInfo":true}},"query":"query getVisitorReviews($input: VisitorReviewsInput) {\\n  visitorReviews(input: $input) {\\n    items {\\n      id\\n      rating\\n      author {\\n        id\\n        nickname\\n        from\\n        imageUrl\\n        objectId\\n        url\\n        review {\\n          totalCount\\n          imageCount\\n          avgRating\\n          __typename\\n        }\\n        theme {\\n          totalCount\\n          __typename\\n        }\\n        __typename\\n      }\\n      body\\n      thumbnail\\n      media {\\n        type\\n        thumbnail\\n        __typename\\n      }\\n      tags\\n      status\\n      visitCount\\n      viewCount\\n      visited\\n      created\\n      reply {\\n        editUrl\\n        body\\n        editedBy\\n        created\\n        replyTitle\\n        __typename\\n      }\\n      originType\\n      item {\\n        name\\n        code\\n        options\\n        __typename\\n      }\\n      language\\n      highlightOffsets\\n      apolloCacheId\\n      translatedText\\n      businessName\\n      showBookingItemName\\n      showBookingItemOptions\\n      bookingItemName\\n      bookingItemOptions\\n      votedKeywords {\\n        code\\n        displayName\\n        __typename\\n      }\\n      userIdno\\n      isFollowing\\n      followerCount\\n      followRequested\\n      loginIdno\\n      __typename\\n    }\\n    starDistribution {\\n      score\\n      count\\n      __typename\\n    }\\n    hideProductSelectBox\\n    total\\n    __typename\\n  }\\n}\\n"},{"operationName":"getVisitorReviews","variables":{"id":"11718339"},"query":"query getVisitorReviews($id: String) {\\n  visitorReviewStats(input: {businessId: $id}) {\\n    id\\n    name\\n    review {\\n      avgRating\\n      totalCount\\n      scores {\\n        count\\n        score\\n        __typename\\n      }\\n      starDistribution {\\n        count\\n        score\\n        __typename\\n      }\\n      imageReviewCount\\n      authorCount\\n      maxSingleReviewScoreCount\\n      maxScoreWithMaxCount\\n      __typename\\n    }\\n    analysis {\\n      themes {\\n        code\\n        label\\n        count\\n        __typename\\n      }\\n      menus {\\n        label\\n        count\\n        __typename\\n      }\\n      votedKeyword {\\n        totalCount\\n        reviewCount\\n        userCount\\n        details {\\n          category\\n          code\\n          displayName\\n          count\\n          previousRank\\n          __typename\\n        }\\n        __typename\\n      }\\n      __typename\\n    }\\n    visitorReviewsTotal\\n    ratingReviewsTotal\\n    __typename\\n  }\\n  visitorReviewThemes(input: {businessId: $id}) {\\n    themeLists {\\n      name\\n      key\\n      __typename\\n    }\\n    __typename\\n  }\\n}\\n"},{"operationName":"getVisitorReviewPhotosInVisitorReviewTab","variables":{"businessId":"11718339","businessType":"restaurant","item":"0","page":1,"display":10},"query":"query getVisitorReviewPhotosInVisitorReviewTab($businessId: String!, $businessType: String, $page: Int, $display: Int, $theme: String, $item: String) {\\n  visitorReviews(input: {businessId: $businessId, businessType: $businessType, page: $page, display: $display, theme: $theme, item: $item, isPhotoUsed: true}) {\\n    items {\\n      id\\n      rating\\n      author {\\n        id\\n        nickname\\n        from\\n        imageUrl\\n        objectId\\n        url\\n        __typename\\n      }\\n      body\\n      thumbnail\\n      media {\\n        type\\n        thumbnail\\n        __typename\\n      }\\n      tags\\n      status\\n      visited\\n      originType\\n      item {\\n        name\\n        code\\n        options\\n        __typename\\n      }\\n      businessName\\n      __typename\\n    }\\n    starDistribution {\\n      score\\n      count\\n      __typename\\n    }\\n    hideProductSelectBox\\n    total\\n    __typename\\n  }\\n}\\n"},{"operationName":"getVisitorRatingReviews","variables":{"input":{"businessId":"11718339","businessType":"restaurant","item":"0","bookingBusinessId":null,"page":1,"display":10,"includeContent":false,"getAuthorInfo":true},"id":"11718339"},"query":"query getVisitorRatingReviews($input: VisitorReviewsInput) {\\n  visitorReviews(input: $input) {\\n    total\\n    items {\\n      id\\n      rating\\n      author {\\n        id\\n        nickname\\n        from\\n        imageUrl\\n        objectId\\n        url\\n        review {\\n          totalCount\\n          imageCount\\n          avgRating\\n          __typename\\n        }\\n        theme {\\n          totalCount\\n          __typename\\n        }\\n        __typename\\n      }\\n      visitCount\\n      visited\\n      originType\\n      reply {\\n        editUrl\\n        body\\n        editedBy\\n        created\\n        replyTitle\\n        __typename\\n      }\\n      votedKeywords {\\n        code\\n        displayName\\n        __typename\\n      }\\n      businessName\\n      status\\n      userIdno\\n      isFollowing\\n      followerCount\\n      followRequested\\n      loginIdno\\n      __typename\\n    }\\n    __typename\\n  }\\n}\\n"}]'
        response = requests.post('https://pcmap-api.place.naver.com/graphql', headers=headers, data=data)

        load = json.loads(response.text)
        total_review = load[0]['data']['visitorReviews']['total']
        print(idx + 1, '/', total_review)

        # 1page당 보여지는 display 수는 100개씩임.
        # 그래서 아래의 for문의 범위 지정을 위한 iter_cnt라는 변수로 page 수를 지정.
        # total review 수를 100개씩 나눠서 보여지게끔 설정
        iter_cnt = int(total_review / 100) + 2

        if total_review < 100:
            display = str(total_review)
        else:
            display = str(100)

        try:
            for page in range(1, iter_cnt):
                print('---' * 30)

                '''
                    businessid 는 식당 고유의 link. ===> csv파일에서 n_link에 해당.
                    json파일로 불러왔을 때 page당 보여지는 리뷰 갯수(display)는 100개씩 보여짐
                        ===> businessid, page, display를 변수 설정
                '''
                data = '[{"operationName":"getVisitorReviews","variables":{"input":{"businessId":"' + businessid + '","businessType":"restaurant","item":"0","bookingBusinessId":null,"page":' + str(
                    page) + ',"display":' + display + ',"isPhotoUsed":false,"includeContent":true,"getAuthorInfo":true}},"query":"query getVisitorReviews($input: VisitorReviewsInput) {\\n  visitorReviews(input: $input) {\\n    items {\\n      id\\n      rating\\n      author {\\n        id\\n        nickname\\n        from\\n        imageUrl\\n        objectId\\n        url\\n        review {\\n          totalCount\\n          imageCount\\n          avgRating\\n          __typename\\n        }\\n        theme {\\n          totalCount\\n          __typename\\n        }\\n        __typename\\n      }\\n      body\\n      thumbnail\\n      media {\\n        type\\n        thumbnail\\n        __typename\\n      }\\n      tags\\n      status\\n      visitCount\\n      viewCount\\n      visited\\n      created\\n      reply {\\n        editUrl\\n        body\\n        editedBy\\n        created\\n        replyTitle\\n        __typename\\n      }\\n      originType\\n      item {\\n        name\\n        code\\n        options\\n        __typename\\n      }\\n      language\\n      highlightOffsets\\n      apolloCacheId\\n      translatedText\\n      businessName\\n      showBookingItemName\\n      showBookingItemOptions\\n      bookingItemName\\n      bookingItemOptions\\n      votedKeywords {\\n        code\\n        displayName\\n        __typename\\n      }\\n      userIdno\\n      isFollowing\\n      followerCount\\n      followRequested\\n      loginIdno\\n      __typename\\n    }\\n    starDistribution {\\n      score\\n      count\\n      __typename\\n    }\\n    hideProductSelectBox\\n    total\\n    __typename\\n  }\\n}\\n"},{"operationName":"getVisitorReviews","variables":{"id":"11718339"},"query":"query getVisitorReviews($id: String) {\\n  visitorReviewStats(input: {businessId: $id}) {\\n    id\\n    name\\n    review {\\n      avgRating\\n      totalCount\\n      scores {\\n        count\\n        score\\n        __typename\\n      }\\n      starDistribution {\\n        count\\n        score\\n        __typename\\n      }\\n      imageReviewCount\\n      authorCount\\n      maxSingleReviewScoreCount\\n      maxScoreWithMaxCount\\n      __typename\\n    }\\n    analysis {\\n      themes {\\n        code\\n        label\\n        count\\n        __typename\\n      }\\n      menus {\\n        label\\n        count\\n        __typename\\n      }\\n      votedKeyword {\\n        totalCount\\n        reviewCount\\n        userCount\\n        details {\\n          category\\n          code\\n          displayName\\n          count\\n          previousRank\\n          __typename\\n        }\\n        __typename\\n      }\\n      __typename\\n    }\\n    visitorReviewsTotal\\n    ratingReviewsTotal\\n    __typename\\n  }\\n  visitorReviewThemes(input: {businessId: $id}) {\\n    themeLists {\\n      name\\n      key\\n      __typename\\n    }\\n    __typename\\n  }\\n}\\n"},{"operationName":"getVisitorReviewPhotosInVisitorReviewTab","variables":{"businessId":"11718339","businessType":"restaurant","item":"0","page":1,"display":10},"query":"query getVisitorReviewPhotosInVisitorReviewTab($businessId: String!, $businessType: String, $page: Int, $display: Int, $theme: String, $item: String) {\\n  visitorReviews(input: {businessId: $businessId, businessType: $businessType, page: $page, display: $display, theme: $theme, item: $item, isPhotoUsed: true}) {\\n    items {\\n      id\\n      rating\\n      author {\\n        id\\n        nickname\\n        from\\n        imageUrl\\n        objectId\\n        url\\n        __typename\\n      }\\n      body\\n      thumbnail\\n      media {\\n        type\\n        thumbnail\\n        __typename\\n      }\\n      tags\\n      status\\n      visited\\n      originType\\n      item {\\n        name\\n        code\\n        options\\n        __typename\\n      }\\n      businessName\\n      __typename\\n    }\\n    starDistribution {\\n      score\\n      count\\n      __typename\\n    }\\n    hideProductSelectBox\\n    total\\n    __typename\\n  }\\n}\\n"},{"operationName":"getVisitorRatingReviews","variables":{"input":{"businessId":"11718339","businessType":"restaurant","item":"0","bookingBusinessId":null,"page":1,"display":10,"includeContent":false,"getAuthorInfo":true},"id":"11718339"},"query":"query getVisitorRatingReviews($input: VisitorReviewsInput) {\\n  visitorReviews(input: $input) {\\n    total\\n    items {\\n      id\\n      rating\\n      author {\\n        id\\n        nickname\\n        from\\n        imageUrl\\n        objectId\\n        url\\n        review {\\n          totalCount\\n          imageCount\\n          avgRating\\n          __typename\\n        }\\n        theme {\\n          totalCount\\n          __typename\\n        }\\n        __typename\\n      }\\n      visitCount\\n      visited\\n      originType\\n      reply {\\n        editUrl\\n        body\\n        editedBy\\n        created\\n        replyTitle\\n        __typename\\n      }\\n      votedKeywords {\\n        code\\n        displayName\\n        __typename\\n      }\\n      businessName\\n      status\\n      userIdno\\n      isFollowing\\n      followerCount\\n      followRequested\\n      loginIdno\\n      __typename\\n    }\\n    __typename\\n  }\\n}\\n"}]'
                response = requests.post('https://pcmap-api.place.naver.com/graphql', headers=headers, data=data)

                response_json = json.loads(response.text)
                response_json = response_json[0]['data']['visitorReviews']['items']

                for index, i in enumerate(response_json):
                    date = i['visited']

                    # date 컬럼의 형태를 YYYY-MM-DD로 맞추기 위한 설정
                    # ========> json 데이터상 해당 년도(2022년)일 경우에는 `1.09.일` 이런 식으로 나옴 <========
                    # 해당 년도 이전의 년도는 `21.12.31.금` 형식으로 나오기 때문에 아래와 같이 설정
                    if len(date) > 7:
                        date_slicing = date[:-2].split('.')
                        date_time = datetime.date(int('20' + date_slicing[0]), int(date_slicing[1]),
                                                  int(date_slicing[2]))
                    else:
                        date_slicing = date[:-2].split('.')
                        date_time = datetime.date(int('2022'), int(date_slicing[0]), int(date_slicing[1]))

                    score = i['rating']
                    # 후에 전처리 하기 수월하고 저장된 csv에서 확인을 쉽게 하기 위해 미리 전처리(줄바꿈, 마침표 제거)
                    review = i['body'].replace('\n', '').replace('.', '')
                    data_frame.append([df['store_id'][idx], 1004, date_time, score, review])

                    print(index, date_time)
                    print(score)
                    print(review)
                    # 리뷰 하나당 0에서 2초로 time sleep을 랜덤으로 설정해 줘야 api사용에 차질이 없음.
                    # 그래도 selenium보다 빠름.
                    time.sleep(np.random.randint(0, 2))

        except:
            if businessid == '':
                data_frame.append('')
                pass

        dataset = pd.DataFrame(data_frame, columns=['store_id', 'portal_id', 'date', 'score', 'review'])

    return dataset