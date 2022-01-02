# -*- coding: utf-8 -*-
import pandas as pd
from soynlp.normalizer import *
from hanspell import spell_checker
from pykospacing import Spacing

def action_naver_review_preprocessing(df):
    preprocessing = []
    spacing = Spacing()

    for idx in range(len(df)):
        try:
            sent = df['review'][idx].strip().replace(' ','')
            # 맞춤법 검사기
            spelled_sent = spell_checker.check(sent)
            checked_sent = spelled_sent.checked
            # 띄어쓰기 검사기
            space = spacing(checked_sent)
            # 한글만 출력하는 soynlp의 only_hangle 함수
            hangeul = only_hangle(space)

            # 가성비라는 단어를 맞춤법 검사기를 거치면 구성비로 번역됨.
            # 가독성을 위해 변경
            if '구성비' in hangeul:
                hangeul = hangeul.replace('구성비','가성비')

            print(hangeul)
            preprocessing.append(hangeul)

        except:
            preprocessing.append('')

    preprocessed = pd.DataFrame(preprocessing)
    df['preprocessed_review'] = preprocessed
    print(df['preprocessed_review'])

    df.to_csv('sample_preprocessing.csv', encoding='utf-8', index=False)

    return df