# new store info
import numpy as np
import pandas as pd
from tqdm import tqdm

def avg_score_to_csv(path):
    df = pd.read_csv(path)
    df = df.sort_values(by=['store_id'], axis=0)
    # print(df)

    info_df = pd.read_csv('./data/storeInfo_1.csv')
    info_df['new_score'] = ''
    info_df['review_cnt'] = ''

    no_store_review_lst = []
    for store_id in tqdm(info_df['store_id']):
        review_df = df[df['store_id'] == store_id]

        if review_df.empty:
            no_store_review_lst.append(store_id)
            continue

        review_cnt = len(review_df['cluster'])
        if review_cnt == 0 :
            pass
        else:
            new_score = np.round(sum(review_df['cluster']) / review_cnt, 2)
            info_df.loc[(info_df['store_id'] == store_id), 'new_score'] = new_score
            info_df.loc[(info_df['store_id'] == store_id), 'review_cnt'] = review_cnt

    for store_id in info_df['store_id']:
        if store_id in no_store_review_lst:
            info_df.loc[(info_df['store_id'] == store_id), 'new_score'] = ''
            info_df.loc[(info_df['store_id'] == store_id), 'review_cnt'] = 0
    info_df.to_csv('new_storeInfo.csv', encoding='utf-8', index=False)
