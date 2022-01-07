import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

#kobert
from kobert.utils.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from sklearn.model_selection import train_test_split
from bertmodel.kobert import kobert_train_test
from kobert.kobert_tokenizer import KoBERTTokenizer

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from bertmodel.kobert import kobert_train_test, BERTDataset, BERTClassifier
from bertmodel.eval import append_new_score
from bertmodel.newscore import avg_score_to_csv
from bertmodel.eval import predict

'''
   1. data_dict에 cluster : percent_list로 데이터 삽입
   2. 클러스터 리스트 생성 [0, 1, 2, 3, 4]
   3. final_cluster_data 생성 = 마지막에 클러스터가 몇점인지 저장하는 dict type
   4. get_sore_at_cluster(data_dict, cluster_list, final_cluster_data)
       - 클러스터 별로 가장 높은 퍼센티지 인덱스 추출
       - 인덱스가 겹치지 않는 클러스터는 인덱스를 통해 평점을 매겨주주고 final_cluster_data에 cluster : 평점으로 데이터 삽입
       - final_cluster_data에 들어있는 Key(클러스터)는 cluster_list에서 삭제
       - score_max = 각 클러스터 별 가장 퍼센티지가 높은 인덱스를 모아놓은 리스트
       - return : cluster_list, final_cluster_data, cluster_max
   5. keys = final_cluster_data의 key 리스트로 받기
   6. 중복이 가장 많은 인덱스 찾기, 같을 경우엔 높은 순서로 찾기
   7. 중복이 가낭 많은 인덱스에서, 클러스터 리스트에 남아있는 점수들 중 가장 퍼센티지가 높은 클러스터를 찾고, 해당 클러스터의 점수를 부여한 후 클러스터 리스트에서 삭제
   8. 남은 클러스터 리스트 중에서 keys에서 현재 없는 점수를 찾고, 점수가 부여되지 않은 클러스터끼리 해당 점수의 퍼센티지를 비교하고, 
       퍼센티지가 더 높은 클러스터에 점수를 할당하고 final_cluster_data에 점수 부여하고
       cluster_list에서 해당 데이터 삭제
       cluster_list가 사라질 때까지 반복
   9. final_clster_data를 기준으로 리뷰 데이터 선택
   10. len이 가장 작은 데이터 기준으로 pd 데이터 슬라이싱 후
   11. 코버트 모델에 파라미터로 인계 .
'''

def get_dict_for_kobert_data(data):
    data_dict = {}
    for cluster in data['cluster'].unique():
        score_percent = []
        score_count = []
        for score in range(5):
            c_df_size = len(data[data['cluster'] == cluster])
            c_s_df_size = len(data[(data['cluster'] == cluster) & (data['score'] == score + 1)])
            score_count.append(c_s_df_size)
            # 각 스코어 df 갯수 / cluster 갯수 * 100 으로 각 스코어 당 퍼센티지 구하기
            score_percent.append((c_s_df_size / c_df_size) * 100)
        data_dict[cluster] = score_percent
    return data_dict

def get_score_at_cluster(data_dict, cluster_list, final_cluster_data):
    cluster_max = []

    for i in range(5):
        # 해당 클러스터에서 퍼센티지가 가장 높은 평점 구하기
        cluster_max.append(data_dict.get(i).index(max(data_dict.get(i))))
    # 클러스터당 가장 높은 평점이 겹치지 않을경우 클러스터 리스트에서 빼고 클러스터 점수 정하기
    for i in range(5):
        if cluster_max.count(i) == 1:
            final_cluster_data[i] =i+1
            del cluster_list[cluster_list.index(i)]
    return cluster_list, final_cluster_data, cluster_max

def get_final_score_at_cluster(data_dict, cluster_list, final_cluster_data):
    # 가장 중복되는 클러스터 찾기
    # one-hot : 클러스터 별 list.count(cluster) 만들어서 점수 높은 부분부터 찾기
    one_hot_clustet_duplicate = [
        cluster_max.count(i)
        for i in range(5)
    ]

    # 가장 중복이 많이된 클러스터 인덱스 찾기
    find_cluster_index = 0
    # 비교할 데이터 선언
    find_cluster_value = 0

    for idx, val in enumerate(one_hot_clustet_duplicate):
        if val >= find_cluster_value:
            find_cluster_value = val
            find_cluster_index = idx

    # 중복 많이 된 클러스터 인덱스 기준으로 클러스터 리스트 안에서 가장 높은퍼센트 뽑아낸 후 final_dict에 삽입 후 cluster_list에서 삭제
    percent_list = [
        data_dict.get(i)[find_cluster_index]
        for i in cluster_list
    ]
    final_cluster_data[cluster_list[percent_list.index(max(percent_list))]] = find_cluster_index + 1
    del cluster_list[cluster_list.index(cluster_list[percent_list.index(max(percent_list))])]

    # 남은 클러스터 끼리 반복문 돌면서 높은 점수부터 비중이 높은 퍼센티지 매칭해서 final_cluster_data 완성

    while len(cluster_list) > 0:
        # 남은 final_dict key에 없는 데이터 찾기
        for val in range(5, 0, -1):
            values = list(final_cluster_data.values())
            keys = list(final_cluster_data.keys())
            # 파이널 리스트에 해당 점수가 없으면
            if val not in values:
                compare_list = []
                cluster_list = []
                for key in range(4, -1, -1):
                    # 아직 데이터가 없는 클러스터를 찾아서
                    if key not in keys:
                        # 해당 클러스터에 퍼센티지 데이터 뽑아오기
                        # print(f' cluster : {key} data : {data_dict.get(key)[val - 1]} index : {val - 1}')
                        compare_list.append(data_dict.get(key)[val - 1])
                        cluster_list.append(key)
                max_cluster = cluster_list[compare_list.index(max(compare_list))]
                final_cluster_data[max_cluster] = val
                del cluster_list[cluster_list.index(cluster_list[compare_list.index(max(compare_list))])]

    return final_cluster_data

DATA = pd.read_csv('dec_test.csv')
DATA = DATA[['score', 'cluster', 'review']]
DATA = DATA.dropna()

# 각 클러스터당 데이터 세분화시킨 dict 데이터 구하기
data_dict = get_dict_for_kobert_data(DATA)

final_cluster_data = {}
cluster_list = [0,1,2,3,4]

# 중복되지 않은 클러스터들은 final_cluster_data에 할당
cluster_list, final_cluster_data, cluster_max = get_score_at_cluster(data_dict, cluster_list, final_cluster_data)

# 남은 클러스터에 자체 알고리즘으로 데이터 할당
final_cluster_data = get_final_score_at_cluster(data_dict, cluster_list, final_cluster_data)

# 해당 클러스터, 점수별 데이터 길이 최솟값 구하기

data_length = [
    len(DATA[(DATA['cluster']==cluster) & (DATA['score']==final_cluster_data.get(cluster))])
    for cluster in final_cluster_data.keys()
]

slice_value = min(data_length)

# 최솟값 기준으로 pandas dataframe에 각 클러스터별 데이터 입력
df_list = [
    DATA[(DATA['cluster']==cluster) & (DATA['score'] == final_cluster_data.get(cluster))][:slice_value]
    for cluster in final_cluster_data.keys()
]

# 위의 데이터프레임 합쳐서 kobert 학습용 데이터 프레임 생성
final_df = pd.concat(df_list)
final_df.to_csv('final_df.csv', encoding='UTF-8', index=False)

device = torch.device("cpu")

bertmodel, vocab = get_pytorch_kobert_model()
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

data_list = []
for q, label in zip(DATA['review'], DATA['score']):
    data = []
    data.append(q)
    data.append(str(int(label)))

    data_list.append(data)

kobert_train_test(device, data_list, bertmodel, vocab)
#
# end = 1
# while end == 1:
#     sentence = input("테스트해보고 싶은 리뷰를 입력해주세요 : ")
#     if sentence == 0:
#         break
#     predict(sentence)
#     print("\n")

# append_new_score(model, sentence)
avg_score_to_csv('dec_test.csv')
