from crawler.store_crawler import Crawler
from crawler.review_crawler import action_naver_review_crawler
from preprocessing.review_preprocessing import action_naver_review_preprocessing
from embedding.embedding_doc2vec import action_naver_review_tokenizing, \
                                        action_naver_review_embedding, \
                                        action_naver_review_modeling

from dec.DEC.DEC.DEC import dec
import pandas as pd
from konlpy.tag import Mecab
import socket
socket.getaddrinfo('localhost', 8080)

if __name__ == '__main__':

    # 최초 storeinfo csv 가져오기
    df = pd.read_csv('./data/storeInfo_1.csv')
    crawler = Crawler()

    # naver get store_info
    store_info_naver = crawler.action_naver_store_info(df)
    print(store_info_naver)
    name = "naver_store_info.csv"
    store_info_naver.to_csv(name, index=False, encoding='UTF-8')

    # naver review Crawling
    df = pd.read_csv(name)
    naver_review = action_naver_review_crawler(df)
    print(naver_review)
    name = "naver_review.csv"
    naver_review.to_csv(name, index=False, encoding='UTF-8')

    # review preprocessing
    df = pd.read_csv(name)
    preprocess_review = action_naver_review_preprocessing(df)

    # review tokenizing
    mecab = Mecab(dicpath=r"C:/mecab/mecab-ko-dic")
    tokenize_review = action_naver_review_tokenizing(preprocess_review, mecab)

    # preparing embedding
    tokenize_review.dropna(inplace=True)
    tokenize_review.reset_index(drop=True, inplace=True)
    tokenized = action_naver_review_embedding(tokenize_review)

    # learned model
    model = action_naver_review_modeling(tokenized)

    # dec
    data = dec(df=tokenize_review, n_clusters=5, model_name = 'my_doc2vec_model', size=100)
    print(data)

    # kobert 모델 학습시킬 데이터 사전 추출
    data = data[['score', 'cluster', 'review']]

    # 각 클러스터 당 데이터 세분화 시킨 dict 데이터 구하기
    # data_dict = get_dict_for_kobert_data(data)
    #
    # final_cluster_data = {}
    #
    # cluster_list = [0, 1, 2, 3, 4]
    #
    # # 중복되지 않은 클러스터들은 final_cluster_data에 할당
    # cluster_list, final_cluster_data, cluster_max = get_score_at_cluster(data_dict, cluster_list, final_cluster_data)
    #
    # # 남은 클러스터에 자체 알고리즘으로 데이터 할당
    # final_cluster_data = get_final_score_at_cluster(data_dict, cluster_list, final_cluster_data)
    #
    # # 해당 클러스터, 점수 별 데이터 길이 최솟값 구하기
    # data_length = [
    #     len(data[(data['cluster'] == cluster) & (data['score'] == final_cluster_data.get(cluster))])
    #     for cluster in final_cluster_data.keys()
    # ]
    #
    # slice_value = min(data_length)
    #
    # # 최솟값 기준으로 판다스 데이터프레임에 각 클러스터별 데이터 입력
    # df_list = [
    #     data[(data['cluster'] == cluster) & (data['score'] == final_cluster_data.get(cluster))][:slice_value]
    #     for cluster in final_cluster_data.keys()
    # ]
    # # 위의 데이터프레임 합쳐서 kobert 학습용 데이터 프레임 생성
    # final_df = pd.concat(df_list)