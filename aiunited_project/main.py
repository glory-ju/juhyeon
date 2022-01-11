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
    preprocess_review.to_csv('preprocessed_review.csv', index=False, encoding='UTF-8')

    # review tokenizing
    mecab = Mecab(dicpath=r"C:/mecab/mecab-ko-dic")
    tokenize_review = action_naver_review_tokenizing(preprocess_review, mecab)
    tokenize_review.to_csv('tokenized_review.csv', index=False, encoding='UTF-8')

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