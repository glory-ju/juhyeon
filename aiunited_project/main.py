from crawler.store_crawler import Crawler
from crawler.review_crawler import action_naver_review_crawler
from preprocessing.review_preprocessing import action_naver_review_preprocessing
from embedding.embedding_doc2vec import action_naver_review_tokenizing, \
                                        action_naver_review_embedding, \
                                        action_naver_review_modeling

from dec.DEC.DEC.DEC import dec
import pandas as pd
import requests, datetime, json, time, math
from soynlp.normalizer import *
from hanspell import spell_checker
from pykospacing import Spacing
from konlpy.tag import Mecab
import matplotlib.pyplot as plt
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from sklearn.manifold import TSNE

if __name__ == '__main__':

    # 최초 storeinfo csv 가져오기
    df = pd.read_csv('C:/Users/140252/PycharmProjects/juhyeon/aiunited/data/storeInfo_1.csv')
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
    tokenize_review.dropna(subset=['score'], inplace=True)
    tokenize_review.reset_index(drop=True, inplace=True)
    tokenized = action_naver_review_embedding(tokenize_review)

    # learned model
    model = action_naver_review_modeling(tokenized)

    # dec
    data = dec(tokenize_review, )