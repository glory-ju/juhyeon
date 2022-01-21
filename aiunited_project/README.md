# _Project For Alluser.net_

* [Description](#for-allusernets-project)
  * [Outline / Background](#outline--background)
    * [Why'?'](#why)
    * [Requirements](#requirements)
    * [How to install](#how-to-install)
  * [How to use](#how-to-use)
  * [KoBERT](#kobert)
    * [Using with PyTorch](#using-with-pytorch)
    * [Tokenizer](#tokenizer)
* [Demo](#simulation-video)
  * [Web](#web-function-implementationvia-django)

---

## Outline / Background

### Why'?'

* 광고를 보고 혹은 검색해보고 찾아갔던 맛집이라고 불리는 식당들?
```markdown
맛의 척도는 상대적이기 때문에 맛집이 모두에게 통용될 수 없음. 
하지만 허위 및 과대 광고로 인한 피해 또한 분명함
                                            `목표`
-> 네이버, 구글, 식신, 다이닝 코드 포털 식당들의 리뷰와 평점 진정성 모델 클러스터링 <-
```

* 학습셋

| 포털     | 데이터   |
|--------|-------|
| 네이버    | 2.2M  |
| 구글     | 1.4M  |
| 식신     | 0.08M |
| 다이닝 코드 | 0.03M |


* 학습 환경
  * Intel(R) Core(TM) i5-1035G7 CPU

### Requirements

* [requirements_dec.txt](https://github.com/glory-ju/juhyeon/blob/main/aiunited_project/requirements_dec.txt) - _main.py_
  * 버전 충돌 때문에 dec와 kobert를 한 번에 돌릴 수 없음.
  * ###가상환경 두 개로 나누어서 DEC / KOBERT 진행.
  * requirements_dec.text : crawling ~ preprocessing ~ embedding ~ modeling ~ DEC
* [requirements_kobert.txt](https://github.com/glory-ju/juhyeon/blob/main/aiunited_project/requirements_kobert.txt) - _main_bert.py_

### How to install

* If you want to modify source codes, please clone this repository

  ```sh
  git clone https://github.com/glory-ju/juhyeon.git
  cd aiunited_project
  pip install -r requirements_dec.txt
  pip install -r requirements_kobert.txt
                          +
  한국어 형태소 분석기 Mecab 윈도우 설치 가이드 링크
  https://uwgdqo.tistory.com/363
  ```

### Virtual Environment and Requirements Detail
* requirements_dec ( 1st venv : python version 3.8 )
  ```python
  >>> pip install 
  
  requests pandas tqdm soynlp matplotlib gensim konlpy torch gluonnlp
  git+https://github.com/haven-jeon/PyKoSpacing.git
  git+https://github.com/ssut/py-hanspell.git
  
  ```
  
* requirements_kobert ( 2nd venv : python version 3.6 )
  ```python
  >>> pip install 
  
  mxnet pandas gluonnlp numpy tqdm jupyter scikit-learn transformers boto3 torch sentencepiece
  ```
---

## How to use

### _샘플값 추출을 위해 naver 포털 데이터 값으로만 진행_

`from crawler.store_crawler import Crawler` -- line 41
```python
for i in tqdm(range(10)):
sample 추출을 위해 임의 설정.
len(df)나 원하는 범위 (0, 19163 사이값) 설정 가능.
```
`Crawler : store_name, n_link 추출` -- review crawling하기 위한 선작업

`action_naver_review_crawler` -- 리뷰 크롤링(copy as cURL(bash))

`action_naver_review_preprocessing` 
* pykospacing ( 띄어쓰기 검사기 )
* soynlp ( 한글만 추출 )
* py-hanspell ( 맞춤법 검사기 )

`action_naver_review_tokenizing` -- konlpy 형태소 분석기 Mecab 사용

`action_naver_review_embedding` 
* 리뷰(문장) 임베딩에 word2vec 보다 doc2vec 성능이 더 높음
* taggeddocument로 model 준비
* dec의 파라미터로 인계하기 위해 return 값을 
  * tokenized : [tokezined된 리뷰, score(라벨)]

`dec.py`
```python
data = dec(df=tokenize_review, n_clusters=5, model_name = 'my_doc2vec_model', size=100)
```
* n_cluster : 평점 1~5점에 맞는 dec clustering 클러스터 값을 5개로 설정
* model_name : 사전 학습된 doc2vec model 사용.
* size : operands broadcasting : 100

---

## KoBERT

---
### Using with PyTorch

```python
>>> import torch
>>> from kobert import get_pytorch_kobert_model
>>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
>>> input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
>>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
>>> model, vocab  = get_pytorch_kobert_model()
>>> sequence_output, pooled_output = model(input_ids, input_mask, token_type_ids)
>>> pooled_output.shape
torch.Size([2, 768])
>>> vocab
Vocab(size=8002, unk="[UNK]", reserved="['[MASK]', '[SEP]', '[CLS]']")
>>> # Last Encoding Layer
>>> sequence_output[0]
tensor([[-0.2461,  0.2428,  0.2590,  ..., -0.4861, -0.0731,  0.0756],
        [-0.2478,  0.2420,  0.2552,  ..., -0.4877, -0.0727,  0.0754],
        [-0.2472,  0.2420,  0.2561,  ..., -0.4874, -0.0733,  0.0765]],
       grad_fn=<SelectBackward>)
```

`model`은 디폴트로 `eval()`모드로 리턴됨, 따라서 학습 용도로 사용시 `model.train()`명령을 통해 학습 모드로 변경할 필요가 있다.

* Naver Sentiment Analysis Fine-Tuning with pytorch
  * Colab에서 [런타임] - [런타임 유형 변경] - 하드웨어 가속기(GPU) 사용을 권장합니다.
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)

### Tokenizer

* Pretrained [Sentencepiece](https://github.com/google/sentencepiece) tokenizer

```python
>>> from gluonnlp.data import SentencepieceTokenizer
>>> from kobert import get_tokenizer
>>> tok_path = get_tokenizer()
>>> sp  = SentencepieceTokenizer(tok_path)
>>> sp('한국어 모델을 공유합니다.')
['▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.']
```

---

# Simulation Video

https://drive.google.com/file/d/1vC53wQ2cWlLHMOeFc2g805omJ3yiMQVF/view?usp=sharing

# Web Function Implementation(via Django)

https://github.com/boredjuju/store_recommendation
