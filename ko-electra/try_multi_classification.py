import torch

from transformers import ElectraTokenizer
from transformers import ElectraForSequenceClassification, AdamW, BertConfig,ElectraModel, ElectraForMultipleChoice
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

n_devices = torch.cuda.device_count()
print(n_devices)

for i in range(n_devices):
    print(torch.cuda.get_device_name(i))

train = pd.read_csv('./datasets/train.tsv',sep='\t')
test = pd.read_csv('./datasets/dev.tsv', sep='\t')

def labeling(df):

    df.loc[(df.contain_gender_bias==False)&(df.bias=='none')&(df.hate=='none'), 'label'] = 0
    df.loc[(df.contain_gender_bias==False)&(df.bias=='none')&(df.hate=='offensive'), 'label'] = 1
    df.loc[(df.contain_gender_bias==True)&(df.bias=='gender')&(df.hate=='hate'), 'label'] = 2
    df.loc[(df.contain_gender_bias==False)&(df.bias=='others')&(df.hate=='offensive'), 'label'] = 3
    df.loc[(df.contain_gender_bias==False)&(df.bias=='others')&(df.hate=='hate'), 'label'] = 4
    df.loc[(df.contain_gender_bias==False)&(df.bias=='none')&(df.hate=='hate'), 'label'] = 5
    df.loc[(df.contain_gender_bias==True)&(df.bias=='gender')&(df.hate=='offensive'), 'label'] = 6
    df.loc[(df.contain_gender_bias==False)&(df.bias=='others')&(df.hate=='none'), 'label'] = 7
    df.loc[(df.contain_gender_bias==True)&(df.bias=='gender')&(df.hate=='none'), 'label'] = 8

    return df

train = labeling(train)
test = labeling(test)


train['label'] = train['label'].astype('int64')
# train['label'] = train['label'].astype(str)
train = train[['comments', 'label']]
test['label'] = test['label'].astype('int64')
# test['label'] = test['label'].astype(str)
test = test[['comments', 'label']]

# LABEL_DICT = { True:1, False:0}
# train['label'] = train['label'].map(lambda x:LABEL_DICT[x])

import re

def clean_text(text):
    corpus = []
    review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(text)) #remove punctuation
    review = re.sub(r'\d+','', review)# remove number
    review = review.lower() #lower case
    review = re.sub(r'\s+', ' ', review) #remove extra space
    review = re.sub(r'<[^>]+>','',review) #remove Html tags
    review = re.sub(r'\s+', ' ', review) #remove spaces
    review = re.sub(r"^\s+", '', review) #remove space from start
    review = re.sub(r'\s+$', '', review) #remove space from the end
    review = re.sub(r'^[A-Za-z+]*$', '', review) #remove alphabet
    review = re.sub(r'[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', review) #remove special characters
    review = re.sub(r'[^\w\s]', '', review) #remove punctuation
    corpus.append(review)

    return review

train['comments'] = train['comments'].apply(lambda x: clean_text(x))
test['comments'] = test['comments'].apply(lambda x: clean_text(x))

# print(train.shape, test.shape)

document_bert = ["[CLS] " + str(s) + " [SEP]" for s in train['comments']]
# print(document_bert[:5])

tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
tokenized_texts = [tokenizer.tokenize(s) for s in document_bert]
# print(tokenized_texts[0])

MAX_LEN = 32
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
# print(input_ids[0])

attention_masks = []

for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

# print(attention_masks[0])

train_inputs, validation_inputs, train_labels, validation_labels = \
train_test_split(input_ids, train['label'].values, random_state=42, test_size=0.2)
print(train_inputs.shape, train_labels.shape)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                       random_state=42,
                                                       test_size=0.2)

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

BATCH_SIZE = 128

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

sentences = test['comments']
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
labels = test['label'].values

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

attention_masks = []
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

model = ElectraForMultipleChoice.from_pretrained('monologg/koelectra-base-v3-discriminator', problem_type="multi_label_classification")
print(model)
# model.cuda()

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps= 0 ,
                                            num_training_steps= total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

seed_val = 42
random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()

# 에폭만큼 반복
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 시작 시간 설정
    t0 = time.time()

    # 로스 초기화
    total_loss = 0

    # 훈련모드로 변경
    model.train()

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(train_dataloader):
        # 경과 정보 표시
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 배치를 GPU에 넣음
        # batch = tuple(t.to(device) for t in batch)
        batch = tuple(t for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch
        # print(f'batch: {batch}')
        # print(f'b_input_ids: {b_input_ids} \n b_input_mask: {b_input_mask} \n b_labels: {b_labels} ')

        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # 로스 구함
        loss = outputs[0]

        # 총 로스 계산
        total_loss += loss.item()

        # Backward 수행으로 그래디언트 계산
        loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 그래디언트를 통해 가중치 파라미터 업데이트
        optimizer.step()

        # 스케줄러로 학습률 감소
        scheduler.step()

        # 그래디언트 초기화
        model.zero_grad()

    # 평균 로스 계산
    avg_train_loss = total_loss / len(train_dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    # 시작 시간 설정
    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 변수 초기화
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for batch in validation_dataloader:
        # 배치를 GPU에 넣음
        # batch = tuple(t.to(device) for t in batch)
        batch = tuple(t for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

        # 그래디언트 계산 안함
        with torch.no_grad():
            # Forward 수행
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # 로스 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 출력 로짓과 라벨을 비교하여 정확도 계산
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")