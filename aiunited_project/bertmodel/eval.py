import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm

#kobert
from kobert.utils.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.kobert_tokenizer import KoBERTTokenizer
from bertmodel.kobert import BERTClassifier, BERTDataset

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

bertmodel, vocab = get_pytorch_kobert_model()
device = torch.device("cpu")
# inference
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

max_len = 64
batch_size = 64

def append_new_score(model, sentence):
    model.load_state_dict(torch.load('model_for_inference.pt'))
    model

    # new review dataframe
    df = pd.read_csv('dec_test.csv')
    df = df.dropna()

    df['new_score'] = df['preprocessed_review'].apply(predict(sentence)) # predict의 인자로 df['preprocessed_review'](type:str)이 들어가고 리턴값이 apply에 들어감
    df.to_csv('_review.csv', encoding='utf-8', index=False)

def predict(sentence):
    data = [sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("최악입니다. 0점")
            elif np.argmax(logits) == 1:
                test_eval.append("노맛입니다. 1점")
            elif np.argmax(logits) == 2:
                test_eval.append("평범합니다. 2점")
            elif np.argmax(logits) == 3:
                test_eval.append("맛있습니다. 3점")
            elif np.argmax(logits) == 4:
                test_eval.append("쫀맛입니다. 4점")

    print(">> 입력하신 리뷰는 " + test_eval[0])