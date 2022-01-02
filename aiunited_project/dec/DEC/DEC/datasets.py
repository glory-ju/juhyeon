import numpy as np
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec


def get_features(model, words, size):
    feature_vector = np.zeros((size), dtype=np.float32)
    num_words = 0
    word_set = set(model.wv.index_to_key)

    for w in words:
        if w in word_set:
            num_words += 1
            feature_vector = np.add(feature_vector, model[w])

    if num_words != 0:
        feature_vector = np.divide(feature_vector, num_words)
    else:
        pass

    return feature_vector

def get_dataset2(model, reviews, size):
    dataset = list()

    for i in reviews:
        dataset.append(get_features(model, i[0], size))

    return np.stack(dataset)

def load_doc2vec(df, **param):
    tokenized_review = df['tokenized']
    model = Doc2Vec.load('weights/' + param['model_name'])

    x = get_dataset2(model, list(tokenized_review), param['size'])
    y = df['score'].to_numpy()

    return x.astype(float), y

def load_kobert(df):
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    import gluonnlp as nlp
    from kobert.utils import get_tokenizer
    from kobert.pytorch_kobert import get_pytorch_kobert_model
    bertmodel, vocab = get_pytorch_kobert_model()

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    class BERTDataset(Dataset):
        def __init__(self, dataset, sent_idx, bert_tokenizer, max_len, pad, pair):
            transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

            self.sentences = [transform([row[sent_idx]]) for i, row in dataset.iterrows()]

        def __getitem__(self, i):
            return (self.sentences[i])

        def __len__(self):
            return (len(self.sentences))

    class BERTClassifier(nn.Module):
        def __init__(self,
                     bert,
                     hidden_size=768,
                     num_classes=None,
                     dr_rate=None,
                     params=None):
            super(BERTClassifier, self).__init__()
            self.bert = bert
            self.dr_rate = dr_rate

            self.classifier = nn.Linear(hidden_size, num_classes)
            if dr_rate:
                self.dropout = nn.Dropout(p=dr_rate)

        def gen_attention_mask(self, token_ids, valid_length):
            attention_mask = torch.zeros_like(token_ids)
            for i, v in enumerate(valid_length):
                attention_mask[i][:v] = 1
            return attention_mask.float()

        def forward(self, token_ids, valid_length, segment_ids):
            attention_mask = self.gen_attention_mask(token_ids, valid_length)

            _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                                  attention_mask=attention_mask.float().to(token_ids.device))
            if self.dr_rate:
                out = self.dropout(pooler)
            return self.classifier(out)

    data_x = BERTDataset(df, 'review', tok, 128, True, False)
    data_x = torch.utils.data.DataLoader(data_x, batch_size=128, num_workers=8)

    x = []
    for token_ids, valid_length, segment_ids in data_x:
        for token_id in token_ids:
            tmp = []
            for i in token_id:
                if int(i) == 2:
                    pass
                elif int(i) == 3 or int(i) == 1:
                    tmp.append(0)
                else:
                    tmp.append(int(i) / 10000)
            x.append(np.array(tmp))

    x = np.array(x)
    y = df['score'].to_numpy()

    return x.astype(float), y


def load_data(dataset_name, data, **param):
    if dataset_name == 'kobert':
        return load_kobert(data)
    elif dataset_name == 'doc2vec':
        return load_doc2vec(data, **param)
    else:
        print('Not defined for loading', dataset_name)
        exit(0)