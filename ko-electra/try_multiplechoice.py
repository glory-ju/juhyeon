import pandas as pd
import torch
from torch.nn import MultiLabelSoftMarginLoss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW, ElectraModel, ElectraTokenizer,ElectraForMultipleChoice
from tqdm.notebook import tqdm

# GPU 사용
device = torch.device("cuda")

class HateSpeech(Dataset):

    def __init__(self, tsv_file):
        self.dataset = pd.read_csv(tsv_file, sep='\t')
        self.dataset.loc[(self.dataset.contain_gender_bias==False)&(self.dataset.bias=='none')&(self.dataset.hate=='none'), 'label'] = 0
        self.dataset.loc[(self.dataset.contain_gender_bias==False)&(self.dataset.bias=='none')&(self.dataset.hate=='offensive'), 'label'] = 1
        self.dataset.loc[(self.dataset.contain_gender_bias==True)&(self.dataset.bias=='gender')&(self.dataset.hate=='hate'), 'label'] = 2
        self.dataset.loc[(self.dataset.contain_gender_bias==False)&(self.dataset.bias=='others')&(self.dataset.hate=='offensive'), 'label'] = 3
        self.dataset.loc[(self.dataset.contain_gender_bias==False)&(self.dataset.bias=='others')&(self.dataset.hate=='hate'), 'label'] = 4
        self.dataset.loc[(self.dataset.contain_gender_bias==False)&(self.dataset.bias=='none')&(self.dataset.hate=='hate'), 'label'] = 5
        self.dataset.loc[(self.dataset.contain_gender_bias==True)&(self.dataset.bias=='gender')&(self.dataset.hate=='offensive'), 'label'] = 6
        self.dataset.loc[(self.dataset.contain_gender_bias==False)&(self.dataset.bias=='others')&(self.dataset.hate=='none'), 'label'] = 7
        self.dataset.loc[(self.dataset.contain_gender_bias==True)&(self.dataset.bias=='gender')&(self.dataset.hate=='none'), 'label'] = 8

        self.dataset['label'] = self.dataset['label'].astype('int64')
        self.dataset = self.dataset[['comments', 'label']]
        self.tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-discriminator')

        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 0:2].values
        text = row[0]
        y = row[1]

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=8,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0]
        print(f'input_ids : {len(input_ids)}')
        attention_mask = inputs['attention_mask'][0]
        print(f'attention_mask : {len(attention_mask)}')

        return input_ids, attention_mask, y

train_dataset = HateSpeech('train.tsv')
test_dataset  = HateSpeech('dev.tsv')

# model = ElectraForSequenceClassification.from_pretrained('monologg/koelectra-base-v3-discriminator', problem_type="multi_label_classification").to(device)
model = ElectraForMultipleChoice.from_pretrained('monologg/koelectra-base-v3-discriminator').to(device)
# text, attention_mask, y = train_dataset[0]
# model(text.unsqueeze(0).to(device), attention_mask=attention_mask.unsqueeze(0).to(device))

epochs = 3
batch_size = 8

optimizer = AdamW(model.parameters(), lr=1e-5)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
from torch.nn import functional as F
gc.collect()
torch.cuda.empty_cache()

losses = []
accuracies = []

for i in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    model.train()

    for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader):
        optimizer.zero_grad()
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        print(y_pred)
        loss = F.cross_entropy(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y_batch).sum()
        total += len(y_batch)

        batches += 1
        if batches % 100 == 0 :
            print('Batch Loss:', total_loss, 'Accuracy:', correct.float() / total)

    losses.append(total_loss)
    accuracies.append(correct.float() / total)
    print('Train Loss:', total_loss, 'Accuracy:', correct.float() / total)