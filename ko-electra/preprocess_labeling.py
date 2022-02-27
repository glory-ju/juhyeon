from soynlp.normalizer import *
from hanspell import spell_checker
import kss
import re
import pandas as pd

df = pd.read_csv('data/kocohub_train.tsv', sep='\t')

'''
    TEXT PREPROCESSING
'''


# using api
def preprocess(text):
    sent = text.strip().replace('\n','').replace('&', '')
    spelled_sent = spell_checker.check(sent)
    checked_sent = spelled_sent.checked
    rp_norm = repeat_normalize(checked_sent, num_repeats=3)
    spacing = ' '.join(kss.split_sentences(rp_norm))
    hangeul = only_hangle_number(spacing)

    return hangeul

# using regular expression
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

df['comments'] = df['comments'].apply(lambda x:clean_text(x))

'''
    LABELING
'''

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

df = labeling(df)
df['label'] = df['label'].astype('int64')
print(df)