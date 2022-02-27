from transformers import pipeline
import pandas as pd

# use train.gender_bias.binary.csv / train.bias.ternary.csv / train.hate.csv
pipe = pipeline('text-classification', model='monologg/koelectra-base-v3-discriminator', device=0)

df = pd.read_csv('data/test.csv')

df['label'] = df['comments'].map(lambda x:pipe(x)[0])['label']

print(df.label.value_counts())

# ex) hate-speech-detection
LABEL_DIC = {
    'none':0,
    'offensive':1,
    'hate':2,
}

df['label'] = df['label'].map(lambda x:LABEL_DIC[x])

df.to_csv('./submission.csv', index=None)