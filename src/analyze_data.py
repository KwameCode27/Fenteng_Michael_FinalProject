import pandas as pd
import numpy as np

df = pd.read_csv('data/train.tsv')

print('SAMPLE TEXTS BY CLASS')
print('='*60)
print('\nNegative Class Samples:')
neg_samples = df[df['label'] == 'Negative'].head(3)
for idx, row in neg_samples.iterrows():
    print(f"  {row['text'][:80]}...")

print('\nPositive Class Samples:')
pos_samples = df[df['label'] == 'Positive'].head(3)
for idx, row in pos_samples.iterrows():
    print(f"  {row['text'][:80]}...")

# Check for duplicates
print(f'\n\nDATA QUALITY CHECKS')
print('='*60)
duplicates = df.duplicated(subset=['text']).sum()
print(f'Duplicate texts: {duplicates} ({100*duplicates/len(df):.2f}%)')

# Check label consistency
print(f'\nTexts with multiple labels:')
text_labels = df.groupby('text')['label'].nunique()
multi_label = (text_labels > 1).sum()
print(f'  {multi_label} texts have conflicting labels')

# Check for near-empty texts
short_texts = df[df['text'].astype(str).str.len() < 15]
print(f'\nVery short texts (<15 chars): {len(short_texts)} ({100*len(short_texts)/len(df):.2f}%)')
print(short_texts.head(3))

# Check unique values in label column
print(f'\nUnique labels: {df["label"].unique()}')
print(f'Label value counts:')
print(df['label'].value_counts())
