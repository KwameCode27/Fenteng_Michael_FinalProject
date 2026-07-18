import pandas as pd
from src.utils import preprocessor

# Check what preprocessing does to sample texts
df = pd.read_csv('data/train.tsv')

print('PREPROCESSING ANALYSIS')
print('='*70)

# Negative samples
print('\nNegative Texts (BEFORE & AFTER preprocessing):')
neg_df = df[df['label'] == 'Negative'].head(3)
for idx, row in neg_df.iterrows():
    original = row['text']
    processed = preprocessor.preprocess(original)
    print(f"\n  ORIGINAL ({len(original)} chars): {original[:70]}...")
    print(f"  PROCESSED ({len(processed)} chars): {processed[:70]}...")

# Positive samples
print('\n\nPositive Texts (BEFORE & AFTER preprocessing):')
pos_df = df[df['label'] == 'Positive'].head(3)
for idx, row in pos_df.iterrows():
    original = row['text']
    processed = preprocessor.preprocess(original)
    print(f"\n  ORIGINAL ({len(original)} chars): {original[:70]}...")
    print(f"  PROCESSED ({len(processed)} chars): {processed[:70]}...")

# Analyze feature extraction
print('\n\nFEATURE EXTRACTION TEST')
print('='*70)
sample_text = "Jin dadi! Wannan aiki ne kyau sosai, an yi albarka!"
features = preprocessor.extract_features(sample_text)
print(f'Sample text: {sample_text}')
print(f'\nExtracted features:')
for feat, val in features.items():
    print(f'  {feat}: {val}')
