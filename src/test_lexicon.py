"""
Quick test to verify expanded sentiment lexicon
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import preprocessor
import pandas as pd

# Load a small sample
df = pd.read_csv('data/twitter/hausa_aug_lex_train.csv')
sample_df = df.sample(min(100, len(df)), random_state=42)

print(f"Sentiment indicator set sizes:")
print(f"  Positive indicators: {len(preprocessor.positive_indicators)}")
print(f"  Negative indicators: {len(preprocessor.negative_indicators)}")

# Count coverage
pos_with_indicator = 0
neg_with_indicator = 0

for _, row in sample_df.iterrows():
    text_lower = row['text'].lower()
    
    pos_found = sum(1 for w in preprocessor.positive_indicators if w in text_lower)
    neg_found = sum(1 for w in preprocessor.negative_indicators if w in text_lower)
    
    if row['label'] == 'Positive' and pos_found > 0:
        pos_with_indicator += 1
    if row['label'] == 'Negative' and neg_found > 0:
        neg_with_indicator += 1

n_pos = (sample_df['label'] == 'Positive').sum()
n_neg = (sample_df['label'] == 'Negative').sum()

print(f"\nIn 100-sample test:")
print(f"  Positive with indicators: {pos_with_indicator}/{n_pos} ({100*pos_with_indicator/n_pos:.1f}%)")
print(f"  Negative with indicators: {neg_with_indicator}/{n_neg} ({100*neg_with_indicator/n_neg:.1f}%)")

print("\n✓ Lexicon loaded successfully")
