"""
Diagnose why sentiment indicators aren't improving model performance
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from utils import preprocessor

print("="*70)
print("DEEP DIAGNOSTIC: WHY SENTIMENT FEATURES AREN'T HELPING")
print("="*70)

df = pd.read_csv('data/train.tsv')
sample = df.sample(min(5000, len(df)), random_state=42)

# Check 1: Are sentiment indicators actually different between classes?
print("\n[CHECK 1] Sentiment Indicator Distribution")
print("-" * 70)

pos_samples = sample[sample['label'] == 'Positive']
neg_samples = sample[sample['label'] == 'Negative']

pos_indicator_counts = []
neg_indicator_counts = []

for _, row in pos_samples.iterrows():
    text_lower = row['text'].lower()
    pos_count = sum(1 for w in preprocessor.positive_indicators if w in text_lower)
    pos_indicator_counts.append(pos_count)

for _, row in neg_samples.iterrows():
    text_lower = row['text'].lower()
    pos_count = sum(1 for w in preprocessor.positive_indicators if w in text_lower)
    neg_indicator_counts.append(pos_count)

print(f"POSITIVE samples:")
print(f"  Mean positive indicators: {np.mean(pos_indicator_counts):.2f}")
print(f"  Median: {np.median(pos_indicator_counts):.2f}")
print(f"  Std: {np.std(pos_indicator_counts):.2f}")

print(f"\nNEGATIVE samples:")
print(f"  Mean positive indicators: {np.mean(neg_indicator_counts):.2f}")
print(f"  Median: {np.median(neg_indicator_counts):.2f}")
print(f"  Std: {np.std(neg_indicator_counts):.2f}")

diff = np.mean(pos_indicator_counts) - np.mean(neg_indicator_counts)
print(f"\nDifference in means: {diff:.3f}")
if abs(diff) < 0.1:
    print("⚠️  WARNING: Positive indicators appear equally in both classes!")

# Check 2: Sample some texts to see what's happening
print("\n\n[CHECK 2] Sample Analysis")
print("-" * 70)

print("\n>>> Positive samples WITH positive indicators:")
count = 0
for _, row in pos_samples.iterrows():
    text_lower = row['text'].lower()
    pos_found = [w for w in preprocessor.positive_indicators if w in text_lower]
    if pos_found and count < 3:
        print(f"\nText: {row['text'][:100]}...")
        print(f"Indicators found: {pos_found}")
        count += 1

print("\n>>> Negative samples WITH positive indicators:")
count = 0
for _, row in neg_samples.iterrows():
    text_lower = row['text'].lower()
    pos_found = [w for w in preprocessor.positive_indicators if w in text_lower]
    if pos_found and count < 3:
        print(f"\nText: {row['text'][:100]}...")
        print(f"Indicators found: {pos_found}")
        print("⚠️  This is a NEGATIVE sample with positive indicators!")
        count += 1

# Check 3: Is the data itself balanced and consistent?
print("\n\n[CHECK 3] Data Label Consistency")
print("-" * 70)

label_dist = sample['label'].value_counts()
print(f"Label distribution:")
for label, count in label_dist.items():
    print(f"  {label}: {count} ({100*count/len(sample):.1f}%)")

# Check 4: Text length comparison
print("\n\n[CHECK 4] Text Length Characteristics")
print("-" * 70)

pos_lengths = pos_samples['text'].str.len()
neg_lengths = neg_samples['text'].str.len()

print(f"POSITIVE texts:")
print(f"  Mean length: {pos_lengths.mean():.1f}")
print(f"  Median: {pos_lengths.median():.1f}")

print(f"\nNEGATIVE texts:")
print(f"  Mean length: {neg_lengths.mean():.1f}")
print(f"  Median: {neg_lengths.median():.1f}")

if abs(pos_lengths.mean() - neg_lengths.mean()) < 5:
    print(f"\nNote: Very similar text lengths between classes")

# Check 5: Word content similarity
print("\n\n[CHECK 5] Vocabulary Overlap Analysis")
print("-" * 70)

from collections import Counter

pos_text = ' '.join(pos_samples['text'].head(1000))
neg_text = ' '.join(neg_samples['text'].head(1000))

pos_words = set(pos_text.lower().split())
neg_words = set(neg_text.lower().split())

overlap = len(pos_words & neg_words)
total_unique = len(pos_words | neg_words)
overlap_ratio = overlap / total_unique if total_unique > 0 else 0

print(f"Unique words in positive samples: {len(pos_words)}")
print(f"Unique words in negative samples: {len(neg_words)}")
print(f"Shared words: {overlap}")
print(f"Overlap ratio: {overlap_ratio:.1%}")

if overlap_ratio > 0.7:
    print("\n⚠️  WARNING: Classes have very similar vocabulary!")
    print("    This suggests the data may be too similar for binary classification")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("""
If the checks above show:
1. Similar indicator distributions between classes → Sentiment indicators aren't predictive
2. High overlap in vocabulary → Classes use similar language
3. Similar text lengths → No meaningful structural difference

Then the problem may be that your data classes are genuinely hard to distinguish,
or the labels may not be reliable.
""")
