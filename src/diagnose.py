"""
Diagnostic script to identify why model performance is at 50%
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import numpy as np
from collections import Counter
from utils import preprocessor

def analyze_preprocessing_impact():
    """Show preprocessing impact on samples"""
    print("="*70)
    print("PREPROCESSING IMPACT ANALYSIS")
    print("="*70)
    
    # Load training data
    df = pd.read_csv('data/train_hausa.csv')
    
    # Sample from each class
    positive_samples = df[df['label'] == 'Positive'].sample(3, random_state=42)
    negative_samples = df[df['label'] == 'Negative'].sample(3, random_state=42)
    
    print("\n>>> POSITIVE CLASS SAMPLES:")
    for idx, (_, row) in enumerate(positive_samples.iterrows(), 1):
        original = row['text']
        preprocessed = preprocessor.preprocess(original, remove_stopwords=False)
        print(f"\n[Positive Sample {idx}]")
        print(f"Original length: {len(original)}")
        print(f"Original: {original[:100]}...")
        print(f"Processed length: {len(preprocessed)}")
        print(f"Processed: {preprocessed[:100]}...")
        print(f"Loss: {(1 - len(preprocessed)/len(original))*100:.1f}%")
        
        # Check sentiment indicators
        text_lower = original.lower()
        pos_found = [w for w in preprocessor.positive_indicators if w in text_lower]
        neg_found = [w for w in preprocessor.negative_indicators if w in text_lower]
        print(f"Sentiment indicators found: {len(pos_found)} positive, {len(neg_found)} negative")
        if pos_found:
            print(f"  Positive: {pos_found}")
        if neg_found:
            print(f"  Negative: {neg_found}")
    
    print("\n\n>>> NEGATIVE CLASS SAMPLES:")
    for idx, (_, row) in enumerate(negative_samples.iterrows(), 1):
        original = row['text']
        preprocessed = preprocessor.preprocess(original, remove_stopwords=False)
        print(f"\n[Negative Sample {idx}]")
        print(f"Original length: {len(original)}")
        print(f"Original: {original[:100]}...")
        print(f"Processed length: {len(preprocessed)}")
        print(f"Processed: {preprocessed[:100]}...")
        print(f"Loss: {(1 - len(preprocessed)/len(original))*100:.1f}%")
        
        # Check sentiment indicators
        text_lower = original.lower()
        pos_found = [w for w in preprocessor.positive_indicators if w in text_lower]
        neg_found = [w for w in preprocessor.negative_indicators if w in text_lower]
        print(f"Sentiment indicators found: {len(pos_found)} positive, {len(neg_found)} negative")
        if pos_found:
            print(f"  Positive: {pos_found}")
        if neg_found:
            print(f"  Negative: {neg_found}")


def analyze_sentiment_distribution():
    """Check if sentiment indicators correlate with labels"""
    print("\n\n" + "="*70)
    print("SENTIMENT INDICATOR CORRELATION WITH LABELS")
    print("="*70)
    
    df = pd.read_csv('data/train_hausa.csv')
    
    # Sample to avoid processing all 670k
    sample_df = df.sample(min(5000, len(df)), random_state=42)
    
    print(f"\nAnalyzing {len(sample_df)} samples...")
    
    pos_with_pos_indicators = 0
    pos_with_neg_indicators = 0
    neg_with_pos_indicators = 0
    neg_with_neg_indicators = 0
    
    pos_indicator_counts = []
    neg_indicator_counts = []
    
    for _, row in sample_df.iterrows():
        text_lower = row['text'].lower()
        label = row['label']
        
        pos_count = sum(1 for w in preprocessor.positive_indicators if w in text_lower)
        neg_count = sum(1 for w in preprocessor.negative_indicators if w in text_lower)
        
        if label == 'Positive':
            if pos_count > 0:
                pos_with_pos_indicators += 1
            if neg_count > 0:
                pos_with_neg_indicators += 1
            pos_indicator_counts.append((pos_count, neg_count))
        else:
            if pos_count > 0:
                neg_with_pos_indicators += 1
            if neg_count > 0:
                neg_with_neg_indicators += 1
            neg_indicator_counts.append((pos_count, neg_count))
    
    n_pos = int((sample_df['label'] == 'Positive').sum())
    n_neg = int((sample_df['label'] == 'Negative').sum())
    
    print(f"\n>>> POSITIVE CLASS ({n_pos} samples):")
    print(f"  - Have positive indicators: {pos_with_pos_indicators}/{n_pos} ({100*pos_with_pos_indicators/n_pos:.1f}%)")
    print(f"  - Have negative indicators: {pos_with_neg_indicators}/{n_pos} ({100*pos_with_neg_indicators/n_pos:.1f}%)")
    
    print(f"\n>>> NEGATIVE CLASS ({n_neg} samples):")
    print(f"  - Have positive indicators: {neg_with_pos_indicators}/{n_neg} ({100*neg_with_pos_indicators/n_neg:.1f}%)")
    print(f"  - Have negative indicators: {neg_with_neg_indicators}/{n_neg} ({100*neg_with_neg_indicators/n_neg:.1f}%)")
    
    # Analyze indicator coverage
    if pos_indicator_counts:
        pos_vals = np.array(pos_indicator_counts)
        print(f"\n>>> POSITIVE INDICATORS IN POSITIVE CLASS:")
        print(f"  - Mean: {pos_vals[:, 0].mean():.2f}")
        print(f"  - Median: {np.median(pos_vals[:, 0]):.2f}")
        print(f"  - Max: {pos_vals[:, 0].max()}")
        
        print(f"\n>>> NEGATIVE INDICATORS IN POSITIVE CLASS:")
        print(f"  - Mean: {pos_vals[:, 1].mean():.2f}")
        print(f"  - Median: {np.median(pos_vals[:, 1]):.2f}")
        print(f"  - Max: {pos_vals[:, 1].max()}")


def find_missing_sentiment_words():
    """Try to identify sentiment words not in current lists"""
    print("\n\n" + "="*70)
    print("IDENTIFYING MISSING SENTIMENT WORDS")
    print("="*70)
    
    df = pd.read_csv('data/train_hausa.csv')
    
    # Sample and tokenize
    sample = df.sample(min(2000, len(df)), random_state=42)
    
    positive_words = Counter()
    negative_words = Counter()
    
    print("\nExtracting words from sample...")
    
    for _, row in sample.iterrows():
        text = row['text'].lower()
        words = text.split()
        
        if row['label'] == 'Positive':
            positive_words.update(words)
        else:
            negative_words.update(words)
    
    # Find words that appear more in one class
    all_words = set(positive_words.keys()) | set(negative_words.keys())
    
    # Calculate bias (p count - n count)
    word_bias = {}
    for word in all_words:
        p_count = positive_words[word]
        n_count = negative_words[word]
        total = p_count + n_count
        
        if total >= 10:  # Only words with enough samples
            bias = (p_count - n_count) / total
            word_bias[word] = (bias, p_count, n_count)
    
    # Sort by bias
    sorted_words = sorted(word_bias.items(), key=lambda x: x[1][0], reverse=True)
    
    print("\n>>> TOP 30 WORDS BIASED TOWARD POSITIVE:")
    for word, (bias, p_count, n_count) in sorted_words[:30]:
        # Skip if already in lists
        if word in preprocessor.positive_indicators or word in preprocessor.negative_indicators:
            continue
        print(f"  {word:20} bias={bias:6.2f}  P={p_count:5d}  N={n_count:5d}")
    
    print("\n>>> TOP 30 WORDS BIASED TOWARD NEGATIVE:")
    for word, (bias, p_count, n_count) in reversed(sorted_words[-30:]):
        # Skip if already in lists
        if word in preprocessor.positive_indicators or word in preprocessor.negative_indicators:
            continue
        print(f"  {word:20} bias={bias:6.2f}  P={p_count:5d}  N={n_count:5d}")


if __name__ == "__main__":
    try:
        analyze_preprocessing_impact()
        analyze_sentiment_distribution()
        find_missing_sentiment_words()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
