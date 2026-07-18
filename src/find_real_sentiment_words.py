"""
Find ACTUAL sentiment words by analyzing statistical difference between classes
Uses TF-IDF and chi-square to identify truly discriminative words
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FINDING ACTUAL SENTIMENT WORDS")
print("="*70)

df = pd.read_csv('data/train_hausa.csv')
# Use larger sample for better statistics
sample = df.sample(min(50000, len(df)), random_state=42)

print(f"\nAnalyzing {len(sample)} samples...")

# Vectorize with minimal preprocessing
vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 1),
    max_features=10000,
    min_df=5,
    max_df=0.7,
    lowercase=True
)

X = vectorizer.fit_transform(sample['text'])
y_encoded = (sample['label'] == 'Positive').astype(int)

# Get feature names
feature_names = np.array(vectorizer.get_feature_names_out())

# Calculate chi-square scores
chi2_scores, p_values = chi2(X, y_encoded)

# Get top words for each class
top_positive_indices = np.argsort(-chi2_scores)[:100]
top_positive_words = feature_names[top_positive_indices]

print("\n" + "="*70)
print("TOP 50 WORDS MOST ASSOCIATED WITH POSITIVE SENTIMENT")
print("="*70)
print("\nThese are statistically most predictive of positive class:")
for i, word in enumerate(top_positive_words[:50], 1):
    if len(word) > 2:  # Skip very short words
        print(f"{i:2d}. {word}")

print("\n" + "="*70)
print("TOP 50 WORDS MOST ASSOCIATED WITH NEGATIVE SENTIMENT")
print("="*70)

top_negative_indices = np.argsort(chi2_scores)[:100]
top_negative_words = feature_names[top_negative_indices]

print("\nThese are statistically most predictive of negative class:")
for i, word in enumerate(top_negative_words[:50], 1):
    if len(word) > 2:  # Skip very short words
        print(f"{i:2d}. {word}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print(f"""
The words listed above are statistically more predictive of each class.

NEXT STEPS:
1. Review these words - do they make sense as sentiment indicators?
2. If yes: Update src/utils.py with these words instead of generic ones
3. If no: The dataset may not have clear sentiment patterns
4. Consider: Are these labels actually for sentiment or some other task?

You can copy the words from above into the positive_indicators and
negative_indicators sets in src/utils.py for better results.
""")
