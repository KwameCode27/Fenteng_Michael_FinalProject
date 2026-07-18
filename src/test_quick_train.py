"""
Fast test of model with expanded sentiment lexicon
Using smaller sample for quick verification
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from utils import preprocessor

def extract_sentiment_features(X):
    """Extract lightweight sentiment features (all non-negative)."""
    n_samples = len(X) if isinstance(X, list) else len(X)
    features = np.zeros((n_samples, 7))
    
    for i, text in enumerate(X):
        text_lower = str(text).lower()
        pos_count = sum(1 for w in preprocessor.positive_indicators if w in text_lower)
        neg_count = sum(1 for w in preprocessor.negative_indicators if w in text_lower)
        
        features[i, 0] = pos_count
        features[i, 1] = neg_count
        features[i, 2] = max(pos_count - neg_count, 0)
        features[i, 3] = max(neg_count - pos_count, 0)
        features[i, 4] = text_lower.count('!')
        features[i, 5] = text_lower.count('?')
        features[i, 6] = len(text) / 100.0
    
    return features

# Test on smaller sample (50k instead of 670k)
print("[INFO] Loading sample data (50k)...")
df = pd.read_csv('data/train_hausa.csv')
df = df.sample(min(50000, len(df)), random_state=42)

print("[INFO] Preprocessing...")
df["text"] = df["text"].astype(str).map(
    lambda x: preprocessor.preprocess(x, remove_stopwords=False)
)

print("[INFO] Train/test split (80/20)...")
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

X_train, y_train = train_df["text"], train_df["label"]
X_test, y_test = test_df["text"], test_df["label"]

print("[INFO] Creating feature pipeline...")
features = FeatureUnion([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.9
    )),
    ("sentiment", FunctionTransformer(extract_sentiment_features, validate=False))
])

print("\n" + "="*60)
print("TESTING MODELS ON 50K SAMPLE")
print("="*60)

models = {
    "LogisticRegression": LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        solver="liblinear",
        random_state=42
    )
}

for name, clf in models.items():
    print(f"\n[{name}]")
    
    pipeline = Pipeline([("features", features), ("clf", clf)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

print("\n✓ Test completed successfully")
print("If accuracy > 55%, the expanded lexicon is helping!")
