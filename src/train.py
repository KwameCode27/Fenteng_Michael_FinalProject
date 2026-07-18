import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from utils import load_sentiment_dataset, preprocessor


# ------------------------
# Data Loading
# ------------------------
def load_data(csv_path: Path, lexicon_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and preprocess dataset."""
    return load_sentiment_dataset(csv_path, lexicon_path=str(lexicon_path) if lexicon_path else None)


# ------------------------
# Custom Feature Extractor
# ------------------------
def get_text_length(X):
    """Return text lengths as numeric feature."""
    return np.array([len(t) for t in X]).reshape(-1, 1)


def get_lexicon_features(X):
    """Return lexicon-derived numeric features for each sample."""
    rows = []
    for text in X:
        features = preprocessor.extract_features(text)
        rows.append([
            features["positive_indicators"],
            features["negative_indicators"],
            features["sentiment_polarity"],
            features["text_length"],
            features["word_count"],
            features["avg_word_length"],
            features["unique_word_ratio"],
            features["exclamation_count"],
            features["question_count"],
            features["caps_ratio"],
            features["hausa_char_ratio"],
        ])
    return np.array(rows, dtype=float)


# ------------------------
# Training
# ------------------------
def train_model(train_csv: Path, dev_csv: Path, model_path: Path, results_path: Path, lexicon_csv: Optional[Path] = None):
    print(f"[INFO] Loading training data from {train_csv}...")
    train_df = load_data(train_csv, lexicon_path=lexicon_csv)
    print(f"[INFO] Loading validation data from {dev_csv}...")
    test_df = load_data(dev_csv, lexicon_path=lexicon_csv)

    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    # FeatureUnion: char n-grams + word n-grams + text length
    features = FeatureUnion([
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=50000
        )),
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=30000
        )),
        ("length", Pipeline([
            ("extract", FunctionTransformer(get_text_length, validate=False))
        ])),
        ("lexicon_features", Pipeline([
            ("extract", FunctionTransformer(get_lexicon_features, validate=False))
        ]))
    ])

    pipeline = Pipeline([
        ("features", features),
        ("clf", LinearSVC(class_weight="balanced", random_state=42))
    ])


    # Hyperparameter tuning
    param_grid = {
        "clf__C": [0.1, 1, 10],
        "features__char_tfidf__max_features": [20000, 50000],
        "features__word_tfidf__max_features": [10000, 30000],
    }

# display progress
    print("[INFO] Running GridSearchCV...")
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"[INFO] Best Params: {grid.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save metrics
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "best_params": grid.best_params_,
            "metrics": report
        }, f, indent=4)
    print(f"[INFO] Metrics saved → {results_path}")

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, model_path)
    print(f"[INFO] Model saved → {model_path}")


# ------------------------
# CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a Hausa sentiment classifier with FeatureUnion + tuning.")
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/train.tsv",
        help="Path to training CSV/TSV (default: data/train.tsv)"
    )
    parser.add_argument(
        "--dev_csv",
        type=str,
        default="data/dev.tsv",
        help="Path to validation CSV/TSV (default: data/dev.tsv)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/hausa_model.joblib",
        help="Where to save the trained model (default: models/hausa_model.joblib)"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="reports/metrics.json",
        help="Where to save training results (default: reports/metrics.json)"
    )
    parser.add_argument(
        "--lexicon_csv",
        type=str,
        default="data/hausa_aug_lex_train.csv",
        help="Path to a sentiment lexicon CSV/TSV used for feature engineering (default: data/hausa_aug_lex_train.csv)"
    )
    args = parser.parse_args()

    train_model(
        Path(args.train_csv),
        Path(args.dev_csv),
        Path(args.model_path),
        Path(args.results_path),
        Path(args.lexicon_csv) if args.lexicon_csv else None,
    )


if __name__ == "__main__":
    main()
