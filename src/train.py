import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from utils import preprocessor


# ------------------------
# Data Loading
# ------------------------
def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess dataset."""
    df = pd.read_csv(csv_path)
    if "tweet" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"tweet": "text"})
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label (or tweet,label)")

    # Hausa preprocessing
    df["text"] = df["text"].astype(str).map(preprocessor.preprocess)
    return df


# ------------------------
# Custom Feature Extractor
# ------------------------
def get_text_length(X):
    """Return text lengths as numeric feature."""
    return np.array([len(t) for t in X]).reshape(-1, 1)


# ------------------------
# Training
# ------------------------
def train_model(train_csv: Path, model_path: Path, results_path: Path):
    print(f"[INFO] Loading training data from {train_csv}...")
    df = load_data(train_csv)

    # Train/Test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Save splits
    train_split_path = Path("data/train_split.csv")
    test_split_path = Path("data/test.csv")
    train_df.to_csv(train_split_path, index=False)
    test_df.to_csv(test_split_path, index=False)
    print(f"[INFO] Saved train split → {train_split_path}")
    print(f"[INFO] Saved test split → {test_split_path}")

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
        default="data/train_hausa.csv",
        help="Path to training CSV (default: data/train_hausa.csv)"
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
    args = parser.parse_args()

    train_model(Path(args.train_csv), Path(args.model_path), Path(args.results_path))


if __name__ == "__main__":
    main()
