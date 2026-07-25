import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from utils import load_sentiment_dataset, preprocessor, get_lexicon_features, get_text_length


MODEL_DEFINITIONS = {
    "multinomial_nb": {
        "display_name": "Multinomial Naïve Bayes",
        "estimator": MultinomialNB(),
        "param_grid": {
            "clf__alpha": [0.1, 0.5, 1.0],
        },
    },
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "estimator": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            tol=1e-4,
        ),
        "param_grid": {
            "clf__C": [0.1, 1, 10],
        },
    },
}


# ------------------------
# Data Loading
# ------------------------
def load_data(csv_path: Path, lexicon_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and preprocess dataset."""
    return load_sentiment_dataset(csv_path, lexicon_path=str(lexicon_path) if lexicon_path else None)


# ------------------------
# Training
# ------------------------
def build_model_pipeline(model_name: str = "multinomial_nb", vectorizer_type: str = "tfidf") -> Pipeline:
    if model_name not in MODEL_DEFINITIONS:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if vectorizer_type not in ["tfidf", "count"]:
        raise ValueError(f"Unsupported vectorizer: {vectorizer_type}")

    # Select vectorizer class based on type
    vectorizer_class = TfidfVectorizer if vectorizer_type == "tfidf" else CountVectorizer

    char_vec = ("char_vec", vectorizer_class(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=50000
    ))
    word_vec = ("word_vec", vectorizer_class(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=30000
    ))

    if model_name == "multinomial_nb":
        features = FeatureUnion([char_vec, word_vec])
    else:
        features = FeatureUnion([
            char_vec,
            word_vec,
            ("length", Pipeline([
                ("extract", FunctionTransformer(get_text_length, validate=False)),
                ("scale", StandardScaler(with_mean=False))
            ])),
            ("lexicon_features", Pipeline([
                ("extract", FunctionTransformer(get_lexicon_features, validate=False)),
                ("scale", StandardScaler(with_mean=False))
            ]))
        ])

    pipeline = Pipeline([
        ("features", features),
        ("clf", MODEL_DEFINITIONS[model_name]["estimator"])
    ])
    return pipeline


def train_model(train_csv: Path, dev_csv: Path, model_path: Path, results_path: Path, lexicon_csv: Optional[Path] = None, model_name: str = "multinomial_nb", vectorizer_type: str = "tfidf"):
    print(f"[INFO] Loading training data from {train_csv}...")
    train_df = load_data(train_csv, lexicon_path=lexicon_csv)
    print(f"[INFO] Loading validation data from {dev_csv}...")
    test_df = load_data(dev_csv, lexicon_path=lexicon_csv)

    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    pipeline = build_model_pipeline(model_name, vectorizer_type=vectorizer_type)
    model_config = MODEL_DEFINITIONS[model_name]
    param_grid = {
        **model_config["param_grid"],
        "features__char_vec__max_features": [20000, 50000],
        "features__word_vec__max_features": [10000, 30000],
    }

    print(f"[INFO] Training {model_config['display_name']} with {vectorizer_type.upper()} vectorizer...")
    print("[INFO] Running GridSearchCV...")
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=2,
        n_jobs=1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"[INFO] Best Params: {grid.best_params_}")

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "model": model_name,
            "model_display_name": model_config["display_name"],
            "vectorizer": vectorizer_type,
            "best_params": grid.best_params_,
            "metrics": report
        }, f, indent=4)
    print(f"[INFO] Metrics saved → {results_path}")

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
    parser.add_argument(
        "--model_name",
        type=str,
        default="multinomial_nb",
        choices=list(MODEL_DEFINITIONS.keys()),
        help="Model to train: multinomial_nb or logistic_regression"
    )
    parser.add_argument(
        "--vectorizer_type",
        type=str,
        default="tfidf",
        choices=["tfidf", "count", "both"],
        help="Vectorizer type: tfidf, count, or both (default: tfidf)"
    )
    args = parser.parse_args()

    if args.vectorizer_type == "both":
        # Train with both vectorizers
        print("\n" + "="*60)
        print("TRAINING WITH TF-IDF VECTORIZER")
        print("="*60 + "\n")
        train_model(
            Path(args.train_csv),
            Path(args.dev_csv),
            Path(args.model_path),
            Path(args.results_path),
            Path(args.lexicon_csv) if args.lexicon_csv else None,
            model_name=args.model_name,
            vectorizer_type="tfidf",
        )
        
        print("\n" + "="*60)
        print("TRAINING WITH COUNT VECTORIZER")
        print("="*60 + "\n")
        
        # Update paths for count vectorizer models
        count_model_path = str(args.model_path).replace(".joblib", "_count.joblib")
        count_results_path = str(args.results_path).replace(".json", "_count.json")
        
        train_model(
            Path(args.train_csv),
            Path(args.dev_csv),
            Path(count_model_path),
            Path(count_results_path),
            Path(args.lexicon_csv) if args.lexicon_csv else None,
            model_name=args.model_name,
            vectorizer_type="count",
        )
    else:
        train_model(
            Path(args.train_csv),
            Path(args.dev_csv),
            Path(args.model_path),
            Path(args.results_path),
            Path(args.lexicon_csv) if args.lexicon_csv else None,
            model_name=args.model_name,
            vectorizer_type=args.vectorizer_type,
        )


if __name__ == "__main__":
    main()
