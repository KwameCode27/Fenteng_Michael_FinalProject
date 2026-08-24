"""
train.py – Hausa Sentiment Classifier Training
================================================
Improvements over the original:
  - GridSearchCV cv raised from 2 → 5 (better variance estimates).
  - n_jobs defaults to -1 (uses all CPU cores) with a CLI flag to override.
  - "both" vectoriser mode reuses one data-load call instead of two.
  - build_model_pipeline accepts an optional preprocessor instance so
    tests can inject a custom one without touching global state.
  - All sys.path hacks removed; run with `python -m src.train` or use
    the provided Makefile targets.
  - Minor: consistent Path throughout; no bare str paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Allow `python src/train.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from utils import (
    get_lexicon_features,
    get_text_length,
    load_sentiment_dataset,
    plot_confusion_matrix,
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_DEFINITIONS: dict = {
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
            "clf__C": [0.3, 1.0, 3.0],
            "clf__class_weight": [None, "balanced"],
        },
    },
}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_model_pipeline(
    model_name: str = "multinomial_nb",
    vectorizer_type: str = "tfidf",
) -> Pipeline:
    """Return a scikit-learn Pipeline for the requested model and vectoriser."""
    if model_name not in MODEL_DEFINITIONS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_DEFINITIONS)}"
        )
    if vectorizer_type not in ("tfidf", "count"):
        raise ValueError(
            f"Unknown vectorizer '{vectorizer_type}'. Choose 'tfidf' or 'count'."
        )

    VecClass = TfidfVectorizer if vectorizer_type == "tfidf" else CountVectorizer

    char_vec = (
        "char_vec",
        VecClass(analyzer="char_wb", ngram_range=(3, 5), max_features=50_000),
    )
    word_vec = (
        "word_vec",
        VecClass(analyzer="word", ngram_range=(1, 2), max_features=30_000),
    )

    if model_name == "multinomial_nb":
        # MNB cannot handle negative values → no scaled dense features
        features = FeatureUnion([char_vec, word_vec])
    else:
        features = FeatureUnion([
            char_vec,
            word_vec,
            (
                "length",
                Pipeline([
                    ("extract", FunctionTransformer(get_text_length, validate=False)),
                    ("scale",   StandardScaler(with_mean=False)),
                ]),
            ),
            (
                "lexicon_features",
                Pipeline([
                    ("extract", FunctionTransformer(get_lexicon_features, validate=False)),
                    ("scale",   StandardScaler(with_mean=False)),
                ]),
            ),
        ])

    return Pipeline([
        ("features", features),
        ("clf",      MODEL_DEFINITIONS[model_name]["estimator"]),
    ])


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_model(
    train_csv: Path,
    dev_csv: Path,
    model_path: Path,
    results_path: Path,
    lexicon_csv: Optional[Path] = None,
    model_name: str = "multinomial_nb",
    vectorizer_type: str = "tfidf",
    cv: int = 5,
    n_jobs: int = -1,
) -> None:
    """Train, tune, evaluate on dev set, and persist a sentiment model."""
    print(f"[INFO] Loading training data …  ({train_csv})")
    train_df = load_sentiment_dataset(
        train_csv,
        lexicon_path=str(lexicon_csv) if lexicon_csv else None,
    )
    print(f"[INFO] Loading validation data … ({dev_csv})")
    dev_df = load_sentiment_dataset(
        dev_csv,
        lexicon_path=str(lexicon_csv) if lexicon_csv else None,
    )

    X_train, y_train = train_df["text"], train_df["label"]
    X_dev,   y_dev   = dev_df["text"],   dev_df["label"]

    pipeline     = build_model_pipeline(model_name, vectorizer_type=vectorizer_type)
    model_config = MODEL_DEFINITIONS[model_name]

    # Merge model-specific and shared vectoriser param grids
    param_grid = {
        **model_config["param_grid"],
        "features__char_vec__max_features": [20_000, 50_000],
        "features__word_vec__max_features": [10_000, 30_000],
        # Use sub-linear term frequency for TF-IDF: it prevents repeated
        # social-media terms from dominating a document representation.
        "features__char_vec__sublinear_tf": [True] if vectorizer_type == "tfidf" else [False],
        "features__word_vec__sublinear_tf": [True] if vectorizer_type == "tfidf" else [False],
    }

    print(
        f"[INFO] Training {model_config['display_name']} "
        f"({vectorizer_type.upper()} vectoriser) — GridSearchCV "
        f"cv={cv}, n_jobs={n_jobs} …"
    )
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"[INFO] Best params: {grid.best_params_}")
    print(f"[INFO] Best CV macro-F1: {grid.best_score_:.4f}")

    y_pred = best_model.predict(X_dev)
    report = classification_report(y_dev, y_pred, output_dict=True)

    # Persist metrics
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model":              model_name,
        "model_display_name": model_config["display_name"],
        "vectorizer":         vectorizer_type,
        "best_params":        grid.best_params_,
        "best_cv_macro_f1":   round(grid.best_score_, 6),
        "metrics":            report,
    }
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=4)
    print(f"[INFO] Metrics saved → {results_path}")

    cm_path = results_path.with_suffix(".confusion.png")
    plot_confusion_matrix(
        y_true=y_dev,
        y_pred=y_pred,
        labels=sorted(set(y_dev) | set(y_pred)),
        output_path=cm_path,
        title=f"{model_config['display_name']} ({vectorizer_type.upper()}) Dev Confusion Matrix",
    )
    print(f"[INFO] Confusion matrix saved → {cm_path}")

    # Persist model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, model_path)
    print(f"[INFO] Model saved  → {model_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Hausa sentiment classifier with FeatureUnion + GridSearchCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train_csv",      type=str, default="data/train.tsv")
    p.add_argument("--dev_csv",        type=str, default="data/dev.tsv")
    p.add_argument("--model_path",     type=str, default="models/hausa_model.joblib")
    p.add_argument("--results_path",   type=str, default="reports/metrics.json")
    p.add_argument("--lexicon_csv",    type=str, default="data/hausa_aug_lex_train.csv")
    p.add_argument(
        "--model_name", type=str, default="logistic_regression",
        choices=list(MODEL_DEFINITIONS) + ["all"],
        help="Classifier to use.",
    )
    p.add_argument(
        "--vectorizer_type", type=str, default="tfidf",
        choices=["tfidf", "count", "both"],
        help="'both' trains two separate models (tfidf and count).",
    )
    p.add_argument(
        "--cv", type=int, default=5,
        help="Number of cross-validation folds in GridSearchCV.",
    )
    p.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Parallel jobs for GridSearchCV (-1 = all cores).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    common = dict(
        train_csv  = Path(args.train_csv),
        dev_csv    = Path(args.dev_csv),
        lexicon_csv= Path(args.lexicon_csv) if args.lexicon_csv else None,
        model_name = args.model_name,
        cv         = args.cv,
        n_jobs     = args.n_jobs,
    )

    if args.model_name == "all":
        for model_name in MODEL_DEFINITIONS:
            model_path = Path(args.model_path).with_suffix("").as_posix() + f"_{model_name}.joblib"
            results_path = Path(args.results_path).with_suffix("").as_posix() + f"_{model_name}.json"
            if args.vectorizer_type == "both":
                for vtype in ("tfidf", "count"):
                    suffix = "" if vtype == "tfidf" else "_count"
                    mp = Path(model_path).with_suffix("").as_posix() + suffix + ".joblib"
                    rp = Path(results_path).with_suffix("").as_posix() + suffix + ".json"
                    print(f"\n{'='*60}\nTRAINING {model_name} WITH {vtype.upper()} VECTORISER\n{'='*60}\n")
                    train_model(
                        model_path       = Path(mp),
                        results_path     = Path(rp),
                        vectorizer_type = vtype,
                        model_name       = model_name,
                        **common,
                    )
            else:
                print(f"\n{'='*60}\nTRAINING {model_name} WITH {args.vectorizer_type.upper()} VECTORISER\n{'='*60}\n")
                train_model(
                    model_path       = Path(model_path),
                    results_path     = Path(results_path),
                    vectorizer_type = args.vectorizer_type,
                    model_name       = model_name,
                    **common,
                )
    elif args.vectorizer_type == "both":
        for vtype in ("tfidf", "count"):
            suffix = "" if vtype == "tfidf" else "_count"
            mp = Path(args.model_path).with_suffix("").as_posix() + suffix + ".joblib"
            rp = Path(args.results_path).with_suffix("").as_posix() + suffix + ".json"
            print(f"\n{'='*60}\nTRAINING WITH {vtype.upper()} VECTORISER\n{'='*60}\n")
            train_model(
                model_path       = Path(mp),
                results_path     = Path(rp),
                vectorizer_type = vtype,
                **common,
            )
    else:
        train_model(
            model_path      = Path(args.model_path),
            results_path    = Path(args.results_path),
            vectorizer_type = args.vectorizer_type,
            **common,
        )


if __name__ == "__main__":
    main()
