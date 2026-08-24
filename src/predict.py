"""
predict.py – Batch Inference on Unlabelled Text
================================================
Improvements over the original:
  - The file contained two full copies of the code (a duplication bug).
    The clean single implementation is kept here.
  - Original metadata columns (title, date, source, url, etc.) are
    preserved alongside the prediction in the output CSV so the reader
    does not lose context.
  - content column alias added.
  - Encoding='utf-8' on CSV write.
  - Type hint uses the | union syntax for Python 3.10+; falls back
    gracefully on 3.9 via __future__ annotations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from joblib import load

from utils import get_preprocessor

# Text-column aliases accepted by this script
_TEXT_ALIASES = ("text", "tweet", "article", "content", "title")


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_prediction_dataset(csv_path: Path) -> pd.DataFrame:
    """Load an unlabelled dataset and return it with a normalised 'text' column.

    All original columns are preserved; the recognised text column is also
    exposed as 'text' for model inference.
    """
    csv_path = Path(csv_path)
    sep = "\t" if csv_path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(csv_path, sep=sep)

    text_col: Optional[str] = None
    for alias in _TEXT_ALIASES:
        if alias in df.columns:
            text_col = alias
            break

    if text_col is None:
        raise ValueError(
            f"Could not find a text column in {csv_path}. "
            f"Expected one of: {_TEXT_ALIASES}. Found: {list(df.columns)}"
        )

    _pp = get_preprocessor()
    df["text"] = (
        df[text_col]
        .fillna("")
        .astype(str)
        .map(_pp.preprocess)
    )
    return df


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    model_path: str | Path,
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Run the saved model on *input_path* and return a results DataFrame.

    The output contains all original columns plus a ``predicted_label`` column.
    If *output_path* is provided the result is also saved as CSV.
    """
    model    = load(model_path)
    input_df = load_prediction_dataset(Path(input_path))

    preds = model.predict(input_df["text"])
    result_df = input_df.copy()
    result_df["predicted_label"] = preds

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out, index=False, encoding="utf-8")
        print(f"[INFO] Predictions saved → {out}")

    return result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run batch sentiment inference on an unlabelled text file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path",  type=str, default="models/hausa_model_lr.joblib")
    p.add_argument("--input_path",  type=str, default="data/hausa_news_articles.csv")
    p.add_argument("--output_path", type=str, default="reports/predictions.csv")
    args = p.parse_args()

    result = predict(args.model_path, args.input_path, args.output_path)
    print(result[["text", "predicted_label"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
