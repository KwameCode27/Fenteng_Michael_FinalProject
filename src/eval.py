"""
eval.py – Model Evaluation on Held-Out Test Set
================================================
Improvements:
  - zero_division=0 passed to all sklearn metrics (avoids UndefinedMetricWarning
    when a class has no predicted samples).
  - Full text classification_report is also printed to stdout.
  - report_path defaults to reports/metrics_eval.json (avoids overwriting the
    training metrics.json by accident).
  - Encoding='utf-8' explicitly on JSON write.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from utils import load_sentiment_dataset, plot_confusion_matrix


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_path: Path,
    test_csv: Path,
    report_path: Path,
) -> None:
    """Load a saved model, run it on test_csv, and write metrics to report_path."""
    print(f"[INFO] Loading model …  {model_path}")
    model = load(model_path)

    print(f"[INFO] Loading test data … {test_csv}")
    test_df = load_sentiment_dataset(test_csv)
    X_test, y_test = test_df["text"], test_df["label"]

    print("[INFO] Running inference …")
    preds = model.predict(X_test)

    kw = dict(zero_division=0)
    acc       = accuracy_score(y_test, preds)
    f1        = f1_score(y_test, preds, average="macro", **kw)
    precision = precision_score(y_test, preds, average="macro", **kw)
    recall    = recall_score(y_test, preds, average="macro", **kw)
    cls_report = classification_report(y_test, preds, output_dict=True, **kw)
    cls_report_text = classification_report(y_test, preds, **kw)

    # Save confusion matrix plot next to the JSON report
    try:
        cm_path = Path(report_path).with_suffix('.confusion.png')
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=preds,
            labels=sorted(set(y_test) | set(preds)),
            output_path=cm_path,
            title=f"Confusion Matrix: {Path(model_path).name}",
        )
        print(f"[INFO] Confusion matrix saved → {cm_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Could not save confusion matrix: {exc}")

    metrics = {
        "task":                    "sentiment_analysis",
        "language":                "hausa",
        "model_path":              str(model_path),
        "test_set":                str(test_csv),
        "accuracy":                round(acc, 6),
        "macro_f1":                round(f1, 6),
        "macro_precision":         round(precision, 6),
        "macro_recall":            round(recall, 6),
        "classification_report":   cls_report,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=4)

    print("\n=== Evaluation Results ===")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Macro-F1:          {f1:.4f}")
    print(f"  Macro-Precision:   {precision:.4f}")
    print(f"  Macro-Recall:      {recall:.4f}")
    print(f"\n{cls_report_text}")
    print(f"[INFO] Report saved → {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a trained Hausa sentiment model on a test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path",  type=str, default="models/hausa_model.joblib")
    p.add_argument("--test_csv",    type=str, default="data/test.tsv")
    p.add_argument("--report_path", type=str, default="reports/metrics_eval.json")
    args = p.parse_args()

    evaluate(
        Path(args.model_path),
        Path(args.test_csv),
        Path(args.report_path),
    )


if __name__ == "__main__":
    main()
