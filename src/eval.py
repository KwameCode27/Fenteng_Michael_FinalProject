import argparse
from pathlib import Path
import pandas as pd
import json
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from joblib import load

from utils import preprocessor , get_text_length # HausaTextPreprocessor instance


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess dataset for evaluation."""
    df = pd.read_csv(csv_path)
    if "tweet" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"tweet": "text"})
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label (or tweet,label)")

    # Apply Hausa preprocessing
    df["text"] = df["text"].astype(str).map(preprocessor.preprocess)
    return df


def evaluate(model_path: Path, test_csv: Path, report_path: Path) -> None:
    """Evaluate a trained model on a test set and save metrics."""
    print(f"Loading model from {model_path}...")
    model = load(model_path)

    print(f"Loading test data from {test_csv}...")
    test_df = load_data(test_csv)

    X_test, y_test = test_df["text"], test_df["label"]

    print("Evaluating model...")
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    precision = precision_score(y_test, preds, average="macro")
    recall = recall_score(y_test, preds, average="macro")

    # Save metrics to JSON
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "task": "sentiment_analysis",
        "language": "hausa",
        "accuracy": acc,
        "macro_f1": f1,
        "precision": precision,
        "recall": recall,
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro-F1:  {f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}\n")
    print(f"Metrics saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Hausa sentiment model.")
    parser.add_argument("--model_path", type=str, default="models/hausa_model.joblib", help="Path to trained model (.joblib)")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to test CSV file (default: data/test.csv)")
    parser.add_argument("--report_path", type=str, default="reports/metrics.json", help="Path to save evaluation metrics JSON")
    args = parser.parse_args()

    evaluate(Path(args.model_path), Path(args.test_csv), Path(args.report_path))


if __name__ == "__main__":
    main()
