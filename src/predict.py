import argparse
from pathlib import Path

import pandas as pd
from joblib import load

from utils import preprocessor


def load_prediction_dataset(csv_path: Path) -> pd.DataFrame:
    """Load a plain-text dataset for inference.

    Accepts files with either a ``text`` column or a common article-like column such as
    ``article`` or ``tweet`` and returns a dataframe with a single ``text`` column.
    """
    csv_path = Path(csv_path)
    sep = "\t" if csv_path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(csv_path, sep=sep)

    if "text" in df.columns:
        text_col = "text"
    elif "article" in df.columns:
        text_col = "article"
    elif "tweet" in df.columns:
        text_col = "tweet"
    elif "content" in df.columns:
        text_col = "content"
    else:
        raise ValueError("Prediction dataset must contain a text/article/tweet/content column")

    df = pd.DataFrame({"text": df[text_col].fillna("").astype(str)})
    df["text"] = df["text"].map(lambda value: preprocessor.preprocess(value))
    return df


def predict_from_file(model_path: str, input_path: str) -> pd.Series:
    model = load(model_path)
    df = load_prediction_dataset(Path(input_path))
    return model.predict(df["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a plain-text dataset")
    parser.add_argument("--model_path", type=str, default="models/hausa_model.joblib")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    preds = predict_from_file(args.model_path, args.input_path)

    if args.output_path:
        pd.DataFrame({"prediction": preds}).to_csv(args.output_path, index=False)
    else:
        print(preds)


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path

import pandas as pd
from joblib import load

from utils import preprocessor


def load_prediction_dataset(csv_path: Path) -> pd.DataFrame:
    """Load a plain-text dataset for prediction.

    Supports files with a text/tweet/article column and returns a dataframe with
    a single column named text for model prediction.
    """
    csv_path = Path(csv_path)
    sep = "\t" if csv_path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(csv_path, sep=sep)

    if "tweet" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"tweet": "text"})
    elif "article" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"article": "text"})
    elif "title" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"title": "text"})

    if "text" not in df.columns:
        raise ValueError("Prediction dataset must contain a text/tweet/article/title column")

    df = df[["text"]].dropna()
    df["text"] = df["text"].astype(str).map(lambda text: preprocessor.preprocess(text))
    return df


def predict(model_path: str, input_path: str, output_path: str | None = None) -> pd.DataFrame:
    model = load(model_path)
    input_df = load_prediction_dataset(Path(input_path))
    predictions = model.predict(input_df["text"])

    result_df = pd.DataFrame({"text": input_df["text"], "prediction": predictions})
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False, encoding="utf-8")

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictions on a plain-text dataset")
    parser.add_argument("--model_path", type=str, default="models/hausa_model.joblib")
    parser.add_argument("--input_path", type=str, default="data/hausa_news_articles.csv")
    parser.add_argument("--output_path", type=str, default="reports/predictions.csv")
    args = parser.parse_args()

    result = predict(args.model_path, args.input_path, args.output_path)
    print(result.head())


if __name__ == "__main__":
    main()
