import argparse
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump

from utils import preprocessor 

def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess training dataset."""
    df = pd.read_csv(csv_path)
    if "tweet" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"tweet": "text"})
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label (or tweet,label)")

    # Apply Hausa preprocessing
    df["text"] = df["text"].astype(str).map(preprocessor.preprocess)
    return df


def train_model(train_csv: Path, model_path: Path):
    """Train a Logistic Regression model with TF-IDF + n-grams."""
    print(f"Loading training data from {train_csv}...")
    df = load_data(train_csv)


      # Split into train/test
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    # Save splits for later evaluation
    train_split_path = Path("data/train_split.csv")
    test_split_path = Path("data/test.csv")
    train_df.to_csv(train_split_path, index=False)
    test_df.to_csv(test_split_path, index=False)
    print(f"Saved train split to {train_split_path}")
    print(f"Saved test split to {test_split_path}")

    
    X_train, y_train = train_df["text"], train_df["label"]

    # Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
       
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),    # character 3–5 grams
            max_features=200000
        )),
        ("clf", LinearSVC(class_weight="balanced", random_state=42))
    ])
    

    # # Hyperparameter tuning
    # params = {
    #     "clf__C": [0.1, 1, 10]
    # }
    # grid = GridSearchCV(
    # pipeline,
    # param_grid=params,
    # scoring="f1_macro",
    # cv=5,
    # n_jobs=1,   # 👈 run sequentially, avoids multiprocessing issue
    # verbose=2
    # )


    print("Training model with GridSearchCV...")
    pipeline.fit(X_train, y_train)

    # Best model
    best_model = pipeline
    # print(f"Best Params: {grid.best_params_}")

    # # Evaluate on dev set
    # dev_preds = best_model.predict(X_dev)
    # dev_f1 = f1_score(y_dev, dev_preds, average="macro")
    # print(f"Validation Macro-F1: {dev_f1:.3f}")

    # Save trained model
    dump(best_model, model_path)
    print(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Hausa sentiment classifier.")
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/train_hausa.csv",
        help="Path to training CSV file (default: data/train_hausa.csv)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/hausa_model.joblib",
        help="Where to save the trained model (default: models/hausa_model.joblib)"
    )
    args = parser.parse_args()

    train_model(Path(args.train_csv), Path(args.model_path))


if __name__ == "__main__":
    main()
