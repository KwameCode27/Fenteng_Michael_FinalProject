# Machine Learning-Based Sentiment Analysis for Hausa Social Media Content in Ghana.

This repository implements a sentiment analysis pipeline for Hausa and English social-media text. It supports data loading, preprocessing, model training, evaluation, and prediction for tasks such as polarity classification.

## Project overview

The current workflow uses a scikit-learn pipeline with:

- custom Hausa text preprocessing
- TF-IDF features
- optional lexicon-based features
- two supported classifiers:
  - Multinomial Naive Bayes
  - Logistic Regression

The project is designed to work with CSV and TSV datasets that contain at least a text column and a label column.

---

## Project structure

```text
project/
├── data/                  # Training, validation, test, and lexicon datasets
├── models/                # Trained model artifacts (.joblib)
├── reports/               # Evaluation metrics and prediction outputs
├── src/
│   ├── train.py           # Training script
│   ├── eval.py            # Evaluation script
│   ├── predict.py         # Prediction/inference script
│   ├── utils.py           # Preprocessing utilities and dataset loader
│   └── ...
├── tests/                 # Regression and model-selection tests
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Requirements

- Python 3.9+
- Tested with Python 3.13

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
git clone <repository-url>
cd Fenteng_Michael_FinalProject
pip install -r requirements.txt
```

---

## Training a model

Train a model with the main training script:

```bash
py -3 src/train.py \
  --train_csv data/train.tsv \
  --dev_csv data/dev.tsv \
  --model_path models/hausa_model.joblib \
  --results_path reports/metrics.json \
  --lexicon_csv data/hausa_aug_lex_train.csv \
  --model_name multinomial_nb
```

Supported model names:

- multinomial_nb
- logistic_regression

The training script will:

- load and preprocess the data
- build a scikit-learn pipeline
- train the selected classifier
- save the model to models/
- save evaluation results to reports/

---

## Evaluating a model

Evaluate a trained model on a test set:

```bash
py -3 src/eval.py \
  --model_path models/hausa_model.joblib \
  --test_csv data/test.tsv \
  --report_path reports/metrics.json
```

The evaluation output includes:

- accuracy
- macro F1-score
- precision
- recall
- a detailed classification report

---

## Running predictions

Run inference on a new file containing text data:

```bash
py -3 src/predict.py \
  --model_path models/hausa_model.joblib \
  --input_path data/hausa_news_articles.csv \
  --output_path reports/predictions.csv
```

The input file may contain columns such as text, tweet, article, or content.

---

## Preprocessing

Text preprocessing is handled in src/utils.py through the HausaTextPreprocessor class. It includes:

- text cleaning
- punctuation handling
- stopword removal
- normalization of Hausa and English social-media text
- optional lexicon-based feature extraction

---

## Testing

Run the test suite with:

```bash
py -3 -m pytest -q
```

---

## Notes

- The logistic regression path now uses scaled numeric features to improve convergence stability.
- Large datasets may require additional memory and time during training.
- For best results, keep the input data consistent and ensure the required columns are present.

---

## License

MIT License © 2025



