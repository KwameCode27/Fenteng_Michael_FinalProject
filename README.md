# Hausa Sentiment Analysis for Ghana 🇬🇭

> **Machine Learning-Based Sentiment Analysis for Hausa Social Media Content in Ghana**
> Department of Computer Science · Research Project · 2026

---

## Overview

This project builds a supervised machine learning pipeline that classifies
Hausa-language text from social media and news sources into three sentiment
classes: **positive**, **negative**, and **neutral**.

The pipeline is designed around two complementary goals:

1. **Academic benchmarking** – train and evaluate on the
   [AfriSenti Hausa dataset](https://github.com/afrisenti-nlp/afrisenti-semeval-2023)
   so results can be compared against published state-of-the-art systems.
2. **Applied inference** – run the trained model on unlabelled Ghana-focused
   Hausa news articles (scraped from BBC Hausa, VOA Hausa, and DW Hausa) to
   analyse real-world sentiment trends relevant to Ghanaian communities.

---

## Architecture

```
Raw Text (TSV / CSV)
        │
        ▼
┌─────────────────────────────────────────┐
│  HausaTextPreprocessor  (src/utils.py)  │
│  • Lowercase & strip noise              │
│  • Remove URLs / mentions / hashtags    │
│  • Normalise repeated characters        │
│  • Tokenise & remove stopwords          │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   FeatureUnion      │
        │  ┌───────────────┐  │
        │  │ Char TF-IDF   │  │  3–5-grams
        │  │ Word TF-IDF   │  │  unigrams + bigrams
        │  │ Length feat.  │  │  scaled
        │  │ Lexicon feat. │  │  11 numeric features, scaled
        │  └───────────────┘  │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │    Classifier       │
        │  MultinomialNB  OR  │
        │  LogisticRegression │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Predicted Label    │
        │  positive / neutral │
        │  / negative         │
        └─────────────────────┘
```

---

## Results

All results below are from the **held-out AfriSenti Hausa test set** (5,303 samples).

| Model | Vectoriser | Accuracy | Macro-F1 | Macro-P | Macro-R |
|---|---|---|---|---|---|
| Multinomial Naïve Bayes | TF-IDF | 74.7% | 0.748 | 0.750 | 0.747 |
| Multinomial Naïve Bayes | Count  | 73.4% | 0.736 | 0.738 | 0.735 |
| **Logistic Regression** | **TF-IDF** | **77.6%** | **0.777** | **0.780** | **0.776** |
| Logistic Regression | Count  | 77.3% | 0.773 | 0.776 | 0.773 |

**Current best model: Logistic Regression + TF-IDF**, macro-F1 = **0.777**.
The training workflow also includes a tuned Linear SVM challenger; evaluate it
on the held-out test set before replacing the current best model.
(exceeds the practical utility threshold of 0.70 set in the project methodology).

### Per-class breakdown (best model)

| Class    | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
| Negative | 0.778     | 0.711  | 0.743 | 1 759   |
| Neutral  | 0.709     | 0.792  | 0.748 | 1 789   |
| Positive | 0.854     | 0.826  | 0.840 | 1 755   |

> The **neutral** class is the hardest to classify – a known challenge in
> three-class Hausa sentiment benchmarks reported in the AfriSenti literature.

---

## Project Structure

```
Hausa_Sentiment_AnalysisProject/
├── data/
│   ├── train.tsv                  # AfriSenti Hausa training split
│   ├── dev.tsv                    # AfriSenti Hausa validation split
│   ├── test.tsv                   # AfriSenti Hausa held-out test split
│   ├── hausa_aug_lex_train.csv    # Hausa sentiment lexicon
│   └── hausa_news_articles.csv   # Scraped Ghana-focused news articles
│
├── models/
│   ├── hausa_model.joblib         # Best NB  model (TF-IDF)
│   ├── hausa_model_lr.joblib      # Best LR  model (TF-IDF)  ← recommended
│   ├── hausa_model_count.joblib   # NB  model (Count)
│   └── hausa_model_lr_count.joblib# LR  model (Count)
│
├── reports/
│   ├── metrics.json               # NB  + TF-IDF dev metrics
│   ├── metrics_lr.json            # LR  + TF-IDF dev metrics
│   ├── metrics_nb.json            # NB  test metrics
│   ├── metrics_lr_eval.json       # LR  test metrics  ← best
│   ├── metrics_count_eval.json    # NB  + Count test metrics
│   ├── metrics_lr_count_eval.json # LR  + Count test metrics
│   └── predictions.csv            # Predicted labels on Ghana news articles
│
├── src/
│   ├── utils.py                   # Preprocessor, feature helpers, data loader
│   ├── train.py                   # Training script with GridSearchCV
│   ├── eval.py                    # Evaluation on held-out test set
│   └── predict.py                 # Batch inference on unlabelled text
│
├── tests/
│   ├── test_training_model_selection.py
│   ├── test_lexicon_features.py
│   └── test_prediction_dataset_loader.py
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9 or later (tested on 3.9 – 3.13)
- pip

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd Hausa_Sentiment_AnalysisProject

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The original `requirements.txt` included `torch`, `torchvision`,
> and `torchaudio` (~2 GB). These are **not used** by the current pipeline and
> have been removed. Re-add them only if you extend the project with
> transformer-based models (e.g. AfriBERTa).

---

## Usage

All scripts are run from the **project root** directory.

### Train a model

```bash
# Logistic Regression + TF-IDF (recommended — best results)
python src/train.py \
  --train_csv  data/train.tsv \
  --dev_csv    data/dev.tsv \
  --model_path models/hausa_model_lr.joblib \
  --results_path reports/metrics_lr.json \
  --lexicon_csv data/hausa_aug_lex_train.csv \
  --model_name  logistic_regression

# Multinomial Naïve Bayes + TF-IDF (faster, good baseline)
python src/train.py \
  --model_name multinomial_nb

# Train both models with both vectorisers in one command
python src/train.py \
  --model_name logistic_regression \
  --vectorizer_type both
```

**CLI options for `train.py`:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `logistic_regression` | `multinomial_nb` or `logistic_regression` |
| `--vectorizer_type` | `tfidf` | `tfidf`, `count`, or `both` |
| `--cv` | `5` | GridSearchCV cross-validation folds |
| `--n_jobs` | `-1` | Parallel workers (`-1` = all CPU cores) |
| `--train_csv` | `data/train.tsv` | Training data path |
| `--dev_csv` | `data/dev.tsv` | Validation data path |
| `--model_path` | `models/hausa_model.joblib` | Output model path |
| `--results_path` | `reports/metrics.json` | Output metrics path |
| `--lexicon_csv` | `data/hausa_aug_lex_train.csv` | Sentiment lexicon path |

---

### Evaluate a model

```bash
python src/eval.py \
  --model_path  models/hausa_model_lr.joblib \
  --test_csv    data/test.tsv \
  --report_path reports/metrics_lr_eval.json
```

The script prints accuracy, macro-F1, precision, recall, and a full
per-class classification report to the console, and saves the same data
to the JSON report file.

---

### Predict on new text

```bash
# Ghana news articles (unlabelled)
python src/predict.py \
  --model_path  models/hausa_model_lr.joblib \
  --input_path  data/hausa_news_articles.csv \
  --output_path reports/predictions.csv
```

The input CSV must contain one of the following column names:
`text`, `tweet`, `article`, `content`, or `title`.

The output CSV contains **all original columns** plus a `predicted_label` column.

---

### Run the test suite

```bash
python -m pytest -q
```

Expected output: 5 tests passing (< 10 s).

---

## Key Design Decisions

### Why TF-IDF over transformer models?

Transformer-based models such as AfriBERTa (Ogueji et al., 2021) and
AfroXLMR (Alabi et al., 2022) consistently achieve higher macro-F1 on the
AfriSenti benchmark (~80 % vs ~78 % here). This project uses classical
TF-IDF classifiers for three reasons:

1. **Computational accessibility** – no GPU required; training completes in
   minutes on a standard laptop.
2. **Interpretability** – feature importance can be inspected directly from
   the vectoriser vocabulary.
3. **Baseline value** – these results serve as the "classical ML baseline"
   against which the research methodology compares transformer approaches.

### Why two vectorisers?

The pipeline combines **character-level TF-IDF** (3–5 n-grams) with
**word-level TF-IDF** (unigrams + bigrams):

- *Character n-grams* capture Hausa morphological patterns (prefixes, suffixes,
  root forms) that are not visible at the word level.
- *Word n-grams* capture lexical collocations and sentiment phrases.

Together they give the classifier access to both sub-word morphology and
surface lexical meaning, which is especially important for a morphologically
rich language like Hausa.

### Why lexicon-based features?

The Hausa training corpus is relatively small by NLP standards. Adding 11
handcrafted numeric features (positive/negative indicator counts, text length,
unique-word ratio, exclamation/question marks, etc.) from a curated Hausa
sentiment lexicon gives the classifier explicit sentiment signals that
statistical features alone may underweight for low-frequency sentiment words.

---

## Data Sources

| Dataset | Source | Licence |
|---------|--------|---------|
| AfriSenti Hausa (train/dev/test) | [AfriSenti-SemEval-2023](https://github.com/afrisenti-nlp/afrisenti-semeval-2023) | CC-BY-4.0 |
| Hausa sentiment lexicon | Included in repository | Research use |
| Ghana Hausa news articles | BBC Hausa, VOA Hausa, DW Hausa (public web) | Public domain / scraped |

---

## Limitations

1. **Domain mismatch** – the training data is Nigerian Twitter content.
   Ghanaian Hausa (Gaananci) differs in vocabulary and code-switching patterns,
   so real-world performance on Ghanaian text may be lower than test-set numbers
   suggest.

2. **Unlabelled inference** – predicted labels on news articles are
   approximate; the model was not trained on labelled Ghanaian Hausa data.

3. **Classical ML ceiling** – macro-F1 ≈ 0.78 is strong for TF-IDF + LR but
   approximately 2–3 points below the best transformer systems on AfriSenti.

4. **Lexicon coverage gaps** – the curated lexicon does not cover all
   informal social-media vocabulary or Ghanaian Hausa loanwords.

---

## Extending the Project

| Goal | Recommended next step |
|------|-----------------------|
| Higher accuracy | Fine-tune AfriBERTa or AfroXLMR on AfriSenti + a Ghana-specific labelled subset |
| Ghana-specific model | Annotate a corpus of Ghanaian Hausa social media posts; apply LAFT pipeline (see Sani & Muhammad, 2025) |
| Real-time monitoring | Wrap `predict.py` in a FastAPI service; connect to platform streaming APIs |
| Multilingual extension | Add Twi / Akan via the GhanaNLP dataset; train a shared multilingual model |

---

## References

- Muhammad et al. (2022). NaijaSenti. *LREC 2022*.
- Muhammad et al. (2023). AfriSenti. *EMNLP 2023*.
- Ogueji et al. (2021). AfriBERTa. *MRL Workshop, EMNLP 2021*.
- Alabi et al. (2022). AfroXLMR. *COLING 2022*.
- Sani & Muhammad (2025). LAFT for Hausa. *arXiv:2501.11023*.

---

## Licence

MIT Licence © 2026 – Fenteng Michael
