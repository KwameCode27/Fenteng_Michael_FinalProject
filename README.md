# Hausa & English Sentiment Analysis

This project implements a **sentiment analysis system for Hausa and English tweets/text**.  
The pipeline covers preprocessing, training, evaluation, and reporting.  
It is built to handle large datasets (600k+ rows) and can be extended to support other African languages.

---

## 📌 Project Structure

```

project/
├── data/               # Raw and processed data (train.csv, test.csv)
├── models/             # Saved models (.joblib)
├── reports/            # Evaluation reports (metrics.json, classification reports)
├── src/
│   ├── train.py        # Training script
│   ├── eval.py         # Evaluation script
│   ├── utils.py        # HausaTextPreprocessor + helper functions
│   └── ...
└── README.md           # This file

````

---

## ⚙️ Requirements

- Python **3.9+** (tested up to 3.13)
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `joblib`
  - `numpy`

---

## 📥 Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/your-username/hausa-sentiment-analysis.git
cd hausa-sentiment-analysis
pip install -r requirements.txt
````

---

## 🚀 Usage

### 1. Training a Model

Run `train.py` with your dataset:

```bash
python src/train.py --train_csv data/train.csv --model_path models/svm_tfidf.joblib
```

* The script:

  * Loads and preprocesses text (Hausa & English).
  * Splits dataset into train/test (80/20).
  * Trains a **LinearSVC with TF-IDF char n-grams**.
  * Saves the model to `models/`.

---

### 2. Evaluating a Model

Run `eval.py` on the saved model:

```bash
python src/eval.py --model_path models/svm_tfidf.joblib --test_csv data/test.csv
```

* The script:

  * Loads the trained model and test set.
  * Evaluates with **Accuracy, Macro-F1, Precision, Recall**.
  * Saves results in `reports/metrics.json`.

Example output:

```json
{
  "accuracy": 0.68,
  "macro_f1": 0.62,
  "precision": 0.64,
  "recall": 0.61
}
```

---

### 3. Preprocessing

Text preprocessing is handled in `utils.py` using a custom `HausaTextPreprocessor`, which:

* Normalizes Hausa text.
* Removes stopwords, URLs, and emojis.
* Applies consistent tokenization.

---

## 📊 Results

* With **LinearSVC + TF-IDF char 3–5 grams**, we achieved:

  * **Macro-F1:** \~0.51 on initial runs (600k rows).
* Future improvements:

  * Use **AfriBERTa** or **XLM-R** for deep learning fine-tuning.
  * Experiment with data augmentation.

---

## 👩‍💻 Developer Guide

* Add new preprocessing rules → edit `utils.py`.
* Change ML model → modify `train.py` (e.g., swap `LinearSVC` for `LogisticRegression` or `XGBoost`).
* New datasets:

  * Place them in `data/`
  * Ensure they have at least two columns: `text`, `label`.

---

## 📌 Notes

* Large datasets (\~600k rows) may require high memory (8GB+ RAM).
* For best performance, train with a GPU if moving to Transformer-based models.

---

## 📄 License

MIT License © 2025



