# CHAPTER 3: METHODOLOGY

This chapter explains how the Hausa sentiment analysis project was carried out in a step-by-step and reproducible manner. The aim is to ensure that another researcher can follow the same workflow from data preparation to model training, evaluation, and prediction.

## 3.1 Research Design

This study adopts an applied experimental research design in which a supervised machine learning approach is used to classify Hausa text into sentiment classes. The project is designed as a text-classification task with the following classes: positive, neutral, and negative.

The study compares two machine learning classifiers within the same preprocessing and feature-engineering pipeline:

1. Multinomial Naive Bayes
2. Logistic Regression

The research design is empirical because the performance of each model is evaluated using a held-out test set and standard classification metrics. The workflow is structured in a way that allows another researcher to replicate the experiment from the repository using the provided datasets and scripts.

## 3.2 Data and Source Description

The project uses both labeled benchmark data and text data stored in the repository’s data folder. The main datasets used for training and evaluation are in CSV/TSV format and contain at least two important columns: text and label. The repository includes files such as train.tsv, dev.tsv, test.tsv, and a lexicon file named hausa_aug_lex_train.csv.

The labeled data are used for supervised learning. Each sample consists of a text instance and its corresponding sentiment label. The label space includes positive, neutral, and negative classes. These datasets provide the ground truth needed for training and evaluation.

In addition to the labeled datasets, the project also supports prediction on new text files such as hausa_news_articles.csv. These inputs are used to demonstrate how the trained model can be applied to unseen text data.

The data sources therefore include:

- labeled training, validation, and test datasets stored locally in the project repository
- a sentiment lexicon used for additional linguistic feature extraction
- optionally, external article-style text data for inference and testing

The data are stored in a structured tabular form so that they can be easily loaded using pandas and processed by the preprocessing pipeline.

## 3.3 Tools and Techniques

The project was implemented in Python using a combination of data-processing, machine learning, and evaluation libraries. The main tools and techniques used are listed below.

### Software and programming language
- Python 3.9 or later, with Python 3.13 used in the current environment
- Jupyter or terminal-based execution for running scripts

### Libraries and frameworks
- pandas for loading and manipulating tabular data
- numpy for numerical processing and feature handling
- scikit-learn for preprocessing, feature extraction, model training, hyperparameter tuning, and evaluation
- joblib for saving and loading the trained model
- sklearn.feature_extraction.text.TfidfVectorizer for creating text-based features
- sklearn.pipeline.Pipeline and FeatureUnion for combining multiple feature sources
- sklearn.model_selection.GridSearchCV for model selection and tuning

### Technical techniques
- Text cleaning and normalization
- Tokenization and stopword removal
- TF-IDF vectorization using word and character n-grams
- Lexicon-based feature engineering using handcrafted sentiment indicators
- Feature scaling for numeric features before classification
- Hyperparameter tuning using macro-F1 as the optimization criterion

## 3.4 Procedure

The procedure followed in this project is presented below in the same order in which it was executed.

### Step 1: Prepare the working environment
A researcher first installs the required dependencies using the requirements.txt file. The project is then run from the repository root so that relative file paths such as data/ and models/ resolve correctly.

### Step 2: Load the datasets
The training, validation, and test datasets are loaded from the data folder using a custom loading function. The loader checks for the required columns and ensures that each dataset contains a text column and a label column.

### Step 3: Preprocess the text data
Each text sample is cleaned and standardized. The preprocessing steps include:
- converting text to string format
- removing URLs, emails, mentions, hashtags, numbers, and emojis
- normalizing repeated characters
- preserving punctuation that may carry sentiment meaning such as ! and ?
- tokenizing the text
- removing stopwords where appropriate

The preprocessing function also creates additional numeric features such as:
- positive indicator count
- negative indicator count
- sentiment polarity score
- text length
- word count
- average word length
- unique-word ratio
- exclamation count
- question count
- capitalization ratio
- Hausa-character ratio

### Step 4: Build the feature representation
The cleaned text is converted into two TF-IDF-based feature representations:
- character-level TF-IDF features using character n-grams
- word-level TF-IDF features using unigrams and bigrams

These representations are combined with the lexicon-based numeric features in a single pipeline. The resulting feature set provides both statistical and linguistic information for classification.

### Step 5: Train the classifiers
Two classifiers are trained within the same pipeline structure:
- Multinomial Naive Bayes
- Logistic Regression

The training process is performed using GridSearchCV, which searches over a set of hyperparameters and identifies the best configuration based on macro-F1 score.

### Step 6: Save the best model
After tuning, the best estimator from GridSearchCV is saved to the models folder as a joblib file. This ensures that the final model can be reused for evaluation and inference without retraining.

### Step 7: Evaluate the trained model
The saved model is loaded and tested on the held-out test set. The evaluation script computes the standard classification metrics and stores the results in the reports folder.

### Step 8: Use the model for prediction
A new input file containing text data can be passed to the prediction script. The model predicts the sentiment label for each input sample and writes the output to a CSV file.

The main commands used in the workflow are:

```bash
py -3 src/train.py --train_csv data/train.tsv --dev_csv data/dev.tsv --model_path models/hausa_model.joblib --results_path reports/metrics.json --lexicon_csv data/hausa_aug_lex_train.csv --model_name logistic_regression
```

```bash
py -3 src/eval.py --model_path models/hausa_model.joblib --test_csv data/test.tsv --report_path reports/metrics.json
```

```bash
py -3 src/predict.py --model_path models/hausa_model.joblib --input_path data/hausa_news_articles.csv --output_path reports/predictions.csv
```

## 3.5 Evaluation Metrics

The performance of the model is measured using a set of standard evaluation metrics for classification tasks.

### Accuracy
Accuracy measures the proportion of all predictions that are correct.

### Precision
Precision measures how many of the samples predicted as a given class were actually correct. Macro precision is used to give equal importance to each class.

### Recall
Recall measures how many of the true samples in a class were correctly identified by the model. Macro recall is used to avoid class imbalance bias.

### Macro-F1 score
Macro-F1 is the main metric used for model selection because it balances both precision and recall across all classes. It is particularly useful in multi-class sentiment tasks where some classes may be less frequent than others.

### Classification report
A detailed classification report is also generated for each evaluation run. This report provides per-class precision, recall, and F1-score, making it easier to identify which sentiment classes are predicted most accurately.

## 3.6 Ethics and Limitations

This project uses publicly available text data and does not involve direct interaction with human participants. Therefore, the ethical risk is relatively low. However, careful attention is still required when using online text data because such data may contain bias, offensive language, or incomplete context.

The main ethical considerations are:
- using publicly available data responsibly
- avoiding the disclosure of sensitive or personal information
- ensuring that the model is not used to misrepresent or harm individuals or communities

The study also has several limitations. First, the quality of the model depends heavily on the quality and coverage of the training data. Second, the handcrafted lexicon may not capture all Hausa expressions, especially informal or region-specific terms. Third, the project focuses on a supervised machine learning approach rather than more advanced deep learning models, which may limit performance on complex or highly contextual language. Finally, the results may vary depending on the specific dataset split, preprocessing choices, and the chosen classification algorithm.

Despite these limitations, the methodology is transparent and reproducible, and it provides a practical foundation for future work on Hausa sentiment analysis.
