from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import preprocessor


def test_lexicon_feature_counts_are_loaded_from_dataset():
    lexicon_path = ROOT / 'data' / 'hausa_aug_lex_train.csv'
    df = pd.read_csv(lexicon_path)
    assert {'Hausa', 'Polarity'} <= set(df.columns)

    positive_words = {row['Hausa'].strip().lower() for _, row in df.iterrows() if str(row['Polarity']).lower() == 'positive'}
    negative_words = {row['Hausa'].strip().lower() for _, row in df.iterrows() if str(row['Polarity']).lower() == 'negative'}

    assert positive_words
    assert negative_words
    assert any(word in preprocessor.positive_indicators for word in positive_words)
    assert any(word in preprocessor.negative_indicators for word in negative_words)
