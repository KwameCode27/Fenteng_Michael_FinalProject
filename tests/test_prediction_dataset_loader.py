from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict import load_prediction_dataset


def test_load_prediction_dataset_uses_article_column(tmp_path):
    input_path = tmp_path / 'articles.csv'
    pd.DataFrame({
        'article': ['Matsala ce a birnin Accra', 'Kyau ne ga al umma'],
        'source': ['BBC', 'VOA'],
    }).to_csv(input_path, index=False)

    df = load_prediction_dataset(input_path)

    assert list(df.columns) == ['text']
    assert len(df) == 2
    assert df['text'].astype(str).str.len().ge(0).all()
