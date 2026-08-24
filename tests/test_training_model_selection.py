import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train import MODEL_DEFINITIONS, build_model_pipeline
from utils import get_text_length, get_lexicon_features


def test_model_definitions_include_requested_algorithms():
    assert {'multinomial_nb', 'logistic_regression'} <= set(MODEL_DEFINITIONS)
    assert MODEL_DEFINITIONS['multinomial_nb']['display_name'] == 'Multinomial Naïve Bayes'
    assert MODEL_DEFINITIONS['logistic_regression']['display_name'] == 'Logistic Regression'


def test_feature_helpers_are_available_from_utils_module():
    assert callable(get_text_length)
    assert callable(get_lexicon_features)


def test_build_model_pipeline_returns_expected_classifier_type():
    nb_pipeline = build_model_pipeline('multinomial_nb')
    lr_pipeline = build_model_pipeline('logistic_regression')

    assert nb_pipeline.named_steps['clf'].__class__.__name__ == 'MultinomialNB'
    assert lr_pipeline.named_steps['clf'].__class__.__name__ == 'LogisticRegression'
    assert lr_pipeline.named_steps['clf'].solver == 'lbfgs'
    assert lr_pipeline.named_steps['clf'].max_iter >= 5000
    assert lr_pipeline.named_steps['features'].transformer_list[2][1].named_steps['scale'].__class__.__name__ == 'StandardScaler'
    assert lr_pipeline.named_steps['features'].transformer_list[3][1].named_steps['scale'].__class__.__name__ == 'StandardScaler'


