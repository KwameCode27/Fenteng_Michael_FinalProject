from pathlib import Path
import sys
sys.path.insert(0, str(Path('src').resolve()))
from train import build_model_pipeline
from utils import load_sentiment_dataset

train_df = load_sentiment_dataset(Path('data/train.tsv'), lexicon_path='data/hausa_aug_lex_train.csv')
train_df = train_df.sample(n=250, random_state=42)
pipeline = build_model_pipeline('logistic_regression')
pipeline.fit(train_df['text'], train_df['label'])
print('solver=', pipeline.named_steps['clf'].solver)
print('n_iter=', pipeline.named_steps['clf'].n_iter_)
print('classes=', pipeline.named_steps['clf'].classes_)
