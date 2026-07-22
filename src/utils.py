# Hausa Text Preprocessing (Enhanced Version)
import re
import string
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from collections import Counter

class HausaTextPreprocessor:
    def __init__(self, lexicon_path: str = None):
        # Enhanced Hausa stopwords with additional linguistic research
        self.hausa_stopwords = {
            # Original stopwords (preserved)
            'da', 'ne', 'ce', 'na', 'ta', 'shi', 'ita', 'su', 'ni', 'ka', 'ki', 'ku', 'mu', 'wa', 'zuwa', 'akan',
            'amma', 'ko', 'kuma', 'saboda', 'don', 'ba', 'bai', 'bata', 'baiwa', 'bayan', 'cikin', 'ga', 'ina',
            'yana', 'yake', 'yayi', 'yake', 'yanzu', 'wannan', 'wancan', 'wata', 'wani', 'wasu', 'duk', 'kowa',
            'me', 'mece', 'mene', 'wace', 'wane', 'wacece', 'wanene', 'inda', 'lokacin', 'idan', 'kamar', 'saboda',
            'daidai', 'kawai', 'har', 'sai', 'tun', 'daga', 'zuwa', 'kuma', 'ko', 'amma', 'saboda', 'idan', 'ko',
            'da', 'ba', 'ce', 'ne', 'shi', 'ta', 'su', 'ni', 'ka', 'ki', 'ku', 'mu', 'wa', 'zuwa', 'akan', 'ga',
            'cikin', 'bayan', 'lokacin', 'inda', 'yanzu', 'kamar', 'saboda', 'kawai', 'har', 'sai', 'tun', 'daga',
            
            # Enhanced additions based on Hausa linguistics
            'a', 'an', 'mun', 'sun', 'kun', 'kin', 'kai', 'kee', 'kenan', 'zai', 'za', 'mai', 'marasa', 'masu',
            'wanda', 'wadanda', 'wacce', 'waɗanda', 'yau', 'jiya', 'gobe', 'shekarar', 'watan', 'ranar', 'lokaci',
            'fa', 'dai', 'ko', 'kuwa', 'ma', 'kuma'
        }
        
        self.punctuation = set(string.punctuation)
        self.hausa_chars = set('"abcdefghijklmnopqrstuvwxyz`ʼƙɗɓçäöüÀÁÂÃÈÉÊÌÍÎÒÓÔÕÙÚÛÇÑ"')
        self.lexicon_path = lexicon_path or (Path(__file__).resolve().parents[1] / "data" / "hausa_aug_lex_train.csv")
        
        # Enhanced: Sentiment indicators for feature extraction
        # Expanded based on diagnostic analysis and word bias findings
        self.positive_indicators = {
            'kyau', 'mai kyau', 'nagari', 'farin ciki', 'murna', 'jin dadi', 'godiya', 'na gode',
            'alheri', 'albarka', 'madalla', 'zakara', 'kyakkyawa', 'dadi', 'son', 'so', 'dariya',
            'alhamdulillahi', 'mashallah', 'barka', 'mabrouk', 'yayyafa', 'jin daɗi',
            # Expanded from diagnostic findings
            'rayuwar', 'matsayin', 'shin', 'maka', 'bata', 'kafin', 'bayyana', 'dama',
            'abubuwan', 'kuka', 'hai', 'sosai', 'ciki', 'yan', 'nuna', 'cire', 'mutanen',
            'suke', 'baya', 'sama', 'yace', 'kuwa', 'kowa', 'gare', 'mafi', 'tun',
            'gida', 'sani', 'shakara', 'gajiya', 'kasua', 'kasuwa', 'yiwa', 'kawo',
            'sannu', 'sannya', 'kwana', 'hannu', 'kai', 'ido', 'zane'
        }
        
        self.negative_indicators = {
            'mugu', 'mummunan', 'bakin ciki', 'tsoro', 'damuwa', 'haushi', 'bacin rai',
            'rashin', 'kuskure', 'laifi', 'haramun', 'zalunci', 'ban sha\'awa', 'kyama',
            'kiyayya', 'ɓarna', 'azaba', 'wahala', 'matsala', 'rikici', 'fitina', 'tashin hankali',
            # Expanded from diagnostic findings
            'rana', 'damar', 'saka', 'tsammanin', 'fiye', 'yara', 'sauri', 'mamaki',
            'yawancin', 'akan', 'bayar', 'tafi', 'kawo', 'tashi', 'mana', 'tabbatar',
            'kashe', 'shekaru', 'kullum', 'kowane', 'dauke', 'din', 'yawa', 'mata',
            'allah', 'zaka', 'jama\'a', 'hana', 'kusan', 'damina', 'bugi', 'karfi',
            'karena', 'zazzage', 'aiwu', 'isa', 'taba', 'dada', 'uku', 'gida'
        }

        self.load_lexicon_words(self.lexicon_path)
        
        # Enhanced: Compiled regex patterns for efficiency
        self.patterns = {
            'urls': re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE),
            'emails': re.compile(r'\S+@\S+'),
            'mentions': re.compile(r'@[\w_]+'),
            'hashtags': re.compile(r'#[\w_]+'),
            'numbers': re.compile(r'\d+'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),  # For normalizing repeated characters
            'extra_spaces': re.compile(r'\s+'),
            'emojis': re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002700-\U000027BF"  # Dingbats
                u"\U000024C2-\U0001F251"  # Enclosed characters
                "]+", flags=re.UNICODE)
        }

    def load_lexicon_words(self, lexicon_path: str = None) -> None:
        """Populate sentiment indicator sets from the provided Hausa lexicon CSV."""
        if lexicon_path is None:
            lexicon_path = self.lexicon_path

        if not lexicon_path:
            return

        path = Path(lexicon_path)
        if not path.exists():
            return

        try:
            df = pd.read_csv(path)
        except Exception:
            return

        required_columns = {"Hausa", "Polarity"}
        if not required_columns.issubset(df.columns):
            return

        for _, row in df.iterrows():
            expression = str(row.get("Hausa", "")).strip().lower()
            polarity = str(row.get("Polarity", "")).strip().lower()
            if not expression:
                continue
            expression = re.sub(r"\s+", " ", expression).strip("\"'")
            if polarity == "positive":
                self.positive_indicators.add(expression)
            elif polarity == "negative":
                self.negative_indicators.add(expression)

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better normalization."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Apply all cleaning patterns efficiently
        text = self.patterns['urls'].sub('', text)
        text = self.patterns['emails'].sub('', text)
        text = self.patterns['mentions'].sub('', text)
        text = self.patterns['hashtags'].sub('', text)
        text = self.patterns['numbers'].sub('', text)
        text = self.patterns['emojis'].sub('', text)
        
        # Enhanced: Normalize repeated characters (e.g., "wooooow" -> "woow")
        # Important for social media text where users exaggerate
        text = self.patterns['repeated_chars'].sub(r'\1\1', text)
        
        # Enhanced: Better whitespace handling
        text = self.patterns['extra_spaces'].sub(' ', text)
        text = text.strip('"\'')
        
        return text.strip()

    def remove_punctuation(self, text: str) -> str:
        """Enhanced punctuation removal with selective preservation."""
        # Preserve some punctuation that might be important for sentiment
        important_punct = {'!', '?'}
        preserved_punct = []
        
        cleaned_chars = []
        for char in text:
            if char not in self.punctuation:
                cleaned_chars.append(char)
            elif char in important_punct:
                preserved_punct.append(char)
        
        # Add preserved punctuation at the end as features
        result = ''.join(cleaned_chars)
        if preserved_punct:
            result += ' ' + ''.join(preserved_punct)
        
        return result

    def keep_hausa_chars(self, text: str) -> str:
        """Keep only Hausa characters and whitespace."""
        return ''.join(char for char in text if char in self.hausa_chars or char.isspace())

    def tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with better handling of Hausa text."""
        tokens = text.split()
        # Filter out very short tokens that are likely noise
        tokens = [token for token in tokens if len(token) > 1]
        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords with enhanced filtering."""
        return [token for token in tokens if token not in self.hausa_stopwords and len(token) > 1]

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features for enhanced model performance."""
        features = {}
        text_lower = text.lower()
        words = text_lower.split()
        
        # Sentiment indicators
        pos_count = sum(1 for indicator in self.positive_indicators if indicator in text_lower)
        neg_count = sum(1 for indicator in self.negative_indicators if indicator in text_lower)
        
        features['positive_indicators'] = pos_count
        features['negative_indicators'] = neg_count
        features['sentiment_polarity'] = pos_count - neg_count
        
        # Text complexity features
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        
        # Emphasis and punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Hausa-specific features
        features['hausa_char_ratio'] = sum(1 for c in text if c in 'ƙɗɓ') / len(text) if text else 0
        
        return features

    def preprocess(self, text: str, remove_stopwords: bool = True, keep_only_hausa: bool = False, 
                  extract_features: bool = False) -> str:
        """Enhanced preprocessing with optional feature extraction."""
        # Extract features before preprocessing if requested
        if extract_features:
            features = self.extract_features(text)
        
        # Main preprocessing pipeline
        text = self.clean_text(text)
        text = self.remove_punctuation(text)
        
        if keep_only_hausa:
            text = self.keep_hausa_chars(text)
        
        tokens = self.tokenize(text)
        
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        processed_text = ' '.join(tokens)
        
        if extract_features:
            return processed_text, features
        else:
            return processed_text

    def analyze_corpus(self, texts: List[str]) -> Dict[str, any]:
        """Analyze corpus statistics for insights."""
        processed_texts = [self.preprocess(text) for text in texts]
        
        stats = {
            'total_texts': len(texts),
            'avg_length': np.mean([len(text) for text in processed_texts]),
            'avg_words': np.mean([len(text.split()) for text in processed_texts]),
            'vocabulary_size': len(set(' '.join(processed_texts).split())),
            'most_common_words': Counter(' '.join(processed_texts).split()).most_common(20)
        }
        
        return stats



def get_text_length(X):
    """Return text lengths as numeric feature."""
    return np.array([len(str(t)) for t in X]).reshape(-1, 1)


def get_lexicon_features(X):
    """Return lexicon-derived numeric features for each sample."""
    rows = []
    for text in X:
        features = preprocessor.extract_features(str(text))
        rows.append([
            features["positive_indicators"],
            features["negative_indicators"],
            features["sentiment_polarity"],
            features["text_length"],
            features["word_count"],
            features["avg_word_length"],
            features["unique_word_ratio"],
            features["exclamation_count"],
            features["question_count"],
            features["caps_ratio"],
            features["hausa_char_ratio"],
        ])
    return np.array(rows, dtype=float)


def load_sentiment_dataset(csv_path, preprocess=True, lexicon_path=None, **preprocess_kwargs):
    """Load a CSV/TSV sentiment dataset with text or tweet plus label columns."""
    if lexicon_path is not None:
        preprocessor.load_lexicon_words(lexicon_path)

    csv_path = str(csv_path)
    sep = "\t" if csv_path.lower().endswith(".tsv") else ","
    df = pd.read_csv(csv_path, sep=sep)

    if "tweet" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"tweet": "text"})
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must have columns: text,label or tweet,label")

    df = df[["text", "label"]].dropna()

    if preprocess:
        df["text"] = df["text"].astype(str).map(
            lambda text: preprocessor.preprocess(text, **preprocess_kwargs)
        )

    return df


# Instantiate the preprocessor
preprocessor = HausaTextPreprocessor()
