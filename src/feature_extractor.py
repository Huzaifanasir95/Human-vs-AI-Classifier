import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Dict, Union
import logging
from collections import Counter
import math

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extracts various features from text for AI vs Human classification.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.tfidf_config = config.get('features', {}).get('tfidf', {})
        self.vectorizer = TfidfVectorizer(
            max_features=self.tfidf_config.get('max_features', 5000),
            ngram_range=tuple(self.tfidf_config.get('ngram_range', [1, 3])),
            min_df=self.tfidf_config.get('min_df', 2),
            max_df=self.tfidf_config.get('max_df', 0.95)
        )
        
    def fit_tfidf(self, texts: List[str]):
        """Fits TF-IDF vectorizer on training texts."""
        logger.info("Fitting TF-IDF vectorizer...")
        self.vectorizer.fit(texts)
        
    def get_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Transforms texts to TF-IDF matrix."""
        return self.vectorizer.transform(texts).toarray()

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extracts linguistic and readability features.
        """
        features = {}
        
        # Sentence and word counts
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        features['sentence_count'] = len(sentences)
        features['word_count'] = len(words)
        features['avg_sentence_len'] = len(words) / len(sentences) if len(sentences) > 0 else 0
        features['avg_word_len'] = sum(len(w) for w in words) / len(words) if len(words) > 0 else 0
        
        # Vocabulary richness (Type-Token Ratio)
        unique_words = set(words)
        features['ttr'] = len(unique_words) / len(words) if len(words) > 0 else 0
        
        # Readability scores
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        features['automated_readability_index'] = textstat.automated_readability_index(text)
        
        return features

    def calculate_entropy(self, text: str) -> float:
        """Calculates Shannon entropy of word distribution."""
        words = word_tokenize(text.lower())
        if not words:
            return 0
        counts = Counter(words)
        probs = [c/len(words) for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    def calculate_burstiness(self, text: str) -> float:
        """
        Simplified burstiness: coefficient of variation of sentence lengths.
        AI text often has very uniform sentence lengths (low burstiness).
        """
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return 0
        lens = [len(word_tokenize(s)) for s in sentences]
        mean_len = np.mean(lens)
        std_len = np.std(lens)
        return std_len / mean_len if mean_len > 0 else 0

    def get_handcrafted_features(self, texts: List[str]) -> pd.DataFrame:
        """Extracts handcrafted features for a list of texts."""
        logger.info("Extracting handcrafted features...")
        all_features = []
        for text in texts:
            feat = self.extract_linguistic_features(text)
            feat['entropy'] = self.calculate_entropy(text)
            feat['burstiness'] = self.calculate_burstiness(text)
            all_features.append(feat)
        return pd.DataFrame(all_features)

    def combine_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        Combines TF-IDF and handcrafted features into a single matrix.
        """
        if fit:
            self.fit_tfidf(texts)
            
        tfidf_feats = self.get_tfidf_features(texts)
        handcrafted_df = self.get_handcrafted_features(texts)
        
        # Normalize handcrafted features
        # (Usually better to use a StandardScaler, but we'll do simple manual normalization or leave for model)
        handcrafted_feats = handcrafted_df.values
        
        return np.hstack([tfidf_feats, handcrafted_feats])
