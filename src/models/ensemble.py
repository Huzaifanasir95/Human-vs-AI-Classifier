import numpy as np
import joblib
from sklearn.ensemble import VotingClassifier
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    Ensemble methods for combining multiple classifiers.
    """
    def __init__(self, models: List[Any], voting: str = 'soft', weights: Optional[List[float]] = None):
        self.models = models
        self.voting = voting
        self.weights = weights
        self.ensemble = None

    def fit(self, X, y):
        """Fits the ensemble (if meta-learner needed or just wrapping)."""
        # For simplicity, we use Sklearn's VotingClassifier
        # We need to provide (name, model) tuples
        estimators = [(f"model_{i}", m) for i, m in enumerate(self.models)]
        self.ensemble = VotingClassifier(
            estimators=estimators, 
            voting=self.voting, 
            weights=self.weights
        )
        logger.info(f"Fitting {self.voting} ensemble...")
        self.ensemble.fit(X, y)
        return self

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)

    def save(self, path: str):
        joblib.dump(self.ensemble, path)
        logger.info(f"Ensemble saved to {path}")
