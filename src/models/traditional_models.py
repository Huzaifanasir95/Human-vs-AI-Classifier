from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import joblib
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles training, saving, and loading of traditional ML models.
    """
    
    def __init__(self, model_type: str, config: Dict):
        self.model_type = model_type
        self.config = config
        self.model = self._initialize_model()
        self.models_dir = config.get('output', {}).get('models_dir', 'models/saved_models')
        os.makedirs(self.models_dir, exist_ok=True)

    def _initialize_model(self):
        """Initializes model based on type and config."""
        params = self.config.get('traditional_models', {}).get(self.model_type, {})
        
        if self.model_type == 'logistic_regression':
            return LogisticRegression(**params)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**params)
        elif self.model_type == 'svm':
            # Ensure probability=True for ensemble soft voting later
            if 'probability' not in params:
                params['probability'] = True
            return SVC(**params)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X_train, y_train):
        """Trains the model."""
        logger.info(f"Training {self.model_type}...")
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Predicts labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predicts probabilities."""
        return self.model.predict_proba(X)

    def save(self, filename: Optional[str] = None):
        """Saves model to disk."""
        if filename is None:
            filename = f"{self.model_type}.pkl"
        path = os.path.join(self.models_dir, filename)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Loads model from disk."""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return self
