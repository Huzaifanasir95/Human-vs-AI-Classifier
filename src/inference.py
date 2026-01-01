import joblib
import os
import sys
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List
import logging

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import FeatureExtractor
from src.data.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class InferencePipeline:
    """
    End-to-end pipeline for predicting Human vs AI on new text.
    """
    
    def __init__(self, model_path: str, config_path: str = 'configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.preprocessor = TextPreprocessor()
        self.extractor = FeatureExtractor(self.config)
        
        # Load the model
        logger.info(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        
        # In a real scenario, we'd also need to load a fitted TF-IDF vectorizer and Scaler
        # For this walkthrough, we'll assume the model is a pipeline or we have them saved
        # If the model is an Sklearn Pipeline, it handles features internally.
        # Otherwise, we might need a separate 'fitted_extractor.pkl'
        self.fitted_objects_path = model_path.replace('.pkl', '_fitted.pkl')
        if os.path.exists(self.fitted_objects_path):
            fitted = joblib.load(self.fitted_objects_path)
            self.extractor.vectorizer = fitted['vectorizer']
            self.scaler = fitted.get('scaler')
        else:
            self.scaler = None

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Preprocesses text, extracts features, and returns prediction with probability.
        """
        # 1. Clean and Preprocess
        cleaned_text = self.preprocessor.preprocess(text)
        
        # 2. Extract Features
        # Handcrafted
        handcrafted = self.extractor.get_handcrafted_features([text])
        # TF-IDF
        tfidf = self.extractor.get_tfidf_features([cleaned_text])
        
        # Combine
        X = np.hstack([tfidf, handcrafted.values])
        
        # 3. Scale if needed
        if self.scaler:
            X = self.scaler.transform(X)
            
        # 4. Predict
        prob = self.model.predict_proba(X)[0]
        prediction = int(self.model.predict(X)[0])
        
        label = "AI-Generated" if prediction == 1 else "Human-Written"
        confidence = prob[1] if prediction == 1 else prob[0]
        
        return {
            'text': text,
            'prediction': prediction,
            'label': label,
            'confidence': float(confidence),
            'probabilities': {
                'human': float(prob[0]),
                'ai': float(prob[1])
            },
            'linguistic_features': handcrafted.iloc[0].to_dict()
        }

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch prediction on multiple texts."""
        return [self.predict(t) for t in texts]
