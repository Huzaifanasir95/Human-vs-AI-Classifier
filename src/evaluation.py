from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Comprehensive evaluation for text classification models.
    """
    
    def __init__(self, output_dir: str = 'outputs/reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.viz_dir = 'outputs/visualizations'
        os.makedirs(self.viz_dir, exist_ok=True)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculates standard classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_prob is not None:
            # Handle binary probability (second column)
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                prob = y_prob[:, 1]
            else:
                prob = y_prob
            metrics['auc_roc'] = roc_auc_score(y_true, prob)
            
        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Plots and saves confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0.5, 1.5], ['Human', 'AI'])
        plt.yticks([0.5, 1.5], ['Human', 'AI'])
        
        path = os.path.join(self.viz_dir, f'cm_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Confusion matrix saved to {path}")

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str):
        """Plots and saves ROC curve."""
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            prob = y_prob[:, 1]
        else:
            prob = y_prob
            
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {model_name}')
        plt.legend(loc="lower right")
        
        path = os.path.join(self.viz_dir, f'roc_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"ROC curve saved to {path}")

    def save_report(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Saves classification report to text file."""
        report = classification_report(y_true, y_pred, target_names=['Human', 'AI'])
        path = os.path.join(self.output_dir, f'report_{model_name.lower().replace(" ", "_")}.txt')
        with open(path, 'w') as f:
            f.write(f"Classification Report: {model_name}\n")
            f.write("="*40 + "\n")
            f.write(report)
        logger.info(f"Report saved to {path}")

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray], model_name: str):
        """Full evaluation suite."""
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        self.plot_confusion_matrix(y_true, y_pred, model_name)
        if y_prob is not None:
            self.plot_roc_curve(y_true, y_prob, model_name)
        self.save_report(y_true, y_pred, model_name)
        return metrics
