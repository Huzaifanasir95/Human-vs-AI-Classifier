import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from datasets import load_dataset
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Utility class for loading and preparing datasets for Human vs AI Classification.
    Supports local files and Hugging Face datasets.
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.raw_path = self.config.get('data', {}).get('raw_path', 'data/raw')
        self.processed_path = self.config.get('data', {}).get('processed_path', 'data/processed')
        
    def load_from_hf(self, path: str, name: Optional[str] = None, split: str = 'train') -> pd.DataFrame:
        """
        Loads a dataset from Hugging Face.
        
        Args:
            path: Path to the dataset on HF (e.g., 'Hello-SimpleAI/HC3')
            name: Specific configuration name if any
            split: Dataset split to load
            
        Returns:
            pd.DataFrame: Loaded dataset as a pandas DataFrame
        """
        logger.info(f"Loading dataset {path} (split: {split}) from Hugging Face...")
        try:
            dataset = load_dataset(path, name, split=split)
            df = dataset.to_pandas()
            logger.info(f"Successfully loaded {len(df)} records.")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset from Hugging Face: {e}")
            raise

    def load_local(self, filename: str) -> pd.DataFrame:
        """Loads a local CSV or Parquet file."""
        file_path = os.path.join(self.raw_path, filename)
        logger.info(f"Loading local file from {file_path}...")
        
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")

    def prepare_daigt_data(self) -> pd.DataFrame:
        """
        Specific helper for DAIGT V2 dataset format.
        Commonly used in Kaggle competitions for AI text detection.
        """
        # Suggesting DAIGT V2 as it's very relevant for human vs AI essays
        try:
            # Note: In a real scenario, the user would download this or we'd uses HF mirror
            # For this walkthrough, we'll demonstrate loading a common HF version
            df = self.load_from_hf('pminervini/DAIGT-V2-Dataset', split='train')
            
            # Map columns to a standard format: 'text', 'label' (1 for AI, 0 for Human)
            # DAIGT-V2 usually has 'text' and 'label'
            return df[['text', 'label']]
        except Exception:
            logger.warning("DAIGT-V2 not available on HF, returning empty template.")
            return pd.DataFrame(columns=['text', 'label'])

    def prepare_hc3_data(self) -> pd.DataFrame:
        """
        Specific helper for HC3 (Human ChatGPT Comparison Corpus).
        """
        try:
            # HC3 has 'human_answers' and 'chatgpt_answers'
            dataset = load_dataset('Hello-SimpleAI/HC3', 'all', split='train')
            df = dataset.to_pandas()
            
            # Reformat HC3: it has lists of answers per question
            # We flatten it into (text, label) pairs
            human_texts = []
            ai_texts = []
            
            for _, row in df.iterrows():
                human_texts.extend(row['human_answers'])
                ai_texts.extend(row['chatgpt_answers'])
                
            human_df = pd.DataFrame({'text': human_texts, 'label': 0})
            ai_df = pd.DataFrame({'text': ai_texts, 'label': 1})
            
            combined_df = pd.concat([human_df, ai_df], ignore_index=True)
            logger.info(f"HC3 dataset prepared with {len(combined_df)} records.")
            return combined_df
        except Exception as e:
            logger.error(f"Error preparing HC3 data: {e}")
            raise

def save_processed_data(df: pd.DataFrame, file_path: str):
    """Saves DataFrame to CSV."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info(f"Data saved to {file_path}")
