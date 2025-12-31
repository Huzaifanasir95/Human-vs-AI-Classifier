# AI-Generated Text Detection System

A comprehensive machine learning system to detect AI-generated text by training classifiers on human vs AI-written essays. This project implements multiple model architectures including traditional ML and deep learning approaches.

## 🎯 Project Overview

This system can distinguish between human-written and AI-generated essays with high accuracy using:
- Traditional ML models (Logistic Regression, Random Forest, SVM, XGBoost)
- Deep learning models (LSTM, BERT, RoBERTa)
- Advanced feature engineering (TF-IDF, statistical, linguistic features)
- Ensemble methods for improved performance
- Interactive web interface for real-time predictions

## 📁 Project Structure

```
Human-vs-AI-Classifier/
├── data/
│   ├── raw/                    # Raw datasets
│   └── processed/              # Preprocessed data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_traditional_models.ipynb
│   ├── 05_deep_learning_models.ipynb
│   ├── 06_ensemble_models.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_final_report.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── preprocessor.py     # Text preprocessing
│   ├── models/
│   │   ├── traditional_models.py
│   │   ├── deep_models.py
│   │   └── ensemble.py
│   ├── feature_extractor.py    # Feature engineering
│   ├── evaluation.py           # Model evaluation
│   ├── inference.py            # Prediction pipeline
│   └── utils.py                # Utility functions
├── models/
│   └── saved_models/           # Trained model checkpoints
├── outputs/
│   ├── visualizations/         # Plots and charts
│   ├── reports/                # Evaluation reports
│   └── predictions/            # Prediction results
├── scripts/
│   ├── train_models.py         # Training script
│   └── evaluate_models.py      # Evaluation script
├── configs/
│   └── config.yaml             # Configuration file
├── tests/                      # Unit tests
├── app.py                      # Streamlit web interface
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Huzaifanasir95/Human-vs-AI-Classifier.git
cd Human-vs-AI-Classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLP models:
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 💻 Usage

### 1. Data Preparation

Place your dataset in `data/raw/` and run the preprocessing notebook:
```bash
jupyter notebook notebooks/02_data_preprocessing.ipynb
```

### 2. Train Models

Run the training script:
```bash
python scripts/train_models.py --config configs/config.yaml
```

Or use individual notebooks for step-by-step training.

### 3. Evaluate Models

```bash
python scripts/evaluate_models.py --model-path models/saved_models/best_model.pkl
```

### 4. Web Interface

Launch the Streamlit app for interactive predictions:
```bash
streamlit run app.py
```

## 📊 Features

### Feature Engineering
- **TF-IDF Vectorization**: Captures word importance
- **Statistical Features**: Perplexity, burstiness, entropy
- **Linguistic Features**: POS tags, sentence length, vocabulary richness
- **Readability Scores**: Flesch-Kincaid, SMOG, Coleman-Liau

### Models Implemented
1. **Traditional ML**:
   - Logistic Regression (baseline)
   - Random Forest
   - Support Vector Machine (SVM)
   - XGBoost & LightGBM

2. **Deep Learning**:
   - BiLSTM with Attention
   - BERT (bert-base-uncased)
   - RoBERTa (roberta-base)
   - DistilBERT (lightweight)

3. **Ensemble Methods**:
   - Voting Classifier
   - Stacking Ensemble
   - Weighted Averaging

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |
| BERT | TBD | TBD | TBD | TBD | TBD |
| Ensemble | TBD | TBD | TBD | TBD | TBD |

*Results will be updated after model training*

## 🔬 Methodology

1. **Data Collection**: Using publicly available datasets (HC3, GPT-2 Output, DAIGT)
2. **Preprocessing**: Text cleaning, normalization, tokenization
3. **Feature Extraction**: Multiple feature types for comprehensive representation
4. **Model Training**: Cross-validation with hyperparameter tuning
5. **Evaluation**: Comprehensive metrics and error analysis
6. **Deployment**: User-friendly web interface

## 🛠️ Configuration

Edit `configs/config.yaml` to customize:
- Model hyperparameters
- Training settings
- Feature extraction parameters
- Data paths

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v --cov=src
```

## 📝 Documentation

For detailed walkthrough, see [WALKTHROUGH.md](WALKTHROUGH.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Huzaifa Nasir**
- GitHub: [@Huzaifanasir95](https://github.com/Huzaifanasir95)

## 🙏 Acknowledgments

- Datasets: HC3, GPT-2 Output Dataset, DAIGT
- Hugging Face Transformers library
- Scikit-learn community

## 📚 References

1. OpenAI GPT models
2. BERT: Pre-training of Deep Bidirectional Transformers
3. Detecting AI-Generated Text: A Survey

---

**Note**: This is a Gen AI project demonstrating practical applications of machine learning in text classification.
