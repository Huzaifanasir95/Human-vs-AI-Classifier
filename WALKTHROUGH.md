# AI-Generated Text Detection Project: A Deep Walkthrough

This document provides a comprehensive, step-by-step walkthrough of the **Human vs AI Classifier** project. We explore the methodology, architecture, and implementation details for detecting AI-generated text.

---

## 📋 Table of Contents
1. [Introduction](#1-introduction)
2. [Project Architecture](#2-project-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Development](#5-model-development)
6. [Evaluation and Analysis](#6-evaluation-and-analysis)
7. [Web Application](#7-web-application)
8. [Conclusion and Future Work](#8-conclusion-and-future-work)

---

## 1. Introduction
With the rise of Large Language Models (LLMs) like ChatGPT, Claude, and Gemini, the ability to distinguish between human-written and AI-generated content has become critical for academic integrity, misinformation detection, and content moderation.

This project implements a multi-layered approach using:
- **Traditional ML**: For speed and interpretability.
- **Deep Learning**: For capturing complex semantic patterns.
- **Linguistic Analysis**: For identifying "signatures" of AI text.

---

## 2. Project Architecture
The project is structured following clean coding principles and modular design:

- `src/data/`: Data loading and text preprocessing.
- `src/models/`: Implementation of various ML/DL architectures.
- `src/feature_extractor.py`: The heart of the feature engineering layer.
- `notebooks/`: Sequential, documented steps for experimentation.
- `app.py`: The deployment-ready interface.

---

## 3. Data Pipeline
The pipeline handles raw text and transforms it into a clean, normalized format.

### Key Datasets:
- **HC3 (Human ChatGPT Comparison Corpus)**: Direct human vs AI answers.
- **DAIGT V2**: A large collection of essays used in detection benchmarks.

### Preprocessing Steps:
1. **Cleaning**: URL removal, lowering, punctuation stripping.
2. **Normalization**: Tokenization, stopword removal, and lemmatization using `NLTK`.
3. **Splitting**: Stratified 70/15/15 splits to ensure class balance across training, validation, and testing.

---

## 4. Feature Engineering
We use a **Hybrid Feature Strategy**:

### A. TF-IDF (Term Frequency-Inverse Document Frequency)
Captures the importance of specific words and n-grams. AI often uses a different vocabulary distribution than humans.

### B. Statistical Features
- **Entropy**: Measures the randomness of word choice.
- **Burstiness**: Measures the variation in sentence lengths. AI text is often "flat" with very consistent sentence structures.

### C. Linguistic Features
- **TTR (Type-Token Ratio)**: Measures vocabulary richness.
- **Readability Scores**: Flesch-Kincaid, ARI, SMOG. AI text often targets specific readability bands.

---

## 5. Model Development
We implement and compare three categories of models:

1. **Baseline**: Logistic Regression with TF-IDF.
2. **Intermediate**: Random Forest and XGBoost using combined features.
3. **Advanced**: 
   - **BiLSTM with Attention**: To capture sequential word patterns.
   - **BERT (Transformer)**: Fine-tuning `bert-base-uncased` for semantic understanding.

---

## 6. Evaluation and Analysis
Success is measured using:
- **Accuracy**: Overall correct predictions.
- **F1-Score**: Balancing precision and recall (critical for unbalanced or sensitive scenarios).
- **AUC-ROC**: Ability of the model to distinguish between classes across thresholds.
- **Confusion Matrix**: Visualizing where our model gets confused (e.g., formal human writing vs AI).

---

## 7. Web Application
The final product is a **Streamlit App** that allows users to:
1. Paste unknown text.
2. Select a trained model.
3. See a real-time prediction with a confidence gauge.
4. Review the "Linguistic Profile" of the text (Avg sentence length, readability, etc.).

---

## 8. Conclusion and Future Work
Detecting AI text is a "cat-and-mouse" game. As LLMs evolve, detection models must become more sophisticated. 

**Future Improvements:**
- Incorporating Watermarking detection.
- Multi-lingual support.
- Explanation layers (SHAP/LIME) to show *why* a text was flagged.

---

*This project was developed as a walkthrough for a GenAI Portfolio.*
