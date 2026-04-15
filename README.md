# SMS Spam Classifier — Character N-gram Models

SMS spam detection using a Multinomial Naive Bayes classifier trained on character n-gram features, achieving **98.75% accuracy** on the SMS Spam Collection dataset.

---

## Overview

Spam filtering is a classic text classification problem with direct real-world impact. This project implements a character-level n-gram approach rather than the more common word-level model, enabling the classifier to capture morphological patterns and partial word sequences — making it robust to deliberate misspellings and obfuscated spam text commonly found in SMS messages.

The pipeline uses scikit-learn's `CountVectorizer` to extract character bigrams, trigrams, and four-grams, which are fed into a Multinomial Naive Bayes classifier trained on the UCI SMS Spam Collection dataset.

---

## Pipeline

```
Raw SMS Messages
        │
        ▼
Label Encoding
(ham → 0, spam → 1)
        │
        ▼
Train / Test Split
(80% training / 20% testing, random_state=42)
        │
        ▼
Character N-gram Feature Extraction
CountVectorizer(analyzer='char', ngram_range=(2,4))
Vocabulary learned from training data only
        │
        ▼
Multinomial Naive Bayes Training
(Laplace smoothing, alpha=1.0)
        │
        ▼
Prediction on Test Set
        │
        ▼
Evaluation — Accuracy, Confusion Matrix, Classification Report
```

---

## Why Character N-grams?

| Approach | Advantage |
|---|---|
| Word-level n-grams | Captures semantic meaning |
| **Character n-grams** | **Robust to misspellings, abbreviations, and obfuscated spam text** |

Character-level features allow the model to generalise across SMS-specific language patterns such as `fr33`, `txt2win`, and `ur` — patterns that word-level models struggle to handle.

---

## Model Configuration

| Parameter | Value | Description |
|---|---|---|
| `analyzer` | `char` | Character-level n-gram extraction |
| `ngram_range` | `(2, 4)` | Extracts bigrams, trigrams, and four-grams |
| Classifier | `MultinomialNB` | Suited for discrete count features |
| Smoothing | Laplace (α = 1.0) | Prevents zero probability for unseen n-grams |
| Test size | 20% | 80/20 train-test split |
| Random state | 42 | Fixed seed for reproducibility |

---

## Results

**Overall accuracy: 98.75%**

### Confusion Matrix

| | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 960 | 6 |
| **Actual Spam** | 10 | 139 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Ham | 0.99 | 0.99 | 0.99 | 966 |
| Spam | 0.96 | 0.93 | 0.95 | 149 |
| **Weighted avg** | **0.99** | **0.99** | **0.99** | **1115** |

### Key Observations

- Only **6 ham messages** were incorrectly classified as spam (false positives) — critical in real-world deployment where legitimate messages must not be blocked
- Only **10 spam messages** were missed (false negatives) out of 149
- The character n-gram approach proves highly effective even without any preprocessing such as stemming or stop-word removal

---

## Project Structure

```
sms-spam-classifier-ngram/
├── sms_spam_classifier.py   # Full pipeline — loading, training, evaluation
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

## Requirements

```
scikit-learn>=0.24.0
pandas>=1.3.0
numpy>=1.21.0
```

Install dependencies:
```bash
pip install scikit-learn pandas numpy
```

---

## Dataset Setup

Download the SMS Spam Collection dataset from the UCI repository:
[https://archive.ics.uci.edu/ml/datasets/sms+spam+collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

Place `smsspamcollection.zip` in the project root directory.

Dataset statistics:
- Total messages: 5,574
- Ham (legitimate): 4,827 (86.6%)
- Spam: 747 (13.4%)

---

## Usage

```bash
python sms_spam_classifier.py
```

The script outputs training info, vocabulary size, accuracy, confusion matrix, and a full classification report.

---

## Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat-square)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat-square)
![pandas](https://img.shields.io/badge/-pandas-150458?logo=pandas&logoColor=white&style=flat-square)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat-square)

---

## Academic Context

Machine Learning — M.Eng. Information Technology
SRH Hochschule Heidelberg, Germany
Supervised by Prof. Dr. Milan Gnjatovic

---

## References

- Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering. *Proceedings of the ACM DOCENG 2011*.
- UCI Machine Learning Repository — SMS Spam Collection Dataset
