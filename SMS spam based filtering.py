"""
SMS Spam Classifier using Character N-gram Models
==================================================
Filters SMS spam messages using a Multinomial Naive Bayes classifier
trained on character n-gram features extracted via CountVectorizer.
 
Achieves 98.75% accuracy on the SMS Spam Collection dataset.
 
Dataset: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
Download the dataset zip file and place it in the project root before running.
 
Usage:
    python sms_spam_classifier.py
"""
 
import zipfile
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ZIP   = 'smsspamcollection.zip'
DATASET_FILE  = 'SMSSpamCollection'
NGRAM_RANGE   = (2, 4)      # Extract character bigrams, trigrams, four-grams
ANALYZER      = 'char'      # Character-level n-gram analysis
TEST_SIZE     = 0.2         # 80% training / 20% testing split
RANDOM_STATE  = 42          # Seed for reproducibility
 
 
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(zip_path: str, filename: str) -> pd.DataFrame:
    """
    Extract and load the SMS Spam Collection dataset from a zip archive.
 
    Args:
        zip_path : Path to the dataset zip file.
        filename : Name of the TSV file inside the archive.
 
    Returns:
        df : DataFrame with columns ['label', 'message'].
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall('.')
 
    df = pd.read_csv(
        filename,
        sep='\t',
        header=None,
        names=['label', 'message'],
        encoding='latin-1'
    )
 
    print(f"[INFO] Dataset loaded — {len(df)} messages")
    print(f"[INFO] Class distribution:\n{df['label'].value_counts()}\n")
    return df
 
 
# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    """
    Map string labels to binary integers and split into features and labels.
 
    Args:
        df : Raw DataFrame with 'label' and 'message' columns.
 
    Returns:
        X : Series of SMS message strings.
        y : Series of binary labels (0 = ham, 1 = spam).
    """
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df['message'], df['label']
 
 
# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(X_train, X_test):
    """
    Transform raw SMS text into character n-gram count matrices.
 
    Uses CountVectorizer with character-level n-grams (bigrams to four-grams).
    Vocabulary is learned from training data only to prevent data leakage.
 
    Args:
        X_train : Training messages.
        X_test  : Test messages.
 
    Returns:
        X_train_counts : Sparse matrix of n-gram counts for training data.
        X_test_counts  : Sparse matrix of n-gram counts for test data.
        vectorizer     : Fitted CountVectorizer instance.
    """
    vectorizer = CountVectorizer(
        analyzer    = ANALYZER,
        ngram_range = NGRAM_RANGE
    )
 
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts  = vectorizer.transform(X_test)
 
    print(f"[INFO] Vocabulary size : {len(vectorizer.vocabulary_)} n-grams")
    print(f"[INFO] Training matrix : {X_train_counts.shape}")
    print(f"[INFO] Test matrix     : {X_test_counts.shape}\n")
 
    return X_train_counts, X_test_counts, vectorizer
 
 
# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train_counts, X_test_counts, y_train, y_test):
    """
    Train a Multinomial Naive Bayes classifier and evaluate on the test set.
 
    Multinomial NB is well-suited for text classification with count features.
    Laplace smoothing (alpha=1.0) is applied by default to avoid zero probabilities
    for n-grams unseen during training.
 
    Args:
        X_train_counts : Sparse training feature matrix.
        X_test_counts  : Sparse test feature matrix.
        y_train        : Training labels.
        y_test         : True test labels.
    """
    # Train
    clf = MultinomialNB()
    clf.fit(X_train_counts, y_train)
    print("[INFO] Model training complete.\n")
 
    # Predict
    y_pred = clf.predict(X_test_counts)
 
    # Results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy:.4%}\n")
 
    print("Confusion Matrix:")
    print(f"{'':>20} {'Predicted Ham':>15} {'Predicted Spam':>15}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'Actual Ham':>20} {cm[0][0]:>15} {cm[0][1]:>15}")
    print(f"{'Actual Spam':>20} {cm[1][0]:>15} {cm[1][1]:>15}\n")
 
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Ham', 'Spam'],
        digits=4
    ))
 
    return clf
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load data
    df = load_dataset(DATASET_ZIP, DATASET_FILE)
 
    # Preprocess
    X, y = preprocess(df)
 
    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE
    )
    print(f"[INFO] Train size : {len(X_train)} | Test size : {len(X_test)}\n")
 
    # Feature extraction
    X_train_counts, X_test_counts, _ = extract_features(X_train, X_test)
 
    # Train and evaluate
    train_and_evaluate(X_train_counts, X_test_counts, y_train, y_test)
 
 
if __name__ == '__main__':
    main()
