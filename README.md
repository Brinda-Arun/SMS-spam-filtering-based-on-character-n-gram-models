# SMS-spam-filtering-based-on-character-n-gram-models
## SMS Spam Classifier

This program uses the SMS Spam Collection dataset to train a Naive Bayes classifier for detecting spam messages. The dataset consists of labeled SMS messages, where "ham" indicates non-spam messages and "spam" indicates spam messages.

### Files

- `sms_spam_classifier.py`: The main script that loads the dataset, preprocesses the data, trains the classifier, and evaluates its performance.
- `smsspamcollection.zip`: The compressed zip file containing the SMS Spam Collection dataset.

**Functionality**

1.Importing the necessary libraries:

pandas: Library for data manipulation and analysis.
sklearn.feature_extraction.text.CountVectorizer: Class for extracting character n-grams from text data.
sklearn.naive_bayes.MultinomialNB: Class for training a Multinomial Naive Bayes classifier.
sklearn.metrics: Module for evaluating classification metrics.
sklearn.model_selection.train_test_split: Function for splitting the dataset into training and test data.
zipfile: Library for working with zip files.
Loading the SMS Spam Collection dataset from the zip file:

2.The dataset is extracted from the provided zip file using zipfile.ZipFile.
Loading the dataset into a Pandas dataframe:

3.The dataset is read into a Pandas dataframe using pd.read_csv.
The column names are assigned as "label" and "message".
Preprocessing the dataset:

4.The "label" column is converted to numeric values (0 for "ham" and 1 for "spam") using map.
Splitting the dataset into training and test data:

5.The dataset is split into training and test data using train_test_split from sklearn.model_selection.
The test data size is set to 20% of the original dataset, and a random state is specified for reproducibility.
Vectorizing the text data:

6.A CountVectorizer is created with the ngram_range parameter set to extract character n-grams (2-grams to 4-grams) from the messages.
The vectorizer is fitted on the training data using fit to learn the vocabulary.
The training and test data are transformed into character n-gram count representations using transform.
Training a Multinomial Naive Bayes classifier:

7. An instance of the MultinomialNB classifier is created.
The classifier is trained on the character n-gram count representations of the training data using fit.
Predicting labels and evaluating performance:

8.The labels of the test data are predicted using the trained classifier's predict method.
The accuracy of the classifier is computed using accuracy_score from sklearn.metrics.
The confusion matrix is computed using confusion_matrix from sklearn.metrics and printed.
The classification report, including precision, recall, F1-score, and support, is computed using classification_report from sklearn.metrics and printed.

### Acknowledgments

The implementation of the Naive Bayes classifier and the usage of scikit-learn modules are based on the scikit-learn documentation and examples.

Please note that the code provided here is a simplified implementation for demonstration purposes and may not include all possible optimizations or advanced techniques. It is recommended to consult the scikit-learn documentation and relevant research papers for a comprehensive understanding of text classification algorithms and best practices.
