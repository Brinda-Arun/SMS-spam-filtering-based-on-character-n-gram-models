import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import zipfile

# Load the SMS Spam Collection dataset from the zip file
with zipfile.ZipFile("C:/Users/arunk/OneDrive/Desktop/Brinda/smsspamcollection.zip", "r") as zip_ref:
    zip_ref.extractall()

# Load the dataset into a Pandas dataframe
df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

df["label"] = df["label"].map({"ham": 0, "spam": 1})
#Split the dataset into training and test data


train_data, test_data, train_labels, test_labels = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# Assign the variables to the original names if needed
X_train = train_data
X_test = test_data
y_train = train_labels
y_test = test_labels

#Use a CountVectorizer to extract character n-grams from the messages
vectorizer = CountVectorizer(ngram_range=(2, 4), analyzer="char")
vectorizer.fit(X_train)
X_train_counts = vectorizer.transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier on the character n-gram counts
clf = MultinomialNB().fit(X_train_counts, y_train)

# Predict the labels of the test data
y_pred = clf.predict(X_test_counts)


# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Print the confusion matrix
print("Confusion matrix:")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"      Actual\\Predicted | {'Spam':<5} {'Ham':<5}")
print(f"      -----------------|---------")
print(f"               Spam    | {tp:<5} {fn:<5}")
print(f"               Ham     | {fp:<5} {tn:<5}")

# Print the classification report
print("Classification report:")
report = classification_report(y_test, y_pred)
print(report)


