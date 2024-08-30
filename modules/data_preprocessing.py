import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

def load_and_clean_data(filepath):
    #Data Frame = Two-dimensional table with rows and columns, similar to a spreadsheet in Excel or a table in SQL.
    data_frame = pd.read_csv(filepath, delimiter="\t", encoding="latin-1", header=None) # 'latin-1' handles special characters in Western European languages

    # Rename the columns
    data_frame.columns = ["label", "text"]

    # Convert to lowercase
    data_frame["text"] = data_frame["text"].str.lower()

    # Remove punctuation and special characters
    data_frame["text"] = data_frame["text"].str.replace("[^\w\s]", "", regex=True)

    # Remove stop words
    stop_words = set(stopwords.words("english"))

    #
    data_frame["text"] = data_frame["text"].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    data_frame["text"] = data_frame["text"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

    # Tokenization
    data_frame["tokens"] = data_frame["text"].apply(lambda x: x.split())

    # Label encoding
    data_frame["label"] = data_frame["label"].map({"ham": 0, "spam": 1})

    # Splitting the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(data_frame["text"], data_frame["label"], test_size = 0.2, random_state = 42)

    # Text vectorization
    vectorizer = CountVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    return x_train_vec, x_test_vec, y_train, y_test

file_path = "data\SMSSpamCollection.csv"

x_train_vec, x_test_vec, y_train, y_test = load_and_clean_data(file_path)

print(f"X_train_vec shape: {x_train_vec.shape}")
print(f"X_test_vec shape: {x_test_vec.shape}")
print(f"y_train sample: {y_train.head()}")
print(f"y_test sample: {y_test.head()}")
