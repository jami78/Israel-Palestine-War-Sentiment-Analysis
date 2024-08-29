##Importing the libraries

import numpy as np
import pandas as pd

import string
import spacy

from collections import Counter


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

##Reading the dataset
df=pd.read_excel('train3.xlsx')

##Train_Test Split
# Separate the data based on sentiment
neutral_df = df[df['Sentiment'] == 'Neutral']
pro_palest_df = df[df['Sentiment'] == 'Pro-Palest']
pro_israel_df = df[df['Sentiment'] == 'Pro-Israel']

# Sample 400 rows of Neutral data for training
neutral_train_df = neutral_df.sample(n=400, random_state=42)

# Use 80% of Pro-Palest and Pro-Israel data for training
pro_palest_train_df, pro_palest_test_df = train_test_split(pro_palest_df, test_size=0.2, random_state=42)
pro_israel_train_df, pro_israel_test_df = train_test_split(pro_israel_df, test_size=0.2, random_state=42)

# Combine the training data
train_df = pd.concat([neutral_train_df, pro_palest_train_df, pro_israel_train_df])

# Combine the testing data
test_df = pd.concat([neutral_df.drop(neutral_train_df.index), pro_palest_test_df, pro_israel_test_df])

# Define features and labels for training and testing
X_train = train_df['Message']
y_train = train_df['Sentiment']
X_test = test_df['Message']
y_test = test_df['Sentiment']

##Character-Level TF-IDF as Features
# Character level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer = 'word',
                             token_pattern = r'\w{1,}',
                             max_features = 5000)
print(tfidf_vect_ngram_chars)

tfidf_vect.fit(df['Message'])
X_train_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_train)
X_test_tfidf_ngram_chars  = tfidf_vect_ngram_chars.transform(X_test)


###Other TF-IDF methods (word, n-gram) was also used as feature. But Character-Level TF-IDF with RandomForest had the highest accuracy (0.92)

##Random Forest

# Helper function

def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    return accuracy_score(predictions, y_test)

# Bagging (Random Forest) on Character Level TF IDF Vectors
rfct= RandomForestClassifier(n_estimators = 100)
accuracy4 = train_model(rfct, X_train_tfidf_ngram_chars, y_train, X_test_tfidf_ngram_chars)
print('RF, WordLevel TF-IDF : %.4f\n' % accuracy4)

 
