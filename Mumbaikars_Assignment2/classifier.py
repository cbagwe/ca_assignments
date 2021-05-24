#import libraries
import json
import pandas
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer, PatternAnalyzer
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt


def clean_text(text):
    # Parse the text using the English language model
    # The returned object is an iterator over all tokens
    parsed_text = nlp(text)
    #print(parsed_text)
    # Initialize a list which will later hold the tokens of the text
    tokenized_clean_text = []
    
    # For each token in the text...
    for token in parsed_text:
        # If the token is _not_ one of the following, append it to
        # the final list of tokens; continue otherwise
        if (not token.is_punct and  # Punctuation
                not token.is_space and  # Whitespace of any kind
                not token.like_url and # Anything that looks like an url
                not token.is_stop):  # Stopwords
            tokenized_clean_text.append(token.text.lower())
    
    # Return the cleaned version for this text
    return ' '.join(tokenized_clean_text)

special_verbs_list = ["because","should","would","that","must","could","believe","wish","belief"]
def check_spezial_words(text):
    #return_data = False
    if any(x in text for x in special_verbs_list):
        #print("\n----------------" + text + " True\n")
        return 1
    #print("\n----------------" + text + " False\n")
    return 0

#def check_sentiments(text):
#    polarity_val = TextBlob(text).sentiment.polarity
#    if polarity_val < 0:
#        return 0
#    return 1

#get training set
with open("train-data-prepared.json", "r") as f:
    train_data = json.load(f)

#get validation set
with open("val-data-prepared.json", "r") as f:
    val_data = json.load(f)

#print(train_data)
#get keys of training set
train_keys = list(train_data[0].keys())
#initialize pandas dataframe
train_df = pandas.DataFrame(columns=train_keys,data=train_data)

#get keys of training set
val_keys = list(val_data[0].keys())
#initialize pandas dataframe
val_df = pandas.DataFrame(columns=val_keys,data=val_data)

#print(train_keys)
#print(train_df)
#load english language
nlp = spacy.load("en_core_web_sm")
#clean the training set
train_df["cleaned_text"] = train_df["text"].apply(clean_text)
val_df["cleaned_text"] = val_df["text"].apply(clean_text)
# print(train_df)

#Feature Extraction
#Number of characters
#train_df["char_count"] = train_df["text"].apply(len)
#val_df["char_count"] = val_df["text"].apply(len)
#print(charCount)
train_df["contains_verbs"] = train_df["text"].apply(check_spezial_words)
val_df["contains_verbs"] = val_df["text"].apply(check_spezial_words)
#Count Vectorization
vectorizer = CountVectorizer()
X_train_vectorised = vectorizer.fit_transform(train_df["cleaned_text"])

vectorizer3 = CountVectorizer()
X_test_vectorised = vectorizer.fit_transform(val_df["cleaned_text"])

#print("X")
#print(vectorizer.get_feature_names())
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 2))
X2 = vectorizer2.fit_transform(train_df["text"])
# print("X2")
# print(vectorizer2.get_feature_names())

#sentiment Analysis
#train_df["sentiment"] = train_df["text"].apply(check_sentiments)
#val_df["sentiment"] = val_df["text"].apply(check_sentiments)
#print(train_df["sentiment"])

train_df["sentiment"] = train_df["text"].apply(lambda x: 
                   TextBlob(x, analyzer=NaiveBayesAnalyzer()).sentiment.p_pos)
val_df["sentiment"] = val_df["text"].apply(lambda x: 
                   TextBlob(x, analyzer=NaiveBayesAnalyzer()).sentiment.p_pos)


train_df["noun_count"] = train_df["text"].apply(lambda x: 
                   len(TextBlob(x).noun_phrases))
val_df["noun_count"] = val_df["text"].apply(lambda x: 
                   len(TextBlob(x).noun_phrases))

#Classification
X_train = np.array([train_df["sentiment"]])
X_train = np.transpose(X_train)
Y_train = train_df["label"]
Y_train = Y_train.values.reshape(len(Y_train),)

clf = SVC()
clf.fit(X_train,Y_train)

X_test = np.array([val_df["sentiment"]])
X_test = np.transpose(X_test)
Y_test = val_df["label"]
Y_test = Y_test.values.reshape(len(Y_test),)


Y_pred = clf.predict(X_test)

print(Y_test)
print(Y_pred)

print(train_df["sentiment"])
print(val_df["sentiment"])

# Evaluate the predictions and print the result
print(f1_score(Y_test, Y_pred))