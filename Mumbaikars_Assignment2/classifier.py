import json
import nltk
import numpy as np
import pandas as pd
import spacy
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

with open("train-data-prepared.json", "r") as f:
    train_data = json.load(f)
    
#get validation set
with open("val-data-prepared.json", "r") as f:
    val_data = json.load(f)

def clean_text(text):
    # Parse the text using the English language model
    # The returned object is an iterator over all tokens
    parsed_text = nlp_english(text)
    # Initialize a list which will later hold the tokens of the text
    tokenized_clean_text = []
    tokenized_stemclean_text = []
    tokenized_pos_tag = []
    #lemmatized_output = []

    #tokens = word_tokenize(parsed_text)
    #print(tokens)''
    ps = PorterStemmer()
    #stemmer = SnowballStemmer(language='english')
    # For each token in the text...
    for token in parsed_text:
        # If the token is _not_ one of the following, append it to
        # the final list of tokens; continue otherwise
        if (not token.is_punct and  # Punctuation
                not token.is_space and  # Whitespace of any kind
                not token.like_url and # Anything that looks like an url
                not token.is_stop):  # Stopwords
            tokenized_clean_text.append(token.text.lower())

    
    #nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    #parsed_text = nlp(' '.join(tokenized_clean_text))
    #print(parsed_text)
    #lemmatized_output.append(token.lemma_ for token in parsed_text) 

    for token in tokenized_clean_text:
    	tokenized_stemclean_text.append(ps.stem(token))

    #for token in tokenized_stemclean_text:
    tokenized_pos_tag = nltk.pos_tag(tokenized_stemclean_text)
    #print(nltk.ne_chunk(tokenized_pos_tag).toarray())
    #return nltk.ne_chunk(tokenized_pos_tag).toarray()
    # Return the cleaned version for this text
    return tokenized_pos_tag

corpus_data = train_data + val_data

train_keys = list(train_data[0].keys())
#val_keys = list(val_data[0].keys())

#initialize pandas dataframe
train_df = pd.DataFrame(columns=train_keys,data=corpus_data)
#val_df = pd.DataFrame(columns=train_keys,data=val_data)

nlp_english = spacy.load("en_core_web_sm")
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
messages_bow = CountVectorizer(analyzer=clean_text).fit_transform(train_df["text"])
#tf_transformer = TfidfTransformer(use_idf=False).fit(messages_bow)
#messages_tfidf = tf_transformer.transform(messages_bow)


train_size = len(train_data)
test_size = len(val_data)

x_train,x_test,y_train,y_test = train_test_split(
    messages_bow, 
    train_df['label'], 
    test_size = test_size, 
    train_size = train_size, 
    random_state=0,
    shuffle=False)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = SVC(kernel="linear",gamma="scale",C=100.0)
clf.fit(x_train,y_train)

#clf = MultinomialNB().fit(x_train,y_train)

# Defining the grids that should be searched
# param_grids = [
# {
# "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
# "gamma": ["scale"],
# "kernel": ["linear", "poly"]
# },
# {
# "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
# "gamma": [0.001, 0.0001],
# "kernel": ["rbf"]
# }
# ]
# # Initializing a new classifier
# grid_clf = MultinomialNB()

# # Initializing the grid search class
# grid_search = GridSearchCV(grid_clf, param_grid=param_grids, cv=5, n_jobs=8,scoring="f1")
# # Starting the grid search
# grid_search.fit(x_train,y_train)
# # Print the best parameter combination
# print(grid_search.best_params_)

# print(
# cross_val_score(
# MultinomialNB(**grid_search.best_params_),
# x_train,y_train,
# scoring="f1",
# cv=5))


#clf = KNeighborsClassifier(4).fit(x_train,y_train)

x_train_pred = clf.predict(x_train) 
x_test_pred = clf.predict(x_test)

print(x_train_pred)
print(y_train.values)

print("Accuracy for train data:",metrics.accuracy_score(y_train, x_train_pred))
print("Accuracy for test data:",metrics.accuracy_score(y_test, x_test_pred))

print("Precision train:",metrics.precision_score(y_train, x_train_pred))
print("Precision test:",metrics.precision_score(y_test, x_test_pred))

print("Recall train:",metrics.recall_score(y_train, x_train_pred))
print("Recall test:",metrics.recall_score(y_test, x_test_pred))

print("F1 score train:",metrics.f1_score(y_train, x_train_pred))
print("F1 score test:",metrics.f1_score(y_test, x_test_pred))

# SVM + Bag Of Words = 33%
# SVM + TDIDF = 27%
# NB + BOW = 36%
# NB + TDIDF = 12%
# NB + TDIDF + BOW = 11%
# SVM + TDIDF + BOW = 30%
# SVM + TDIDF + BOW + Stem= 34%
# NB + BOW + Stem= 39%
# NB + BOW + Stem + POS = 41%
# SVM + BOW + Stem + POS = 39%
# SVM + TDIDF + BOW + Stem + POS = 32%
# SVM + BOW + Stem + POS + GridCV = 47%