# import modules
import argparse
import json
import nltk
import pandas as pd
import spacy

from nltk.stem import PorterStemmer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# combine training and test data
def get_corpus_info(train_data, test_data):
    corpus_data = train_data + test_data
    keys = list(corpus_data[0].keys())
    corpus_df = pd.DataFrame(columns=keys, data=corpus_data)
    return corpus_df

# split corpus into training and test data sets based on their size/length
def split_corpus(messages_bow, train_size, test_size, corpus_df): 
    x_train,x_test,y_train,y_test = train_test_split(
        messages_bow, 
        corpus_df['label'], 
        test_size = test_size, 
        train_size = train_size, 
        random_state=0,
        shuffle=False)
    return [x_train,x_test,y_train,y_test]

# get part of speech
def part_of_speech_text(text):
    tokenized_pos_tag = nltk.pos_tag(stem_text(text))
    return tokenized_pos_tag

# perform stemming on the words
def stem_text(text):
    tokenized_stem_text = []
    ps = PorterStemmer()
    for token in clean_text(text):
    	tokenized_stem_text.append(ps.stem(token))

    return tokenized_stem_text

# remove punctuation, space, urls, stopwords from text
def clean_text(text):
    parsed_text = nlp_english(text)
    clean_text = []
    for token in parsed_text:
        stop_flag = (token.is_punct or token.is_space or  
                 token.like_url or token.is_stop)
        if (not stop_flag):
            clean_text.append(token.text.lower())
    return clean_text

# Preprocess and convert the corpus into bag of words
def transform_to_bow(corpus_df):
    return CountVectorizer(
    analyzer=part_of_speech_text,
    ).fit_transform(corpus_df["text"])
    
# fit training data and predict for test set
def classify(x_train,x_test,y_train,y_test):
    # parameters updated using cross validation
    clf = MultinomialNB(alpha=0.1)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    # print evaluation result to console
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    print("F1:",metrics.f1_score(y_test, y_pred))
    return y_pred

# Write final prediction output in result.json file
def write_to_json_file(test_data, y_pred):
    output_dict = {}
    for i in range(len(test_data)):
        output_dict[test_data[i]["id"]] = int(y_pred[i])
        
    with open('result.json', 'w') as fp:
        json.dump(output_dict, fp)

def main():
    # get training dataset 
    with open(args.train, "r") as f:
        train_data = json.load(f)
    # get testing dataset 
    with open(args.test, "r") as f:
        test_data = json.load(f)
    # get size of each data, size will be used later to split data 
    train_size = len(train_data)
    test_size = len(test_data)
    # combine data 
    corpus_df = get_corpus_info(train_data, test_data)
    # apply cleaning and feature engineering techniques
    messages_bow = transform_to_bow(corpus_df)
    # split data into training and test sets again
    data_splits = split_corpus(messages_bow, 
    train_size, test_size, corpus_df)
    # classify and predict output
    y_pred = classify(data_splits[0], data_splits[1], 
    data_splits[2], data_splits[3])
    # generate desired output file
    write_to_json_file(test_data, y_pred)

if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser("Script to classify texts in a test set into claims or non-claims.")

    parser.add_argument(
        "--train",
        "-x",
        required=True,
        help="Path to the training dataset",
        metavar="TRAINING_DATA")
    parser.add_argument(
        "--test",
        "-y",
        required=True,
        help="Path to the test dataset",
        metavar="TESTING_DATA")
    args = parser.parse_args()
    nlp_english = spacy.load("en_core_web_sm")
    main()
    print("Done.")
