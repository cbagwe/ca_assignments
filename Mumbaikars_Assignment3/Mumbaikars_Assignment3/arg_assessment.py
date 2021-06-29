#import statements
import argparse
import json
import numpy as np
import re
import spacy

from nltk.stem import PorterStemmer
from numpy import dot
from numpy.linalg import norm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from textblob import TextBlob
from tqdm import tqdm

# remove punctuation, space, urls and stop words from text
def clean_text(text):
    parsed_text = nlp_english(text)
    clean_text = []
    for token in parsed_text:
        stop_flag = (token.is_punct or token.is_space or  
                 token.like_url)
        if (not stop_flag):
            clean_text.append(re.sub('[^A-Za-z0-9]+', ' ',token.text.lower()))
            
    return clean_text

# get the root of each word in the text
def stem_text(text):
    return [stemmer.stem(word) for word in clean_text(text)]

# counts the number of abusive words in a text
def count_insults(text):
    insult_words = ["ass", "idiot", "fuck", "shit"]
    counter = 0
    for word in text:
        if word in insult_words:
            counter = counter + 1
            
    return counter

# counts the number of pos tag in a text
def get_number(list_of_tuple, key):
    counter = 0
    for (word,tag) in list_of_tuple:
        if key in tag:
            counter = counter + 1
    return counter

# for each thread gather primary features for each dialogue in the thread
# primary features -> length of argument, insults, pos tags,sentiment
def gather_data(thread):
    returnObj = {}
    for i in range(len(thread["preceding_posts"])):
        comment_data = {}
        comment = thread["preceding_posts"][i]
        # clean text
        comment_data["text"] = stem_text(comment["body"])
        # length just in case of Godwin's Law
        comment_data["char_length_vec"] = [len("".join(comment_data["text"]))]
        # check for some common insults
        comment_data["insults_vec"] = [count_insults(comment_data["text"])]
        # calculate number of POS tags
        sentence_tags = TextBlob(comment["body"]).tags
        comment_data["count_pos"] = [
            get_number(sentence_tags, 'NN'), 
            get_number(sentence_tags, 'VBP'),
            get_number(sentence_tags, 'MD'), 
            get_number(sentence_tags, 'PRP')
            ]
        # get sentiment data
        sentiment = TextBlob(' '.join(comment_data["text"])).sentiment
        comment_data["sentiment"] =  [sentiment.polarity, sentiment.subjectivity]
        # merge into one vector
        feature_vec = comment_data["char_length_vec"] + comment_data["insults_vec"] + comment_data["sentiment"] + comment_data["count_pos"]
        returnObj[" ".join(comment_data["text"])] = feature_vec
        
    return returnObj

# checks if the dialogues in a given thread are 
# strictly increasing or decreasing in a particular property
def is_increased(array, index):
    values = [x[index] for x in array]
    ret_answer = values[0] < values[1]
    return int(ret_answer)

# to find trends of the thread,
# calculate secondary features from the previously calculated primary features
# secondary features = cosine similarity, avg polarity, avg insults, 
# increasing or decreasing trends 
def combine_vectors(ddict):
    (insults_index, polar_index) = (1,2)
    feature_vectors = list(ddict.values())
    # for cosine similarity and cosine distance
    # calculate dot product and product of norm of vectors 
    dot_product = dot(feature_vectors[0], feature_vectors[-1])
    norms_product = (norm(feature_vectors[0])*norm(feature_vectors[-1]))
    # For some vectors, the product of norm is so less 
    # that python "considers" it as zero. 
    # Eg: Long float point numbers with very large negative exponents
    if norms_product == 0:
        cos_sim = 1
    else:
        cos_sim = dot_product/norms_product
    # average and increasing thrend in number of insults
    avg_insults = np.average([x[insults_index] for x in feature_vectors])
    is_increasing_insults = is_increased(feature_vectors, insults_index)
    # average and increasing thrend in polarity
    avg_polarity = np.average([x[polar_index] for x in feature_vectors])
    is_increasing_polarity = is_increased(feature_vectors, polar_index)
    # final vector for one thread
    return [avg_insults, is_increasing_insults, avg_polarity, is_increasing_polarity,  cos_sim]

# concatinate dialogues for bow
def concatAllStringForBoW(listOfDict):
    return_obj = []
    for d_dict in listOfDict:
        return_obj.append(" ".join(list(d_dict.keys())))
        
    return return_obj  

# write final prediction output in result.json file
def write_to_json_file(dataset, key, pred):
    ids_list = dataset[key + "_ids"]
    output_dict = {}
    for i in range(len(ids_list)):
        output_dict[ids_list[i]] = int(pred[i])

    with open(key + '_result.json', 'w') as fp:
        json.dump(output_dict, fp)

# print statistics of the model
def print_statistics(a, b):
    print("Accuracy:",metrics.accuracy_score(a, b))
    print("Precision:",metrics.precision_score(a, b))
    print("Recall:",metrics.recall_score(a, b))
    print("F1 score:",metrics.f1_score(a, b))

def main():
    # get training dataset 
    with open(args.train, "r") as f:
        train_data = json.load(f)
    # get training dataset 
    with open(args.val, "r") as f:
        val_data = json.load(f)
    # get testing dataset 
    with open(args.test, "r") as f:
        test_data = json.load(f)

    # gather all data in a single dictionary 
    entire_dataset = {
        'train_ids': [thread["id"] for thread in train_data],
        'train_posts': [thread["preceding_posts"] for thread in train_data],
        'train_label': [thread["label"] for thread in train_data],
        
        'val_ids': [thread["id"] for thread in val_data],
        'val_posts': [thread["preceding_posts"] for thread in val_data],
        'val_label': [thread["label"] for thread in val_data],
        
        'test_ids': [thread["id"] for thread in test_data],
        'test_posts': [thread["preceding_posts"] for thread in test_data],
        'test_label': [thread["label"] for thread in test_data],
    }

    # gather primary features for each thread in each dataset
    entire_dataset["train_prep"] = [gather_data(thread) for thread in tqdm(train_data)]
    entire_dataset["val_prep"] = [gather_data(thread) for thread in tqdm(val_data)]
    entire_dataset["test_prep"] = [gather_data(thread) for thread in tqdm(test_data)]

    # generate training data
    x_train = [combine_vectors(thread) for thread in tqdm(entire_dataset['train_prep'])]
    x_val = [combine_vectors(thread) for thread in tqdm(entire_dataset['val_prep'])]
    x_test = [combine_vectors(thread) for thread in tqdm(entire_dataset['test_prep'])]
    # concatinate all dialogues to pass as input for bow
    train_bow_input = concatAllStringForBoW(entire_dataset["train_prep"])
    val_bow_input = concatAllStringForBoW(entire_dataset["val_prep"])
    test_bow_input = concatAllStringForBoW(entire_dataset["test_prep"])
    # bow
    vectorizer = CountVectorizer()
    train_bow = vectorizer.fit_transform(train_bow_input).toarray().tolist()
    val_bow = vectorizer.transform(val_bow_input).toarray().tolist()
    test_bow = vectorizer.transform(test_bow_input).toarray().tolist()
    # merge bow and computed "secondary" vector
    [x_train[i].extend(train_bow[i]) for i in tqdm(range(len(x_train)))]
    [x_val[i].extend(val_bow[i]) for i in tqdm(range(len(x_val)))]
    [x_test[i].extend(test_bow[i]) for i in tqdm(range(len(x_test)))] 

    y_train = entire_dataset["train_label"]
    y_val = entire_dataset["val_label"]
    y_test = entire_dataset["test_label"]

    # initialize classifier
    clf = SVC()
    # train the model
    clf.fit(x_train,y_train)
    # prediction phase
    val_pred = clf.predict(x_val)
    test_pred = clf.predict(x_test)

    # results
    print("Validation Data Statistics")
    print_statistics(y_val, val_pred)
    print("Test Data Statistics")
    print_statistics(y_test, test_pred)

    write_to_json_file(entire_dataset, 'val', val_pred)
    write_to_json_file(entire_dataset, 'test', test_pred)

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
        "--val",
        "-y",
        required=True,
        help="Path to the validation dataset",
        metavar="VALIDATION_DATA")

    parser.add_argument(
        "--test",
        "-z",
        required=True,
        help="Path to the test dataset",
        metavar="TESTING_DATA")
    args = parser.parse_args()
    #create spacy object
    nlp_english = spacy.load("en_core_web_sm")
    #create Stemmer object
    stemmer = PorterStemmer()
    main()
    print("Done.")