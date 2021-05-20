#import libraries
import json
import pandas
import spacy

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

#get training set
with open("train-data-prepared.json", "r") as f:
    train_data = json.load(f)
#print(train_data)
#get keys of training set
train_keys = list(train_data[0].keys())
#initialize pandas dataframe
train_df = pandas.DataFrame(columns=train_keys,data=train_data)
#print(train_keys)
#print(train_df)
#load english language
nlp = spacy.load("en_core_web_sm")
#clean the training set
train_df["cleaned_text"] = train_df["text"].apply(clean_text)
#print(train_df)