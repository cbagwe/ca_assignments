import json
import argparse
import numpy as np
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

import nltk
nltk.download('stopwords')

# write final prediction output in result.json file
def write_to_json_file(args_list, args_id, key):
    output_dict = {}
    for i in range(len(args_list)):
        output_dict[args_id[i]] = args_list[i]

    with open(key+'_result.json', 'w') as fp:
        json.dump(output_dict, fp)

def text_summary(argument):
    parser=PlaintextParser.from_string(argument,Tokenizer('english'))
    # creating the summarizer
    rank_summarizer=LexRankSummarizer()
    summary= rank_summarizer(parser.document,2)
    summary = [str(text) for text in summary]
    #print(summary)
    return ' '.join(summary)


def main():
    # get training dataset 
    # with open(train_file, "r") as f:
    #     train_data = json.load(f)
    # get training dataset 
    with open(args.val, "r") as f:
        val_data = json.load(f)
    # get testing dataset 
    with open(args.test, "r") as f:
        test_data = json.load(f)

    #train_args = [obj.argument for obj in train_data]
    val_args = [obj["argument"] for obj in val_data]
    test_args = [obj["argument"] for obj in test_data]

    #train_ids = [obj.id for obj in train_data]
    val_ids = [obj["id"] for obj in val_data]
    test_ids = [obj["id"] for obj in test_data]



    val_final = [text_summary(args) for args in tqdm(val_args)]

    test_final = [text_summary(args) for args in tqdm(test_args)]

    write_to_json_file(val_final, val_ids, 'val')
    write_to_json_file(test_final, test_ids, 'test')
    print("Done.")


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser("Script to classify texts in a test set into claims or non-claims.")

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
    main()