Assignment 2 - Argument Mining
==============================

## Evaluate your results
The validation set of the data should be used for evaluating your approach. You can use our evaluation script to check the performance of your classifier on that set. After saving the predictions of your classifier in the format specified below, you can evaluate them on the validation set with the provided `eval.py`. You can use it by running `python eval.py -t <path-to-ground-truth-file> -p <path-to-predictions-file>`. The ground truth data should be in the original format, the predictions file in the format defined below.
We will use the same script to evaluate your approach but with a separate test set.


## Output format
```json
{
    "<sentence_id_1>": "<label>" # where <label> is either 1 (contains claim) or 0
    "<sentence_id_2>": "<label>" # where <label> is either 1 (contains claim) or 0
    ...
}
```
