import pandas as pd
import csv
sents = pd.read_csv('valid_clean.txt', header=None, delimiter='\n', error_bad_lines=False, quoting=csv.QUOTE_NONE)
sents.columns = ["Sentence"]

sents.to_csv('train.csv', index=None)