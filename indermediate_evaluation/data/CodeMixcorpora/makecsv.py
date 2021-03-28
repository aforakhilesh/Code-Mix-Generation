import pandas as pd

sents = pd.read_csv('temp.txt', header=None, delimiter='\n', error_bad_lines=False)
sents.columns = ["Sentence"]

sents.to_csv('temp.csv', index=None)