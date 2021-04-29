import pandas

file = pandas.read_csv('train_clean2.csv')

sents = []

for sentence in file['Sentence']:
    sents.append('# ' + sentence + ' @')

dict = {
    'Sentence' : sents
}

df = pandas.DataFrame(dict)

df.to_csv('train_clean3 .csv')