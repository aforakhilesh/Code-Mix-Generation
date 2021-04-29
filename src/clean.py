import re
import pandas

file=pandas.read_csv('train.csv')

sents = []
for i in range(len(file['Sentence'])):
    print(i)
    test=re.sub('[]!@#$%^&*()-_=+<>:;\{\}()"]',' ',file["Sentence"][i])
    sents.append('# '+re.sub('\s+',' ',test) + ' @')

dict = {
    'Sentence' : sents
}

df = pandas.DataFrame(dict)

df.to_csv('train_clean2.csv')