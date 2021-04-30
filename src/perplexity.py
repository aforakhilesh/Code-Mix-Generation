import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
import random
import pandas as pd
import re
import random
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def perp(dataset, model, text):
    target = text.split(' ')
    next_words = len(target)
    model.eval()
    words = [target[0]]
    state_h, state_c = model.init_state(len(words))
    prob=1
    prob1=1
    prob2=1

    for i in range(0,next_words-1):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        if target[i+1] in dataset.word_to_index.keys():
            word_exp = dataset.word_to_index[target[i+1]]
        else:
            word_exp = dataset.word_to_index['the']
        
        if prob<10**(-200):
            if prob1<10^(-200):
                prob2*=p[word_exp]
            else:
                prob1*=p[word_exp]
        else:
            prob*=p[word_exp]

        words.append(dataset.index_to_word[word_exp])

    perp = ((1/prob)**(1/(next_words)))*((1/prob1)**(1/(next_words)))*((1/prob2)**(1/(next_words)))

    return words,prob,perp

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()

dataset = Dataset(args)

model = torch.load("/home/aaradhya/Desktop/Academics/2.2/NLP/Code-Mix-Generation/src/models/colab_train_model_512_10_epoch_6gram_256lstmsize")

test_file = pd.read_csv("./Data/validate_clean2.csv")

perplexity = {}

i=0
total=0
for sentence in test_file['Sentence'][:5000]:

    perplexity[i]={}
    words,prob,perpl = perp(dataset,model,sentence)
    perplexity[i]['perplexity']=perpl
    perplexity[i]['probability']=prob
    print(perpl)
    total+=perpl
    i+=1

print("average perplexity : ",total/i)

with open("./perplexity_train_15_512_6gram.json",'w') as f:
    json.dump(perplexity, f,indent=4)