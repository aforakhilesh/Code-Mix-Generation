import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
import random
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(dataset, model, text, next_words):
    words = text.split(' ')
    model.eval()

    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        # print(p[word_index])
        words.append(dataset.index_to_word[word_index])

    return words

def perp(dataset, model, text):
    target = text.split(' ')
    next_words = len(target)
    model.eval()
    words = [target[0]]
    state_h, state_c = model.init_state(len(words))
    prob=1

    for i in range(0,next_words-1):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_exp = dataset.word_to_index[target[i+1]]
        word_index = np.random.choice(len(last_word_logits), p=p)
        
        prob*=p[word_exp]
        words.append(dataset.index_to_word[word_exp])

    perp = (1/prob)**1/(next_words)

    return words,prob,perp



parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()

dataset = Dataset(args)

model = torch.load("./models/test_15_512_epoch_model_gpu")
text=input("enter Prompt : ")
words = text.split(' ')
num_words = int(input("enter the number of words to be predicted : "))
sentence=""
predictions=predict(dataset, model, text,num_words)
for w in predictions:
    sentence=sentence+" "+w

print("the generated text is : ",sentence)

words, prob, perpl =perp(dataset,model,text)
print("perplexity of prompt is : ",perpl)
print("probability of prompt is : ",prob)

# validation=pd.read_csv('./data/CodeMixcorpora/test.csv')
# validation['Sentence'][0]
# perplexity=0

# for i in range(1000):
#     try:
#         words, prob, perpl =perp(dataset,model, validation['Sentence'][random.randrange(0,len(validation['Sentence']))])
#         # print(perpl)
#  
#         perplexity+=perpl
#     except: 
#         print("unk word")
#         i+=1
    

# print("average perplexity is : ", perplexity/1000)

