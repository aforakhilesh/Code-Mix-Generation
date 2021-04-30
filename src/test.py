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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        prob*=p[word_exp]
        words.append(dataset.index_to_word[word_exp])

    perp = (1/prob)**(1/(next_words))

    return words,prob,perp

def predict(dataset, model, text, next_words):
    words = text.split(' ')
    print("normal")

    model.eval()

    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(device)
        # print(x)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        # print(type(p))
        word_index = np.random.choice(len(last_word_logits), p=p)
        # word_index = np.argmax(p)
        # print(p[word_index])
        words.append(dataset.index_to_word[word_index])
        if words[-1]=='@':
            break

    return words

def beamsearch(dataset,model,text,next_words):
    # print()
    words = text.split(' ')
    model.eval()
    print("beam")
    state_h, state_c = model.init_state(len(words))

    sequences=[[words,1.0]]

    for i in range(0,next_words):
        all_candidates=[]
        for bw in range(len(sequences)):
            seq,prob = sequences[bw]
            # print(seq)
            for j in range(100):
                candidate=[]
                x = torch.tensor([[dataset.word_to_index[w] for w in seq[i:]]]).to(device)
                y_pred, (state_h, state_c) = model(x, (state_h, state_c))
                last_word_logits = y_pred[0][-1]
                p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
                # word_index = np.random.choice(len(last_word_logits), p=p)
                
                idx = np.argpartition(p,-10)[-10:]

                cm,ne,nh=cmi(seq)
                # print(cm)

                if cm<15:
                    if ne>nh:
                        for wrd_idx in idx:
                            if bool(re.match(("[\u0900-\u097F]+"), dataset.index_to_word[wrd_idx])):
                                word_index = wrd_idx
                            else:
                                rng=random.randint(0, 4)
                                word_index = idx[rng]
                    else :
                        for wrd_idx in idx:
                            if bool(re.match(("[a-z]+"), dataset.index_to_word[wrd_idx])):
                                word_index = wrd_idx
                            else:
                                rng=random.randint(0, 4)
                                word_index = idx[rng]
                else:
                    rng=random.randint(0, 4)
                    word_index = idx[rng]

                seq.append(dataset.index_to_word[word_index])
                temp=seq.copy()
                # print(temp)
                seq.pop()
                candidate = [temp, prob*p[word_index]]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[1])

        sequences = ordered[:5]
        # print("seq:",sequences)

    for test in sequences:
        # print(test)
        sentence=""
        for w in test[0]:
            sentence=sentence+" "+w
        print(sentence)
        words, prob, perpl =perp(dataset,model,sentence)
        print("probability is : ",prob)
        print("perplexity is : ",perpl)
        

def beamsearch_nocmi(dataset,model,text,next_words):
    # print()
    words = text.split(' ')
    model.eval()
    print("beam with no cmi forcing")
    state_h, state_c = model.init_state(len(words))

    sequences=[[words,1.0]]

    for i in range(0,next_words):
        all_candidates=[]
        for bw in range(len(sequences)):
            seq,prob = sequences[bw]
            # print(seq)
            for j in range(100):
                candidate=[]
                x = torch.tensor([[dataset.word_to_index[w] for w in seq[i:]]]).to(device)
                y_pred, (state_h, state_c) = model(x, (state_h, state_c))
                last_word_logits = y_pred[0][-1]
                p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
                # word_index = np.random.choice(len(last_word_logits), p=p)
                
                idx = np.argpartition(p,-10)[-10:]


        
                rng=random.randint(0, 4)
                word_index = idx[rng]

                seq.append(dataset.index_to_word[word_index])
                temp=seq.copy()
                # print(temp)
                seq.pop()
                candidate = [temp, prob*p[word_index]]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[1])

        sequences = ordered[:5]
        # print("seq:",sequences)

    for test in sequences:
        # print(test)
        sentence=""
        for w in test[0]:
            sentence=sentence+" "+w
        print(sentence)
        words, prob, perpl =perp(dataset,model,sentence)
        print("probability is : ",prob)
        print("perplexity is : ",perpl)
        

def cmi(sentence):

    n_e=0
    n_h=0
    u=0

    for word in sentence:
        # print(word)
        if bool(re.match("[a-z]+", word)):
            n_e+=1
        elif word.isnumeric():
            print(word)
            u+=1
        elif bool(re.match(("[\u0900-\u097F]+"), word)):
            n_h+=1

    if (n_h+n_e)==0:
        return 0

    return ((100*(1-(max(n_e,n_h)/(n_h+n_e)))),n_e,n_h)

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()

dataset = Dataset(args)

model = torch.load("/home/aaradhya/Desktop/Academics/2.2/NLP/Code-Mix-Generation/src/models/colab_train_model_512_10_epoch_6gram_256lstmsize")


text=input("enter Prompt : ")
words = text.split(' ')
num_words = int(input("enter the number of words to be predicted : "))
sentence=""
beamsearch(dataset, model, "# "+text,num_words)
print()
print()
beamsearch_nocmi(dataset, model, "# "+text,num_words)
print()
print()

predictions=predict(dataset, model, "# "+text,num_words)
predictions.append(" @")
for w in predictions:
    sentence=sentence+" "+w

print("the generated text is : ",sentence)

cmi(predictions[1:-1])

words, prob, perpl =perp(dataset,model,sentence)
print("perplexity is : ",perpl)
print("probability is : ",prob)



