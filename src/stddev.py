import statistics
import json

with open("perplexity_train_15_512_6gram.json") as f:
    data = json.load(f)

perp=[]


for sent in data.keys():
    perp.append(data[sent]['perplexity'])


# print(perp)

print(statistics.stdev(perp))