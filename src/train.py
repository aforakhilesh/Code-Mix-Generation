import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset

if torch.cuda.is_available():
    print("SD")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataset, model, args):
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    try :
        for epoch in range(args.max_epochs):
            state_h, state_c = model.init_state(args.sequence_length)

            for batch, (x, y) in enumerate(dataloader):

                optimizer.zero_grad()

                y_pred, (state_h, state_c) = model(x, (state_h, state_c))
                loss = criterion(y_pred.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()
                optimizer.step()

                print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
    except:
        torch.save(model,"./test_15_256_epoch_model_gpu")
        

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=15)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()

dataset = Dataset(args)
model = Model(dataset).to(device)

train(dataset, model, args)

torch.save(model,"./model/test_10_512_epoch_model_gpu")

