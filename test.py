import torch
from data_loader import Tokenizer, Data_generator, create_data
from torch.utils.data import DataLoader, Dataset
import os
import time

from environment import *


num_cores = os.cpu_count()
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)


class Model(torch.nn.Module):
  def __init__(self, units=[1000, 300, 100, 100, 1000]):
    super().__init__()
    self.embedding = torch.nn.Linear(in_features=units[0], out_features=units[1], bias=False)
    self.hidden_size = units[2]
    self.rnn = torch.nn.RNN(input_size=units[1], hidden_size=self.hidden_size, batch_first=True)
    self.dence1 = torch.nn.Linear(in_features=units[2], out_features=units[3])
    self.dence2 = torch.nn.Linear(in_features=units[3], out_features=units[4])
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    x = self.embedding(x)
    h0 = torch.autograd.Variable(torch.zeros((1, x.shape[0], self.hidden_size)))
    rnn_out, x = self.rnn(x, h0)
    x = torch.reshape(x, (x.shape[1], x.shape[2]))
    x = self.sigmoid(self.dence1(x))
    x = self.dence2(x)
    return x


def build_model():
  tokenizer = Tokenizer()
  out_len = len(tokenizer)
  
  model = Model(units=[out_len, EMBEDDINGS_SIZE, 300, 500, out_len])
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)
  return model, optimizer


def time_func(func, text, *args):
  t0 = time.time()
  out = func(*args)
  t1 = time.time()
  print(text, t1 - t0)
  return out


if __name__ == '__main__':
  model, optimizer = build_model()
  dataset = Data_generator()  
  data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_cores)
  
  # test = next(iter(data_loader))
  loss_fn = torch.nn.CrossEntropyLoss()
  for epoch in range(EPOCHS):
    losses = []
    
    for data in iter(data_loader):
      context, target = data[:,:-1,:], data[:,-1,:]

      predicted = model(context)
      loss = loss_fn(predicted, target)
      losses.append(loss.item())
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    print(epoch, sum(losses) / len(losses))
torch.save(model, MODEL_PATH)
