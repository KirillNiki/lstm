import torch
from data_loader import Tokenizer, Data_generator
from torch.utils.data import DataLoader
import os

import time


num_cores = os.cpu_count()
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)


DATA_FILE_PATH = '/home/kirill/develop/python/diffrent_things/lstm/data/dataset.txt'
SOURCE_DATA_FILE_PATH = '/home/kirill/develop/python/diffrent_things/lstm/source_data/DaNetQA/train.jsonl'
PAD_WORD = '<pad>'

BATCH_SIZE = 16
MAX_WINDOW_LEN = 11
EMBEDDINGS_SIZE = 200
EPOCHS = 100
LR = 1e-3


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
  
  model = Model(units=[out_len, EMBEDDINGS_SIZE, 300, 300, out_len])
  optimizer = torch.optim.SGD(model.parameters(), lr=LR)
  return model, optimizer
    
    
def time_func(func, text, *args):
  t0 = time.time()
  out = func(*args)
  t1 = time.time()
  print(text, t1 - t0)
  return out

    
if __name__ == '__main__':
  model, optimizer = build_model()
  dataset = Data_generator(DATA_FILE_PATH, MAX_WINDOW_LEN, PAD_WORD)
  data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_cores)
  
  loss_fn = torch.nn.CrossEntropyLoss()
  for epoch in range(EPOCHS):
    losses = []
    
    for index, (context, target) in enumerate(data_loader):
      predicted = time_func(model, '>>>>0', context)
      loss = time_func(loss_fn, '>>>>1', predicted, target)
      time_func(losses.append, '>>>2', loss.item())
      time_func(optimizer.zero_grad, '>>>3')
      time_func(loss.backward, '>>>4')
      time_func(optimizer.step, '>>>5')
      # predicted = model(context)
  	  # loss = loss_fn(predicted, target)
  	  # losses.append(loss.item())

  	  # optimizer.zero_grad()
  	  # loss.backward()
  	  # optimizer.step()

  print(epoch, sum(losses) / len(losses))
	