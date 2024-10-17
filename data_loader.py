import json
import unicodedata as ud
import regex as re
import numpy as np
import torch
from torch.utils.data import Dataset

from environment import *


class Tokenizer():
  def __init__(self):
    try:
      with open(WORD2IND_PATH, 'r') as file:
        string = file.readline()
      self.word2ind = json.loads(string)
    except:
      self.word2ind = {}
    self.index = len(self.word2ind)
    
  def generate_dict(self):
    self.add_words([PAD_WORD])
    with open(TOKENIZER_DATA_PATH, 'r') as data_file:
      for line in data_file.readlines():
        words = line.split()
        self.add_words(words)
    self.save()
  
  def __len__(self):
    return len(self.word2ind)
  
  def add_words(self, words):
    for word in words:
      try:
        self.word2ind[word]
      except:
        self.word2ind[word] = self.index
        self.index += 1

  def __call__(self, word):
    try:
      index = self.word2ind[word]
    except:
      index = None
    return index
  
  def save(self):
    with open(WORD2IND_PATH, 'w') as file:
      string = json.dumps(self.word2ind)
      file.writelines(string)


class Data_generator(Dataset):
  def __init__(self):
    self.data = np.load(DATA_FILE_PATH)

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    return self.data[index]



def prepare_string(text):
    text = ud.normalize('NFC', text)
    text = text.lower()
    
    text = text.replace('ё', 'е')
    for letter in ['а', 'е', 'и', 'о', 'у', 'ы', 'э', 'я', 'ю']:
      text = text.replace(letter + '\u0301', letter)
      
    words = re.findall('[а-я\-?0-9]+', text)
    return words


def create_data():
  tokenizer = Tokenizer()
  tokenizer.generate_dict()
  tokenizer_length = len(tokenizer)
  pad_word_index = tokenizer(PAD_WORD)
  
  with open(SOURCE_DATA_FILE_PATH, 'r') as source_file:
    json_strings = source_file.readlines()
    data_count = len(json_strings) * (MAX_WINDOW_LEN-1)
    # add_length = 1 if data_count % BATCH_SIZE > 0 else 0
    data = np.zeros((data_count, MAX_WINDOW_LEN, tokenizer_length), dtype=np.float32)

    for sent_ind, json_string in enumerate(json_strings):
      json_data = json.loads(json_string)
      string = json_data['passage']
      
      words = prepare_string(string)
      words = words if len(words) <= MAX_WINDOW_LEN else words[:MAX_WINDOW_LEN]
      word_inds = [tokenizer(word) for word in words]
      
      for in_sent_ind in range(1, MAX_WINDOW_LEN):
        data_index = sent_ind*(MAX_WINDOW_LEN-1) + in_sent_ind -1
        # index = (data_index) // BATCH_SIZE
        # batch_index = (data_index) % BATCH_SIZE

        word_one_inds = word_inds[:in_sent_ind + 1]
        while (len(word_one_inds) < MAX_WINDOW_LEN): word_one_inds.insert(0, pad_word_index)
        for i in range(len(word_one_inds)):
          data[data_index][i][word_one_inds[i]] = 1.
  np.save(DATA_FILE_PATH, data)


if __name__ == '__main__':
  # create_data()
  gen = Data_generator()
  print(len(gen))
  test = gen.__getitem__(0)
  print(test.shape)
