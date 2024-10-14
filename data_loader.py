import json
import unicodedata as ud
import regex as re
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from environment import *


DATASET_FILE_PATH = 'data/data.csv'


class Tokenizer():
  def __init__(self, word2ind_path='/home/kirill/develop/python/diffrent_things/lstm/data/word2ind.json'):
    self.word2ind_path = word2ind_path
    try:
      with open(self.word2ind_path, 'r') as file:
        string = file.readline()
      self.word2ind = json.loads(string)
    except:
      self.word2ind = {}
      
    self.index = len(self.word2ind)
    
  def generate_dict(self, data_file_path, pad_word):
    self.add_words([pad_word])
    
    with open(data_file_path, 'r') as data_file:
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
    with open(self.word2ind_path, 'w') as file:
      string = json.dumps(self.word2ind)
      file.writelines(string)


class Data_generator(Dataset):
  def __init__(self, data_file_path, max_sentence_len, pad_word):
    self.tokenizer = Tokenizer()
    self.data_file_path = data_file_path
    self.max_sentence_len = max_sentence_len
    self.pad_word = pad_word
    
    self.tokenizer_length = len(self.tokenizer)
    self.pad_word_index = self.tokenizer(self.pad_word)
    
    with open(self.data_file_path, 'r') as data_file:
      lines = data_file.readlines()
    self.ind_lines = []
      
    for line in lines:
      ind_line = []
      for word in line.split():
        ind_line.append(self.tokenizer(word))
      self.ind_lines.append(ind_line)
      
  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, index):
    sentence_ind = index // (self.max_sentence_len -1)
    word_in_sent_ind = index % (self.max_sentence_len - 1) + 1
    context = torch.zeros((1, self.max_sentence_len-1, self.tokenizer_length))
    target = torch.zeros((1, self.tokenizer_length))
    context_ind = self.max_sentence_len - 2
    
    for i in range(word_in_sent_ind-1, -1, -1):
      context_word_ind = self.ind_lines[sentence_ind][i]
      context[0][context_ind][context_word_ind] = 1.
      context_ind -= 1
    while (context_ind > -1):
      context[0][context_ind][self.pad_word_index] = 1.
      context_ind -= 1
    target[0][self.ind_lines[sentence_ind][word_in_sent_ind]] = 1.
    
    return context, target


def prepare_string(text):
    text = ud.normalize('NFC', text)
    text = text.lower()
    
    text = text.replace('ё', 'е')
    for letter in ['а', 'е', 'и', 'о', 'у', 'ы', 'э', 'я', 'ю']:
      text = text.replace(letter + '\u0301', letter)
      
    words = re.findall('[а-я\-?0-9]+', text)
    return words


def create_datafile(target_file_path, source_file_path, max_sentence_len):
  with open(target_file_path, 'w') as target_file:
    target_file.write('')
  
  with open(target_file_path, 'a') as target_file:
    with open(source_file_path, 'r') as source_file:
      json_strings = source_file.readlines()
  
      for json_string in json_strings:
        json_data = json.loads(json_string)
        string = json_data['passage']

        words = prepare_string(string)
        words = words if len(words) <= max_sentence_len else words[:max_sentence_len]
        string = ' '.join(words)
        target_file.write(f'{string}\n')

if __name__ == '__main__':
  # create_datafile(DATA_FILE_PATH, SOURCE_DATA_FILE_PATH, MAX_WINDOW_LEN)
  test = Data_generator(DATA_FILE_PATH, MAX_WINDOW_LEN, PAD_WORD)
  test.__getitem__(0)
  print()
