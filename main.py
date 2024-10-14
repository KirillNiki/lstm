import tensorflow as tf
import keras
import numpy as np

from data_loader import create_datafile, Data_generator,  Tokenizer
from environment import *


class Model(keras.Model):
  def __init__(self, rate=0.1, units=[1000, 300, 400, 400, 400, 1000]):
    super().__init__()

    self.embedding = keras.layers.Embedding(units[0], units[1], input_length=MAX_WINDOW_LEN-1)
    self.rnn = keras.layers.LSTM(units=units[2])
    # self.dense1 = keras.layers.Dense(units=units[3], activation='sigmoid')
    # keras.layers.Dropout(rate=rate)
    # self.dense2 = keras.layers.Dense(units=units[4], activation='sigmoid')
    # keras.layers.Dropout(rate=rate)
    self.dense3 = keras.layers.Dense(units=units[5], activation='softmax')

  def call(self, x):
    x = self.embedding(x)
    x = tf.reduce_sum(x, 2)
    x = self.rnn(x)
    # x = self.dense1(x)
    # x = self.dense2(x)
    x = self.dense3(x)
    return x


def create_model(input, out_len):
  model = Model(units=[out_len, EMBEDDINGS_SIZE, 150, 1000, 1000, out_len])
  model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
  )

  test_in = tf.convert_to_tensor([input])
  test_out = model(test_in)
  return model


if __name__ == "__main__":
  create_datafile(DATA_FILE_PATH, SOURCE_DATA_FILE_PATH, MAX_WINDOW_LEN)
  tokenizer = Tokenizer()
  tokenizer.generate_dict(DATA_FILE_PATH, PAD_WORD)
  out_len = len(tokenizer)
  
  generator = Data_generator(DATA_FILE_PATH, MAX_WINDOW_LEN, PAD_WORD)
  x, y = generator.get_data()
  X = np.array(x, dtype=float)
  Y = np.array(y, dtype=float)
  
  model = create_model(X[0], out_len)
  model.summary()
  
  
  # optimizer = keras.optimizers.Adam(learning_rate=0.01)
  # for epoch in range(EPOCHS):
  #   losses = []
    
  #   for index in range(64):
  #     context = tf.constant([x[index]], dtype=tf.float32)
  #     target = tf.constant([y[index]])
      
  #     with tf.GradientTape() as tape:
  #       tape.watch(context)
  #       output = model(context)
  #       loss = tf.keras.losses.categorical_crossentropy(target, output)
  #       losses.append(loss.numpy())
      
  #     grads = tape.gradient(loss, model.trainable_variables)
  #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
      
  #   print(epoch, np.mean(losses))

  
  inv_map = {v: k for k, v in tokenizer.word2ind.items()}
  text1 = 'темные аллеи сборник'
  for _ in range(5):
    
    context = [[tokenizer(word)] for word in text1.split()]
    context = np.concatenate(([[0]] * (MAX_WINDOW_LEN - len(context)), context))
    context = np.array([context])
    
    output = model(context).numpy()[0]
    index = np.where(output == max(output))[0][0]
    word = inv_map[index]
    text1 += f' {word}'

  print(text1)
  
  
  model.fit(X, Y, epochs=20)
  
  
  inv_map = {v: k for k, v in tokenizer.word2ind.items()}
  text2 = 'темные аллеи сборник'
  for _ in range(5):
    
    context = [[tokenizer(word)] for word in text2.split()]
    context = np.concatenate(([[0]] * (MAX_WINDOW_LEN - len(context)), context))
    context = np.array([context])
    
    output = model(context).numpy()[0]
    index = np.where(output == max(output))[0][0]
    word = inv_map[index]
    text2 += f' {word}'

  print(text1)
  print(text2)