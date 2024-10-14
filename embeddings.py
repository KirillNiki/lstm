from navec import Navec
from environment import EMBEDDINGS_SIZE


class Embeddings_loader():
  def __init__(self, embiddings_path='navec_hudlit_v1_12B_500K_300d_100q.tar'):
    self.navec_embeddings = Navec.load(embiddings_path)

  def __len__(self):
    return len(self.navec_embeddings.vocab.words)
    
  def __call__(self, words):
    embeddings = []
    for word in words:
      try:
        embedding = self.navec_embeddings[word]
      except:
        embedding = [0 for i in range(EMBEDDINGS_SIZE)]
      
      embeddings.append(embedding)
    return embeddings
