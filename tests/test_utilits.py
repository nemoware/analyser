import numpy as np

from embedding_tools import AbstractEmbedder
from text_tools import Tokens


class FakeEmbedder(AbstractEmbedder):

  def __init__(self, default_point):
    self.default_point = default_point

  def embedd_tokens(self, tokens: Tokens) -> np.ndarray:
    return self.embedd_tokenized_text([tokens], [len(tokens)])[0]

  def embedd_strings(self, strings: Tokens) -> np.ndarray:

    ret = self.embedd_tokens(strings)
    return ret

  def embedd_tokenized_text(self, tokenized_sentences_list, lens):
    # def get_embedding_tensor(self, tokenized_sentences_list):
    tensor = []
    for sent in tokenized_sentences_list:
      sentense_emb = []
      for token in sent:
        token_emb = self.default_point
        sentense_emb.append(token_emb)
      tensor.append(sentense_emb)

    return np.array(tensor)
