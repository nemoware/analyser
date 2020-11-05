import json
import os
import pickle

import numpy as np
from bson import json_util

from analyser.contract_parser import ContractDocument
from analyser.embedding_tools import AbstractEmbedder, Embeddings
from analyser.text_tools import Tokens


def load_json_sample(fn: str) -> dict:
  pth = os.path.dirname(__file__)
  with open(os.path.join(pth, fn), 'rb') as handle:
    # jsondata = json.loads(json_string, object_hook=json_util.object_hook)
    data = json.load(handle, object_hook=json_util.object_hook)

  return data


def get_a_contract() -> ContractDocument:
  pth = os.path.dirname(__file__)
  with open(pth + '/2. Договор по благ-ти Радуга.docx.pickle', 'rb') as handle:
    doc = pickle.load(handle)

  return doc


class FakeEmbedder(AbstractEmbedder):

  def __init__(self, default_point):
    self.default_point = default_point

  def embedd_tokens(self, tokens: Tokens) -> Embeddings:
    return self.embedd_tokenized_text([tokens], [len(tokens)])[0]

  def embedd_strings(self, strings: Tokens) -> Embeddings:
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
