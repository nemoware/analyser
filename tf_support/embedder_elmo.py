from typing import List

import tensorflow as tf
import tensorflow_hub as hub

from embedding_tools import AbstractEmbedder
from text_tools import Tokens
import numpy as np

class ElmoEmbedder(AbstractEmbedder):

  def __init__(self, layer_name="elmo",
               module_url: str = 'https://storage.googleapis.com/az-nlp/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz'):

    self.layer_name = layer_name
    self.module_url = module_url
    self.elmo = None
    self.text_input = None
    self.text_lengths = None
    self.embedded_out = None
    self.session = None

    self.build_graph()

  def build_graph(self):
    embedding_graph = tf.Graph()

    with embedding_graph.as_default():
      self.elmo = hub.Module(self.module_url, trainable=False)

      # inputs:--------------------------------------------------------------------
      self.text_input = tf.placeholder(dtype=tf.string, name="text_input")
      self.text_lengths = tf.placeholder(dtype=tf.int32, name='text_lengths')

      self.embedded_out = self.elmo(
        inputs={
          "tokens": self.text_input,
          "sequence_len": self.text_lengths
        },
        signature="tokens",
        as_dict=True)["elmo"]

      init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

      self.session = tf.Session(graph=embedding_graph)
      self.session.run(init_op)

    embedding_graph.finalize()
    return embedding_graph

  def embedd_tokenized_text(self, words: [Tokens], text_lens: List[int]) -> (np.ndarray, Tokens):
    feed_dict = {
      self.text_input: words,  # text_input
      self.text_lengths: text_lens,  # text_lengths
    }

    out = self.session.run(self.embedded_out, feed_dict=feed_dict)

    return out, words


if __name__ == '__main__':
  ee = ElmoEmbedder(layer_name='elmo')
  ee.embedd_tokenized_text([['просто', 'одно', 'предложение']], [3])
