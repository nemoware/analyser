from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from analyser.embedding_tools import AbstractEmbedder
from analyser.text_tools import Tokens

_e_instance: AbstractEmbedder = None


class ElmoEmbedderWrapper(AbstractEmbedder):
  def __init__(self, instance: AbstractEmbedder, layer: str):
    self.instance: AbstractEmbedder = instance
    self.layer_name = layer

  def embedd_tokens(self, tokens: Tokens) -> np.ndarray:
    if self.layer_name == 'elmo':
      return self.instance.embedd_tokenized_text([tokens], [len(tokens)])[0]
    else:
      return self.instance.embedd_strings(tokens)

  def embedd_tokenized_text(self, words: [Tokens], text_lens: List[int]) -> np.ndarray:
    return self.instance.embedd_tokenized_text(words, text_lens)

  def embedd_strings(self, strings: Tokens) -> np.ndarray:
    return self.instance.embedd_strings(strings)


class ElmoEmbedderImpl(AbstractEmbedder):

  def __init__(self, module_url: str = 'https://storage.googleapis.com/az-nlp/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz'):
    self.module_url = module_url
    self.elmo = None
    self.session = None

  def _build_session_and_graph(self):

    embedding_graph = tf.compat.v1.Graph()

    with embedding_graph.as_default():
      print(f'< loading ELMO module {self.module_url}')
      self.elmo = hub.Module(self.module_url, trainable=False)
      print(f'ELMO module loaded >')

      self.text_input = tf.compat.v1.placeholder(dtype='string', name="text_input")
      self.text_lengths = tf.compat.v1.placeholder(dtype='int32', name='text_lengths')

    inputs_elmo = {
      "tokens": self.text_input,
      "sequence_len": self.text_lengths
    }

    inputs_default = {
      "strings": self.text_input
    }

    with embedding_graph.as_default():
      print(f'ELMO: creating embedded_out_elmo')
      self.embedded_out_elmo = self.elmo(
        inputs=inputs_elmo,
        signature="tokens",
        as_dict=True)['elmo']

      print(f'ELMO: embedded_out_defaut embedded_out_elmo')
      self.embedded_out_defaut = self.elmo(
        inputs=inputs_default,
        signature="default",
        as_dict=True)['default']

    with embedding_graph.as_default():
      self.session = tf.compat.v1.Session(graph=embedding_graph)
      init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
      self.session.run(init_op)

    embedding_graph.finalize()

  def embedd_tokens(self, tokens: Tokens) -> np.ndarray:
    if self.session is None:
      self._build_session_and_graph()  # lazy init

    return self.embedd_tokenized_text([tokens], [len(tokens)])[0]

  def embedd_tokenized_text(self, words: [Tokens], text_lens: List[int]) -> np.ndarray:
    if self.session is None:
      self._build_session_and_graph()  # lazy init

    feed_dict = {
      self.text_input: words,  # text_input
      self.text_lengths: text_lens,  # text_lengths
    }

    out = self.session.run(self.embedded_out_elmo, feed_dict=feed_dict)
    return out

  def embedd_strings(self, strings: Tokens) -> np.ndarray:
    if self.session is None:
      self._build_session_and_graph()  # lazy init

    _strings = []
    for s in strings:
      if s == '':
        _strings.append(' ')
      else:
        _strings.append(s)

    feed_dict = {
      self.text_input: _strings,  # text_input
    }

    out = self.session.run(self.embedded_out_defaut, feed_dict=feed_dict)

    return out


class ElmoEmbedder:

  @staticmethod
  def get_instance(layer) -> AbstractEmbedder:
    global _e_instance
    if _e_instance is None:
      print('creating ElmoEmbedderImpl instance')
      _e_instance = ElmoEmbedderImpl()

    wrapper = ElmoEmbedderWrapper(_e_instance, layer)
    return wrapper


if __name__ == '__main__':
  ee = ElmoEmbedder.get_instance('default')
  x = ee.embedd_tokenized_text([['просто', 'одно', 'предложение']], [3])
  print(x)
  ee2 = ElmoEmbedder.get_instance('elmo')
  x = ee2.embedd_strings(['просто одно предложение'])
  print(x)
