from abc import abstractmethod

import tensorflow as tf
import tensorflow_hub as hub

from text_tools import *


def embedd_tokenized_sentences_list(embedder, tokenized_sentences_list):
  maxlen = 0
  lens = []

  for s in tokenized_sentences_list:
    lens.append(len(s))
    if len(s) > maxlen:
      maxlen = len(s)

  _strings = []
  for i in range(len(tokenized_sentences_list)):
    s = tokenized_sentences_list[i]
    s.extend([' '] * (maxlen - len(s)))
    _strings.append(s)

  _strings = np.array(_strings)

  # ======== call TENSORFLOW -----==================
  sentences_emb, wrds = embedder.embedd_tokenized_text(_strings, lens)
  # ================================================
  return sentences_emb, wrds, lens


class AbstractEmbedder:

  @abstractmethod
  def get_embedding_tensor(self, tokenized_sentences_list):
    pass

  @abstractmethod
  def embedd_tokenized_text(self, words: [Tokens], lens: List[int]) -> tuple:
    return None, None

  def embedd_sentence(self, _str):
    words = tokenize_text(_str)
    return self.embedd_tokenized_text([words], [len(words)])

  def embedd_contextualized_patterns(self, patterns):
    tokenized_sentences_list: [Tokens] = []
    regions = {}

    i = 0
    maxlen = 0
    lens = []
    for (ctx_prefix, pattern, ctx_postfix) in patterns:
      # sentence = ' '.join((ctx_prefix, pattern, ctx_postfix))

      prefix_tokens = tokenize_text(ctx_prefix)
      pattern_tokens = tokenize_text(pattern)
      suffix_tokens = tokenize_text(ctx_postfix)

      start = len(prefix_tokens)
      end = start + len(pattern_tokens)

      sentence_tokens = prefix_tokens + pattern_tokens + suffix_tokens

      # print('embedd_contextualized_patterns', (sentence, start, end))

      regions[i] = (start, end)
      tokenized_sentences_list.append(sentence_tokens)
      lens.append(len(sentence_tokens))
      if len(sentence_tokens) > maxlen:
        maxlen = len(sentence_tokens)

      i = i + 1

    # print('maxlen=', maxlen)
    _strings = []

    for s in tokenized_sentences_list:
      s.extend([' '] * (maxlen - len(s)))
      _strings.append(s)
      # print(s)
    _strings = np.array(_strings)

    # ======== call TENSORFLOW -----==================
    sentences_emb, wrds = self.embedd_tokenized_text(_strings, lens)
    # ================================================

    # print(sentences_emb.shape)
    #     assert len(sentence_tokens) == sentences_emb

    patterns_emb = []

    for i in regions:
      start, end = regions[i]

      sentence_emb = sentences_emb[i]
      pattern_emb = sentence_emb[start:end]

      patterns_emb.append(pattern_emb)

    return np.array(patterns_emb)


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
