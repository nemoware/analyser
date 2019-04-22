import gc
from abc import abstractmethod

from text_tools import *

import time


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

  ## ======== call TENSORFLOW -----==================
  sentences_emb, wrds = embedder.embedd_tokenized_text(_strings, lens)
  ## ================================================
  return sentences_emb, wrds, lens


class AbstractEmbedder:

  @abstractmethod
  def get_embedding_tensor(self, tokenized_sentences_list):
    pass

  @abstractmethod
  def embedd_tokenized_text(self, words, lens):
    pass

  def embedd_sentence(self, _str):
    words = tokenize_text(_str)
    return self.embedd_tokenized_text([words], [len(words)])

  def embedd_contextualized_patterns(self, patterns):
    tokenized_sentences_list = []
    regions = {}

    i = 0
    maxlen = 0
    lens = []
    for (ctx_prefix, pattern, ctx_postfix) in patterns:
      sentence = ' '.join((ctx_prefix, pattern, ctx_postfix))

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

    ## ======== call TENSORFLOW -----==================
    sentences_emb, wrds = self.embedd_tokenized_text(_strings, lens)
    ## ================================================

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

  def __init__(self, elmo, tf, layer_name, create_module_method):
    self.create_module_method = create_module_method
    self.elmo = elmo
    self.config = tf.ConfigProto()
    self.config.gpu_options.allow_growth = True


    self.layer_name = layer_name
    self.tf = tf

    self.session = tf.Session(config=self.config)

    self.sessionruns = 0

  def embedd_tokenized_text(self, words, lens):
    # with self.tf.Session(config=self.config) as sess:
    print(f'🐌 Embedding { np.nansum(lens) } words... it takes time (☕️?)..')

    embeddings = self.elmo(
      inputs={
        "tokens": words,
        "sequence_len": lens
      },
      signature="tokens",
      as_dict=True)[self.layer_name]

    self.session.run(self.tf.global_variables_initializer())
    out = self.session.run(embeddings)
    print(f'Embedding complete 🐌 ; the shape is { out.shape }')
    self.reset_maybe()

    return out, words


  def get_embedding_tensor(self, str, signature="default"):
    embedding_tensor = self.elmo(str, signature=signature, as_dict=True)[self.layer_name]

    # with self.tf.Session(config=self.config) as sess:
    self.session.run(self.tf.global_variables_initializer())
    embedding_ = self.session.run(embedding_tensor)
    embedding = np.array(embedding_)
    del(embedding_)
    self.reset_maybe()

    #       sess.close()

    return embedding

  def reset_maybe(self):
    self.sessionruns += 1

    if self.sessionruns > 14:
      self.reset()



  def reset(self):
    self.session.close()

    del self.elmo
    del self.session
    self.elmo = None
    self.session = None

    print(gc.collect())
    gc.enable()

    print('clean-up ------------- 🐌 -SLEEP: give it a time')
    time.sleep(10)

    self.elmo = self.create_module_method()
    self.session = self.tf.Session(config=self.config)
    # self.session.run(self.tf.global_variables_initializer())


    self.sessionruns = 0
    # self.session = self.tf.Session(config=self.config)
