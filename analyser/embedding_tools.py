import warnings
from abc import abstractmethod

import numpy as np

from analyser.documents import TextMap
from analyser.hyperparams import work_dir
from analyser.ml_tools import Embeddings
from analyser.text_tools import Tokens


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
  sentences_emb = embedder.embedd_tokenized_text(_strings, lens)
  # ================================================
  return sentences_emb, None, lens


import os


class AbstractEmbedder:

  def __cache_fn(self, checksum):
    return os.path.join(work_dir, f'cache-{checksum}-embeddings-{type(self).__name__}.npy')

  def get_cached_embedding(self, checksum) -> Embeddings or None:
    fn = self.__cache_fn(checksum)
    if os.path.isfile(fn):
      print(f'skipping embedding doc {checksum} ...., {fn} exits, loading')
      e = np.load(fn)
      print('loaded embedding shape is:', e.shape)
      return e

    return None

  def cache_embedding(self, checksum, embeddings):
    fn = self.__cache_fn(checksum)
    np.save(fn, embeddings)

  @abstractmethod
  def embedd_tokens(self, tokens: Tokens) -> Embeddings:
    raise NotImplementedError()

  @abstractmethod
  def embedd_tokenized_text(self, words: [Tokens], lens: [int]) -> Embeddings:
    raise NotImplementedError()

  @abstractmethod
  def embedd_strings(self, strings: Tokens) -> Embeddings:
    raise NotImplementedError()

  def embedd_contextualized_patterns(self, patterns, trim_padding=True):

    tokenized_sentences_list: [Tokens] = []
    regions = []

    i = 0
    maxlen = 0
    lens = []
    for (ctx_prefix, pattern, ctx_postfix) in patterns:
      # sentence = ' '.join((ctx_prefix, pattern, ctx_postfix))

      prefix_tokens = TextMap(ctx_prefix).tokens  # tokenize_text(ctx_prefix)
      pattern_tokens = TextMap(pattern).tokens
      suffix_tokens = TextMap(ctx_postfix).tokens

      start = len(prefix_tokens)
      end = start + len(pattern_tokens)

      sentence_tokens = prefix_tokens + pattern_tokens + suffix_tokens

      # print('embedd_contextualized_patterns', (sentence, start, end))

      regions.append((start, end))
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
    sentences_emb = self.embedd_tokenized_text(_strings, lens)
    # ================================================

    # print(sentences_emb.shape)
    #     assert len(sentence_tokens) == sentences_emb

    patterns_emb = []

    if trim_padding:
      for i, (start, end) in enumerate(regions):
        sentence_emb = sentences_emb[i]
        pattern_emb = sentence_emb[start:end]

        patterns_emb.append(pattern_emb)
      patterns_emb = np.array(patterns_emb)
    else:
      patterns_emb = sentences_emb

    return patterns_emb, regions

  def embedd_large(self, text_map, max_tokens=8000, verbosity=2):
    overlap = max_tokens // 20

    number_of_windows = 1 + len(text_map) // max_tokens
    window = max_tokens

    if verbosity > 1:
      msg = f"WARNING: Document is too large for embedding: {len(text_map)} tokens. Splitting into {number_of_windows} windows overlapping with {overlap} tokens "
      warnings.warn(msg)

    start = 0
    embeddings = None
    # tokens = []
    while start < len(text_map):

      subtokens: Tokens = text_map[start:start + window + overlap]
      if verbosity > 2:
        print("Embedding region:", start, len(subtokens))

      sub_embeddings = self.embedd_tokens(subtokens)[0:window]

      if embeddings is None:
        embeddings = sub_embeddings
      else:
        embeddings = np.concatenate([embeddings, sub_embeddings])

      start += window

    return embeddings
    # self.tokens = tokens
