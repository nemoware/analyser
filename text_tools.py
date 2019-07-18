#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import os
import pickle
from typing import List

import nltk, warnings
import numpy as np
import scipy.spatial.distance as distance

nltk.download('punkt')

from nltk.tokenize import _treebank_word_tokenizer

nltk_treebank_word_tokenizer = _treebank_word_tokenizer

Tokens = List[str]


def find_ner_end(tokens, start, max_len=20):
  for i in range(start, len(tokens)):
    if tokens[i] == '"':
      return i

    if tokens[i] == '»':
      return i

    if tokens[i] == '\n':
      return i

    if tokens[i] == '.':
      return i

    if tokens[i] == ';':
      return i

  return min(len(tokens), start + max_len)


def to_float(string):
  try:
    return float(string.replace(" ", "").replace(",", "."))
  except:
    return np.nan


def replace_with_map(txt, replacements):
  a = txt
  for (src, target) in replacements:
    a = a.replace(src, target)

  return a


def remove_empty_lines(original_text):
  a = "\n".join([ll.strip() for ll in original_text.splitlines() if ll.strip()])
  return a.replace('\t', ' ')


# ----------------------------------------------------------------
# DISTANCES
# ----------------------------------------------------------------

def dist_hausdorff(u, v):
  return max(distance.directed_hausdorff(v, u, 4)[0], distance.directed_hausdorff(u, v, 4)[0]) / 40


def dist_mean_cosine(u, v):
  return distance.cosine(u.mean(0), v.mean(0))


def dist_mean_eucl(u, v):
  return distance.euclidean(u.mean(0), v.mean(0))


def dist_sum_cosine(u, v):
  return distance.cosine(u.sum(0), v.sum(0))


# not in use
def dist_correlation_min_mean(u, v):
  if u.shape[0] > v.shape[0]:
    return distance.cdist(u, v, 'correlation').min(0).mean()
  else:
    return distance.cdist(v, u, 'correlation').min(0).mean()


def dist_cosine_min_mean(u, v):
  if u.shape[0] > v.shape[0]:
    return distance.cdist(u, v, 'cosine').min(0).mean()
  else:
    return distance.cdist(v, u, 'cosine').min(0).mean()


# not in use
def dist_euclidean_min_mean(u, v):
  if u.shape[0] > v.shape[0]:
    return distance.cdist(u, v, 'euclidean').min(0).mean()
  else:
    return distance.cdist(v, u, 'euclidean').min(0).mean()


"""

Kind of Moving Earth (or Fréchet distance)
ACHTUNG! This is not WMD

inspired by https://en.wikipedia.org/wiki/Earth_mover%27s_distance https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html

Compute matrix of pair-wize distances between words of 2 sentences (each 2 each)
For each word in sentence U, find the Distance to semantically nearest one in the other sentence V
The Sum of these minimal distances (or mean) is sort of effort required to transform U sentence to another, V sentence.
For balance (symmetry) swap U & V and find the effort required to strech V sentence to U.


"""


def dist_frechet_cosine_directed(u, v):
  d_ = distance.cdist(u, v, 'cosine')
  return d_.min(0).sum()


def dist_frechet_cosine_undirected(u, v):
  d1 = dist_frechet_cosine_directed(u, v)
  d2 = dist_frechet_cosine_directed(v, u)
  return round((d1 + d2) / 2, 2)


def dist_frechet_eucl_directed(u, v):
  d_ = distance.cdist(u, v, 'euclidean')
  return d_.min(0).sum()


def dist_frechet_eucl_undirected(u, v):
  d1 = dist_frechet_eucl_directed(u, v)
  d2 = dist_frechet_eucl_directed(v, u)
  return round((d1 + d2) / 2, 2)


def dist_mean_cosine_frechet(u, v):
  return dist_frechet_cosine_undirected(u, v) + dist_mean_cosine(u, v)


def dist_cosine_housedorff_directed(u, v):
  d_ = distance.cdist(u, v, 'cosine')
  return d_.min(0).max()


def dist_cosine_housedorff_undirected(u, v):
  d1 = dist_cosine_housedorff_directed(u, v)
  d2 = dist_cosine_housedorff_directed(v, u)
  return round((d1 + d2) / 2, 2)


# ----------------------------------------------------------------
# MISC
# ----------------------------------------------------------------


def norm_matrix(mtx):
  mtx = mtx - mtx.min()
  mtx = mtx / np.abs(mtx).max()
  return mtx


def min_index(sums):
  min_i = 0
  min_d = float('Infinity')

  for d in range(0, len(sums)):
    if sums[d] < min_d:
      min_d = sums[d]
      min_i = d

  return min_i


# ----------------------------------------------------------------
# TOKENS
# ----------------------------------------------------------------

def sentence_similarity_matrix(emb, distance_function):
  mtx = np.zeros(shape=(emb.shape[0], emb.shape[0]))

  # TODO: write it Pythonish!!
  for u in range(emb.shape[0]):
    for v in range(emb.shape[0]):
      mtx[u, v] = distance_function(emb[u], emb[v])

  # TODO: no norm here
  return mtx  # norm_matrix(mtx)


my_punctuation = r"""!"#$%&'*+,-./:;<=>?@[\]^_`{|}~"""


def untokenize(tokens: Tokens) -> str:
  warnings.warn("deprecated", DeprecationWarning)
  return "".join([" " + i if not i.startswith("'") and i not in my_punctuation else i for i in tokens]).strip()


def tokenize_text(text):
  warnings.warn("deprecated", DeprecationWarning)
  sentences = text.split('\n')
  result = []
  for i in range(len(sentences)):
    sentence = sentences[i]
    result += nltk.word_tokenize(sentence)
    if i < len(sentences) - 1:
      result += ['\n']

  return result


def find_token_before_index(tokens: Tokens, index, token, default_ret=-1):
  for i in reversed(range(index)):
    if tokens[i] == token:
      return i
  return default_ret


def find_token_after_index(tokens: Tokens, index, token, default_ret=-1):
  for i in range(index, len(tokens)):
    if tokens[i] == token:
      return i
  return default_ret


def get_sentence_bounds_at_index(index, tokens):
  start = find_token_before_index(tokens, index, '\n', 0)
  end = find_token_after_index(tokens, index, '\n', len(tokens) - 1)
  return start + 1, end


def get_sentence_slices_at_index(index, tokens) -> slice:
  start = find_token_before_index(tokens, index, '\n')
  end = find_token_after_index(tokens, index, '\n')
  if start < 0:
    start = 0
  if end < 0:
    end = len(tokens)
  return slice(start + 1, end)


def min_index_per_row(rows):
  indexes = []
  for row in rows:
    indexes.append(np.argmin(row))

  return indexes


def hot_quotes(tokens: Tokens) -> (np.ndarray, np.ndarray):
  q_re_open = '\'\"«<{['
  q_re_close = '\'\"»>]'
  _quotes_open = np.zeros(len(tokens))
  _quotes_closing = np.zeros(len(tokens))

  quotes_attention = 1
  for i in range(len(tokens)):
    if tokens[i][0] in q_re_open:
      _quotes_open[i] = quotes_attention
    if tokens[i][0] in q_re_close:
      _quotes_closing[i] = quotes_attention

  return _quotes_open, _quotes_closing


def hot_punkt(tokens: Tokens) -> np.ndarray:
  _hot_punkt = np.zeros(len(tokens))

  for i in range(len(tokens)):
    if tokens[i][0] in my_punctuation + '–«»()[] ':
      _hot_punkt[i] = 1

  return _hot_punkt


def acronym(n):
  return ''.join([x[0] for x in n.split(' ') if len(x) > 1]).upper()


def token_at_index(index: int, txt: str) -> int:
  warnings.warn("deprecated", DeprecationWarning)
  _txt = txt[0:index]
  head_tokens = tokenize_text(_txt)
  return len(head_tokens)


def tokens_in_range(span: List[int], tokens: Tokens, txt: str = None) -> slice:
  warnings.warn("deprecated", DeprecationWarning)
  if txt is None:
    txt = untokenize(tokens)
  a = token_at_index(span[0], txt)
  b = token_at_index(span[1], txt)
  return slice(a, b)


def replace_tokens(tokens: Tokens, replacements_map):
  result = []
  for t in tokens:
    key = t.lower()
    if key in replacements_map:
      result.append(replacements_map[key])
    else:
      result.append(key)
  return result


class CaseNormalizer:
  __shared_state = {}  ## see http://code.activestate.com/recipes/66531/

  def __init__(self):
    self.__dict__ = self.__shared_state
    if 'replacements_map' not in self.__dict__:
      __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
      p = os.path.join(__location__, 'vocab', 'word_cases_stats.pickle')
      print('loading word cases stats model', p)

      with open(p, 'rb') as handle:
        self.replacements_map = pickle.load(handle)

  def normalize_tokens(self, tokens: Tokens) -> Tokens:
    return replace_tokens(tokens, self.replacements_map)

  def normalize_text(self, text: str) -> str:
    tokens = tokenize_text(text)
    tokens = self.normalize_tokens(tokens)
    return untokenize(tokens)

  def normalize_word(self, token: str) -> str:
    if token.lower() in self.replacements_map:
      return self.replacements_map[token.lower()]
    else:
      return token


if __name__ == '__main__':
  # warnings.warn('стороны Заключили ')

  cn = CaseNormalizer()
  print(cn.normalize_tokens(['стороны', 'Заключили', 'договор', 'уррраа!! ']))
  print(cn.normalize_text('стороны Заключили (ХОРОШИЙ)договор, (уррраа!!) ПРЕДМЕТ ДОГОВОРА'))
  print(cn.normalize_word('ДОГОВОР'))
