from typing import Dict

from ml_tools import *
# ==============================================================================
from patterns import FuzzyPattern
from text_tools import tokenize_text


def select_best_attention_and_relu_it(original, improved, relu_th=0.8):
  if improved is None:
    return original

  i = np.argmax(original)
  if original[i] > relu_th:
    selection = improved
  else:
    selection = original

  return selection


class AttentionVectors:
  def __init__(self):
    self.vectors: Dict[str, FixedVector] = {}
    self.improved: Dict[str, FixedVector] = {}
    self.size = None

  def add(self, name: str, x: FixedVector, x_improved: FixedVector = None):
    if self.size is None:
      self.size = len(x)

    assert len(x) == self.size

    self.vectors[name] = np.array(x)
    self.vectors[name].flags.writeable = False

    if x_improved is not None:
      assert len(x_improved) == self.size
      self.improved[name] = np.array(x_improved)
      self.improved[name].flags.writeable = False
    else:
      self.improved[name] = None

  def has(self, name):
    if name in self.vectors:
      return True
    elif name[-1] == '*':
      names = [x for x in self.get_names_by_pattern(name[:-1])]
      # TODO: improve this
      return len(names) > 0
    #       return len(self.get_names_by_pattern(name[:-1])) > 0

    return False

  def get_names_by_pattern(self, prefix) -> Iterable[str]:
    for nm in self.vectors.keys():
      if nm[:len(prefix)] == prefix:
        yield nm

  def get_by_name(self, name) -> FixedVector:
    return self.vectors[name]

  def get_by_prefix(self, prefix) -> FixedVectors:
    for name in self.get_names_by_pattern(prefix):
      yield self.get_by_name(name)

  def get_improved_by_name(self, name) -> FixedVector:
    return self.improved[name]

  def get_best(self, name, relu_th=0.7) -> FixedVector:
    return select_best_attention_and_relu_it(self.get_by_name(name), self.get_improved_by_name(name), relu_th=relu_th)

  def get_bests(self, prefix, relu_th=0.7) -> FixedVectors:
    for name in self.get_names_by_pattern(prefix):
      yield self.get_best(name, relu_th=relu_th)


from fuzzy_matcher import AttentionVectors
from ml_tools import sum_probabilities, subtract_probability, relu, momentum_p
import numpy as np
from ml_tools import FixedVector, FixedVectors, Tokens


def group_indices(_indices, cut_threshold=2):
  indices = sorted(_indices)

  _start = indices[0]
  _prev = indices[0]
  slices = []

  _n = len(indices)

  for i_i in range(_n):
    i = indices[i_i]

    if i - _prev > cut_threshold:
      slices.append(slice(_start, _prev + 1))
      _start = i

    if i_i == _n - 1:
      slices.append(slice(_start, i + 1))

    _prev = i

  return slices


# group_indices([2,8])


class FuzzyMatcher:
  def __init__(self, av: AttentionVectors):
    self.av = av
    self.constraints = []
    self.incusions = []

  def _add(self, name, max_tokens, multiplyer, to_left):
    assert self.av.has(name), f'there is no vector named {name}'
    self.constraints.append((name, max_tokens, multiplyer, to_left))

  def _include(self, name, include):
    assert self.av.has(name), f'there is no vector named {name}'
    self.incusions.append((name, include))

  def after(self, name, max_tokens: int, multiplyer=1) -> 'FuzzyMatcher':
    self._add(name, max_tokens, multiplyer, False)
    return self

  def before(self, name, max_tokens: int, multiplyer=1) -> 'FuzzyMatcher':
    self._add(name, max_tokens, multiplyer, True)
    return self

  def excluding(self, name) -> 'FuzzyMatcher':
    self._include(name, False)
    return self

  def including(self, name) -> 'FuzzyMatcher':
    self._include(name, True)
    return self

  def _get_by_prefix(self, prefix) -> FixedVectors:
    for name in self.av.get_names_by_pattern(prefix):
      yield self.av.get_by_name(name)

  def _get(self, name, p_threshold=0.5):
    _relu = 0.8
    if name[-1] == '*':
      vectors = self._get_by_prefix(name[:-1])
      v = sum_probabilities([relu(v, p_threshold) for v in vectors])
    else:
      v = self.av.get_by_name(name)
      v = relu(v, p_threshold)

    return v

  def extract_name(self, attention: FixedVector, tokens: Tokens, cut_threshold = 2) -> [slice]:
    best_indices = []
    #     best_indices = sorted(np.argsort(attention)[::-1][:20])

    for i in np.argsort(attention)[::-1][:20]:
      if attention[i] > 0.001:
        best_indices.append(i)
    best_indices = sorted(best_indices)

    if len(best_indices) == 0:
      return []


    slices = group_indices(best_indices, cut_threshold)

    _dict = {}
    for s in slices:
      key = '__'.join(tokens[s])
      _sum = np.max(attention[s])
      _dict[key] = (_sum, s)

    _arr = [x[1] for x in sorted(_dict.values(), key=lambda item: -item[0])]

    return _arr

  def compile(self, p_threshold=0.33, bias=-0.33) -> np.ndarray:

    # regions
    _momented = []
    _momented.append(np.zeros(self.av.size))
    for name, max_tokens, multiplyer, to_left in self.constraints:
      v = self._get(name, p_threshold) + bias

      v = relu(v, p_threshold)  # TODO: remove? do not ReLu here?

      momented = momentum_p(v, half_decay=max_tokens, left=to_left)
      momented -= v
      momented *= multiplyer
      _momented.append(momented)

    m_result = sum_probabilities(_momented)

    # incusions
    _included = []
    _included.append(np.zeros(self.av.size))
    for name, include in self.incusions:
      v = self._get(name, p_threshold) + bias
      if include:
        _included.append(relu(v, p_threshold))
    i_result = sum_probabilities(_included)

    # exclusions
    _excluded = []
    _excluded.append(np.zeros(self.av.size))
    for name, include in self.incusions:
      v = self._get(name, p_threshold) + bias
      if not include:
        _excluded.append(relu(v, p_threshold))
    e_result = sum_probabilities(_excluded)

    m_result = sum_probabilities([m_result, i_result])

    m_result = subtract_probability(m_result, e_result)

    #     result = relu(result, 0.001)
    return m_result


def prepare_patters_for_embedding(patterns):
  tokenized_sentences_list = []
  regions = []

  maxlen = 0
  lens = []
  for p in patterns:
    ctx_prefix, pattern, ctx_postfix = p.prefix_pattern_suffix_tuple

    sentence = ' '.join((ctx_prefix, pattern, ctx_postfix))

    prefix_tokens = tokenize_text(ctx_prefix)
    pattern_tokens = tokenize_text(pattern)
    suffix_tokens = tokenize_text(ctx_postfix)

    start = len(prefix_tokens)
    end = start + len(pattern_tokens)

    sentence_tokens = prefix_tokens + pattern_tokens + suffix_tokens

    # print('embedd_contextualized_patterns', (sentence, start, end))

    regions.append([start, end])
    tokenized_sentences_list.append(sentence_tokens)
    lens.append(len(sentence_tokens))
    if len(pattern_tokens) > maxlen:
      maxlen = len(sentence_tokens)

  _strings = []

  for s in tokenized_sentences_list:
    s.extend(['\n'] * (maxlen - len(s)))
    _strings.append(s)
    # print(s)
  _strings = np.array(_strings)

  return _strings, lens, regions, maxlen


def find_patterns(embedding_session, in_out, text_tokens: Tokens, patterns: List[FuzzyPattern]) -> AttentionVectors:
  patterns_tokens, patterns_lengths, pattern_slices, patterns_max_len = prepare_patters_for_embedding(patterns)

  runs = in_out[0]
  feeds_t = in_out[1]
  feeds_p = in_out[2]
  attentions, improved_attentions = embedding_session.run(in_out[0], feed_dict={
    feeds_t[0]: [text_tokens],  # text_input
    feeds_t[1]: [len(text_tokens)],  # text_lengths

    feeds_p[0]: len(patterns),
    feeds_p[1]: patterns_tokens,
    feeds_p[2]: patterns_lengths,
    feeds_p[3]: pattern_slices,
    feeds_p[4]: patterns_max_len

  })

  av = AttentionVectors()

  for i in range(len(patterns)):
    pattern = patterns[i]
    av.add(pattern.name, attentions[i], improved_attentions[i])

  return av
