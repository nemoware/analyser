import warnings
from typing import List, TypeVar, Iterable

import numpy as np

from text_tools import Tokens

FixedVector = TypeVar('FixedVector', List[float], np.ndarray)
Vector = TypeVar('Vector', FixedVector, Iterable[float])
Vectors = TypeVar('Vectors', List[Vector], Iterable[Vector])
FixedVectors = TypeVar('FixedVectors', List[FixedVector], Iterable[FixedVector])


class ProbableValue:
  def __init__(self, value, confidence: float):
    self.confidence: float = confidence
    self.value = value

def select_most_confident_if_almost_equal(a: ProbableValue, alternative: ProbableValue, equality_range=0.0)->ProbableValue:
  try:
    if abs(a.value.value - alternative.value.value) < equality_range:
      if a.confidence > alternative.confidence:
        return a
      else:
        return alternative
  except:
    #TODO: why dan hell we should have an exception here??
    return a

  return a


def split_by_token(tokens: List[str], token):
  res = []
  sentence = []
  for i in tokens:
    if i == token:
      res.append(sentence)
      sentence = []
    else:
      sentence.append(i)

  res.append(sentence)
  return res


def split_by_token_into_ranges(tokens: List, token) -> List[slice]:
  warnings.warn("deprecated", DeprecationWarning)
  res = []

  p = 0
  for i in range(len(tokens)):
    if tokens[i] == token:
      res.append(slice(p, i + 1))
      p = i + 1

  if p != len(tokens):
    res.append(slice(p, len(tokens)))
  return res


def estimate_threshold(a, min_th=0.3):
  return max(min_th, np.max(a) * 0.7)


def normalize(x: FixedVector, out_range=(0, 1)):
  domain = np.min(x), np.max(x)
  if (domain[1] - domain[0]) == 0:
    # all same
    return np.full(len(x), out_range[0])
    # raise ValueError('all elements are same')

  y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
  return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def smooth_safe(x: FixedVector, window_len=10, window='hanning'):
  _blur = int(min([window_len, 2 + len(x) / 3.0]))
  _blur = int(_blur / 2) * 2
  if (_blur > (len(x))):
    return x
  return smooth(x, window_len=_blur, window=window)


def smooth(x: FixedVector, window_len=11, window='hanning'):
  """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

  if x.ndim != 1:
    raise ValueError("smooth only accepts 1 dimension arrays.")

  if x.size < window_len:
    raise ValueError("Input vector needs to be bigger than window size.")

  if window_len < 3:
    return x

  if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

  s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
  # print(len(s))
  if window == 'flat':  # moving average
    w = np.ones(window_len, 'd')
  else:
    w = eval('np.' + window + '(window_len)')

  y = np.convolve(w / w.sum(), s, mode='valid')
  #     return y
  halflen = int(window_len / 2)
  #     return y[0:len(x)]
  return y[(halflen - 1):-halflen]


def relu(x: np.ndarray, relu_th: float = 0.0) -> np.ndarray:
  assert type(x) is np.ndarray

  _relu = x * (x > relu_th)
  return _relu


def extremums(x: FixedVector) -> List[int]:
  _extremums = [0]
  for i in range(1, len(x) - 1):
    if x[i - 1] < x[i] > x[i + 1]:
      _extremums.append(i)
  return _extremums


def softmax(v: np.ndarray) -> np.ndarray:
  x = normalize(v)
  x /= len(x)
  return x


def make_echo(av: FixedVector, k=0.5) -> np.ndarray:
  innertia = np.zeros(len(av))
  sum = 0

  for i in range(len(av)):
    if av[i] > k:
      sum = av[i]
    innertia[i] = sum
  #     sum-=0.0005
  return innertia


def momentum_(x: FixedVector, decay=0.99) -> np.ndarray:
  innertia = np.zeros(len(x))
  m = 0
  for i in range(len(x)):
    m += x[i]
    innertia[i] = m
    m *= decay

  return innertia


def momentum(x: FixedVector, decay=0.999) -> np.ndarray:
  innertia = np.zeros(len(x))
  m = 0
  for i in range(len(x)):
    m = max(m, x[i])
    innertia[i] = m
    m *= decay

  return innertia


import math


def momentum_t(x: FixedVector, half_decay: int = 10, left=False) -> np.ndarray:
  assert half_decay > 0

  decay = math.pow(2, -1 / half_decay)
  innertia = np.zeros(len(x))
  m = 0
  _r = range(len(x))
  if left:
    _r = reversed(_r)
  for i in _r:
    m = max(m, x[i])
    innertia[i] = m

    m *= decay

  return innertia


def momentum_p(x, half_decay: int = 10, left=False) -> np.ndarray:
  assert half_decay > 0

  decay = math.pow(2, -1 / half_decay)
  innertia = np.zeros(len(x))
  m = 0
  _r = range(len(x))
  if left:
    _r = reversed(_r)
  for i in _r:
    m = m + x[i] - (m * x[i])
    innertia[i] = m

    m *= decay

  return innertia


def momentum_reversed(x: FixedVector, decay=0.999) -> np.ndarray:
  innertia = np.zeros(len(x))
  m = 0
  for i in reversed(range(0, len(x))):
    m = max(m, x[i])
    innertia[i] = m
    m *= decay

  return innertia


def onehot_column(a: np.ndarray, mask=-2 ** 32, replacement=None):
  """

  Searches for maximum in every column.
  Other elements are replaced with mask

  :param a:
  :param mask:
  :return:
  """
  maximals = np.max(a, 0)

  for i in range(a.shape[0]):
    for j in range(a.shape[1]):
      if a[i, j] < maximals[j]:
        a[i, j] = mask
      else:
        if replacement is not None:
          a[i, j] = replacement

  return a


def most_popular_in(arr: FixedVector) -> int:
  if len(arr) == 0:
    return np.nan

  counts = np.bincount(arr)
  return int(np.argmax(counts))


def remove_similar_indexes(indexes: List[int], min_section_size=20):
  if len(indexes) < 2:
    return indexes

  indexes_zipped = [indexes[0]]

  for i in range(1, len(indexes)):
    if indexes[i] - indexes[i - 1] > min_section_size:
      indexes_zipped.append(indexes[i])
  return indexes_zipped


def cut_above(x: np.ndarray, threshold: float) -> np.ndarray:
  float___threshold = (x * float(-1.0)) + threshold

  return threshold + relu(float___threshold) * -1.0


def put_if_better(destination: dict, key, x, is_better: staticmethod):
  if key in destination:
    if is_better(x, destination[key]):
      destination[key] = x
  else:
    destination[key] = x


def rectifyed_sum(vectors: FixedVectors, relu_th: float = 0.0) -> np.ndarray:
  sum = None

  for x in vectors:
    if sum is None:
      sum = np.zeros(len(x))
    sum += relu(x, relu_th)

  assert sum is not None

  return sum


def filter_values_by_key_prefix(dictionary: dict, prefix: str) -> Vectors:
  for p in dictionary:
    if str(p).startswith(prefix):
      yield dictionary[p]


def max_exclusive_pattern_by_prefix(distances_per_pattern_dict, prefix):
  vectors = filter_values_by_key_prefix(distances_per_pattern_dict, prefix)

  return max_exclusive_pattern(vectors)


def max_exclusive_pattern(vectors: FixedVectors) -> FixedVector:
  _sum = None
  for x in vectors:
    if _sum is None:
      _sum = np.zeros(len(x))

    _sum = np.maximum(_sum, x)

  return _sum


def sum_probabilities(vectors: FixedVectors) -> FixedVector:
  _sum = np.zeros_like(vectors[0])
  for x in vectors:
    a = _sum + x
    b = _sum * x
    _sum = a - b

  return _sum


def subtract_probability(a: FixedVector, b: FixedVector) -> FixedVector:
  return 1.0 - sum_probabilities([1.0 - a, b])


class TokensWithAttention:
  def __init__(self, tokens: Tokens, attention: FixedVector):
    assert len(tokens) == len(attention)
    self.tokens = tokens
    self.attention = attention
