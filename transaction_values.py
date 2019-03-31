import re
from typing import List

from legal_docs import LegalDocument, rectifyed_sum_by_pattern_prefix
from ml_tools import normalize, smooth, extremums
from text_tools import to_float, untokenize, get_sentence_bounds_at_index, np


def extract_sum(sentence: str):
  currency_re = re.compile(r'((^|\s+)(\d+[., ])*\d+)(\s*([(].{0,100}[)]\s*)?(евро|руб|доллар))')
  currency_re_th = re.compile(
    r'((^|\s+)(\d+[., ])*\d+)(\s+(тыс\.|тысяч.{0,2})\s+)(\s*([(].{0,100}[)]\s*)?(евро|руб|доллар))')
  currency_re_mil = re.compile(
    r'((^|\s+)(\d+[., ])*\d+)(\s+(млн\.|миллион.{0,3})\s+)(\s*([(].{0,100}[)]\s*)?(евро|руб|доллар))')

  r = currency_re.findall(sentence)
  f = None
  try:
    number = to_float(r[0][0])
    f = (number, r[0][5])
  except:
    r = currency_re_th.findall(sentence)

    try:
      number = to_float(r[0][0]) * 1000
      f = (number, r[0][5])
    except:
      r = currency_re_mil.findall(sentence)
      try:
        number = to_float(r[0][0]) * 1000000
        f = (number, r[0][5])
      except:
        pass

  return f


def extract_sum_from_tokens(sentence_tokens: List):
  sentence = untokenize(sentence_tokens).lower().strip()
  f = extract_sum(sentence)
  return f, sentence


def _extract_sums_from_distances(doc: LegalDocument, x):
  maximas = extremums(x)

  results = []
  for max_i in maximas:
    start, end = get_sentence_bounds_at_index(max_i, doc.tokens)
    sentence_tokens = doc.tokens[start + 1:end]

    f, sentence = extract_sum_from_tokens(sentence_tokens)

    if f is not None:
      result = {
        'sum': f,
        'region': (start, end),
        'sentence': sentence,
        'confidence': x[max_i]
      }
      results.append(result)

  return results


def extract_sum_from_doc(doc: LegalDocument, attention_mask=None, relu_th=0.5):
  sum_pos, _c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, 'sum_max', relu_th=relu_th)
  sum_neg, _c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, 'sum_max_neg', relu_th=relu_th)

  sum_pos -= sum_neg

  sum_pos = smooth(sum_pos, window_len=8)
  #     sum_pos = relu(sum_pos, 0.65)

  if attention_mask is not None:
    sum_pos *= attention_mask

  sum_pos = normalize(sum_pos)

  return _extract_sums_from_distances(doc, sum_pos), sum_pos


def _extract_sum_from_distances____(doc: LegalDocument, sums_no_padding):
  max_i = np.argmax(sums_no_padding)
  start, end = get_sentence_bounds_at_index(max_i, doc.tokens)
  sentence_tokens = doc.tokens[start + 1:end]

  f, sentence = extract_sum_from_tokens(sentence_tokens)

  return (f, (start, end), sentence)