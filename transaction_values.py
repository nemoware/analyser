#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# transaction_values.py

import re
from typing import List

from text_tools import to_float, untokenize, np

currencly_map = {
  'доллар': 'USD',
  'евро': 'EUR',
  'руб': 'RUR'
}


class ValueConstraint:
  def __init__(self, value: float, currency: str, sign: int, context=None):
    self.value = value
    self.currency = currency
    self.sign = sign
    self.context = context


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


_re_less_then = re.compile(r'(до|менее|не выше|не превыша[а-я]{2,4})')
_re_greather_then = re.compile(r'(от|больше|более|свыше|выше|превыша[а-я]{2,4})')


def detect_sign(prefix: str):
  a = _re_less_then.findall(prefix)
  if len(a) > 0:
    return -1
  else:
    a = _re_greather_then.findall(prefix)
    if len(a) > 0:
      return +1
  return 0




def split_by_number(tokens: List[str], attention: List[float], threshold):
  # TODO: mind decimals!!

  indexes = []
  last_token_is_number = False
  for i in range(len(tokens)):
    if tokens[i].isdigit() and attention[i] > threshold:
      if not last_token_is_number:
        indexes.append(i)
      last_token_is_number = True
    else:
      last_token_is_number = False

  regions = []
  bounds = []
  if len(indexes) > 0:
    for i in range(1, len(indexes)):
      s = indexes[i - 1]
      e = indexes[i]
      regions.append(tokens[s:e])
      bounds.append((s, e))

    regions.append(tokens[indexes[-1]:])
    bounds.append((indexes[-1], len(tokens)))
  return regions, indexes, bounds


VALUE_SIGN_MIN_TOKENS = 4


def extract_sum_and_sign(subdoc, b) -> ValueConstraint:
  subtokens = subdoc.tokens_cc[b[0] - VALUE_SIGN_MIN_TOKENS:b[1]]
  _prefix_tokens = subtokens[0:VALUE_SIGN_MIN_TOKENS + 1]
  _prefix = untokenize(_prefix_tokens)
  _sign = detect_sign(_prefix)
  # ======================================
  sum = extract_sum_from_tokens(subtokens)[0]
  # ======================================

  currency = "UNDEF"
  value = np.nan
  if sum is not None:
    if sum[1] in currencly_map:
      currency = currencly_map[sum[1]]
    value = sum[0]

  vc = ValueConstraint(value, currency, _sign)
  return vc


