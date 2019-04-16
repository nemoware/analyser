#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# transaction_values.py

import re
import math
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

currency_normalizer = {
  'руб':'РУБ',
  'дол':'USD',
  'евр':'EURO',
  'тэн':'KZT',
  'тен':'KZT',
}

complete_re = re.compile(
  # r'(свыше|превыша[а-я]{2,4}|не превыша[а-я]{2,4})?\s+'
  r'(\d+([., ]\d+)*)'                                 # digits
  r'(\s*(тыс[а-я]*|млн|милли[а-я]{0,4})\.?)?'         # *1000 qualifier
  r'(\s*\(.+?\))?\s*'                                 # some shit in parenthesis 
  r'((руб[а-я]{0,4}|доллар[а-я]{1,2}|евро|тенге)\.?)' # currency
  r'(\s*\(.+?\))?'                                    # some shit in parenthesis 
  r'(\s*(\d+)(\s*\(.+?\))?\s*коп[а-я]{0,4})?',        # cents
  re.MULTILINE|re.IGNORECASE
)

def extract_sum(sentence: str):
  r = complete_re.search(sentence)
  # print(r[0])
  number = to_float(r[1])
  r_num = r[4]
  if r_num:
    if r_num.startswith('тыс'):
      number *= 1000
    else:
      if r_num.startswith('м'):
        number *= 1000000

  r_cents = r[10]
  if r_cents:
    frac, whole = math.modf(number)
    if frac==0:
      number += to_float(r_cents) / 100.

  curr = r[7][0:3]

  return (number, currency_normalizer[curr.lower()])



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

if __name__ == '__main__':
    print(extract_sum('\n2.1.  Общая сумма договора составляет 41752 руб. (Сорок одна тысяча семьсот пятьдесят два рубля) '
     '62 копейки, в т.ч. НДС (18%) 6369,05 руб. (Шесть тысяч триста шестьдесят девять рублей) 05 копеек, в'))
    print(extract_sum('эквивалентной 25 миллионам долларов сша'))
    print (extract_sum('взаимосвязанных сделок в совокупности составляет от 1000000 ( одного ) миллиона рублей до 50000000 '))

