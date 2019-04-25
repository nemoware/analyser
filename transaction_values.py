#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# transaction_values.py

import math
import re
from typing import List

from ml_tools import TokensWithAttention
from text_tools import np
from text_tools import to_float, untokenize

currencly_map = {
  'руб': 'RUB',
  'дол': 'USD',
  'евр': 'EURO',
  'тэн': 'KZT',
  'тен': 'KZT',
}


class ValueConstraint:
  def __init__(self, value: float, currency: str, sign: int, context: TokensWithAttention):
    assert context is not None



    self.value = value
    self.currency = currency
    self.sign = sign
    self.context: TokensWithAttention = context




complete_re = re.compile(
  # r'(свыше|превыша[а-я]{2,4}|не превыша[а-я]{2,4})?\s+'
  r'(\d+([., ]\d+)*)'  # digits
  r'(?:\s*\(.+?\)\s*(?:тыс[а-я]*|млн|милли[а-я]{0,4})\.?)?'   # bullshit like 'от 1000000 ( одного ) миллиона рублей'
  r'(\s*(тыс[а-я]*|млн|милли[а-я]{0,4})\.?)?'                 # *1000 qualifier
  r'(\s*\((?:(?!\)).)+?\))?\s*'                               # some shit in parenthesis 
  r'((руб[а-я]{0,4}|доллар[а-я]{1,2}|евро|тенге)[\.,]?)'         # currency
  r'(\s*\((?:(?!\)).)+?\))?\s*'                               # some shit in parenthesis 
  r'(\s*(\d+)(\s*\(.+?\))?\s*коп[а-я]{0,4})?',                # cents
  re.MULTILINE | re.IGNORECASE
)

def extract_sum(sentence: str) -> (float, str):
  r = complete_re.search(sentence)

  if r is None:
    return None

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
    if frac == 0:
      number += to_float(r_cents) / 100.

  curr = r[7][0:3]

  return number, currencly_map[curr.lower()]


def extract_sum_from_tokens(sentence_tokens: List):
  sentence = untokenize(sentence_tokens).lower().strip()
  f = extract_sum(sentence)
  return f, sentence


def extract_sum_from_tokens_2(sentence_tokens: List):
  f, __ = extract_sum_from_tokens(sentence_tokens)
  return f


_re_greather_then_1 = re.compile(r'(не менее|не ниже)', re.MULTILINE)
_re_less_then = re.compile(r'(до\s+|менее|не более|не выше|не превыша[а-я]{2,4})', re.MULTILINE)
_re_greather_then = re.compile(r'(от\s+|больше|более|свыше|выше|превыша[а-я]{2,4})', re.MULTILINE)


def detect_sign(prefix: str):
  a = _re_greather_then_1.findall(prefix)
  if len(a) > 0:
    return +1

  a = _re_less_then.findall(prefix)
  if len(a) > 0:
    return -1
  else:
    a = _re_greather_then.findall(prefix)
    if len(a) > 0:
      return +1
  return 0


number_re = re.compile(r'^\d+[,.]?\d+', re.MULTILINE)


def split_by_number_2(tokens: List[str], attention: List[float], threshold) -> (
        List[List[str]], List[int], List[slice]):
  indexes = []
  last_token_is_number = False
  for i in range(len(tokens)):

    if attention[i] > threshold and len(number_re.findall(tokens[i])) > 0:
      if not last_token_is_number:
        indexes.append(i)
      last_token_is_number = True
    else:
      last_token_is_number = False

  text_fragments = []
  ranges: List[slice] = []
  if len(indexes) > 0:
    for i in range(1, len(indexes)):
      _slice = slice(indexes[i - 1], indexes[i])
      text_fragments.append(tokens[_slice])
      ranges.append(_slice)

    text_fragments.append(tokens[indexes[-1]:])
    ranges.append(slice(indexes[-1], len(tokens)))
  return text_fragments, indexes, ranges


def split_by_number(tokens: List[str], attention: List[float], threshold):
  indexes = []
  last_token_is_number = False
  for i in range(len(tokens)):

    if attention[i] > threshold and len(number_re.findall(tokens[i])) > 0:
      if not last_token_is_number:
        indexes.append(i)
      last_token_is_number = True
    else:
      last_token_is_number = False

  text_fragments = []
  ranges = []
  if len(indexes) > 0:
    for i in range(1, len(indexes)):
      s = indexes[i - 1]
      e = indexes[i]
      text_fragments.append(tokens[s:e])
      ranges.append((s, e))

    text_fragments.append(tokens[indexes[-1]:])
    ranges.append((indexes[-1], len(tokens)))
  return text_fragments, indexes, ranges


VALUE_SIGN_MIN_TOKENS = 4


def extract_sum_and_sign(subdoc, region) -> ValueConstraint:
  subtokens = subdoc.tokens_cc[region[0] - VALUE_SIGN_MIN_TOKENS:region[1]]
  _prefix_tokens = subtokens[0:VALUE_SIGN_MIN_TOKENS + 1]
  _prefix = untokenize(_prefix_tokens)
  _sign = detect_sign(_prefix)
  # ======================================
  _sum = extract_sum_from_tokens(subtokens)[0]
  # ======================================

  currency = "UNDEF"
  value = np.nan
  if _sum is not None:
    currency = _sum[1]
    if _sum[1] in currencly_map:
      currency = currencly_map[_sum[1]]
    value = _sum[0]

  vc = ValueConstraint(value, currency, _sign, TokensWithAttention([''], [0]))

  return vc


def extract_sum_and_sign_2(subdoc, region: slice) -> ValueConstraint:
  _slice = slice(region.start - VALUE_SIGN_MIN_TOKENS, region.stop)
  subtokens = subdoc.tokens_cc[_slice]
  _prefix_tokens = subtokens[0:VALUE_SIGN_MIN_TOKENS + 1]
  _prefix = untokenize(_prefix_tokens)
  _sign = detect_sign(_prefix)
  # ======================================
  _sum = extract_sum_from_tokens_2(subtokens)
  # ======================================

  currency = "UNDEF"
  value = np.nan
  if _sum is not None:
    currency = _sum[1]
    if _sum[1] in currencly_map:
      currency = currencly_map[_sum[1]]
    value = _sum[0]

  vc = ValueConstraint(value, currency, _sign, TokensWithAttention([], []))

  return vc


if __name__ == '__main__':
  ex = """
  одобрение заключения , изменения или расторжения какой-либо сделки общества , не указанной прямо в пункте 17.1 устава или настоящем пункте 22.5 ( за исключением крупной сделки в определении действующего законодательства российской федерации , которая подлежит одобрению общим собранием участников в соответствии с настоящим уставом или действующим законодательством российской федерации ) , если предметом такой сделки ( а ) является деятельность , покрываемая в долгосрочном плане , и сделка имеет стоимость , равную или превышающую 5000000 ( пять миллионов ) долларов сша , либо ( b ) является деятельность , не покрываемая в долгосрочном плане , и сделка имеет стоимость , равную или превышающую 500000 ( пятьсот тысяч ) долларов сша ;
  """
  print(extract_sum('\n2.1.  Общая сумма договора составляет 41752 руб. (Сорок одна т'
                    'ысяча семьсот пятьдесят два рубля) '
                    '62 копейки, в т.ч. НДС (18%) 6369,05 руб. (Шесть тысяч триста шестьдесят девять рублей) 05 копеек, в'))
  print(extract_sum('эквивалентной 25 миллионам долларов сша'))

  print(extract_sum('взаимосвязанных сделок в совокупности составляет от '
                    '1000000 ( одного ) миллиона рублей до 50000000 '))
  print(extract_sum(
    'Общая сумма договора составляет 41752,62 рублей ( Сорок одна тысяча '
    'семьсот пятьдесят два рубля ) 62 копейки , в том числе НДС '))

  print(extract_sum(ex))

  print(extract_sum(
    'одобрение заключения , изменения или расторжения какой-либо сделки общества , '
    'не указанной прямо в пункте 17.1 или настоящем пункте 22.5 ( за исключением '
    'крупной сделки в определении действующего законодательства российской федерации , '
    'которая подлежит одобрению общим собранием участников в соответствии с настоящим '
    'уставом или действующим законодательством российской федерации ) , если предметом '
    'такой сделки ( a ) является деятельность , покрываемая в долгосрочном плане , и '
    'сделка имеет стоимость , равную или превышающую 2000000 ( два миллиона ) долларов '
    'сша , но менее 5000000 ( пяти миллионов ) долларов сша , либо ( b ) деятельность , '
    'не покрываемая в долгосрочном плане , и сделка имеет стоимость , равную или '
    'превышающую 150000 ( сто пятьдесят тысяч ) долларов сша , но менее 500000 ( пятисот тысяч ) долларов сша ;'))

  print(extract_sum(
    '3. нецелевое расходование обществом денежных средств ( расходование не в соответствии'
    ' с утвержденным бизнес-планом или бюджетом ) при совокупности следующих условий : ( i ) размер '
    'таких нецелевых расходов в течение 6 ( шести ) месяцев превышает 25 000 000 ( двадцать пять миллионов ) рублей '
    'или эквивалент данной суммы в иной валюте , ( ii ) такие расходы не были в последующем одобрены советом директоров '
    'в течение 1 ( одного ) года с момента , когда нецелевые расходы превысили вышеуказанную сумму и ( iii ) генеральным директором '
    'не было получено предварительное одобрение на такое расходование от представителей участников , уполномоченных голосовать '
    'на общем собрании участников общества ;'))


  sentence = """ 1.1.1. Счет № 115 на приобретение спортивного оборудования ( теннисный стол, рукоход с перекладинами, шведская стенка ). Стоимость оборудования 80 000,00 ( восемьдесят тысяч рублей рублей 00 копеек ) рублей, НДС не облагается."""
  print(extract_sum(sentence))
