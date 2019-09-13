#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# transaction_values.py

import math
import re
import warnings
from typing import List

from ml_tools import TokensWithAttention, FixedVector
from text_tools import np, Tokens, to_float, untokenize

currencly_map = {
  'руб': 'RUB',
  'дол': 'USD',
  'евр': 'EURO',
  'тэн': 'KZT',
  'тен': 'KZT',
}


class ValueConstraint:
  def __init__(self, value: float, currency: str, sign: int, context: TokensWithAttention):
    warnings.warn("ValueConstraint is deprecated, use TaggedValueConstraint", DeprecationWarning)
    assert context is not None

    self.value: float = value
    self.currency: str = currency
    self.sign: int = sign

    self.context: TokensWithAttention = context

  def __str__(self):
    return f'{self.value} {self.sign} {self.currency}'


complete_re = re.compile(
  # r'(свыше|превыша[а-я]{2,4}|не превыша[а-я]{2,4})?\s+'
  r'(?P<digits>\d+([., ]\d+)*)'  # digits #0
  r'(?:\s*\(.+?\)\s*(?:тыс[а-я]*|млн|милли[а-я]{0,4})\.?)?'  # bullshit like 'от 1000000 ( одного ) миллиона рублей'
  r'(\s*(?P<qualifier>тыс[а-я]*|млн|милли[а-я]{0,4})\.?)?'  # *1000 qualifier
  r'(\s*\((?:(?!\)).)+?\))?\s*'  # some shit in parenthesis 
  r'?((?P<currency>руб[а-я]{0,4}|доллар[а-я]{1,2}|евро|тенге)[\.,]?)'  # currency #7
  r'(\s*\((?:(?!\)).)+?\))?\s*'  # some shit in parenthesis 
  r'(\s*(?P<cents>\d+)(\s*\(.+?\))?\s*коп[а-я]{0,4})?',  # cents
  re.MULTILINE | re.IGNORECASE
)


# for r in re.finditer(complete_re, text):
def extract_sum(_sentence: str) -> (float, str):
  warnings.warn("use find_value_spans", DeprecationWarning)
  r = complete_re.search(_sentence)

  if r is None:
    return None, None

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


def extract_sum_from_tokens(sentence_tokens: Tokens):
  warnings.warn("method relies on untokenize, not good", DeprecationWarning)
  _sentence = untokenize(sentence_tokens).lower().strip()
  f = extract_sum(_sentence)
  return f, _sentence


def extract_sum_from_tokens_2(sentence_tokens: Tokens):
  warnings.warn("method relies on untokenize, not good", DeprecationWarning)
  f, __ = extract_sum_from_tokens(sentence_tokens)
  return f


_re_greather_then_1 = re.compile(r'(не менее|не ниже)', re.MULTILINE)
_re_greather_then = re.compile(r'(\sот\s+|больше|более|свыше|выше|превыша[а-я]{2,4})', re.MULTILINE)
_re_less_then = re.compile(
  r'(до\s+|менее|не может превышать|лимит соглашения[:]*|не более|не выше|не превыша[а-я]{2,4})', re.MULTILINE)


def detect_sign(prefix: str):
  warnings.warn("use detect_sign_2", DeprecationWarning)
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


def split_by_number_2(tokens: List[str], attention: FixedVector, threshold) -> (
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


def extract_sum_and_sign_2(subdoc, region: slice) -> ValueConstraint:
  warnings.warn("deprecated", DeprecationWarning)
  # TODO: rename

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


def find_value_spans(_sentence: str) -> (List[int], float, List[int], str):
  for match in re.finditer(complete_re, _sentence):

    # NUMBER
    number_span = match.span('digits')

    number = to_float(_sentence[number_span[0]:number_span[1]])

    # NUMBER MULTIPLIER
    qualifier_span = match.span('qualifier')
    qualifier = _sentence[qualifier_span[0]:qualifier_span[1]]
    if qualifier:
      if qualifier.startswith('тыс'):
        number *= 1000
      else:
        if qualifier.startswith('м'):
          number *= 1000000

    # FRACTION (CENTS, KOPs)
    cents_span = match.span('cents')
    r_cents = _sentence[cents_span[0]:cents_span[1]]
    if r_cents:
      frac, whole = math.modf(number)
      if frac == 0:
        number += to_float(r_cents) / 100.

    # CURRENCY
    currency_span = match.span('currency')
    currency = _sentence[currency_span[0]:currency_span[1]]
    curr = currency[0:3]
    currencly_name = currencly_map[curr.lower()]

    # TODO: include fration span to the return value
    ret = number_span, number, currency_span, currencly_name

    return ret


if __name__ == '__main__':
  ex = "составит - не более 1661 293,757 тыс. рублей  25 копеек ( с учетом ндс ) ( 0,93 % балансовой стоимости активов)"
  val = find_value_spans(ex)
  print('extract_sum', extract_sum(ex))
  print('val', val)

if __name__ == '__main__X':
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

  s_ = """Стоимость оборудования 80 000,00 ( восемьдесят тысяч рублей рублей 00 копеек ) рублей, НДС не облагается."""
  print(extract_sum(s_))

  s_ = """Стоимость оборудования 80 000,00 (восемьдесят тысяч рублей рублей 00 копеек) рублей,"""
  print(extract_sum(s_))
