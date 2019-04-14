#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

# transaction_values.py

import re
import math
from typing import List

from text_tools import to_float, untokenize

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
  r'(\s*(\d+)(\s*\(.+?\))?\s*коп[а-я]{0,4})?',                     # cents
  re.MULTILINE|re.IGNORECASE
)

def extract_sum(sentence: str):
  r = complete_re.findall(sentence)
  print(r[0])
  number = to_float(r[0][0])
  r_num = r[0][3]
  if r_num:
    if r_num.startswith('тыс'):
      number *= 1000
    else:
      if r_num.startswith('м'):
        number *= 1000000

  r_cents = r[0][9]
  if r_cents:
    frac, whole = math.modf(number)
    if frac==0:
      number += to_float(r_cents) / 100.

  curr = r[0][6][0:3]

  return (number, currency_normalizer[curr.lower()])



def extract_sum_from_tokens(sentence_tokens: List):
  sentence = untokenize(sentence_tokens).lower().strip()
  f = extract_sum(sentence)
  return f, sentence


if __name__ == '__main__':
    print(extract_sum('\n2.1.  Общая сумма договора составляет 41752 руб. (Сорок одна тысяча семьсот пятьдесят два рубля) '
     '62 копейки, в т.ч. НДС (18%) 6369,05 руб. (Шесть тысяч триста шестьдесят девять рублей) 05 копеек, в'))
    print(extract_sum('эквивалентной 25 миллионам долларов сша'))



