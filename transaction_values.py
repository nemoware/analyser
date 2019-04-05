import re
from typing import List

from text_tools import to_float, untokenize


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





