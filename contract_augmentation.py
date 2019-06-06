import random
from random import randint

from documents import MarkedDoc
from text_tools import Tokens


def remove_char_at(token, i):
  return token[:i] + token[i + 1:]


def remove_random_char(token: str):
  if len(token) > 2:
    idx = randint(0, len(token) - 1)
    return remove_char_at(token, idx)
  else:
    return token


def augment_dropout_chars_d(d: MarkedDoc, rate):
  def _rm(token, clazz):
    ntoken = token
    if random.random() < rate:
      ntoken = remove_random_char(token)

    return (ntoken, clazz)

  d.filter(_rm)


def augment_dropout_words_d(d: MarkedDoc, rate):
  def _rm(token, clazz):
    if random.random() < rate:
      return None

    return (token, clazz)

  d.filter(_rm)


def augment_trim(doc: MarkedDoc, max_to_trim=40):
  _slice = slice( randint(0, max_to_trim), -randint(1, max_to_trim))
  doc.trim(_slice)


def augment_alter_case_d(doc: MarkedDoc, rate):
  def _alter_case(t, c):

    if random.random() < rate:
      if random.random() < 0.5:
        t = t.upper()
      else:
        t = t.lower()

    return (t, c)

  return doc.filter(_alter_case)


def augment_dropout_punctuation_d(doc: MarkedDoc, rate):
  def _drop_punkt(t, c):

    if t in ',."«-»–()' and random.random() < rate:
      return None
    else:
      return (t, c)

  return doc.filter(_drop_punkt)


def augment_contract(tokens_: Tokens, categories_vector_):
  doc = MarkedDoc(tokens_, categories_vector_)

  augment_dropout_words_d(doc, 0.05)
  augment_dropout_punctuation_d(doc, 0.15)
  augment_dropout_chars_d(doc, 0.02)
  augment_alter_case_d(doc, 0.15)
  augment_trim(doc, 30)

  return doc.tokens, doc.categories_vector


if __name__ == '__main__':
  doc = MarkedDoc(['12345', '12345', '12345', '12345', '12345'], [1, 2, 3, 4, 5])

  augment_dropout_chars_d(doc, 0.5)
  print(doc.tokens)
  pass
