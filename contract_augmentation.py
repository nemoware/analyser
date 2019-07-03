import random
from random import randint

import rstr

from contract_agents import find_org_names_spans, r_alter
from contract_parser import ContractDocument3
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

    return ntoken, clazz

  d.filter(_rm)


def augment_dropout_words_d(d: MarkedDoc, rate):
  def _rm(token, clazz):
    if random.random() < rate:
      return None

    return token, clazz

  d.filter(_rm)


def augment_trim(_doc: MarkedDoc, max_to_trim=40):
  _slice = slice(randint(0, max_to_trim), -randint(1, max_to_trim))
  _doc.trim(_slice)


def augment_alter_case_d(_doc: MarkedDoc, rate):
  def _alter_case(t, c):
    t = t.upper()
    return t, c

  if random.random() < rate:
    return _doc.filter(_alter_case)
  return _doc



def augment_dropout_punctuation_d(_doc: MarkedDoc, rate):
  def _drop_punkt(t, c):

    if t in ',."«-»–()' and random.random() < rate:
      return None
    else:
      return t, c

  return _doc.filter(_drop_punkt)


def augment_contract(tokens_: Tokens, categories_vector_):
  _doc = MarkedDoc(tokens_, categories_vector_)

  augment_dropout_words_d(_doc, 0.05)
  augment_dropout_punctuation_d(_doc, 0.15)
  augment_dropout_chars_d(_doc, 0.02)
  augment_alter_case_d(_doc, 0.05)
  # augment_trim(_doc, 30)

  return _doc.tokens, _doc.categories_vector


def augment_contract_2(_mdoc: ContractDocument3) -> MarkedDoc:
  _doc = MarkedDoc(_mdoc.tokens, _mdoc.categories_vector)

  augment_dropout_words_d(_doc, 0.05)
  augment_dropout_punctuation_d(_doc, 0.15)
  augment_dropout_chars_d(_doc, 0.02)
  augment_alter_case_d(_doc, 0.05)
  # augment_trim(_doc, 30)

  return _doc


if __name__ == '__main__':
  doc = MarkedDoc(['12345', '12345', '12345', '12345', '12345'], [1, 2, 3, 4, 5])

  augment_dropout_chars_d(doc, 0.5)
  print(doc.tokens)
  pass


def make_random_word(lenn) -> str:
  return ''.join(random.choices('АБВГДЕЖЗИКЛМН', k=1) + random.choices('абвгдежопа', k=max(1, lenn)))


def make_random_name(lenn) -> str:
  #   nwords = int(1+(lenn/8))
  #   words = [  ]
  return ''.join(random.choices('АБВГДЕЖЗИКЛМН', k=1) + random.choices('абвгдежопа ', k=lenn))


def make_random_name_random_len(new_len, maxlen=30) -> str:
  new_len += random.randint(int(-new_len / 2), int(new_len / 2))
  new_len = min(new_len, maxlen)
  return make_random_name(new_len)


ORG_TYPES = [
  'Акционерное общество', 'АО',
  'Закрытое акционерное общество', 'ЗАО',
  'Открытое акционерное общество', 'ОАО',
  'Государственное автономное учреждение',
  'Муниципальное бюджетное учреждение',
  'Общественная организация',
  'Общество с ограниченной ответственностью',
  'Некоммерческая организация',
  'Благотворительный фонд',
  'Индивидуальный предприниматель', 'ИП'
]


def obfuscate_org_types(_doc: ContractDocument3, rate=0.5) -> ContractDocument3:
  txt_a = _doc.normal_text

  for org in _doc.agent_infos:
    if random.random() < rate:
      substr = org['type'][0]
      txt_a = txt_a.replace(substr, random.choice(ORG_TYPES))

  new_doc = ContractDocument3(txt_a)
  new_doc.parse()
  find_org_names_spans(new_doc)
  assert len(_doc.agent_infos) == len(new_doc.agent_infos)
  return new_doc


def obfuscate_contract(_doc: ContractDocument3, rate=0.5):
  new_doc = obfuscate_org_types(_doc, rate)
  txt_a = new_doc.normal_text

  for org in new_doc.agent_infos:

    for e in ['name', 'alias']:

      substr = org[e][0]

      if substr and substr != '' and random.random() < rate:
        repl = make_random_name_random_len(len(substr))
        #         print('obfuscating:', substr, '--->', repl)
        txt_a = txt_a.replace(substr, repl)

        # new_org_infos = find_org_names(txt_a)
        # try:
        #   assert len(new_doc.agent_infos) <= len(new_org_infos)
        # except:
        #   print(new_org_infos)
        #   print(txt_a)
        #   raise ValueError('cannot replace ' + substr + ' with ' + repl)

  for org in new_doc.agent_infos:
    substr = org['alt_name'][0]
    rstr.xeger(r_alter)
    if substr is not None and substr != '' and random.random() < rate:
      repl = rstr.xeger(r_alter)
      txt_a = txt_a.replace(substr, repl)

  # return txt_a, find_org_names(txt_a)

  new_doc = ContractDocument3(txt_a)
  new_doc.parse()
  find_org_names_spans(new_doc)
  return new_doc
