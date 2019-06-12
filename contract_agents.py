#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import re
from typing import AnyStr, Match, Dict, List


from contract_parser import ContractDocument3
from text_normalize import r_group, r_bracketed, r_quoted, r_capitalized_ru, \
  _r_name, r_quoted_name, replacements_regex, ru_cap, r_few_words_s, r_human_name
from text_tools import tokens_in_range



ORG_TYPES_re = [
  ru_cap('Акционерное общество'), 'АО',
  ru_cap('Закрытое акционерное общество'), 'ЗАО',
  ru_cap('Открытое акционерное общество'), 'ОАО',
  ru_cap('Государственное автономное учреждение'),
  ru_cap('Муниципальное бюджетное учреждение'),
  ru_cap('учреждение'),
  ru_cap('Общественная организация'),
  ru_cap('Общество с ограниченной ответственностью'), 'ООО',
  ru_cap('Некоммерческая организация'),
  ru_cap('Благотворительный фонд'),
  ru_cap('Индивидуальный предприниматель'), 'ИП',

  r'[Фф]онд[а-я]{0,2}' + r_few_words_s,

]
_r_types_ = '|'.join([x for x in ORG_TYPES_re])

r_few_words = r'\s+[А-Я]{1}[а-я\-, ]{1,80}'

r_type_ext = r_group(r'[А-Яa-zа-яА-Я0-9\s]*', 'type_ext')
r_name_alias = r_group(_r_name, 'alias')

r_quoted_name_alias = r_group(r_quoted(r_name_alias))
r_alias_prefix = r_group(''
                         + r_group(r'(именуе[а-я]{1,3}\s+)?в?\s*дал[а-я]{2,8}\s?[–\-]?') + '|'
                         + r_group(r'далее\s?[–\-]?\s?'))
r_alias = r_group(r".{0,140}" + r_alias_prefix + r'\s*' + r_quoted_name_alias)

r_types = r_group(f'{_r_types_}', 'type')
r_type_and_name = r_types + r_type_ext + r_quoted_name

r_alter = r_group(r_bracketed(r'.{1,70}') + r'{0,2}', 'alt_name')
complete_re_str = r_type_and_name + '\s*' + r_alter + r_alias + '?'
# ----------------------------------
complete_re = re.compile(complete_re_str, re.MULTILINE)

# ----------------------------------

entities_types = ['type', 'name', 'alt_name', 'alias', 'type_ext']
import random


def make_rnanom_name(lenn) -> str:
  return ''.join(random.choices('АБВГДЕЖЗИКЛМН', k=1) + random.choices('абвгдежопа ', k=lenn))



# def augment_contract(txt: str, org_infos: List[Dict]):
#   txt_a = txt
#   for org in org_infos:
#     for e in ['name', 'alias']:
#       substr = org[e][0]
#       r = re.compile(substr)
#       txt_a = re.sub(r, make_rnanom_name(10), txt_a)
#
#   return txt_a, _find_org_names(txt_a)

def _find_org_names(txt: str) -> List[Dict]:
  def _clean(x):
    if x is None:
      return x
    return x.replace('\t', ' ').replace('\n', ' ').replace(' – ', '-').lower()

  def _to_dict(m: Match[AnyStr]):
    d = {}
    for entity_type in entities_types:
      d[entity_type] = (m[entity_type], m.span(entity_type))

    return d

  org_names = {}
  for r in re.finditer(complete_re, txt):
    org = _to_dict(r)

    # filter similar out
    _name = _clean(org['name'][0])
    if _name not in org_names:
      org_names[_name] = org

  return list(org_names.values())


def find_org_names_spans(doc: ContractDocument3) -> None:
  agent_infos = _find_org_names(doc.normal_text)
  doc.agent_infos = agent_infos
  _convert_char_slices_to_tokens(doc)



def _convert_char_slices_to_tokens(doc: ContractDocument3):
  for org in doc.agent_infos:
    for ent in org:
      span = org[ent][1]

      if span[0] > 0:
        tokens_slice = tokens_in_range(span, doc.tokens_cc, doc.normal_text)
        org[ent] = (org[ent][0], org[ent][1], tokens_slice)
      else:
        org[ent] = (org[ent][0], None, None)


def normalize_contract(_t: str) -> str:
  t = _t
  for (reg, to) in alias_quote_regex + replacements_regex:
    t = reg.sub(to, t)

  return t


r_ip = r_group('(\s|^)' + ru_cap('Индивидуальный предприниматель') + '\s*' + '|(\s|^)ИП\s*', 'ip')
sub_ip_quoter = (re.compile(r_ip + r_human_name), r'\1«\g<human_name>»')
sub_org_name_quoter = (re.compile(r_quoted_name + '\s*' + r_bracketed(r_types)), r'\g<type> «\g<name>» ')

sub_alias_quote = (re.compile(r_alias_prefix + r_group(r_capitalized_ru, '_alias')), r'\1«\g<_alias>»')
alias_quote_regex = [
  sub_alias_quote,
  sub_ip_quoter,
  sub_org_name_quoter
]

if __name__ == '__main__':
  print(r_group(r_capitalized_ru, 'alias'))
  pass
