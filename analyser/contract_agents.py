#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import re
import warnings
from typing import AnyStr, Match, Dict, List

from pyjarowinkler import distance

from analyser.hyperparams import HyperParameters
from analyser.legal_docs import LegalDocument
from analyser.ml_tools import SemanticTag, put_if_better
from analyser.structures import ORG_LEVELS_names, legal_entity_types
from analyser.text_normalize import r_group, r_bracketed, r_quoted, r_capitalized_ru, \
  _r_name, r_quoted_name, ru_cap, normalize_company_name, r_alias_prefix, r_types
from analyser.text_tools import is_long_enough, span_len
from gpn.gpn import subsidiaries


def morphology_agnostic_re(x):
  if len(x) > 2:
    r = f'[{x[0].upper()}{x[0].lower()}]{x[1:-2]}[а-я]{{0,3}}'
    r = f'({r})'
    return r
  else:
    return f'({x})'


def re_legal_entity_type(xx):
  _all = r'\s+'.join(morphology_agnostic_re(x) for x in xx.split(' '))
  return f'(?P<org_type>{_all})'


legal_entity_types_re = {}
for t in legal_entity_types:
  _regex = re_legal_entity_type(t)
  rr = re.compile(_regex, re.IGNORECASE | re.UNICODE)
  legal_entity_types_re[rr] = t
  found = rr.match(t)[0]
  assert t == found

_is_valid = is_long_enough

ORG_LEVELS_re = r_group('|'.join([ru_cap(x) for x in ORG_LEVELS_names]), 'org_structural_level') + r'\s'

r_few_words = r'\s+[А-Я]{1}[а-я\-, ]{1,80}'

r_type_ext = r_group(r'[А-Яa-zа-яА-Я0-9\s]*', 'type_ext')
r_name_alias = r_group(_r_name, 'alias')

r_quoted_name_alias = r_group(r_quoted(r_name_alias), 'r_quoted_name_alias')
r_alias = r_group(r".{0,140}" + r_alias_prefix + r'\s*' + r_quoted_name_alias)

r_type_and_name = r_types + r_type_ext + r_quoted_name

r_alter = r_group(r_bracketed(r'.{1,70}') + r'{0,2}', 'alt_name')
complete_re_str = r_type_and_name + r'\s*' + r_alter + r_alias + '?'
# ----------------------------------
complete_re = re.compile(complete_re_str, re.MULTILINE | re.IGNORECASE)

# ----------------------------------

org_pieces = ['type', 'name', 'alt_name', 'alias', 'type_ext']


class ContractAgent:
  # org_pieces = ['type', 'name', 'alt_name', 'alias', 'type_ext']
  def __init__(self):
    self.name: SemanticTag or None = None
    self.type: SemanticTag or None = None
    self.alt_name: SemanticTag or None = None
    self.alias: SemanticTag or None = None
    self.type_ext: SemanticTag or None = None

  def as_list(self):
    return [self.__dict__[key] for key in org_pieces if self.__dict__[key] is not None]

  def confidence(self):
    confidence = 0
    for key in ['type', 'name']:
      tag = self.__dict__[key]
      if tag is not None:
        confidence += tag.confidence

    return confidence / 2


def clean_value(x: str) -> str or None:
  if x is None:
    return x
  return x.replace('\t', ' ').replace('\n', ' ').replace(' – ', '-').lower()


def _find_org_names(text: str) -> List[Dict]:
  warnings.warn("make semantic tags", DeprecationWarning)

  def _to_dict(m: Match[AnyStr]):
    warnings.warn("make semantic tags", DeprecationWarning)
    d = {}
    for entity_type in org_pieces:
      d[entity_type] = (m[entity_type], m.span(entity_type))

    return d

  org_names = {}

  for r in re.finditer(complete_re, text):
    org = _to_dict(r)

    # filter similar out
    _name = normalize_company_name(org['name'][0])
    if _name not in org_names:
      org_names[_name] = org

  return list(org_names.values())


def find_org_names_in_tag(doc: LegalDocument, parent: SemanticTag, max_names=2, tag_kind_prefix='',
                          decay_confidence=True) -> List[
  SemanticTag]:
  span = parent.span
  return find_org_names(doc[span[0]:span[1]], max_names=max_names, tag_kind_prefix=tag_kind_prefix, parent=parent,
                        decay_confidence=decay_confidence)


def find_org_names(doc: LegalDocument, max_names=2, tag_kind_prefix='', parent=None, decay_confidence=True) -> List[
  SemanticTag]:
  all: [ContractAgent] = find_org_names_raw(doc, max_names, parent, decay_confidence)
  return _rename_org_tags(all, tag_kind_prefix)


def _rename_org_tags(all: [ContractAgent], prefix=''):
  tags = []
  for group in range(len(all)):
    for tag in all[group].as_list():
      tagname = f'{prefix}org-{group + 1}-{tag.kind}'
      tag.kind = tagname
      tags.append(tag)

  return tags


def find_org_names_raw(doc: LegalDocument, max_names=2, parent=None, decay_confidence=True) -> [ContractAgent]:
  all: [ContractAgent] = []

  for m in re.finditer(complete_re, doc.text):
    ca = ContractAgent()
    all.append(ca)
    for kind in org_pieces:
      char_span = m.span(kind)
      span = doc.tokens_map.token_indices_by_char_range(char_span)
      confidence = 1
      if decay_confidence:
        confidence = 1.0 - (span[0] / len(doc))

      val = doc.tokens_map.text_range(span)
      val = val.strip()
      if span_len(char_span) > 1 and _is_valid(val):
        tag = SemanticTag(kind, val, span, parent=parent)
        tag.confidence = confidence
        tag.offset(doc.start)
        ca.__dict__[kind] = tag

  # normalize org_name names by find_closest_org_name
  for ca in all:
    if ca.name is not None:
      legal_entity_type, val = normalize_company_name(ca.name.value)
      ca.name.value = val
      known_org_name, best_similarity = find_closest_org_name(subsidiaries, val,
                                                              HyperParameters.subsidiary_name_match_min_jaro_similarity)
      if known_org_name is not None:
        ca.name.value = known_org_name['_id']
        ca.name.confidence *= best_similarity

    # normalize org_type names by find_closest_org_name
    if ca.type is not None:
      long_, short_, confidence_ = normalize_legal_entity_type(ca.type.value)
      ca.type.value = long_
      ca.type.confidence *= confidence_

  # filter, keep unique names
  _map = {}
  for ca in all:
    if ca.name is not None:
      if ca.confidence() > 0.2:
        put_if_better(_map, ca.name.value, ca, lambda a, b: a.confidence() > b.confidence())

  res = list(_map.values())
  res = sorted(res, key=lambda a:-a.confidence())
  return res[:max_names]
  #


def compare_masked_strings(a, b, masked_substrings):
  a1 = a
  b1 = b
  for masked in masked_substrings:
    if a1.find(masked) >= 0 and b1.find(masked) >= 0:
      a1 = a1.replace(masked, '')
      b1 = b1.replace(masked, '')

  # print(a1, '--', b1)
  return distance.get_jaro_distance(a1, b1, winkler=True, scaling=0.1)


def find_closest_org_name(subsidiaries, pattern, threshold=HyperParameters.subsidiary_name_match_min_jaro_similarity):
  if pattern is None:
    return None, 0
  best_similarity = 0
  finding = None
  _entity_type, pn = normalize_company_name(pattern)

  for s in subsidiaries:
    for alias in s['aliases'] + [s['_id']]:
      similarity = compare_masked_strings(pn, alias, [])
      if similarity > best_similarity:
        best_similarity = similarity
        finding = s

  if best_similarity > threshold:
    return finding, best_similarity
  else:
    return None, best_similarity


def find_known_legal_entity_type(txt) -> [(str, str)]:
  stripped = txt.strip()
  if stripped in legal_entity_types:
    return [(stripped, legal_entity_types[stripped])]

  for t in legal_entity_types:
    if stripped == legal_entity_types[t]:
      return [(t, stripped)]

  found = []
  for r in legal_entity_types_re:

    match = r.match(stripped)
    if (match):
      normalized = legal_entity_types_re[r]
      found.append((normalized, legal_entity_types[normalized]))
  return found


def normalize_legal_entity_type(txt):
  knowns = find_known_legal_entity_type(txt.strip())
  if len(knowns) > 0:
    if len(knowns) == 1:
      k = knowns[0]
      return k[0], k[1], distance.get_jaro_distance(k[0], txt, winkler=True, scaling=0.1)
    else:
      finding = '', '', 0
      for k in knowns:
        d = distance.get_jaro_distance(k[0], txt, winkler=True, scaling=0.1)
        # print( k, d )
        if d > finding[2]:
          finding = k[0], k[1], d
      return finding
  else:
    return txt, '', 0.5


if __name__ == '__main__':
  print(r_group(r_capitalized_ru, 'alias'))
  pass
