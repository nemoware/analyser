#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from analyser.hyperparams import HyperParameters
from analyser.text_tools import compare_masked_strings
from gpn.gpn import subsidiaries

__version__ = "1.6.7"
print(f'Nemoware Analyser v{__version__}')


def all_do_names():
  for s in subsidiaries:
    for alias in s['aliases'] + [s['_id']]:
      yield alias


def estimate_subsidiary_name_match_min_jaro_similarity():
  top_similarity = 0

  for name1 in all_do_names():
    for name2 in all_do_names():
      name1 = name1.replace('»', '').replace('«', '')
      name2 = name2.replace('»', '').replace('«', '')

      if name1.lower() != name2.lower():

        similarity = compare_masked_strings(name1, name2, [])
        if similarity > top_similarity:
          top_similarity = similarity
          print(top_similarity, name1, name2)

  return top_similarity


HyperParameters.subsidiary_name_match_min_jaro_similarity = estimate_subsidiary_name_match_min_jaro_similarity()
print('HyperParameters.subsidiary_name_match_min_jaro_similarity',
      HyperParameters.subsidiary_name_match_min_jaro_similarity)
