#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from analyser.contract_agents import find_closest_org_name, ContractAgent
from analyser.text_tools import compare_masked_strings
from analyser.hyperparams import HyperParameters
from gpn.gpn import subsidiaries


class ContractAgentsTestCase(unittest.TestCase):

  def test_compare_masked_strings(self):
    similarity = compare_masked_strings('Газпромнефть -  МАбильная карта', 'Газпромнефть-Мобильная карта', [])
    print('similarity=', similarity)
    self.assertGreater(similarity, HyperParameters.subsidiary_name_match_min_jaro_similarity)

  def test_find_closest_org_name(self):

    mc = {
      "_id": "Газпромнефть-Мобильная карта",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Мобильная карта"
      ]
    }
    known_org_name, _ = find_closest_org_name([mc], 'Газпромнефть - МАбильная карта',
                                              HyperParameters.subsidiary_name_match_min_jaro_similarity)
    self.assertIsNotNone(known_org_name)

    known_org_name, _ = find_closest_org_name(subsidiaries, 'Газпромнефть - МАбильная карта',
                                              HyperParameters.subsidiary_name_match_min_jaro_similarity)
    self.assertIsNotNone(known_org_name)
    self.assertEqual(mc['_id'], known_org_name['_id'])

  def test_compare_masked_strings_1(self):
    s = compare_masked_strings('Многофункциональный комплекс «Лахта центр»',
                               'Многофункциональный комплекс «Лахта центр»', [])
    print(s)
    for s1 in subsidiaries:
      for name in s1['aliases']:
        s = compare_masked_strings(name, name, [])
        self.assertEqual(s, 1)

  def test_find_closest_org_name_solo(self):
    # s = compare_masked_strings('Многофункциональный комплекс «Лахта центр»',
    #                            'Многофункциональный комплекс «Лахта центр»', [])

    s = find_closest_org_name(subsidiaries, 'Многофункциональный комплекс «Лахта центр»', 0)
    print(s)
    # print(s)
    # for s1 in subsidiaries:
    #   for name in s1['aliases']:

  def test_find_closest_org_names_self(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity
    # finding self
    for s1 in subsidiaries:
      augmented = s1['_id']
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])

  def test_find_closest_org_uppercased(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity

    # finding uppercased
    for s1 in subsidiaries:
      augmented = s1['_id'].upper()
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])

  def test_find_closest_org_postfix(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity

    # finding uppercased
    for s1 in subsidiaries:
      augmented = s1['_id'] + ' x'
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])

  def test_find_closest_org_postfix_2(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity

    # finding uppercased
    for s1 in subsidiaries:
      augmented = s1['_id'] + ' 2'
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])
      print(known_org_name)

  def test_find_closest_org_prefix(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity

    # finding uppercased
    for s1 in subsidiaries:
      augmented = 'c' + s1['_id']
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])

  def test_find_closest_org_names_cut_begin(self):
    _threshold = 0.8

    for s1 in subsidiaries:
      augmented = s1['_id'][1:]
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])

  def test_ContractAgent_as_lsit(self):
    ca = ContractAgent()
    self.assertEqual(0, len(ca.as_list()))


if __name__ == '__main__':
  unittest.main()
