#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from contract_agents import *
from contract_augmentation import *
from documents import MarkedDoc


def n(x):
  return normalize_contract(x)


class TestAugm(unittest.TestCase):

  def test_remove_char(self):
    doc = MarkedDoc(['12345', '12345', '12345', '12345', '12345'], [1, 2, 3, 4, 5])

    augment_dropout_chars_d(doc, 0.5)
    print(doc.tokens)

  def test_remove_tokens(self):
    doc = MarkedDoc(['12345', '12345', '12345', '12345', '12345'], [1, 2, 3, 4, 5])

    augment_dropout_words_d(doc, 0.5)
    print(doc.tokens, doc.categories_vector)

  def test_alter_case(self):
    doc = MarkedDoc(['aaaa', 'AAA', 'bbb', 'BBB', 'cccc'], [1, 2, 3, 4, 5])

    augment_alter_case_d(doc, 1)
    print(doc.tokens, doc.categories_vector)

  def test_drop_punkt(self):
    doc = MarkedDoc([',', 'AAA', '"', 'BBB', '.'], [1, 2, 3, 4, 5])

    augment_dropout_punctuation_d(doc, 1)
    print(doc.tokens, doc.categories_vector)

  def test_concat (self):
    doc = MarkedDoc([',', 'AAA', '"', 'BBB', '.'], [1, 2, 3, 4, 5])
    doc2 = MarkedDoc(['12345', '12345', '12345', '12345', '12345'], [1, 2, 3, 4, 5])
    doc.concat(doc2)
    print(doc.tokens, doc.categories_vector)

  def test_trim(self):
    doc = MarkedDoc(['12345', '12345', '12345', '12345', '12345'], [1, 2, 3, 4, 5])
    augment_trim(doc, 1)
    print('augment_trim', doc.tokens, doc.categories_vector)
    self.assertGreaterEqual(doc.get_len(), 3)

    for c in range(5):

      augment_trim(doc, 1)
      print('augment_trim', doc.tokens, doc.categories_vector)



unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
