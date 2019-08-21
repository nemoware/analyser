#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import pickle
import unittest

from contract_parser import ContractAnlysingContext


class TestContractParser(unittest.TestCase):

  def get_doc(self):
    with open('doc_1.pickle', 'rb') as handle:
      doc = pickle.load(handle)

    with open('contract_pattern_factory.pickle', 'rb') as handle:
      factory = pickle.load(handle)

    self.assertEqual(2637, doc.embeddings.shape[-2])
    self.assertEqual(1024, doc.embeddings.shape[-1])
    return doc, factory

  def test_some(self):
    doc, factory = self.get_doc()
    ctx = ContractAnlysingContext(embedder={}, renderer=None, pattern_factory=factory)
    ctx.verbosity_level = 3

    ctx.sections_finder.find_sections(doc, ctx.pattern_factory, ctx.pattern_factory.headlines, headline_patterns_prefix='headline.')
    values = ctx.find_contract_value_NEW(doc)

    for v in values:
      print(v)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
