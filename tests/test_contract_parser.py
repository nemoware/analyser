#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import pickle
import unittest

from contract_parser import ContractAnlysingContext, ContractDocument
from contract_patterns import ContractPatternFactory
from documents import TextMap
from ml_tools import SemanticTag


class TestContractParser(unittest.TestCase):

  def get_doc(self) -> (ContractDocument, ContractPatternFactory):
    with open('doc_4.pickle', 'rb') as handle:
      doc = pickle.load(handle)

    with open('contract_pattern_factory.pickle', 'rb') as handle:
      factory = pickle.load(handle)


    #self.assertEqual(2637, doc.embeddings.shape[-2])
    self.assertEqual(1024, doc.embeddings.shape[-1])
    print (doc._normal_text)
    return doc, factory

  def print_semantic_tag(self, tag: SemanticTag, map: TextMap):
    print(tag,  f"[{map.text_range(tag.span)}]")

  def test_find_contract_value_NEW(self):
    doc, factory = self.get_doc()


    ctx = ContractAnlysingContext(embedder={}, renderer=None, pattern_factory=factory)
    ctx.verbosity_level = 3

    ctx.sections_finder.find_sections(doc, ctx.pattern_factory, ctx.pattern_factory.headlines,
                                      headline_patterns_prefix='headline.')
    # ----------------------------------------
    values = ctx.find_contract_value_NEW(doc)
    # ----------------------------------------

    for v in values:
      self.print_semantic_tag(v.sign, doc.tokens_map)
      self.print_semantic_tag(v.value, doc.tokens_map)
      self.print_semantic_tag(v.currency, doc.tokens_map)

      print(v.sign, v.currency, v.value)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
