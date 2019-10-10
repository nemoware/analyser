#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import os
import pickle
import unittest

from contract_parser import ContractAnlysingContext, ContractDocument
from contract_patterns import ContractPatternFactory
from legal_docs import LegalDocument


class TestContractParser(unittest.TestCase):

  def get_doc(self, fn) -> (ContractDocument, ContractPatternFactory):
    pth = os.path.dirname(__file__)
    with open(os.path.join(pth, fn), 'rb') as handle:
      doc = pickle.load(handle)

    with open(pth + '/contract_pattern_factory.pickle', 'rb') as handle:
      factory = pickle.load(handle)

    self.assertEqual(1024, doc.embeddings.shape[-1])

    return doc, factory

  def test_load_picke(self):
    doc, factory, ctx = self._get_doc_factory_ctx('Протокол_СД_ 3.docx.pickle')
    doc:LegalDocument  = doc
    for p in doc.paragraphs:
      print('😱 \t', doc.get_tag_text(p.header).strip(),'📂')


  def _get_doc_factory_ctx(self, fn):
    doc, factory = self.get_doc(fn)

    ctx = ContractAnlysingContext(embedder={}, renderer=None, pattern_factory=factory)
    ctx.verbosity_level = 3
    ctx.sections_finder.find_sections(doc, ctx.pattern_factory, ctx.pattern_factory.headlines,
                                      headline_patterns_prefix='headline.')
    return doc, factory, ctx



unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
