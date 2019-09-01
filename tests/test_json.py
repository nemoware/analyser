#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import pickle
import unittest
import warnings

from contract_parser import ContractAnlysingContext, ContractDocument
from contract_patterns import ContractPatternFactory
from documents import TextMap
from legal_docs import LegalDocument, DocumentJson
from ml_tools import SemanticTag

import json
class TestContractParser(unittest.TestCase):

  def _get_doc(self) -> (ContractDocument, ContractPatternFactory):
    with open('doc_4.pickle', 'rb') as handle:
      doc = pickle.load(handle)

    with open('contract_pattern_factory.pickle', 'rb') as handle:
      factory = pickle.load(handle)

    # self.assertEqual(2637, doc.embeddings.shape[-2])
    self.assertEqual(1024, doc.embeddings.shape[-1])
    # print(doc._normal_text)
    return doc, factory

  def _get_doc_factory_ctx(self):

    doc, factory = self._get_doc()

    ctx = ContractAnlysingContext(embedder={}, renderer=None, pattern_factory=factory)
    ctx.verbosity_level = 3
    ctx.sections_finder.find_sections(doc, ctx.pattern_factory, ctx.pattern_factory.headlines,
                                      headline_patterns_prefix='headline.')
    return doc, factory, ctx

  def print_semantic_tag(self, tag: SemanticTag, map: TextMap):
    print('print_semantic_tag:', tag, f"[{map.text_range(tag.span)}]")



  def test_find_contract_subject_region_in_subj_section(self):
    doc, factory, ctx = self._get_doc_factory_ctx()

    ctx.analyze_contract_doc(doc)
    json_struct = DocumentJson(doc)
    _j = json.dumps(json_struct.__dict__, indent=4, ensure_ascii=False, default=lambda o: '<not serializable>')
    # TODO: compare with file

unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
