#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import json
import os
# import json
import pickle
import unittest

from bson import json_util

from analyser.contract_parser import ContractParser, ContractDocument
from analyser.contract_patterns import ContractPatternFactory
from analyser.documents import TextMap
from analyser.legal_docs import DocumentJson
from analyser.ml_tools import SemanticTag
from analyser.parsing import AuditContext

# 5ded4e284ddc27bcf92dd6cf
# 5ded4e284ddc27bcf92dd6ce
class TestJsonExport(unittest.TestCase):

  def _get_doc(self) -> (ContractDocument, ContractPatternFactory):
    pth = os.path.dirname(__file__)
    with open(pth + '/2. Договор по благ-ти Радуга.docx.pickle', 'rb') as handle:
      doc = pickle.load(handle)

    with open(pth + '/contract_pattern_factory.pickle', 'rb') as handle:
      factory = pickle.load(handle)

    # self.assertEqual(2637, doc.embeddings.shape[-2])
    self.assertEqual(1024, doc.embeddings.shape[-1])
    # print(doc._normal_text)
    return doc, factory

  def _get_doc_factory_ctx(self):
    doc, factory = self._get_doc()

    ctx = ContractParser(embedder={} )
    ctx.verbosity_level = 3

    return doc, factory, ctx

  def print_semantic_tag(self, tag: SemanticTag, map: TextMap):
    print('print_semantic_tag:', tag, f"[{map.text_range(tag.span)}]")

  def test_to_json(self):
    doc, factory, ctx = self._get_doc_factory_ctx()

    doc.__dict__['number'] = None  # hack for old pickles
    doc.__dict__['date'] = None  # hack for old pickles
    doc.__dict__['warnings'] = []  # hack for old pickles

    actx = AuditContext()
    ctx.find_attributes(doc, actx)
    json_struct = DocumentJson(doc)
    _j = json_struct.dumps()
    print(_j)
    # TODO: compare with file

  def test_from_json(self):
    doc, factory, ctx = self._get_doc_factory_ctx()

    doc.__dict__['number'] = None  # hack for old pickles
    doc.__dict__['date'] = None  # hack for old pickles
    doc.__dict__['warnings'] = []  # hack for old pickles
    actx = AuditContext()
    ctx.find_attributes(doc, actx)
    json_struct = DocumentJson(doc)
    json_string = json.dumps(json_struct.__dict__, indent=4, ensure_ascii=False, default=json_util.default)

    restored: DocumentJson = DocumentJson.from_json_str(json_string)
    for key in restored.__dict__:
      print(key)
      self.assertIn(key, json_struct.__dict__.keys())

    for key in restored.attributes:
      self.assertIn(key, json_struct.attributes.keys())

    for key in json_struct.attributes:
      self.assertIn(key, restored.attributes.keys())

    # self.assertDictEqual(json_struct.attributes, restored.attributes)

    # TODO: compare with file


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
