#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import os
import pickle
import unittest

from analyser.contract_parser import ContractParser, ContractDocument, nn_find_contract_value
from analyser.contract_patterns import ContractPatternFactory
from analyser.documents import TextMap
from analyser.ml_tools import SemanticTag
from analyser.parsing import AuditContext
from analyser.protocol_parser import ProtocolDocument
from analyser.structures import ContractTags
from tf_support.tf_subject_model import nn_predict


class TestContractParser(unittest.TestCase):

  def get_doc(self, fn) -> (ContractDocument, ContractPatternFactory):
    pth = os.path.dirname(__file__)
    with open(os.path.join(pth, fn), 'rb') as handle:
      doc = pickle.load(handle)

    with open(pth + '/contract_pattern_factory.pickle', 'rb') as handle:
      factory = pickle.load(handle)

    self.assertEqual(1024, doc.embeddings.shape[-1])

    return doc, factory

  def test_find_value_sign_currency(self):

    contract, factory, ctx = self._get_doc_factory_ctx('Договор _2_.docx.pickle')
    contract.__dict__['warnings'] = []  # hack for old pickles
    semantic_map, subj_1hot = nn_predict(ctx.subject_prediction_model, contract)
    r = nn_find_contract_value(contract, semantic_map)
    # r = ctx.find_contract_value_NEW(doc)
    print(len(r))
    for group in r:
      for tag in group.as_list():
        print(tag)

    self.assertLessEqual(len(r), 2)
    # print(r)
    #
    # value = SemanticTag.find_by_kind(r[0], 'value')
    # sign = SemanticTag.find_by_kind(r[0], 'sign')
    # currency = SemanticTag.find_by_kind(r[0], 'currency')
    #
    # print(doc.tokens_map_norm.text_range(value.span))
    # self.assertEqual(price, value.value, text)
    # self.assertEqual(currency_exp, currency.value)
    # print(f'{value}, {sign}, {currency}')

  def _get_doc_factory_ctx(self, fn='2. Договор по благ-ти Радуга.docx.pickle'):
    doc, factory = self.get_doc(fn)

    ctx = ContractParser(embedder={})
    ctx.verbosity_level = 3

    return doc, factory, ctx

  def test_ProtocolDocument3_init(self):
    doc, __ = self.get_doc('2. Договор по благ-ти Радуга.docx.pickle')
    pr = ProtocolDocument(doc)
    print(pr.__dict__['date'])

  def test_contract_analyze(self):
    doc, factory, ctx = self._get_doc_factory_ctx()
    doc.__dict__['number'] = None  # hack for old pickles
    doc.__dict__['date'] = None  # hack for old pickles

    ctx.find_attributes(doc, AuditContext())
    tags: [SemanticTag] = doc.get_tags()

    _tag = SemanticTag.find_by_kind(tags, ContractTags.Value.display_string)
    quote = doc.tokens_map.text_range(_tag.span)
    self.assertEqual('80000,00', quote)

    _tag = SemanticTag.find_by_kind(tags, ContractTags.Currency.display_string)
    quote = doc.tokens_map.text_range(_tag.span)
    self.assertEqual('рублей', quote)

  def print_semantic_tag(self, tag: SemanticTag, map: TextMap):
    print('print_semantic_tag:', tag, f"[{map.text_range(tag.span)}]", tag.parent)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
