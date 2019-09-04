#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import pickle
import unittest
import warnings

from contract_parser import ContractAnlysingContext, ContractDocument
from contract_patterns import ContractPatternFactory
from documents import TextMap
from legal_docs import LegalDocument
from ml_tools import SemanticTag


class TestContractParser(unittest.TestCase):

  def get_doc(self) -> (ContractDocument, ContractPatternFactory):
    with open('doc_4.pickle', 'rb') as handle:
      doc = pickle.load(handle)

    with open('contract_pattern_factory.pickle', 'rb') as handle:
      factory = pickle.load(handle)

    # self.assertEqual(2637, doc.embeddings.shape[-2])
    self.assertEqual(1024, doc.embeddings.shape[-1])
    # print(doc._normal_text)
    return doc, factory

  def get_doc_factory_ctx(self):

    doc, factory = self.get_doc()

    ctx = ContractAnlysingContext(embedder={}, renderer=None, pattern_factory=factory)
    ctx.verbosity_level = 3
    ctx.sections_finder.find_sections(doc, ctx.pattern_factory, ctx.pattern_factory.headlines,
                                      headline_patterns_prefix='headline.')
    return doc, factory, ctx

  def print_semantic_tag(self, tag: SemanticTag, map: TextMap):
    print('print_semantic_tag:', tag, f"[{map.text_range(tag.span)}]", tag.parent)

  def test_find_contract_value(self):
    doc, factory = self.get_doc()

    ctx = ContractAnlysingContext(embedder={}, renderer=None, pattern_factory=factory)
    ctx.verbosity_level = 3
    ctx.sections_finder.find_sections(doc, ctx.pattern_factory, ctx.pattern_factory.headlines,
                                      headline_patterns_prefix='headline.')
    # ----------------------------------------
    values = ctx.find_contract_value_NEW(doc)
    # ----------------------------------------

    self.assertEqual(1, len(values))
    v = values[0]

    value = SemanticTag.find_by_kind(v, 'value')
    sign = SemanticTag.find_by_kind(v, 'sign')
    currency = SemanticTag.find_by_kind(v, 'currency')


    self.print_semantic_tag(sign, doc.tokens_map)
    self.print_semantic_tag(value, doc.tokens_map)
    self.print_semantic_tag(currency, doc.tokens_map)

    self.assertEqual(0, sign.value)
    self.assertEqual(80000, value.value)
    self.assertEqual('RUB', currency.value)


  def test_find_contract_subject(self):
    warnings.warn("use find_contract_subject_region", DeprecationWarning)
    doc, factory, ctx = self.get_doc_factory_ctx()
    # ----------------------------------------
    subjects = ctx.find_contract_subject(doc)
    # ----------------------------------------
    print("SUBJECTS:")
    for subj in subjects:
      print(subj)

  def test_find_contract_subject_region_in_subj_section(self):
    doc, factory, ctx = self.get_doc_factory_ctx()

    subj_section = doc.sections['subj']
    section: LegalDocument = subj_section.body
    # ----------------------------------------
    result = ctx.find_contract_subject_regions(section)
    # ---------------------

    self.print_semantic_tag(result, doc.tokens_map)
    self.assertEqual('1.1 Благотворитель оплачивает следующий счет, выставленный на Благополучателя:',
                     doc.tokens_map.text_range(result.span).strip())

  def test_find_contract_subject_region_in_doc_head(self):
    doc, factory, ctx = self.get_doc_factory_ctx()

    section = doc.subdoc_slice(slice(0, 1500))
    denominator = 0.7

    # subj_section = doc.sections['subj']
    # section: LegalDocument = subj_section.body
    # ----------------------------------------
    result = ctx.find_contract_subject_regions(section, denominator)
    # ---------------------

    self.print_semantic_tag(result, doc.tokens_map)
    self.assertEqual('1.1 Благотворитель оплачивает следующий счет, выставленный на Благополучателя:',
                     doc.tokens_map.text_range(result.span).strip())


  def test_find_contract_subject_region(self):
    doc, factory, ctx = self.get_doc_factory_ctx()


    # ----------------------------------------
    result = ctx.find_contract_subject_region(doc)
    # ---------------------

    self.print_semantic_tag(result, doc.tokens_map)
    self.assertEqual('1.1 Благотворитель оплачивает следующий счет, выставленный на Благополучателя:',
                     doc.tokens_map.text_range(result.span).strip())



unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
