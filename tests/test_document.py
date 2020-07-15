#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

import numpy as np

from analyser.charter_parser import CharterParser, CharterDocument
from analyser.contract_parser import ContractParser, ContractDocument
from analyser.legal_docs import LegalDocument, DocumentJson
from analyser.parsing import AuditContext
from tests.test_utilits import FakeEmbedder


class LegalDocumentTestCase(unittest.TestCase):

  def test_embedd_large(self):
    point1 = [1, 6, 4]
    emb = FakeEmbedder(point1)

    ld = LegalDocument('a b c d e f g h').parse()
    print(ld.tokens)

    emb.embedd_large(ld.tokens_map_norm, 5)

    # print(ld.embeddings)
    print(ld.tokens)

  def test_parse(self):
    d = LegalDocument("a")
    d.parse()
    print(d.tokens)
    self.assertEqual(1, len(d.tokens))

  def test_analyze_contract_0(self):
    point1 = np.random.rand(1024)
    emb = FakeEmbedder(point1)

    ctx = ContractParser(emb)
    contract = ContractDocument("1. ЮРИДИЧЕСКИЙ содержание 4.")
    contract.parse()
    actx = AuditContext()
    ctx.find_org_date_number(contract, actx)
    ctx.find_attributes(contract, actx)

    ctx._logstep("analyze_contract")

  def test_checksum(self):
    d0 = LegalDocument("aasasasasas aasasas").parse()
    d = LegalDocument("aasasasasas aasasas").parse()
    d1 = LegalDocument("bgfgjfgdfg dfgj d gj").parse()

    print(d.checksum)
    self.assertIsNotNone(d.checksum)
    self.assertTrue(d.checksum != 0)

    self.assertEqual(d0.checksum, d.checksum)
    self.assertNotEqual(d0.checksum, d1.checksum)

    self.assertEqual(d0.checksum, DocumentJson(d0).checksum)
    self.assertEqual(d.checksum, DocumentJson(d).checksum)
    self.assertEqual(d1.checksum, DocumentJson(d1).checksum)

    self.assertEqual(d1.checksum, d1.checksum)
    self.assertEqual(d1.get_checksum(), d1.get_checksum())

  def test_charter_parser(self):
    # from renderer import SilentRenderer
    point1 = [1, 6, 4]
    emb = FakeEmbedder(point1)
    legal_doc = LegalDocument("1. ЮРИДИЧЕСКИЙ содержание 4.").parse()
    charter = CharterDocument().parse()
    charter += legal_doc
    charter_parser = CharterParser(emb, emb)
    actx = AuditContext()
    charter_parser.find_org_date_number(charter, actx)
    charter_parser.find_attributes(charter, actx)

    print(charter.warnings)
    # ctx.analyze_charter("1. ЮРИДИЧЕСКИЙ содержание 4.")
    # ctx._logstep("analyze_charter")


if __name__ == '__main__':
  unittest.main()
