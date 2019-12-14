#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from analyser.charter_parser import CharterParser, CharterDocument
from analyser.contract_parser import ContractAnlysingContext, ContractDocument
from analyser.legal_docs import *
from analyser.legal_docs import _embedd_large
from analyser.parsing import AuditContext
from tests.test_utilits import FakeEmbedder


class LegalDocumentTestCase(unittest.TestCase):

  def test_embedd_large(self):
    point1 = [1, 6, 4]
    emb = FakeEmbedder(point1)

    ld = LegalDocument('a b c d e f g h').parse()
    print(ld.tokens)

    _embedd_large(ld.tokens_map_norm, emb, 5)

    # print(ld.embeddings)
    print(ld.tokens)

  def test_parse(self):
    d = LegalDocument("a")
    d.parse()
    print(d.tokens)
    self.assertEqual(1, len(d.tokens))

  def test_analyze_contract_0(self):
    point1 = [1, 6, 4]
    emb = FakeEmbedder(point1)

    ctx = ContractAnlysingContext(emb)
    contract = ContractDocument("1. ЮРИДИЧЕСКИЙ содержание 4.")
    contract.parse()
    actx = AuditContext()
    ctx.find_org_date_number(contract, actx)
    ctx.find_attributes(contract, actx)

    ctx._logstep("analyze_contract")

  def test_charter_parser(self):
    # from renderer import SilentRenderer
    point1 = [1, 6, 4]
    emb = FakeEmbedder(point1)
    legal_doc = LegalDocument("1. ЮРИДИЧЕСКИЙ содержание 4.").parse()
    charter = CharterDocument().parse()
    charter += legal_doc
    charter_parser = CharterParser(emb, emb)

    charter_parser.analyse(charter)
    # ctx.analyze_charter("1. ЮРИДИЧЕСКИЙ содержание 4.")
    # ctx._logstep("analyze_charter")


if __name__ == '__main__':
  unittest.main()
