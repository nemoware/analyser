#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import os
import pickle
import unittest

from contract_parser import ContractDocument
from contract_patterns import ContractPatternFactory
from legal_docs import LegalDocument
from protocol_parser import find_protocol_org


class TestProtocolParser(unittest.TestCase):

  def get_doc(self, fn) -> (ContractDocument, ContractPatternFactory):
    pth = os.path.dirname(__file__)
    with open(os.path.join(pth, fn), 'rb') as handle:
      doc = pickle.load(handle)

    self.assertEqual(1024, doc.embeddings.shape[-1])

    return doc

  def test_load_picke(self):
    doc = self.get_doc('Протокол_СД_ 3.docx.pickle')
    doc: LegalDocument = doc
    for p in doc.paragraphs:
      print('😱 \t', doc.get_tag_text(p.header).strip(), '📂')

  def test_find_protocol_org_1(self):
    txt = '''Протокол № 3/2019 Проведения итогов заочного голосования Совета директоров Общества с ограниченной ответственностью «Технологический центр «Бажен» (далее – ООО «Технологический центр «Бажен») г. Санкт-Петербург Дата составления протокола «__» _______ 2019 года
    Дата окончания приема бюллетеней для голосования членов Совета директоров «___»__________ 2019 года.
    '''
    doc = ContractDocument(txt)
    doc.parse()

    tags = find_protocol_org(doc)
    self.assertEqual('Технологический центр «Бажен»', tags[0].value)
    self.assertEqual('Общества с ограниченной ответственностью', tags[1].value)

  def test_find_protocol_org_2(self):
    doc = self.get_doc('Протокол_СД_ 3.docx.pickle')
    print(doc[0:200].text)
    tags = find_protocol_org(doc)
    self.assertEqual('Технологический центр «Бажен»', tags[0].value)
    self.assertEqual('Общества с ограниченной ответственностью', tags[1].value)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
