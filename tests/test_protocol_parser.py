#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import os
import pickle
import re
import unittest

from analyser.contract_agents import ORG_LEVELS_re
from analyser.contract_parser import ContractDocument
from analyser.contract_patterns import ContractPatternFactory
from analyser.legal_docs import LegalDocument
from analyser.protocol_parser import find_protocol_org, find_org_structural_level, protocol_votes_re
from analyser.structures import OrgStructuralLevel


class TestProtocolParser(unittest.TestCase):

  def get_doc(self, fn) -> (LegalDocument, ContractPatternFactory):
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
    suff = ' ' * 300

    txt = '''Протокол № 3/2019 Проведения итогов заочного голосования Совета директоров Общества с ограниченной ответственностью «Технологический центр «Бажен» (далее – ООО «Технологический центр «Бажен») г. Санкт-Петербург Дата составления протокола «__» _______ 2019 года
    Дата окончания приема бюллетеней для голосования членов Совета директоров «___»__________ 2019 года.
    ''' + suff
    doc = ContractDocument(txt)
    doc.parse()

    tags = find_protocol_org(doc)
    self.assertEqual('Технологический центр «Бажен»', tags[0].value)
    self.assertEqual('Общество с ограниченной ответственностью', tags[1].value)

  def test_find_protocol_org_2(self):
    doc = self.get_doc('Протокол_СД_ 3.docx.pickle')
    print(doc[0:200].text)
    tags = find_protocol_org(doc)
    self.assertEqual('Технологический центр «Бажен»', tags[0].value)
    self.assertEqual('Общество с ограниченной ответственностью', tags[1].value)

  def test_ORG_LEVELS_re(self):
    suff = ' ' * 300
    t = '''
    ПРОТОКОЛ
заседания Совета директоров ООО «Газпромнефть- Корпоративные продажи» (далее – ООО «Газпромнефть- Корпоративные продажи» или «Общество»)
Место проведения заседания:
''' + suff
    r = re.compile(ORG_LEVELS_re, re.MULTILINE | re.IGNORECASE | re.UNICODE)
    x = r.search(t)
    self.assertEqual('Совета директоров', x['org_structural_level'])

  def test_find_org_structural_level(self):
    t = '''
    ПРОТОКОЛ \
    заседания Совета директоров ООО «Газпромнефть - Внеземная Любофьи» (далее – ООО «Газпромнефть-ВНЛ» или «Общество»)\
    Место проведения заседания:
    ''' + ' ' * 900
    doc = LegalDocument(t)
    doc.parse()

    tags = list(find_org_structural_level(doc))
    self.assertEqual(OrgStructuralLevel.BoardOfDirectors, tags[0].value)

  def test_find_org_structural_level_2(self):
    t = '''
    ПРОТОКОЛ ночного заседания Правления общества ООО «Газпромнефть - Внеземная Любофь» (далее – ООО «Газпромнефть- ВНЛ» или «Общество»)\
    Место проведения заседания:
    ''' + ' ' * 900
    doc = LegalDocument(t)
    doc.parse()

    tags = list(find_org_structural_level(doc))
    self.assertEqual(OrgStructuralLevel.BoardOfCompany, tags[0].value)

  def test_find_protocol_votes(self):
    doc = self.get_doc('Протокол_СД_ 3.docx.pickle')
    x = protocol_votes_re.search(doc.text)

    # for f in x:
    print(doc.text[x.span()[0]:x.span()[1]])

  def test_find_protocol_votes_2(self):
    t = '''
Предварительно утвердить годовой отчет Общества за 2017 год.
Итоги голосования:
 «ЗА»              8;
«ПРОТИВ»        нет;
«ВОЗДЕРЖАЛСЯ»  нет.
РЕШЕНИЕ ПРИНЯТО.
Решение, принятое по первому вопросу повестки дня:
Предварительно утвердить годовой отчет Общества за 2017 год.'''

    doc = LegalDocument(t)
    doc.parse()

    x = protocol_votes_re.search(doc.text)

    match = doc.text[x.span()[0]:x.span()[1]]
    print(f'[{match}]')


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
