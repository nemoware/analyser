#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from integration.word_document_parser import WordDocParser
from legal_docs import LegalDocument


class TestContractParser(unittest.TestCase):

  def test_doc_parser(self):
    FILENAME = "/Users/artem/work/nemo/goil/IN/Другие договоры/Договор Формула.docx"

    wp = WordDocParser()
    res = wp.read_doc(FILENAME)

    doc: LegalDocument = LegalDocument('')
    doc.parse()
    last=0
    for p in res['paragraphs']:
      header = LegalDocument(p['paragraphHeader']['text'] + '\n')
      header.parse()
      doc += header
      span = (last, len(doc.tokens_map))
      print(span)


      body = LegalDocument(p['paragraphBody']['text'] + '\n')
      body.parse()
      doc += body
      last = len(doc.tokens_map)

    print('-' * 100)
    print(doc.text)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
