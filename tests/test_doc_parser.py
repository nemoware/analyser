#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from integration.word_document_parser import WordDocParser
from legal_docs import LegalDocument, Paragraph
from ml_tools import SemanticTag


class TestContractParser(unittest.TestCase):

  def test_doc_parser(self):
    FILENAME = "/Users/artem/work/nemo/goil/IN/Другие договоры/Договор Формула.docx"

    wp = WordDocParser()
    res = wp.read_doc(FILENAME)

    doc: LegalDocument = LegalDocument('')
    doc.parse()
    last = 0
    for p in res['paragraphs']:
      header = LegalDocument(p['paragraphHeader']['text'] + '\n')
      header.parse()
      doc += header
      headerspan = (last, len(doc.tokens_map))
      print(headerspan)
      last = len(doc.tokens_map)

      body = LegalDocument(p['paragraphBody']['text'] + '\n')
      body.parse()
      doc += body
      bodyspan = (last, len(doc.tokens_map))

      header_tag = SemanticTag('headline', p['paragraphHeader']['text'], headerspan)
      body_tag = SemanticTag('paragraphBody', None, bodyspan)

      print(header_tag)
      print(body_tag)
      para = Paragraph(header_tag, body_tag)
      doc.paragraphs.append(para)
      last = len(doc.tokens_map)

    print('-' * 100)
    print(doc.text)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
