#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import unittest
# pprint library is used to make the output look more pretty
from pprint import pprint

from pymongo import MongoClient

from integration.word_document_parser import WordDocParser
from legal_docs import LegalDocument, Paragraph
from ml_tools import SemanticTag
from text_normalize import normalize_text, replacements_regex


class TestContractParser(unittest.TestCase):

  def n(self, txt):
    return normalize_text(txt, replacements_regex)






  def test_mongodb_connection(self):
    # connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
    client = MongoClient('mongodb://localhost:27017/')
    db = client['docsdatabase']
    # Issue the serverStatus command and print the results
    serverStatusResult = db.command("serverStatus")
    pprint(serverStatusResult)

  def test_doc_parser(self):
    FILENAME = "/Users/artem/work/nemo/goil/IN/Другие договоры/Договор Формула.docx"

    wp = WordDocParser()
    res = wp.read_doc(FILENAME)

    doc: LegalDocument = LegalDocument('')
    doc.parse()

    last = 0
    for p in res['paragraphs']:
      header_text = p['paragraphHeader']['text'] + '\n'
      body_text = p['paragraphBody']['text'] + '\n'

      header = LegalDocument(header_text)
      header.parse()
      # self.assertEqual(self.n(header_text), header.text)

      doc += header
      headerspan = (last, len(doc.tokens_map))
      print(headerspan)
      last = len(doc.tokens_map)

      body = LegalDocument(body_text)
      body.parse()
      doc += body
      bodyspan = (last, len(doc.tokens_map))

      header_tag = SemanticTag('headline', header_text, headerspan)
      body_tag = SemanticTag('paragraphBody', None, bodyspan)

      print(header_tag)
      # print(body_tag)
      para = Paragraph(header_tag, body_tag)
      doc.paragraphs.append(para)
      last = len(doc.tokens_map)

      h_subdoc = doc.subdoc_slice(para.header.as_slice())
      b_subdoc = doc.subdoc_slice(para.body.as_slice())
      # self.assertEqual(self.n(header_text), h_subdoc.text)
      # self.assertEqual(self.n(body_text), b_subdoc.text)

    print('-' * 100)
    print(doc.text)

    headers = [doc.subdoc_slice(p.header.as_slice()) for p in doc.paragraphs]
    print('-' * 100)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
