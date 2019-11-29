#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from analyser.charter_parser import find_charter_org
from analyser.runner import *
from tests.test_utilits import FakeEmbedder


class TestAnalyse(unittest.TestCase):

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_get_org_name(self):
    embedder = FakeEmbedder([1, 2, 3, 4])
    parser = CharterParser(embedder, embedder)

    audit_id = next(get_audits())['_id']
    docs = get_docs_by_audit_id(audit_id, kind='CHARTER')
    db_document = next(docs)
    db_document = next(docs)
    db_document = next(docs)
    print(db_document['filename'])

    parsed_p_json = db_document['parse']
    charter = join_paragraphs(parsed_p_json, doc_id=db_document['_id'])

    parser.ebmedd(charter)
    parser.analyse(charter)

    tags = find_charter_org(charter)

    for tag in tags:
      print(tag)

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_get_org_names(self):
    embedder = FakeEmbedder([1, 2, 3, 4])
    parser = CharterParser(embedder, embedder)

    audit_id = next(get_audits())['_id']
    docs = get_docs_by_audit_id(audit_id, kind='CHARTER')

    for db_document in docs:
      print(db_document['filename'])

      parsed_p_json = db_document['parse']
      charter = join_paragraphs(parsed_p_json, doc_id=db_document['_id'])

      parser.ebmedd(charter)
      parser.analyse(charter)

      tags = find_charter_org(charter)

      for tag in tags:
        print(tag)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
