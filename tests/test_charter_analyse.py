#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from analyser.charter_parser import CharterDocument
from analyser.runner import *


class TestAnalyse(unittest.TestCase):

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_get_org_names(self):
    parser = CharterParser()

    audit_id = next(get_audits())['_id']
    docs = get_docs_by_audit_id(audit_id, kind='CHARTER')

    for db_document in docs:
      print(db_document['filename'])

      parsed_p_json = db_document['parse']
      charter: CharterDocument = join_paragraphs(parsed_p_json, doc_id=db_document['_id'])

      parser.find_org_date_number(charter, AuditContext())

      for tag in charter.get_tags():
        print(tag)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
