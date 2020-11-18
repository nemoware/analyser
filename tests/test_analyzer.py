#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from bson import ObjectId

from analyser.finalizer import get_doc_by_id, get_audit_by_id
from analyser.log import logger
from analyser.parsing import AuditContext
from analyser.persistence import DbJsonDoc
from analyser.runner import BaseProcessor, document_processors, CONTRACT, PROTOCOL, CHARTER
from integration.db import get_mongodb_connection


class AnalyzerTestCase(unittest.TestCase):

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_analyze_contract(self):
    processor: BaseProcessor = document_processors[CONTRACT]
    doc = get_doc_by_id(ObjectId('5ded004e4ddc27bcf92dd47c'))
    if doc is None:
      raise RuntimeError("fix unit test please")

    audit = get_audit_by_id(doc['auditId'])

    jdoc = DbJsonDoc(doc)
    logger.info(f'......pre-processing {jdoc._id}')
    ctx = AuditContext()
    processor.preprocess(jdoc, context=ctx)
    processor.process(jdoc, audit, ctx)

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_analyze_protocol(self):
    processor: BaseProcessor = document_processors[PROTOCOL]
    doc = get_doc_by_id(ObjectId('5e5de70b01c6c73c19eebd35'))
    if doc is None:
      raise RuntimeError("fix unit test please")

    audit = get_audit_by_id(doc['auditId'])

    jdoc = DbJsonDoc(doc)
    logger.info(f'......pre-processing {jdoc._id}')
    ctx = AuditContext()
    processor.preprocess(jdoc, context=ctx)
    processor.process(jdoc, audit, ctx)

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_analyze_charter(self):
    processor: BaseProcessor = document_processors[CHARTER]
    doc = get_doc_by_id(ObjectId('5e5de70d01c6c73c19eebd48'))
    if doc is None:
      raise RuntimeError("fix unit test please")

    audit = get_audit_by_id(doc['auditId'])

    jdoc = DbJsonDoc(doc)
    logger.info(f'......pre-processing {jdoc._id}')
    ctx = AuditContext()
    processor.preprocess(jdoc, context=ctx)
    processor.process(jdoc, audit, ctx)


#


if __name__ == '__main__':
  unittest.main()
