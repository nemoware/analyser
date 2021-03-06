#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

# os.environ['GPN_DB_HOST']='192.168.10.36'

import unittest

from bson import ObjectId

from analyser.finalizer import get_doc_by_id, get_audit_by_id
from analyser.log import logger
from analyser.parsing import AuditContext
from analyser.persistence import DbJsonDoc
from analyser.runner import BaseProcessor, document_processors, CONTRACT, PROTOCOL, CHARTER
from integration.db import get_mongodb_connection


class AnalyzerTestCase(unittest.TestCase):
  @unittest.skip
  def test_analyse_acontract(self):

    doc = get_doc_by_id(ObjectId('5fdb213f542ce403c92b4530'))
    # _db_client = MongoClient(f'mongodb://192.168.10.36:27017/')
    # _db_client.server_info()

    # db = _db_client['gpn']

    # documents_collection = db['documents']

    # doc = documents_collection.find_one({"_id": ObjectId('5fdb213f542ce403c92b4530')} )
    # audit = db['audits'].find_one({'_id': doc['auditId']})
    audit = get_audit_by_id(doc['auditId'])
    jdoc = DbJsonDoc(doc)
    logger.info(f'......pre-processing {jdoc._id}')
    _audit_subsidiary: str = audit["subsidiary"]["name"]

    ctx = AuditContext(_audit_subsidiary)
    processor: BaseProcessor = document_processors[CONTRACT]
    processor.preprocess(jdoc, context=ctx)
    processor.process(jdoc, audit, ctx)
    print(jdoc)

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
