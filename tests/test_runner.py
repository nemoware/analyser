#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from analyser.runner import *

SKIP_TF = True


def get_runner_instance_no_embedder() -> Runner:
  if TestRunner.default_no_tf_instance is None:
    TestRunner.default_no_tf_instance = Runner(init_embedder=False)
  return TestRunner.default_no_tf_instance


@unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
class TestRunner(unittest.TestCase):
  default_no_tf_instance: Runner = None

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_get_audits(self):
    aa = get_audits()
    for a in aa:
      print(a['_id'])

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_get_docs_by_audit_id(self):
    audit_id = next(get_audits())['_id']
    docs = get_docs_by_audit_id(audit_id, kind='PROTOCOL')
    for a in docs:
      print(a['_id'], a['filename'])

  def _get_doc_from_db(self, kind):
    audits = get_mongodb_connection()['audits'].find().sort([("createDate", pymongo.ASCENDING)]).limit(1)
    for audit in audits:
      for doc in get_docs_by_audit_id(audit['_id'], kind=kind, state=3).limit(1):
        print(doc['_id'])
        yield doc

  def _preprocess_single_doc(self, kind):
    for doc in self._get_doc_from_db(kind):
      processor = document_processors.get(kind, None)
      processor.preprocess(doc)

  # @unittest.skipIf(SKIP_TF, "requires TF")

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_preprocess_single_protocol(self):
    self._preprocess_single_doc('PROTOCOL')

  @unittest.skipIf(get_mongodb_connection() is None is None, "requires mongo")
  def test_preprocess_single_contract(self):
    self._preprocess_single_doc('CONTRACT')

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_process_contracts_phase_1(self):
    runner = Runner.get_instance()

    audit_id = next(get_audits())['_id']
    docs = get_docs_by_audit_id(audit_id, kind='CONTRACT')
    for doc in docs:
      processor = document_processors.get('CONTRACT', None)
      processor.preprocess(doc)

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_process_charters_phase_1(self):
    runner = Runner.get_instance()

    audit_id = next(get_audits())['_id']
    docs = get_docs_by_audit_id(audit_id, kind='CHARTER')
    for doc in docs:
      processor = document_processors.get('CHARTER', None)
      processor.preprocess(doc)

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_process_protocols_phase_1(self):
    runner = get_runner_instance_no_embedder()

    for audit in get_audits():
      audit_id = audit['_id']
      docs = get_docs_by_audit_id(audit_id, kind='PROTOCOL')

      for doc in docs:
        charter = runner.make_legal_doc(doc)
        runner.protocol_parser.find_org_date_number(charter)
        save_analysis(doc, charter, -1)

  # if get_mongodb_connection() is not None:
  unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
# else:
#   warnings.warn('mongo connection is not available')
