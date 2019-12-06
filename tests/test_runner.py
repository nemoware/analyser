#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest
import warnings

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

  @unittest.skipIf(SKIP_TF, "requires TF")
  def test_process_single_protocol(self):
    audit = next(get_audits())
    audit_id = audit['_id']
    docs = get_docs_by_audit_id(audit_id, kind='PROTOCOL')
    doc = next(docs)

    ProtocolProcessor().process(doc, audit)

  @unittest.skipIf(SKIP_TF, "requires TF")
  def test_process_single_contract(self):
    audit = next(get_audits())
    audit_id = audit['_id']
    docs = get_docs_by_audit_id(audit_id, kind='CONTRACT')
    doc = next(docs)
    doc = next(docs)
    doc = next(docs)

    ContractProcessor().process(doc, audit)

  @unittest.skipIf(SKIP_TF, "requires TF")
  def test_process_single_charter(self):
    audit = next(get_audits())
    audit_id = audit['_id']
    docs = get_docs_by_audit_id(audit_id, kind='CHARTER')
    doc = next(docs)

    CharterProcessor().process(doc, audit)

  def test_process_contracts_phase_1(self):
    runner = get_runner_instance_no_embedder()

    for audit in get_audits():
      audit_id = audit['_id']
      docs = get_docs_by_audit_id(audit_id, kind='CONTRACT')

      for doc in docs:
        charter = runner.make_legal_doc(doc)
        runner.contract_parser.find_org_date_number(charter)
        save_analysis(doc, charter)
        print(charter._id)

  def test_process_charters_phase_1(self):
    runner = get_runner_instance_no_embedder()

    for audit in get_audits():
      audit_id = audit['_id']
      docs = get_docs_by_audit_id(audit_id, kind='CHARTER')

      for doc in docs:
        charter = runner.make_legal_doc(doc)
        print(charter._id)
        runner.charter_parser.find_org_date_number(charter)
        save_analysis(doc, charter)

  def test_process_protocols_phase_1(self):
    runner = get_runner_instance_no_embedder()

    audit_id = next(get_audits())['_id']
    docs = get_docs_by_audit_id(audit_id, kind='PROTOCOL')

    for doc in docs:
      charter = runner.make_legal_doc(doc)
      runner.protocol_parser.find_org_date_number(charter)
      save_analysis(doc, charter)

  # def test_process_single_contract(self):
  #   audit_id = next(get_audits())['_id']
  #   docs = get_docs_by_audit_id(audit_id, kind='CONTRACT')
  #   doc = next(docs)
  #   process_document(doc)


if get_mongodb_connection() is not None:
  unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
else:
  warnings.warn('mongo connection is not available')
