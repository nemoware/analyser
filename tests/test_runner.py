#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest
import warnings

from analyser.runner import *

SKIP_TF=True

class TestRunner(unittest.TestCase):

  @unittest.skipIf( get_mongodb_connection() is None, "requires mongo")
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

    runner = Runner.get_instance()

    audit_id = next(get_audits())['_id']
    docs = get_docs_by_audit_id(audit_id, kind='PROTOCOL')
    doc = next(docs)

    runner.process_protocol(doc)

  @unittest.skipIf(SKIP_TF, "requires TF")
  def test_process_single_contract (self):

    runner = Runner.get_instance()

    audit_id = next(get_audits())['_id']
    docs = get_docs_by_audit_id(audit_id, kind='CONTRACT')
    doc = next(docs)
    doc = next(docs)
    doc = next(docs)

    runner.process_contract( doc)

  # @unittest.skipIf(SKIP_TF, "requires TF")
  def test_process_single_charter (self):

    runner = Runner.get_instance()

    audit_id = next(get_audits())['_id']
    docs = get_docs_by_audit_id(audit_id, kind='CHARTER')
    doc = next(docs)


    runner.process_charter(doc)


  # def test_process_single_contract(self):
  #   audit_id = next(get_audits())['_id']
  #   docs = get_docs_by_audit_id(audit_id, kind='CONTRACT')
  #   doc = next(docs)
  #   process_document(doc)


if get_mongodb_connection() is not None:
  unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
else:
  warnings.warn('mongo connection is not available')
