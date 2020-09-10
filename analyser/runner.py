import logging
import traceback
import warnings

import pymongo
from pymongo import CursorType

import analyser
from analyser import finalizer
from analyser.charter_parser import CharterParser
from analyser.contract_parser import ContractParser
from analyser.legal_docs import LegalDocument
from analyser.parsing import AuditContext
from analyser.persistence import DbJsonDoc
from analyser.protocol_parser import ProtocolParser
from analyser.structures import DocumentState
from integration.db import get_mongodb_connection
from tf_support.embedder_elmo import ElmoEmbedder

logger = logging.getLogger('root')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.DEBUG)

logger.addHandler(ch)


class Runner:
  default_instance: 'Runner' = None

  def __init__(self, init_embedder=True):
    self.elmo_embedder: ElmoEmbedder = None
    self.elmo_embedder_default: ElmoEmbedder = None
    if init_embedder:
      self.elmo_embedder = ElmoEmbedder.get_instance('elmo')
      self.elmo_embedder_default = ElmoEmbedder.get_instance('default')

    self.protocol_parser = ProtocolParser(self.elmo_embedder, self.elmo_embedder_default)
    self.contract_parser = ContractParser(self.elmo_embedder)
    self.charter_parser = CharterParser(self.elmo_embedder, self.elmo_embedder_default)

  def init_embedders(self):
    self.elmo_embedder = ElmoEmbedder.get_instance('elmo')
    self.elmo_embedder_default = ElmoEmbedder.get_instance('default')
    self.protocol_parser.init_embedders(self.elmo_embedder, self.elmo_embedder_default)
    self.contract_parser.init_embedders(self.elmo_embedder, self.elmo_embedder_default)
    self.charter_parser.init_embedders(self.elmo_embedder, self.elmo_embedder_default)

  @staticmethod
  def get_instance(init_embedder=False) -> 'Runner':
    if Runner.default_instance is None:
      Runner.default_instance = Runner(init_embedder=init_embedder)
    return Runner.default_instance


class BaseProcessor:
  parser = None

  def preprocess(self, db_document: dict, context: AuditContext):
    jdoc = DbJsonDoc(db_document)
    if jdoc.is_user_corrected():
      logger.info(f"skipping doc {jdoc._id} because it is corrected by user")
      # TODO: update state?
    else:
      legal_doc = jdoc.asLegalDoc()
      Runner.get_instance()
      self.parser.find_org_date_number(legal_doc, context)
      save_analysis(jdoc, legal_doc, state=DocumentState.Preprocessed.value)

  def process(self, db_document: DbJsonDoc, audit, context: AuditContext):
    if db_document.retry_number is None:
      db_document.retry_number = 0

    if db_document.retry_number > 2:
      logger.error(f'document {db_document._id} exceeds maximum retries for analysis and is skipped')
      return None

    legal_doc = db_document.asLegalDoc()
    try:

      # self.parser.find_org_date_number(legal_doc, context) # todo: remove this call
      # save_analysis(db_document, legal_doc, state=DocumentState.InWork.value)

      if audit is None or self.is_valid(audit, db_document):

        if db_document.is_user_corrected():
          logger.info(f"skipping doc {db_document._id} postprocessing because it is corrected by user")
        else:
          self.parser.find_attributes(legal_doc, context)

        save_analysis(db_document, legal_doc, state=DocumentState.Done.value)
        logger.info(f'analysis saved, doc._id={legal_doc._id}')
      else:
        logger.info(f"excluding doc {db_document._id}")
        save_analysis(db_document, legal_doc, state=DocumentState.Excluded.value)

    except Exception as err:
      traceback.print_tb(err.__traceback__)
      logger.exception(f'cant process document {db_document._id}')
      save_analysis(db_document, legal_doc, DocumentState.Error.value, db_document.retry_number + 1)

    return legal_doc

  def is_valid(self, audit, db_document: DbJsonDoc):
    # date must be ok
    _date = db_document.get_date_value()
    if _date is not None:
      date_is_ok = audit["auditStart"] <= _date <= audit["auditEnd"]
    else:
      # if date not found, we keep processing the doc anyway
      date_is_ok = True

    # org filter must be ok
    _audit_subsidiary: str = audit["subsidiary"]["name"]
    org_is_ok = ("* Все ДО" == _audit_subsidiary) or (self._same_org(db_document, _audit_subsidiary))

    return org_is_ok and date_is_ok

  def _same_org(self, db_doc: DbJsonDoc, subsidiary: str) -> bool:
    o1: str = db_doc.get_attribute_value("org-1-name")
    o2: str = db_doc.get_attribute_value("org-2-name")

    return (subsidiary == o1) or (o2 == subsidiary)

  def is_same_org(self, legal_doc, db_doc, subsidiary):
    warnings.warn("use _same_org", DeprecationWarning)
    if db_doc.get("user") is not None and db_doc["user"].get("attributes") is not None and db_doc["user"][
      "attributes"].get("org-1-name") is not None:
      if subsidiary == db_doc["user"]["attributes"]["org-1-name"]["value"]:
        return True
    else:
      return legal_doc.is_same_org(subsidiary)


class ProtocolProcessor(BaseProcessor):
  def __init__(self):
    self.parser = Runner.get_instance().protocol_parser


class CharterProcessor(BaseProcessor):
  def __init__(self):
    self.parser = Runner.get_instance().charter_parser


class ContractProcessor(BaseProcessor):
  def __init__(self):
    self.parser = Runner.get_instance().contract_parser


document_processors = {"CONTRACT": ContractProcessor(), "CHARTER": CharterProcessor(), "PROTOCOL": ProtocolProcessor()}


def get_audits():
  db = get_mongodb_connection()
  audits_collection = db['audits']

  res = audits_collection.find({'status': 'InWork'}, cursor_type=CursorType.EXHAUST).sort(
    [("createDate", pymongo.ASCENDING)])
  return res


def get_docs_by_audit_id(id: str, states=None, kind=None, id_only=False) -> []:
  db = get_mongodb_connection()
  documents_collection = db['documents']

  query = {
    "$and": [
      {'auditId': id},
      {"parserResponseCode": 200},
      {"$or": [{"analysis.version": None},
               {"analysis.version": {"$ne": analyser.__version__}},
               {"state": None}]}
    ]
  }

  if states is not None:
    for state in states:
      query["$and"][2]["$or"].append({"state": state})

  if kind is not None:
    query["$and"].append({'parse.documentType': kind})

  if id_only:
    cursor = documents_collection.find(query, cursor_type=CursorType.EXHAUST, projection={'_id': True})
  else:
    cursor = documents_collection.find(query, cursor_type=CursorType.EXHAUST)

  res = []
  for doc in cursor:
    if id_only:
      res.append(doc["_id"])
    else:
      res.append(doc)
  return res


def save_analysis(db_document: DbJsonDoc, doc: LegalDocument, state: int, retry_number: int = 0):
  analyse_json_obj = doc.to_json_obj()
  db = get_mongodb_connection()
  documents_collection = db['documents']
  db_document.analysis = analyse_json_obj
  db_document.state = state
  db_document.retry_number = retry_number
  documents_collection.update({'_id': doc._id}, db_document.as_dict(), True)


def change_audit_status(audit, status):
  db = get_mongodb_connection()
  db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"status": status}})


def need_analysis(document) -> bool:
  return document["parse"]["documentType"] != "CHARTER" or (document["parserResponseCode"] == 200 and
    (document.get("isActive") is None or document["isActive"]))


def run(run_pahse_2=True, kind=None):
  # phase 1
  print('=' * 80)
  print('PHASE 1')
  runner = Runner.get_instance(init_embedder=False)
  audits = get_audits()
  for audit in audits:

    ctx = AuditContext()
    ctx.audit_subsidiary_name = audit["subsidiary"]["name"]

    print('=' * 80)
    logger.info(f'.....processing audit {audit["_id"]}')

    document_ids = get_docs_by_audit_id(audit["_id"], states=[DocumentState.New.value], kind=kind, id_only=True)
    if audit.get("charters") is not None:
      document_ids.extend(audit["charters"])
    for k, document_id in enumerate(document_ids):
      document = finalizer.get_doc_by_id(document_id)
      processor: BaseProcessor = document_processors.get(document["parse"]["documentType"], None)
      if processor is not None and need_analysis(document) and (document.get("state") == DocumentState.New.value or document.get("state") is None):
        print(f'......pre-processing {k} of {len(document_ids)}  {document["parse"]["documentType"]} {document["_id"]}')
        processor.preprocess(db_document=document, context=ctx)

  charters = get_docs_by_audit_id(id=None, states=[DocumentState.New.value], kind="CHARTER")
  for k, document in enumerate(charters):
    processor: BaseProcessor = document_processors.get(document["parse"]["documentType"], None)
    if processor is not None:
      print(f'......pre-processing {k} of {len(charters)}  {document["parse"]["documentType"]} {document["_id"]}')
      ctx = AuditContext()
      processor.preprocess(db_document=document, context=ctx)

  if run_pahse_2:
    # phase 2
    print('=' * 80)
    print('PHASE 2')
    runner.init_embedders()
    audits = get_audits()
    for audit in audits:
      ctx = AuditContext()
      ctx.audit_subsidiary_name = audit["subsidiary"]["name"]

      print('=' * 80)
      print(f'.....processing audit {audit["_id"]}')

      document_ids = get_docs_by_audit_id(audit["_id"],
                                          states=[DocumentState.Preprocessed.value, DocumentState.Error.value],
                                          kind=kind, id_only=True)
      if audit.get("charters") is not None:
        document_ids.extend(audit["charters"])

      for k, document_id in enumerate(document_ids):
        document = finalizer.get_doc_by_id(document_id)

        processor = document_processors.get(document["parse"]["documentType"], None)
        if processor is not None and need_analysis(document) \
                and (document.get("state") == DocumentState.Preprocessed.value or document.get("state") == DocumentState.Error.value):
          jdoc = DbJsonDoc(document)
          print(f'......processing  {k} of {len(document_ids)}   {document["parse"]["documentType"]} {document["_id"]}')
          processor.process(jdoc, audit, ctx)

      change_audit_status(audit, "Finalizing")  # TODO: check ALL docs in proper state

    charters = get_docs_by_audit_id(id=None, states=[DocumentState.Preprocessed.value, DocumentState.Error.value], kind="CHARTER")
    for k, document in enumerate(charters):
      processor: BaseProcessor = document_processors.get(document["parse"]["documentType"], None)
      if processor is not None:
        jdoc = DbJsonDoc(document)
        print(f'......processing  {k} of {len(charters)}   {document["parse"]["documentType"]} {document["_id"]}')
        ctx = AuditContext()
        processor.process(jdoc, audit=None, context=ctx)

  else:
    warnings.warn("phase 2 is skipped")

  finalizer.finalize()


if __name__ == '__main__':
  run()
