import traceback
import warnings

import pymongo
from pymongo import CursorType

import analyser
from analyser import finalizer
from analyser.charter_parser import CharterParser
from analyser.contract_parser import ContractParser
from analyser.legal_docs import LegalDocument
from analyser.log import logger
from analyser.parsing import AuditContext
from analyser.persistence import DbJsonDoc
from analyser.protocol_parser import ProtocolParser
from analyser.structures import DocumentState
from integration.db import get_mongodb_connection


class Runner:
  default_instance: 'Runner' = None

  def __init__(self, init_embedder=True):
    self.protocol_parser = ProtocolParser()
    self.contract_parser = ContractParser()
    self.charter_parser = CharterParser()

  def init_embedders(self):
    pass

  @staticmethod
  def get_instance(init_embedder=False) -> 'Runner':
    if Runner.default_instance is None:
      Runner.default_instance = Runner(init_embedder=init_embedder)
    return Runner.default_instance


class BaseProcessor:
  parser = None

  def preprocess(self, jdoc: DbJsonDoc, context: AuditContext):
    # phase I
    # TODO: include phase I into phase II, remove phase I
    if jdoc.is_user_corrected():
      logger.info(f"skipping doc {jdoc._id} because it is corrected by user")
      # TODO: update state?
    else:
      legal_doc = jdoc.asLegalDoc()
      self.parser.find_org_date_number(legal_doc, context)
      save_analysis(jdoc, legal_doc, state=DocumentState.Preprocessed.value)

  def process(self, db_document: DbJsonDoc, audit, context: AuditContext):
    # phase II
    if db_document.retry_number is None:
      db_document.retry_number = 0

    if db_document.retry_number > 2:
      logger.error(f'{db_document.documentType} {db_document._id} exceeds maximum retries for analysis and is skipped')
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


def get_charters():
  return get_docs_by_audit_id(id=None, states=[DocumentState.New.value], kind="CHARTER")


def get_docs_by_audit_id(id: str or None, states=None, kind=None, id_only=False) -> []:
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


def need_analysis(document: DbJsonDoc) -> bool:
  _is_not_a_charter = document.documentType != "CHARTER"
  _well_parsed = document.parserResponseCode == 200

  return _well_parsed and (document.isActiveCharter() or _is_not_a_charter)


def audit_phase_1(audit, kind=None):
  logger.info(f'.....processing audit {audit["_id"]}')
  ctx = AuditContext(audit["subsidiary"]["name"])

  document_ids = get_docs_by_audit_id(audit["_id"], states=[DocumentState.New.value], kind=kind, id_only=True)
  _charter_ids = audit.get("charters", [])
  document_ids.extend(_charter_ids)

  for k, document_id in enumerate(document_ids):
    _document = finalizer.get_doc_by_id(document_id)
    jdoc = DbJsonDoc(_document)

    processor: BaseProcessor = document_processors.get(jdoc.documentType)
    if processor is None:
      logger.warning(f'unknown/unsupported doc type: {jdoc.documentType}, cannot process {document_id}')
    else:
      logger.info(f'......pre-processing {k} of {len(document_ids)}  {jdoc.documentType}:{document_id}')
      if need_analysis(jdoc) and jdoc.isNew():
        processor.preprocess(jdoc=jdoc, context=ctx)


def audit_phase_2(audit, kind=None):
  ctx = AuditContext(audit["subsidiary"]["name"])

  print(f'.....processing audit {audit["_id"]}')

  document_ids = get_docs_by_audit_id(audit["_id"],
                                      states=[DocumentState.Preprocessed.value, DocumentState.Error.value],
                                      kind=kind, id_only=True)

  _charter_ids = audit.get("charters", [])
  document_ids.extend(_charter_ids)

  for k, document_id in enumerate(document_ids):
    _document = finalizer.get_doc_by_id(document_id)
    jdoc = DbJsonDoc(_document)

    processor: BaseProcessor = document_processors.get(jdoc.documentType)
    if processor is None:
      logger.warning(f'unknown/unsupported doc type: {jdoc.documentType}, cannot process {document_id}')
    else:
      if need_analysis(jdoc) and jdoc.isPreprocessed():
        logger.info(f'.....processing  {k} of {len(document_ids)}   {jdoc.documentType} {document_id}')
        processor.process(jdoc, audit, ctx)

  change_audit_status(audit, "Finalizing")  # TODO: check ALL docs in proper state


def audit_charters_phase_1():
  """preprocess"""
  charters = get_charters()
  processor: BaseProcessor = document_processors['CHARTER']

  for k, charter in enumerate(charters):
    jdoc = DbJsonDoc(charter)
    logger.info(f'......pre-processing {k} of {len(charters)} CHARTER {jdoc._id}')
    ctx = AuditContext()
    processor.preprocess(jdoc, context=ctx)


def audit_charters_phase_2():
  charters = get_docs_by_audit_id(id=None, states=[DocumentState.Preprocessed.value, DocumentState.Error.value],
                                  kind="CHARTER")

  for k, _document in enumerate(charters):
    jdoc = DbJsonDoc(_document)
    processor: BaseProcessor = document_processors['CHARTER']

    logger.info(f'......processing  {k} of {len(charters)}  CHARTER {jdoc._id}')
    ctx = AuditContext()
    processor.process(jdoc, audit=None, context=ctx)


def run(run_pahse_2=True, kind=None):
  # -----------------------
  # NIL (сорян, в системе римских цифр отсутствует ноль)
  logger.info('-> PHASE 0 (charters)...')
  audit_charters_phase_1()
  if run_pahse_2:
    audit_charters_phase_2()

  # -----------------------
  # I
  logger.info('-> PHASE I...')
  for audit in get_audits():
    audit_phase_1(audit, kind)

  # -----------------------
  # II
  logger.info('-> PHASE II..')
  if run_pahse_2:
    # phase 2
    for audit in get_audits():
      audit_phase_2(audit, kind)

  else:
    logger.info("phase 2 is skipped")

  # -----------------------
  # III
  logger.info('-> PHASE III (finalize)...')
  finalizer.finalize()


if __name__ == '__main__':
  run()
