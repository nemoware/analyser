import warnings

import pymongo

import analyser
from analyser import finalizer
from analyser.charter_parser import CharterParser
from analyser.contract_parser import ContractParser
from analyser.legal_docs import LegalDocument, DbJsonDoc
from analyser.parsing import AuditContext
from analyser.protocol_parser import ProtocolParser
from integration.db import get_mongodb_connection
from integration.word_document_parser import join_paragraphs
from tf_support.embedder_elmo import ElmoEmbedder


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

  def make_legal_doc(self, db_document: DbJsonDoc):
    parsed_p_json = db_document.parse
    legal_doc = join_paragraphs(parsed_p_json, doc_id=db_document._id)
    # save_analysis(db_document, legal_doc)
    # TODO: do not ignore user-corrected attributes here
    return legal_doc


class BaseProcessor:
  parser = None

  def preprocess(self, db_document: DbJsonDoc, context: AuditContext):
    legal_doc = Runner.get_instance().make_legal_doc(db_document)
    self.parser.find_org_date_number(legal_doc, context)
    save_analysis(db_document, legal_doc, state=5)

  def process(self, db_document: DbJsonDoc, audit, context: AuditContext):
    if db_document.retry_number is None:
      db_document.retry_number = 0

    if db_document.retry_number > 2:
      print(f'document {db_document._id} exceeds maximum retries for analysis and is skipped')
      return None
    legal_doc = Runner.get_instance().make_legal_doc(db_document)
    try:
      # todo: remove find_org_date_number call
      self.parser.find_org_date_number(legal_doc, context)
      save_analysis(db_document, legal_doc, state=10)
      if self.is_valid(legal_doc, audit, db_document):
        self.parser.find_attributes(legal_doc, context)
        save_analysis(db_document, legal_doc, state=15)
        print('analysis saved, doc._id=', legal_doc._id)
      else:
        save_analysis(db_document, legal_doc, 12)
    except:
      print(f'cant process document {db_document._id}')

      save_analysis(db_document, legal_doc, 11, db_document.retry_number + 1)
    return legal_doc

  def is_valid(self, legal_doc, audit, db_document):
    if legal_doc.date is not None:
      _date = legal_doc.date.value
      date_is_ok = legal_doc.date is not None or audit["auditStart"] <= _date <= audit["auditEnd"]
    else:
      date_is_ok = True

    return ("* Все ДО" == audit["subsidiary"]["name"] or legal_doc.is_same_org(
      audit["subsidiary"]["name"])) and date_is_ok

  def is_same_org(self, legal_doc, db_doc, subsidiary):
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

  res = audits_collection.find({'status': 'InWork'}).sort([("createDate", pymongo.ASCENDING)])
  return res


def get_docs_by_audit_id(id: str, states=None, kind=None, limit=None) -> [dict]:
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

  cursor = documents_collection.find(query)
  if limit is not None:
    cursor.limit(limit)
  res = []
  # TODO there may be too many docs, might be we should fetch ids only
  for doc in cursor:
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
    print(f'.....processing audit {audit["_id"]}')
    documents = get_docs_by_audit_id(audit["_id"], [0], kind=kind)  # TODO: fetch IDs only
    for k, document in enumerate(documents):
      processor: BaseProcessor = document_processors.get(document["parse"]["documentType"], None)
      if processor is not None:
        print(f'........pre-processing {k} of {len(documents)}  {document["parse"]["documentType"]} {document["_id"]}')
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
      documents = get_docs_by_audit_id(audit["_id"], [5, 11], kind=kind)  # TODO: fetch IDs only
      for k, document in enumerate(documents):
        processor = document_processors.get(document["parse"]["documentType"], None)
        if processor is not None:
          jdoc = DbJsonDoc(document)
          print(f'........processing  {k} of {len(documents)}   {document["parse"]["documentType"]} {document["_id"]}')
          processor.process(jdoc, audit, ctx)

      change_audit_status(audit, "Finalizing")  # TODO: check ALL docs in proper state
  else:
    warnings.warn("phase 2 is skipped")

  finalizer.finalize()


if __name__ == '__main__':
  run()
