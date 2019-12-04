import pymongo

import analyser
from analyser.charter_parser import CharterParser, CharterDocument
from analyser.contract_parser import ContractDocument, ContractAnlysingContext
from analyser.legal_docs import LegalDocument
from analyser.protocol_parser import ProtocolParser, ProtocolDocument
from integration.db import get_mongodb_connection
from integration.word_document_parser import join_paragraphs
from tf_support.embedder_elmo import ElmoEmbedder

class Runner:
  default_instance: 'Runner' = None

  def __init__(self, init_embedder=True):
    self.elmo_embedder: ElmoEmbedder = None
    self.elmo_embedder_default: ElmoEmbedder = None
    if init_embedder:
      self.elmo_embedder = ElmoEmbedder()
      self.elmo_embedder_default = ElmoEmbedder(layer_name="default")

    self.protocol_parser = ProtocolParser(self.elmo_embedder, self.elmo_embedder_default)
    self.contract_parser = ContractAnlysingContext(self.elmo_embedder)
    self.charter_parser = CharterParser(self.elmo_embedder, self.elmo_embedder_default)

  @staticmethod
  def get_instance() -> 'Runner':
    if Runner.default_instance is None:
      Runner.default_instance = Runner()
    return Runner.default_instance

  def make_legal_doc(self, db_document):
    parsed_p_json = db_document['parse']
    legal_doc = join_paragraphs(parsed_p_json, doc_id=db_document['_id'])
    # save_analysis(db_document, legal_doc)
    return legal_doc


class BaseProcessor:
  parser = None

  def preprocess(self, db_document):
    legal_doc = Runner.get_instance().make_legal_doc(db_document)
    self.parser.find_org_date_number(legal_doc)
    save_analysis(db_document, legal_doc, 2)

  def process(self, db_document, audit):
    legal_doc = Runner.get_instance().make_legal_doc(db_document)
    if self.is_valid(legal_doc, audit):
      self.parser.ebmedd(legal_doc)
      self.parser.find_attributes(legal_doc)
      save_analysis(db_document, legal_doc, 3)
      print(legal_doc._id)
    return legal_doc

  def is_valid(self, legal_doc, audit):
    return legal_doc.is_same_org(audit["subsidiary"]["name"]) and audit["auditStart"] <= legal_doc.date <= audit["auditEnd"]


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


def get_docs_by_audit_id(id: str, state=None, kind=None):
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

  if state is not None:
    query["$and"][2]["$or"].append({"state": state})

  if kind is not None:
    query["$and"].append({'parse.documentType': kind})

  res = documents_collection.find(query)
  return res


def save_analysis(db_document, doc: LegalDocument, state: int):
  analyse_json_obj = doc.to_json_obj()
  db = get_mongodb_connection()
  documents_collection = db['documents']
  db_document['analysis'] = analyse_json_obj
  db_document["state"] = state
  documents_collection.update({'_id': doc._id}, db_document, True)


def change_audit_status(audit, status):
  db = get_mongodb_connection()
  db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"status": status}})


def run():
  audits = get_audits()
  for audit in audits:
    documents = get_docs_by_audit_id(audit["_id"])
    for document in documents:
      processor = document_processors.get(document["parse"]["documentType"], None)
      if processor is not None:
        processor.preprocess(db_document=document)

    documents = get_docs_by_audit_id(audit["_id"], 2)
    for document in documents:
      processor = document_processors.get(document["parse"]["documentType"], None)
      if processor is not None:
        processor.process(document, audit)
    change_audit_status(audit, "Finalizing")