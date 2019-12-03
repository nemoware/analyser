import pymongo

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
    self.document_processors = {"CONTRACT": self.process_contract, "CHARTER": self.process_charter, "PROTOCOL": self.process_protocol}

  @staticmethod
  def get_instance() -> 'Runner':
    if Runner.default_instance is None:
      Runner.default_instance = Runner()
    return Runner.default_instance

  def _make_legal_doc(self, db_document):
    parsed_p_json = db_document['parse']
    legal_doc = join_paragraphs(parsed_p_json, doc_id=db_document['_id'])
    # save_analysis(db_document, legal_doc)
    return legal_doc

  def process_protocol(self, db_document) -> ProtocolDocument:
    protocol = self._make_legal_doc(db_document)
    self.protocol_parser.analyse(protocol)
    save_analysis(db_document, protocol)

    print(protocol._id)
    return protocol

  def process_contract(self, db_document) -> ContractDocument:
    contract = self._make_legal_doc(db_document)
    self.contract_parser.analyze_contract_doc(contract)
    save_analysis(db_document, contract)

    print(contract._id)
    return contract

  def process_charter(self, db_document) -> CharterDocument:
    charter = self._make_legal_doc(db_document)
    self.charter_parser.ebmedd(charter)
    self.charter_parser.analyse(charter)
    save_analysis(db_document, charter)

    print(charter._id)
    return charter


def get_audits():
  db = get_mongodb_connection()
  audits_collection = db['audits']

  res = audits_collection.find({'status': 'InWork'}).sort([("createDate", pymongo.ASCENDING)])
  return res


def get_docs_by_audit_id(id: str, kind=None):
  db = get_mongodb_connection()
  documents_collection = db['documents']

  query = {
    'auditId': id,
    "parserResponseCode": 200,
    "analysis.version": None
  }

  if kind is not None:
    query['parse.documentType'] = kind

  res = documents_collection.find(query)
  return res


def save_analysis(db_document, doc: LegalDocument):
  analyse_json_obj = doc.to_json_obj()
  db = get_mongodb_connection()
  documents_collection = db['documents']
  db_document['analysis'] = analyse_json_obj
  documents_collection.update({'_id': doc._id}, db_document, True)


def change_audit_status(audit, status):
  db = get_mongodb_connection()
  db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"status": status}})


def run():
  audits = get_audits()
  for audit in audits:
    documents = get_docs_by_audit_id(audit["_id"])
    for document in documents:
      Runner.get_instance().document_processors.get(document["parse"]["documentType"], lambda x: None)(document)
    change_audit_status(audit, "Finalizing")