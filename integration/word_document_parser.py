import json
import os
import subprocess
import warnings

from analyser.charter_parser import CharterDocument
from analyser.contract_parser import ContractDocument
from analyser.legal_docs import LegalDocument, Paragraph, PARAGRAPH_DELIMITER
from analyser.ml_tools import SemanticTag
from analyser.protocol_parser import ProtocolDocument
from integration.doc_providers import DirDocProvider


class WordDocParser(DirDocProvider):
  version = '1.1.19'

  def __init__(self):

    x = os.system("java -version")
    assert x == 0
    if 'documentparser' in os.environ:
      self.documentparser = os.environ['documentparser']
    else:
      msg = f'please set "documentparser" environment variable to point ' \
            f'document-parser-{self.version} unpacked lib ' \
            f'(downloadable from https://github.com/nemoware/document-parser)'
      warnings.warn(msg)

      self.documentparser = f'../libs/document-parser-{self.version}'

    self.cp = f"{self.documentparser}/classes:{self.documentparser}/lib/*"
    print(self.cp)

  def read_doc(self, fn) -> dict:
    fn = fn.encode('utf-8')

    assert os.path.isfile(fn), f'{fn} does not exist'

    s = ["java", "-cp", self.cp, "com.nemo.document.parser.App", "-i", fn]
    result = subprocess.run(s, stdout=subprocess.PIPE, encoding='utf-8')

    if result.returncode != 0:
      raise RuntimeError(f'cannot execute {result.args}')

    return json.loads(result.stdout)


def join_paragraphs(response, doc_id) -> CharterDocument or ContractDocument or ProtocolDocument:
  # TODO: check type of res

  if response['documentType'] == 'CONTRACT':
    doc: LegalDocument = ContractDocument('')
  elif response['documentType'] == 'PROTOCOL':
    doc: LegalDocument = ProtocolDocument()
  elif response['documentType'] == 'CHARTER':
    doc: LegalDocument = CharterDocument()
  else:
    msg = f"Unsupported document type: {response['documentType']}"
    warnings.warn(msg)
    doc: LegalDocument = LegalDocument('')

  doc.parse()

  fields = ['documentNumber', 'documentType']

  for key in fields:
    doc.__dict__[key] = response[key]

  last = 0

  # remove empty headers
  paragraphs = []
  for _p in response['paragraphs']:
    header_text = _p['paragraphHeader']['text']
    if header_text.strip() != '':
      paragraphs.append(_p)
    else:
      doc.warnings.append('blank header encountered')
      warnings.warn('blank header encountered')

  for _p in paragraphs:

    header_text = _p['paragraphHeader']['text']
    header_text = header_text.replace('\n', ' ').strip() + PARAGRAPH_DELIMITER

    header = LegalDocument(header_text)
    header.parse()

    doc += header
    headerspan = (last, len(doc.tokens_map))

    last = len(doc.tokens_map)

    if _p['paragraphBody']:
      body_text = _p['paragraphBody']['text'] + PARAGRAPH_DELIMITER
      body = LegalDocument(body_text)
      body.parse()
      doc += body

    bodyspan = (last, len(doc.tokens_map))

    header_tag = SemanticTag('headline', header_text, headerspan)
    body_tag = SemanticTag('paragraphBody', None, bodyspan)

    para = Paragraph(header_tag, body_tag)
    doc.paragraphs.append(para)
    last = len(doc.tokens_map)

  doc._id = doc_id
  doc.filename = doc_id
  return doc


if __name__ == '__main__':
  wp = WordDocParser()
  res = wp.read_doc("/Users/artem/work/nemo/goil/IN/Другие договоры/Договор Формула.docx")
  for d in res['documents']:
    print("-" * 100)
    for p in d['paragraphs']:
      print(' 📃 ', p['paragraphHeader']['text'])

  print("=" * 100)
  for d in res['documents']:
    print("-" * 100)
    c = join_paragraphs(d, 'Другие договоры/Договор Формула.docx')
    print(c.text)
    print(c.__dict__.keys())
