import datetime
import json
import os
import subprocess
import warnings

from contract_parser import ContractDocument
from integration.doc_providers import DirDocProvider
from legal_docs import LegalDocument, Paragraph
from ml_tools import SemanticTag


class WordDocParser(DirDocProvider):

  def __init__(self):
    self.version='1.0.8'
    x = os.system("java -version")
    assert x == 0
    if 'documentparser' in os.environ:
      self.documentparser = os.environ['documentparser']
    else:
      msg=f'please set "documentparser" environment variable to point ' \
          f'document-parser-{self.version} unpacked lib ' \
          f'(downloadable from https://github.com/nemoware/document-parser)'
      warnings.warn(msg)

      self.documentparser = f'../tests/libs/document-parser-{self.version}'

    self.cp = f"{self.documentparser}/classes:{self.documentparser}/lib/*"
    print(self.cp)

  def read_doc(self, fn) -> dict:
    FILENAME = fn.encode('utf-8')

    assert os.path.isfile(FILENAME), f'{FILENAME} does not exist'

    s = ["java", "-cp", self.cp, "com.nemo.document.parser.App", "-i", FILENAME]

    # s=['pwd']
    result = subprocess.run(s, stdout=subprocess.PIPE, encoding='utf-8')
    # print(f'result=[{result.stdout}]')

    res = json.loads(result.stdout)

    #
    return res


def join_paragraphs(res, doc_id):
  # TODO: check type of res
  doc: ContractDocument = ContractDocument('').parse()

  last = 0
  for p in res['paragraphs']:
    header_text = p['paragraphHeader']['text'] + '\n'

    header = LegalDocument(header_text)
    header.parse()

    doc += header
    headerspan = (last, len(doc.tokens_map))

    last = len(doc.tokens_map)

    if p['paragraphBody']:
      body_text = p['paragraphBody']['text'] + '\n'
      body = LegalDocument(body_text)
      body.parse()
      doc += body

    bodyspan = (last, len(doc.tokens_map))

    header_tag = SemanticTag('headline', header_text, headerspan)
    body_tag = SemanticTag('paragraphBody', None, bodyspan)

    para = Paragraph(header_tag, body_tag)
    doc.paragraphs.append(para)
    last = len(doc.tokens_map)

  # doc.parse()

  if res["documentDate"]:
    date_time_obj = datetime.datetime.strptime(res["documentDate"], '%Y-%m-%d')
    doc.date = date_time_obj
  doc._id = doc_id
  return doc


if __name__ == '__main__':
  wp = WordDocParser()
  res = wp.read_doc("/Users/artem/work/nemo/goil/IN/Другие договоры/Договор Формула.docx")
  for p in res['paragraphs']:
    print(' 📃 ', p['paragraphHeader']['text'])

  c = join_paragraphs(res, 'Другие договоры/Договор Формула.docx')
  print(c.text)
