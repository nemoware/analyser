import json

from integration.db import get_mongodb_connection
from integration.word_document_parser import WordDocParser
from utilits.read_all_contracts import read_all_docs

# _files_dir = '/Users/artem/Google Drive/GazpromOil/Примеры документов/'


# _files_dir = '/Users/artem/work/nemo/goil/IN/яДата-сет 16.12'
_files_dir = '/Users/artem/work/nemo/goil/IN/'


def make_trainset():
  # -------------------------------
  docs = read_all_docs(_files_dir)
  # -------------------------------

  cntr = {}
  for n, doc in enumerate(docs):
    print(n, '👾 make_trainset\t', doc.filename)
    header = doc.paragraphs[0].header

    print(doc.substr(header))

    cntr[doc.substr(header)] = doc.filename

  for h in sorted(cntr.keys()):
    print(f'👾{h}☣️  {cntr[h]}')


def export_contracts():
  db = get_mongodb_connection()

  criterion = {
    'version': WordDocParser.version
  }

  res = db['legaldocs'].find(criterion)

  arr = {}
  for docs in res:

    for doc_json in docs['documents']:
      k = 0
      if doc_json['documentType'] == 'CONTRACT':
        k += 1
        print(docs['_id'])
        key = f"{k}-{docs['_id']}"
        arr[key] = doc_json

  with open('parsed_docs.json', 'w') as outfile:
    json.dump(arr, outfile, ensure_ascii=False, indent=2)


if __name__ == '__main__':
  make_trainset()
  export_contracts()
