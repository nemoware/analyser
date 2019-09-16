import json
import os
import subprocess
import warnings

from integration.doc_providers import DirDocProvider


class WordDocParser(DirDocProvider):

  def __init__(self):
    x = os.system("java -version")
    assert x == 0
    if 'documentparser' in os.environ:
      self.documentparser = os.environ['documentparser']
    else:
      warnings.warn(
        'please set "documentparser" environment variable to point '
        'document-parser-1.0.2 unpacked lib '
        '(downloadable from https://github.com/nemoware/document-parser)')

      self.documentparser = '../tests/libs/document-parser-1.0.2'

    self.cp = f"{self.documentparser}/classes:{self.documentparser}/lib/*"
    print(self.cp)

  def read_doc(self, fn) -> dict:
    FILENAME = fn.encode('utf-8')

    s = ["java", "-cp", self.cp, "com.nemo.document.parser.App", "-i", FILENAME]

    # s=['pwd']
    result = subprocess.run(s, stdout=subprocess.PIPE, encoding='utf-8')
    print(result.stdout)

    res = json.loads(result.stdout)

    #
    return res


if __name__ == '__main__':
  wp = WordDocParser()
  res = wp.read_doc("/Users/artem/work/nemo/goil/IN/Другие договоры/Договор Формула.docx")
  for p in res['paragraphs']:
    print(p['paragraphHeader']['text'])
