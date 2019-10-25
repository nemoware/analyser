import glob
import json
import os

import docx2txt


def read_doc(fn):
  try:
    text = docx2txt.process(fn)

  except:
    os.system('antiword -w 0 "' + fn + '" > "' + fn + '.txt"')
    with open(fn + '.txt') as f:
      text = f.read()

  return text


def read_documents(filename_prefix):
  texts = []

  filenames = []
  filenames += [file for file in glob.glob(filename_prefix + "*.docx", recursive=True)]
  filenames += [file for file in glob.glob(filename_prefix + "*.doc", recursive=True)]

  for file in filenames:
    text = read_doc(file)

    fi = {
      "filemtime": os.path.getmtime(file),
      "short_filename": os.path.split(file)[-1],
      "len": len(text),
      "filename": file,
      "checksum": hash(text),
      "analyse": {}
    }
    texts.append(fi)
    # print(fi)

  print(len(texts))

  return texts


if __name__ == "__main__":
  texts = read_documents('/Users/artem/work/nemo/goil/IN/**/')

  with open('/Users/artem/work/nemo/goil/gpn-ui/projects/gpn-ui/src/assets/list_documents.json', 'w') as file:
    _j = json.dumps(texts, indent=4, ensure_ascii=False, default=lambda o: '<not serializable>')
    file.write(_j)
