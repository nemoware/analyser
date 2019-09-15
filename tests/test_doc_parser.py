#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import subprocess
import unittest
import json

class TestContractParser(unittest.TestCase):

  def test_doc_parser(self):
    FILENAME = "/Users/artem/work/nemo/goil/IN/Другие договоры/Договор Формула.docx"
    FILENAME = FILENAME.encode('utf-8')

    s = ["java", "-cp", "libs/document-parser-1.0.2/classes:libs/document-parser-1.0.2/lib/*",
         "com.nemo.document.parser.App", "-i", FILENAME]

    # s=['pwd']
    result = subprocess.run(s, stdout=subprocess.PIPE, encoding='utf-8')
    print(result.stdout)

    res = json.loads(result.stdout)

    for p in res['paragraphs']:
      print(p['paragraphHeader'])



unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
