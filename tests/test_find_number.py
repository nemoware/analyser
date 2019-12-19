#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import re
import unittest

from analyser.doc_numbers import document_number_c


class NumbersTestCase(unittest.TestCase):

  def test_find_doc_number(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №3 к Договору о выдаче банковских гарантий №3256-5/876 от 06-02-2013 год, заключенному между '''
    findings = re.finditer(document_number_c, t)

    self.assertEqual('№3 ', next(findings)[0])
    self.assertEqual('№3256-5/876 ', next(findings)[0])


if __name__ == '__main__':
  unittest.main()
