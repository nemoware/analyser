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

    self.assertEqual('3', next(findings)[0])
    self.assertEqual('3256-5/876', next(findings)[0])

  def test_find_doc_number_missing_na(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения № на на ыдаче'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual(0, len(findings))

  def test_find_doc_number_missing___(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения № ____ на ыдаче'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual(0, len(findings))

  def test_find_doc_number_no_dot(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №343434.'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('343434', findings[0][0])

  def test_find_doc_number_two_upper_space(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №ДК 834/34-2.'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('ДК 834/34-2', findings[0][0])

  def test_find_doc_number_two_upper_space_latin(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №XK 834/34-2.'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('ДК XK 834/34-2', findings[0][0])

  def test_find_doc_number_two_upper_space_latin_eol(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №XK 834/34-2'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('ДК XK 834/34-2', findings[0][0])



if __name__ == '__main__':
  unittest.main()
