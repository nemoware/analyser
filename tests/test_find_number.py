#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import re
import unittest

from analyser.doc_numbers import document_number_c


class NumbersTestCase(unittest.TestCase):

  def test_find_doc_number(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №3 к Договору о выдаче банковских гарантий №3256-5/876 от 06-02-2013 год, заключенному между '''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('3', findings[1]['number'][0])
    self.assertEqual('3256-5/876', findings[1]['number'])

  def test_find_doc_number_missing_na(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения № на на ыдаче'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual(0, len(findings))

  def test_find_doc_number_N(self):
    t = 'Договор чего-то-там N 16-89/44 г. Санкт-Петербург    '
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual(1, len(findings))
    self.assertEqual('16-89/44', findings[0]['number'])

  def test_find_doc_number_missing___(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения № ____ на ыдаче'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual(0, len(findings))

  def test_find_doc_number_no_dot(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №343434.'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('343434', findings[0]['number'])

  def test_find_doc_number_same_digits_dot(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №111111111.'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('111111111', findings[0]['number'])

  def test_find_doc_number_uppercased_alpha(self):
    t = '''Одобрить соглашения №БУГАГА и далее'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('БУГАГА', findings[0]['number'])

  def test_find_doc_number_two_upper_space(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №ДК834/34-2.'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('ДК834/34-2', findings[0]['number'])

  def test_find_doc_number_last_g_with_dot(self):
    t = ''''Договор пожертвования N 16-89/44 г. Санкт-Петербург                     «11» декабря 2018 год.\nМуниципальное бюджетное учреждение города Москвы «Радуга» именуемый в дальнейшем «Благополучатель»'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('16-89/44', findings[0]['number'])

  def test_find_doc_number_two_upper_space_latin(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №XK834/34-2.'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('XK834/34-2', findings[0]['number'])

  def test_find_doc_number_two_upper_space_latin_eol(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №XK834/34-2'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('XK834/34-2', findings[0]['number'])

  def test_find_doc_number_two_upper_space_latin_eol_after_spaces(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №  XK834/34-2'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('XK834/34-2', findings[0]['number'])

  def test_find_doc_number_two_upper_dash(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №ДК-834 что-то'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual('ДК-834', findings[0]['number'])

  def test_find_doc_number_license(self):
    t = '''действия на базе лицензнии №ДК-834 от 12.23 что-то'''
    findings = list(re.finditer(document_number_c, t))

    self.assertEqual(0, len(findings))


if __name__ == '__main__':
  unittest.main()
