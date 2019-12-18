#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import re
import unittest

from analyser.dates import find_date, document_number_c


class DatesTestCase(unittest.TestCase):
  # def test_parse_date(self):
  #   parse_date('15» мая 2014 года')

  def test_find_date_0(self):
    txt = '''
    Договор на оказание услуг по телеметрическому и технологическому сопровождению при наклонно-направленном и горизонтальном бурении) №100 - КС между Акционерное общество «Тумманность Рассудка» (АО «Тумманность Рассудка») (наименование дочернего общества ПАО «ГКЧП») и Общество с ограниченной ответственностью «Нептунишка» (ООО «Нептунишка») (наименование контрагента) г. Москва 2019 год.
  
  
  ДОГОВОР №100 - КС на оказание услуг по телеметрическому и технологическому сопровождению при наклонно-направленном и горизонтальном бурении
  
  г. Москва     15 Апреля 2015 год.
   
  Акционерное общество «Тумманность Рассудка» (АО «Тумманность Рассудка»), именуемое в дальнейшем «Тумманность», в лице Генерального директора Леопольда Никанорыча Вжик, действующего на основании Сустава, с одной стороны, 
    '''
    span, date_ = find_date(txt)
    self.validate_date(15, 4, 2015, date_)

    span, date_ = find_date(txt.lower())
    self.validate_date(15, 4, 2015, date_)

    span, date_ = find_date(txt.upper())
    self.validate_date(15, 4, 2015, date_)

  def validate_date(self, d, m, y, date_):
    self.assertIsNotNone(date_)
    self.assertEqual(d, date_.day)
    self.assertEqual(m, date_.month)
    self.assertEqual(y, date_.year)

  def test_find_doc_number(self):
    t = '''Одобрить сделку, связанную с заключением Дополнительного соглашения №3 к Договору о выдаче банковских гарантий №3256-5/876 от 06-02-2013 год, заключенному между '''
    findings = re.finditer(document_number_c, t)

    self.assertEqual('№3 ', next(findings)[0])
    self.assertEqual('№3256-5/876 ', next(findings)[0])

  def test_find_date_1(self):
    txt = '''
      Протокол заседания Совета директоров Общества с ограниченной ответственностью «Учкудук» (ООО «Учкудук» или «кудук»)
  Место проведения заседания:

  г. Москва, 



  Дата и время проведения заседания:

  «15» мая 2014 года, 18 часов 00 минут.



  Форма проведения заседания:

  заочное голосование'''

    span, date_ = find_date(txt)
    self.validate_date(15, 5, 2014, date_)

  def test_find_date_2(self):
    txt = '''1 мая 2014 года'''

    span, date_ = find_date(txt)
    self.validate_date(1, 5, 2014, date_)

  def test_find_date_3(self):
    txt = '''1 марта 2014 года'''

    span, date_ = find_date(txt)
    self.validate_date(1, 3, 2014, date_)

  def test_find_date_3_1(self):
    txt = '''1 янв 2014 года'''

    span, date_ = find_date(txt)
    self.validate_date(1, 1, 2014, date_)

  def test_find_date_3_0(self):
    txt = '''1 января 2014 года'''

    span, date_ = find_date(txt)
    self.validate_date(1, 1, 2014, date_)

  def test_find_date_4(self):
    txt = '''1.12.2019'''

    span, date_ = find_date(txt)
    self.validate_date(1, 12, 2019, date_)

  def test_find_date_4_1(self):
    txt = '''01.12.2019'''

    span, date_ = find_date(txt)
    self.validate_date(1, 12, 2019, date_)

  def test_find_date_5(self):
    txt = '''01-12-2019'''

    span, date_ = find_date(txt)
    self.validate_date(1, 12, 2019, date_)

  def test_find_date_6(self):
    txt = '''1-12-2019'''

    span, date_ = find_date(txt)
    self.validate_date(1, 12, 2019, date_)


if __name__ == '__main__':
  unittest.main()
