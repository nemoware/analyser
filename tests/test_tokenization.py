#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from documents import TextMap


class TopkenizationTestCase(unittest.TestCase):

  def test_map_tokens_in_range(self):
    text = '1.2. мама   да'
    tm = TextMap(text)

    tokens = tm.tokens_in_range([0, 2])
    self.assertEqual(len(tokens), 2)
    self.assertEqual(tokens[0], '1.2.')
    self.assertEqual(tokens[1], 'мама')

  def test_map_text_range(self):
    text = '1.2. мама   молилась Раме\n\nРама -- Вишну, А Вишну ел... черешню? (черешня по 10 руб. 20 коп.)'

    tm = TextMap(text)
    t = tm.text_range([0, 3])
    self.assertEqual(t, '1.2. мама   молилась')

  def test_get_tokens(self):
    text = '1.2. мама   молилась Раме\n\nРама -- Вишну, а Вишну ел... черешню? (черешня по 10 руб. 20 коп.)'

    tm = TextMap(text)

    print(tm.tokens)


  def test_get_len(self):
    text = 'а б с'
    tm = TextMap(text)

    self.assertEqual(3, len(tm))



  def test_get_by_index(self):
    text = 'а б с'
    tm = TextMap(text)

    self.assertEqual(tm[0], 'а')
    self.assertEqual(tm[1], 'б')
    self.assertEqual(tm[2], 'с')

    for x in tm:
      print (x)

    print (tm[0:2])




if __name__ == '__main__':
  unittest.main()
