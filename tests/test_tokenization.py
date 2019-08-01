#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from documents import TextMap, span_tokenize
from legal_docs import CharterDocument


class TopkenizationTestCase(unittest.TestCase):

  def test_normalize_doc_slice(self):
    doc_text = """\n\n\nАкционерное общество «Газпром - Вибраниум и Криптонит» (АО «ГВК»), именуемое в собранием `` акционеров собранием `` акционеров \'\' \
        дальнейшем «Благотворитель», в лице заместителя генерального директора по персоналу и \
        организационному развитию Неизвестного И.И., действующего на основании на основании Доверенности № Д-17 от 29.01.2018г, \
        с одной стороны, и Фонд поддержки социальных инициатив «Интерстеларные пущи», именуемый в дальнейшем «Благополучатель», \
        в лице Генерального директора ____________________действующего на основании Устава, с другой стороны, \
        именуемые совместно «Стороны», а по отдельности «Сторона», заключили настоящий Договор о нижеследующем:
        """
    doc_o = CharterDocument(doc_text)
    doc_o.parse()

    doc = doc_o.subdoc_slice(slice(0, 300))

    self.assertEqual(doc.tokens_map.text.lower(), doc.tokens_map_norm.text.lower())

    for i in range(len(doc.tokens)):
      self.assertEqual(doc.tokens[i].lower(), doc.tokens_cc[i].lower())

  def test_subdoc_slice(self):
    doc_text = """аслово бслово цслово"""

    doc_o = CharterDocument(doc_text)
    doc_o.parse()

    doc = doc_o.subdoc_slice(slice(0, 2))
    self.assertEqual('аслово бслово',doc.text )

    doc1 = doc_o.subdoc_slice(slice(2, 3))
    self.assertEqual('цслово', doc1.text)

    doc2 = doc.subdoc_slice(slice(0, 2))
    self.assertEqual('аслово бслово', doc2.text)

    doc3 = doc2.subdoc_slice(slice(1, 2))
    self.assertEqual('бслово', doc3.text)

    doc4 = doc3.subdoc_slice(slice(5, 6))
    self.assertEqual('', doc4.text)

    # self.assertEqual(doc.tokens_map.text.lower(), doc.tokens_map_norm.text.lower())
    #
    # for i in range(len(doc.tokens)):
    #   self.assertEqual(doc.tokens[i].lower(), doc.tokens_cc[i].lower())

  def test_span_tokenize(self):
    text = 'УТВЕРЖДЕН.\n\nОбщим собранием `` акционеров собранием `` акционеров \'\' '
    spans = span_tokenize(text)

    print(spans)
    for c in spans:
      print(c)

  def test_slice(self):
    text = 'этилен мама   ಶ್ರೀರಾಮ'
    tm = TextMap(text)
    tm2:TextMap = tm.slice(slice(1, 2))

    self.assertEqual(tm2[0], 'мама')
    self.assertEqual(tm2.text, 'мама')

    tm3 = tm2.slice(slice(0, 1))
    self.assertEqual(tm3[0], 'мама')


    self.assertEqual(0, tm.token_index_by_char(1))
    self.assertEqual(0, tm2.token_index_by_char(1))
    self.assertEqual(0, tm3.token_index_by_char(1))

    self.assertEqual('мама', tm3.text)
    self.assertEqual('мама', tm3.text_range([0, 1]))
    self.assertEqual('мама', tm3.text_range([0, 2]))

  def test_sentence_at_index(self):

    tm = TextMap('стороны Заключили\n  договор  ПРЕДМЕТ \nДОГОВОРА')
    for i in range(len(tm)):
      print(i, tm[i])

    bounds = tm.sentence_at_index(0)
    print(bounds)
    print(tm.text_range(bounds))
    for i in range(0, 3):
      bounds = tm.sentence_at_index(i)
      self.assertEqual('стороны Заключили\n', tm.text_range(bounds), str(i))

    for i in range(3, 5):
      bounds = tm.sentence_at_index(i)
      self.assertEqual('договор  ПРЕДМЕТ \n', tm.text_range(bounds))

    for i in range(6, 7):
      bounds = tm.sentence_at_index(i)
      self.assertEqual('ДОГОВОРА', tm.text_range(bounds))

  def test_tokens_in_range(self):
    text = 'мама'
    tm = TextMap(text)

    self.assertEqual(0, tm.token_index_by_char(0))
    self.assertEqual(0, tm.token_index_by_char(1))
    self.assertEqual(0, tm.token_index_by_char(2))
    self.assertEqual(0, tm.token_index_by_char(3))

    text = 'мама выла папу'
    tm = TextMap(text)

    self.assertEqual(1, tm.token_index_by_char(5))
    self.assertEqual(1, tm.token_index_by_char(6))
    self.assertEqual(1, tm.token_index_by_char(7))
    self.assertEqual(1, tm.token_index_by_char(8))

    self.assertEqual(2, tm.token_index_by_char(9))
    self.assertEqual(1, tm.token_index_by_char(4))

  def test_map_tokens_in_range(self):
    text = '1.2. мама   ಶ್ರೀರಾಮ'
    tm = TextMap(text)

    tokens = tm.tokens_by_range([0, 2])
    self.assertEqual(len(tokens), 2)
    self.assertEqual(tokens[0], '1.2.')
    self.assertEqual(tokens[1], 'мама')

  def test_split(self):
    text = '1 2 3\nмама\nಶ್ರೀರಾಮ'
    tm = TextMap(text)

    for k in tm.split('\n'):
      print(k)

  def test_split_span(self):
    text = '1 2 3\nмама\nಶ್ರೀರಾಮ'
    tm = TextMap(text)

    spans = [s for s in tm.split_spans('\n', add_delimiter=True)]
    for k in spans:
      print(tm.text_range(k))

    self.assertEqual('1 2 3\n', tm.text_range(spans[0]))

  def test_map_text_range(self):
    text = """1.2. мама   молилась ಶ್ರೀರಾಮ\n\nРама -- Вишну, А Вишну 
    ел... черешню? (черешня по 10 руб. 20 коп.) '' """

    tm = TextMap(text)
    t = tm.text_range([0, 3])
    self.assertEqual(t, '1.2. мама   молилась')

  def test_get_tokens(self):
    text = 'ಉಂದು ಅರ್ತೊಪೂರ್ಣೊ ವಾಕ್ಯೊಲೆನ್ ಕೊರ್ಪುನ ಸಾಮರ್ತ್ಯೊನು ಹೊಂದೊಂತ್ '
    tm = TextMap(text)
    print(tm.tokens)

  def test_get_len(self):
    text = 'а б с'
    tm = TextMap(text)

    self.assertEqual(3, len(tm))

  def test_get_by_index(self):
    ಶ್ರೀರಾಮ = self
    ಮ = 'ቋንቋ የድምጽ፣ የምልክት ወይም የምስል ቅንብር ሆኖ ለማሰብ'
    ቅ = TextMap(ಮ)
    ಶ್ರೀರಾಮ.assertEqual(ቅ[0], 'ቋንቋ')
    ಶ್ರೀರಾಮ.assertEqual(ቅ[1], 'የድምጽ፣')
    ಶ್ರೀರಾಮ.assertEqual(ቅ[2], 'የምልክት')

    # test iteration
    for x in ቅ:
      print(x)

    # test slicing
    print(ቅ[0:2])


if __name__ == '__main__':
  unittest.main()
