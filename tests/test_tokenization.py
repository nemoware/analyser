#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from nltk import TreebankWordTokenizer

from documents import TextMap, span_tokenize
from legal_docs import CharterDocument, LegalDocument


class TokenisationTestCase(unittest.TestCase):

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
    self.assertEqual('аслово бслово', doc.text)

    doc1 = doc_o.subdoc_slice(slice(2, 3))
    self.assertEqual('цслово', doc1.text)

    doc2 = doc.subdoc_slice(slice(0, 2))
    self.assertEqual('аслово бслово', doc2.text)

    doc3 = doc2.subdoc_slice(slice(1, 2))
    self.assertEqual('бслово', doc3.text)

    doc4 = doc3.subdoc_slice(slice(5, 6))
    self.assertEqual('', doc4.text)

  def test_span_tokenize(self):
    text = 'УТВЕРЖДЕН.\n\nОбщим собранием `` акционеров собранием `` акционеров \'\' '
    spans = span_tokenize(text)

    print(spans)
    for c in spans:
      print(c)

  def test_word_tokenize_quotes(self):
    text = '"сл"'
    tokenizer = TreebankWordTokenizer()
    # _spans = nltk.word_tokenize(text)
    _spans = tokenizer.tokenize(text)

    spans = [s for s in _spans]
    print("".join(spans))
    for c in spans:
      print(len(c))
    self.assertEqual(3, len(spans))

  def test_span_tokenize_quotes(self):
    text = '"слово"'
    _spans = span_tokenize(text)

    spans = [s for s in _spans]
    print(spans)
    self.assertEqual(3, len(spans))

  def test_slice(self):
    text = 'этилен мама   ಶ್ರೀರಾಮ'
    tm = TextMap(text)
    tm2: TextMap = tm.slice(slice(1, 2))

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

  def test_finditer(self):
    from transaction_values import _re_greather_then
    text = """стоимость, равную или превышающую 2000000 ( два миллиона ) долларов сша, но менее"""
    tm = TextMap(text)
    iter = tm.finditer(_re_greather_then)

    results = [t for t in iter]
    results = results[0]

    self.assertEqual('превышающую', tm.text_range(results))
    self.assertEqual(4, results[0])
    self.assertEqual(5, results[1])

  def test_finditer__a(self):
    from transaction_values import _re_greather_then

    text = """стоимость, равную или превышающую 2000000 ( два миллиона ) долларов сша, но менее"""
    __doc = LegalDocument(text)
    __doc.parse()
    doc = __doc.subdoc_slice(slice(2, len(__doc.tokens_map)))
    tm = doc.tokens_map
    iter = tm.finditer(_re_greather_then)

    spans_ = [t for t in iter]
    spans = spans_[0]

    self.assertEqual('превышающую', tm.text_range(spans))
    self.assertEqual(2, spans[0])
    self.assertEqual(3, spans[1])

  def test_slice_doc_1(self):
    text = 'этилен мама этилен'
    __doc = LegalDocument(text)
    __doc.parse()

    subdoc = __doc.subdoc_slice(slice(2, 3))
    self.assertEqual('этилен', subdoc.text)

  def test_slice_doc_2(self):
    text = 'этилен мама этилен'
    __doc = LegalDocument(text)
    __doc.parse()
    tm: TextMap = __doc.tokens_map
    subdoc = __doc.subdoc_slice(slice(0, 1))
    del __doc

    tm2: TextMap = subdoc.tokens_map

    self.assertEqual('этилен', tm2[0])
    self.assertEqual('этилен', tm2.text)

    self.assertEqual(0, tm2.token_index_by_char(1))

  def test_slice_doc_3(self):
    text = 'этилен мама этилен'
    __doc = LegalDocument(text)
    __doc.parse()
    tm: TextMap = __doc.tokens_map
    subdoc = __doc.subdoc_slice(slice(1, 3))
    del __doc

    tm2: TextMap = subdoc.tokens_map

    self.assertEqual('мама', tm2[0])
    self.assertEqual('этилен', tm2[1])
    self.assertEqual('мама этилен', tm2.text)

    self.assertEqual(0, tm2.token_index_by_char(1))
    self.assertEqual(1, tm2.token_index_by_char(6))
    self.assertEqual(1, tm2.token_index_by_char(5))

  def test_token_indices_by_char_range(self):
    text = 'мама'
    span = [0, 4]
    expected = text[span[0]:span[1]]
    print(expected)

    tm = TextMap(text)  # tokenization
    ti = tm.token_indices_by_char_range_2(span)
    self.assertEqual(0, ti[0])
    self.assertEqual(1, ti[1])

    self.assertEqual(expected, tm.text_range(ti))

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
    self.assertEqual(6, len(tm.tokens))

  def test_get_len(self):
    text = 'а б с'
    tm = TextMap(text)

    self.assertEqual(3, len(tm))

  def test_concat_TextMap(self):

    tm1 = TextMap('a')
    tm2 = TextMap('b')

    tm1 += tm2
    self.assertEqual('ab', tm1.text)
    self.assertEqual('a', tm1.tokens[0])
    self.assertEqual('b', tm1.tokens[1])

    self.assertEqual(2, len(tm1))
    self.assertEqual(1, len(tm2))

  def test_concat_TextMap2(self):

    tm1 = TextMap('alpha \n')
    tm2 = TextMap('bet')

    tm1 += tm2

    self.assertEqual(3, len(tm1))
    self.assertEqual(1, len(tm2))

    self.assertEqual('alpha \nbet', tm1.text)
    self.assertEqual('alpha', tm1.tokens[0])
    self.assertEqual('bet', tm1.tokens[2])

  def test_concat_TextMap3(self):

    tm1 = TextMap('text prefix \n')
    tm2 = TextMap('more words')

    N = 10
    expected_tokens = len(tm1.tokens) + N * len(tm2.tokens)
    for i in range(N):
      tm1 += tm2

    self.assertEqual(expected_tokens, len(tm1))

  def test_concat_docs(self):

    tm1 = LegalDocument('text prefix \n').parse()
    tm2 = LegalDocument('more words').parse()


    N = 10
    expected_tokens = len(tm1.tokens) + N * len(tm2.tokens)
    for i in range(N):
      tm1 += tm2

    # //tm1.parse()
    self.assertEqual(expected_tokens, len(tm1.tokens))

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
