#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from embedding_tools import AbstractEmbedder
from legal_docs import *
from patterns import *


class FakeEmbedder(AbstractEmbedder):

  def __init__(self, default_point):
    self.default_point = default_point

  def embedd_tokenized_text(self, tokenized_sentences_list, lens):
    # def get_embedding_tensor(self, tokenized_sentences_list):
    tensor = []
    for sent in tokenized_sentences_list:
      sentense_emb = []
      for token in sent:
        token_emb = self.default_point
        sentense_emb.append(token_emb)
      tensor.append(sentense_emb)

    return np.array(tensor), tokenized_sentences_list


class LegalDocumentTestCase(unittest.TestCase):

  def test_embedd_large(self):
    point1 = [1, 6, 4]
    emb = FakeEmbedder(point1)

    ld = LegalDocument('a b c d e f g h')

    ld.parse()
    print(ld.tokens)
    ld._embedd_large(emb, 5)

    # print(ld.embeddings)
    print(ld.tokens)
    print_prof_data()

    # point1 = [1, 6, 4]
    #
    # PF = AbstractPatternFactory(FakeEmbedder(point1))
    #
    # fp1 = PF.create_pattern('p1', ('prefix', 'pat 2', 'suffix'))
    # fp2 = PF.create_pattern('p2', ('prefix', 'pat', 'suffix 2'))
    # fp3 = PF.create_pattern('p3', ('', 'a b c', ''))
    #
    # self.assertEqual(3, len(PF.patterns))
    #
    # PF.embedd()
    #
    # self.assertEqual(2, len(fp1.embeddings))
    # self.assertEqual(1, len(fp2.embeddings))
    # self.assertEqual(3, len(fp3.embeddings))

  def test_normalize_sentences_bounds(self):
    d = LegalDocument()

    text = ""
    self.assertEqual(text, d.normalize_sentences_bounds(text))

    text = "A"
    self.assertEqual(text, d.normalize_sentences_bounds(text))

    text = "A."
    self.assertEqual(text, d.normalize_sentences_bounds(text))

    text = "Ай да А.С. Пушкин! Ай да сукин сын!"
    print(d.normalize_sentences_bounds(text))

  def test_parse(self):
    d = LegalDocument("a")
    d.parse()
    print(d.tokens)
    self.assertEqual(2, len(d.tokens))


  def test_parse_2(self):
    d = LegalDocument("a\nb")
    d.parse()
    print(d.tokens)
    self.assertEqual(4, len(d.tokens))


    self.assertEqual(2, len(d.structure.structure))

    l0 = d.structure.structure[0]
    lll = d.tokens_cc[l0.span[0]: l0.span[1]]

    print(lll)

    l1 = d.structure.structure[1]

    self.assertEqual('a', l0.to_string(d.tokens_cc))
    self.assertEqual('b', l1.to_string(d.tokens_cc))
    lll = d.tokens_cc[l1.span[0]: l1.span[1]]

    print(lll)


  def test_embedd_headlines(self):
    charter_text_1 = """
    x
        2. ЮРИДИЧЕСКИЙ СТАТУС. 
        что-то
            1. Общество является юридическим лицом согласно законодательству.
        что-то иное 
        3. УСТАВНЫЙ КАПИТАЛ. 
        и более  
        """
    TCD = CharterDocument(charter_text_1)
    TCD.right_padding = 0
    TCD.parse()

    # TCD.structure.print_structured(TCD)

    r, TCD = highlight_doc_structure(TCD)
    headline_indexes = np.nonzero(r['result'])[0]

    # --
    print('headline_indexes', len(headline_indexes))
    for i in headline_indexes:
      l=TCD.structure.structure[i].to_string(TCD.tokens_cc)
      print(l)

    # # point1 = [1, 6, 4]
    # # emb = FakeEmbedder(point1)
    # _embedded_headlines = embedd_headlines(headline_indexes, TCD, None)

if __name__ == '__main__':
  unittest.main()
