#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from charter_parser import CharterDocumentParser
from charter_patterns import CharterPatternFactory
from contract_parser import ContractAnlysingContext

from doc_structure import remove_similar_indexes_considering_weights
from embedding_tools import AbstractEmbedder
from legal_docs import *
from parsing import print_prof_data
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

  def test_remove_similar_indexes_considering_weights(self):
    a = []
    w = []

    remove_similar_indexes_considering_weights(a,w)

  def test_remove_similar_indexes_considering_weights_2(self):
    a = [1,2]
    w = [99,0,1,99]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r,[2])

  def test_remove_similar_indexes_considering_weights_3(self):
    a = [1,2]
    w = [99,1,0,99]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r,[1])

  def test_remove_similar_indexes_considering_weights_4(self):
    a = [1,2,4]
    w = [99,1,0,99,0]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r,[1,4])

  def test_remove_similar_indexes_considering_weights_5(self):
    a = [1,2,4,5]
    w = [99,1,0,99,0,1]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r,[1,5])

  def test_embedd_headlines_0(self):

    from renderer import SilentRenderer
    point1 = [1, 6, 4]
    emb = FakeEmbedder(point1)
    ctx = ContractAnlysingContext(emb, SilentRenderer())
    ctx.analyze_contract("1. ЮРИДИЧЕСКИЙ содержание 4.")

    ctx._logstep("analyze_charter")

  def test_charter_parser (self):
    # from renderer import SilentRenderer
    point1 = [1, 6, 4]

    cpf = CharterPatternFactory(FakeEmbedder(point1))
    ctx = CharterDocumentParser(cpf)

    ctx.analyze_charter("1. ЮРИДИЧЕСКИЙ содержание 4.")

    ctx._logstep("analyze_charter")
    ctx._logstep("analyze_charter 2")


  def test_embedd_headlines(self):
    charter_text_1 = """
        e
    
        1. ЮРИДИЧЕСКИЙ содержание 4.
        2. ЮРИДИЧЕСКИЙ СТАТУС.
         
        что-то
            1. Общество является юридическим лицом согласно законодательству.
        что-то иное 
        3. УСТАВНЫЙ КАПИТАЛ. 
        и более  
        """
    charter_text_1_ = """
                      2. correct
                          no number
                          no number
                        2.1 correct
                        2.2 correct
                        2.3 correct
                          2.3.1 correct
                          - bullet
                          - bullet
                      4 INcorrect (expected: 2.4)
                          no number
                      3. correct
                        3.1 correct
                          3.1.2 correct
                            no number
                          1.1 INcorrect
                            no number:
                              1) ket correct 1
                              2) ket correct 2
                        3.2 correct
                      4. correct
                        1. INcorrect (expected: 4.1)
                        4.2 correct
                          1) ket correct 4.4.1
                          2) ket correct 4.2.2
                      I correct Roman I
                        1). ket correct
                        2). ket correct
                      II correct Roman II
                        1 correct
                        2. correct
                          2.2 correct
                          no number


                    """
    # charter_text_1 = "Заголовок"

    TCD = CharterDocument(charter_text_1)
    TCD.right_padding = 0
    TCD.parse()


    # TCD.structure.print_structured(TCD)

    # r = highlight_doc_structure(TCD)
    # headline_indexes = np.nonzero(r['result'])[0]
    # print(headline_indexes)

    headline_indexes2=TCD.structure.headline_indexes
    print(headline_indexes2)
    # --

    print('-'*50,'lines len' )
    for i in range( len(TCD.structure.structure)):
      # l = TCD.structure.structure[i].to_string(TCD.tokens_cc)
      # print(f'[{l}]')
      li = TCD.structure.structure[i]
      print( f'\t {li.level}\t #{li.number} \t--> {li._possible_levels}\t {li.subtokens(TCD.tokens_cc)}')

    # print('-'*50,'headline_indexes len', len(headline_indexes))
    # for i in headline_indexes:
    #   l=TCD.structure.structure[i].to_string(TCD.tokens_cc)
    #   print(f'[{l}]')

    print('-'*50,'headline_indexes2 len', len(headline_indexes2))
    for i in headline_indexes2:
      TCD.structure.structure[i].print(TCD.tokens_cc)
      # print(f'[{l}]')


    # # point1 = [1, 6, 4]
    # # emb = FakeEmbedder(point1)
    # _embedded_headlines = embedd_headlines(headline_indexes, TCD, None)

if __name__ == '__main__':
  unittest.main()
