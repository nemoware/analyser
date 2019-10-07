#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from doc_structure import remove_similar_indexes_considering_weights, DocumentStructure, get_tokenized_line_number
from legal_docs import LegalDocument


class DocumentStructureTestCase(unittest.TestCase):

  def test_remove_similar_indexes_considering_weights(self):
    a = []
    w = []

    remove_similar_indexes_considering_weights(a, w)

  def test_remove_similar_indexes_considering_weights_2(self):
    a = [1, 2]
    w = [99, 0, 1, 99]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r, [2])

  def test_remove_similar_indexes_considering_weights_3(self):
    a = [1, 2]
    w = [99, 1, 0, 99]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r, [1])

  def test_remove_similar_indexes_considering_weights_4(self):
    a = [1, 2, 4]
    w = [99, 1, 0, 99, 0]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r, [1, 4])

  def test_remove_similar_indexes_considering_weights_5(self):
    a = [1, 2, 4, 5]
    w = [99, 1, 0, 99, 0, 1]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r, [1, 5])

  def test_get_tokenized_line_number(self):
    n, span, level, roman = get_tokenized_line_number('2. correct'.split(' '),0)
    self.assertEqual([2], n)
    self.assertEqual(1, level)
    self.assertEqual(False, roman)

    n, span, level, roman = get_tokenized_line_number('II correct Roman II	'.split(' '), 0)
    self.assertEqual([2], n)
    self.assertEqual(0, level)
    self.assertEqual(True, roman)

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


    ds = DocumentStructure()
    ds.detect_document_structure(LegalDocument(charter_text_1_).parse().tokens_map)

    numbered = ds.get_numbered_lines()
    for n in numbered:
      print(n.number, n.slice)

    # headline_indexes2 = ds.headline_indexes
    # print(headline_indexes2)
    #
    # print('-' * 50, 'lines len')
    # for i in range(len(ds.structure)):
    #   li = ds.structure[i]
    #   print(f'\t {li.level}\t #{li.number} \t--> {li._possible_levels}\t  ')
    #
    # print('-' * 50, 'headline_indexes2 len', len(headline_indexes2))


if __name__ == '__main__':
  unittest.main()
