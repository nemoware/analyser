#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

import numpy as np

from legal_docs import BasicContractDocument, remove_similar_indexes


# from patterns import *


class SplitSectionsTest(unittest.TestCase):

    def test_find_sentence_beginnings(self):
        cd = BasicContractDocument()
        cd.tokens = ['a', 'a', 'a', '\n', 'a', 'a', 'a', '\n', 'a', '\n', 'a']
        row_to_index = [6, 1]

        sentence_starts = cd.find_sentence_beginnings(row_to_index)
        print(sentence_starts)
        self.assertTrue(np.allclose(sentence_starts, [3, 0]))

    def test_find_sentence_beginnings_2(self):
        cd = BasicContractDocument()
        cd.tokens = ['a', 'a', 'a', '\n', 'a', 'a', 'a', '\n', 'a', '\n', 'a']
        row_to_index = [6, 1]
        sentence_starts = cd.find_sentence_beginnings(row_to_index)
        print(sentence_starts)
        self.assertTrue(np.allclose(sentence_starts, [3, 0]))

    def test_remove_near_indexes_1(self):
        indexes = [0,0,0]
        filtered = remove_similar_indexes(indexes)
        print(filtered)
        self.assertTrue(np.allclose(filtered, [ 0]))

    def test_remove_near_indexes_2(self):
        indexes = [100 ,  10,   500]
        filtered = remove_similar_indexes(indexes, min_section_size=100)
        print(filtered)
        self.assertTrue(np.allclose(filtered, [100, 500]))




if __name__ == '__main__':
    unittest.main()
