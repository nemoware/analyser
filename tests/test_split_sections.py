#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from split import *


# from patterns import *


class SplitSectionsTest(unittest.TestCase):

    def test_find_sections_indexes(self):
        cd = BasicContractDocument()

        cd.tokens = ['a', 'a', 'a', '\n', 'a', 'a', 'a']
        dists = np.array(
            [[5.1, np.nan, 3.1, 4, 1],
             [1.1, 2.1, np.nan, 4, 5],
             [5.1, np.nan, 3.1, 4, 1]
             ])
        idxs = cd.find_sections_indexes(dists, min_section_size=1)
        print(idxs)

        self.assertTrue(np.allclose(idxs, [[1, 0], [0, 3]]))

    def test_find_sentence_beginnings(self):
        cd = BasicContractDocument()
        cd.tokens = ['a', 'a', 'a', '\n', 'a', 'a', 'a', '\n', 'a', '\n', 'a']
        row_to_index = [6,1]

        sentence_starts = cd.find_sentence_beginnings(row_to_index)
        print(sentence_starts)
        self.assertTrue(np.allclose(sentence_starts, [3, 0] ))

    def test_find_sentence_beginnings_2(self):
        cd = BasicContractDocument()
        cd.tokens = ['a', 'a', 'a', '\n', 'a', 'a', 'a', '\n', 'a', '\n', 'a']
        row_to_index = [6,1]
        sentence_starts = cd.find_sentence_beginnings(row_to_index)
        print(sentence_starts)
        self.assertTrue(np.allclose(sentence_starts, [3, 0]))

    def test_remove_near_indexes_1(self):
        cd = BasicContractDocument()

        indexes = [[0, 0], [0, 0], [0, 0]]
        filtered = cd.remove_similar_indexes(indexes, 0)
        print(filtered)
        self.assertTrue(np.allclose(filtered, [[0, 0]]))

    def test_remove_near_indexes_2(self):
        cd = BasicContractDocument()

        indexes = [[100, 1], [10, 1], [500, 0]]
        filtered = cd.remove_similar_indexes(indexes, 1, min_section_size=1)
        print(filtered)
        self.assertTrue(np.allclose(filtered, [[100, 1]]))

    def test_remove_near_indexes_3(self):
        cd = BasicContractDocument()

        indexes = [[10, 3000], [100, 2000], [500, 1000]]
        filtered = cd.remove_similar_indexes(indexes, 0, min_section_size=91)
        print(filtered)
        self.assertTrue(np.allclose(filtered, [[10, 3000], [500, 1000]]))

    def test_remove_near_indexes_3(self):
        cd = BasicContractDocument()

        indexes = [[10, 3000], [100, 2000], [500, 1000]]
        filtered = cd.remove_similar_indexes(indexes, 0, min_section_size=1)
        print(filtered)
        self.assertTrue(np.allclose(filtered, indexes))


if __name__ == '__main__':
    unittest.main()
