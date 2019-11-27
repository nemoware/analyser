#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# import json
import unittest

import numpy as  np
import pandas as pd

from analyser.ml_tools import calc_distances_per_pattern, attribute_patternmatch_to_index


class TestMlTools(unittest.TestCase):

  def test_calc_distances_per_pattern(self):
    pattern_names = ['a', 'b', 'c']
    pattern__embeddings = [[1, 0, 0], [1, 1, 0], [0.1, 0, 0.2]]
    sentences_embeddings = np.array([[0.1, 0, 0], [0.1, 0, 0.2]])

    patterns_named_embeddings = pd.DataFrame(pattern__embeddings, columns=pattern_names)

    # ---
    adf = calc_distances_per_pattern(sentences_embeddings, patterns_named_embeddings)
    # ---
    self.assertEqual(len(sentences_embeddings), len(adf))
    self.assertLessEqual(adf['c'][0], 0.01)
    print(adf)
    print(adf['c'][0])

  def test_attribute_patternmatch_to_index(self):
    pattern_names = ['a', 'b', 'c', 'd']
    header_to_pattern_distances_ = pd.DataFrame(np.array( [[0,1],[0,2],[0,4],[0,3]]).T, columns=pattern_names)
    attribute_patternmatch_to_index(header_to_pattern_distances_ ,
                                    threshold=0)

unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
