#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

import numpy as np

from documents import TOKENIZER_DEFAULT
from renderer import HtmlRenderer, to_multicolor_text


class TestRender(unittest.TestCase):

  def test_to_multicolor_text(self):
    r = HtmlRenderer()
    tokens = TOKENIZER_DEFAULT.tokenize('Как ныне')
    print(tokens)
    s = to_multicolor_text(tokens, {'a': np.zeros(len(tokens))}, {'a': (1, 1, 1)})
    print(s)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
