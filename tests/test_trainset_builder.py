#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from contract_agents import *
from contract_augmentation import *
from documents import MarkedDoc

from text_tools import nltk_treebank_word_tokenizer


class TestTrainsetBuilder(unittest.TestCase):


  def test_remove_char_d(self):
    print(nltk_treebank_word_tokenizer.span_tokenize('sfdsf dsf'))



unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
