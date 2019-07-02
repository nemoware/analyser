#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from contract_agents import *
from contract_augmentation import *
from documents import MarkedDoc, SpmGTokenizer
# from text_tools import nltk_treebank_word_tokenizer
from text_tools import token_at_index, token_at_index_


def n(x):
  return normalize_contract(x)


class TestAugm(unittest.TestCase):

  # def test_remove_char_d(self):
  #   print(nltk_treebank_word_tokenizer.span_tokenize('sfdsf dsf'))

  def test_tokenizer(self):
    tz = SpmGTokenizer()
    txt = u"""как   ныне 
    сбирается вещий чувак 
    
    отмстить \n отмстить отмстить xdfg отмстить
    ненасытным баранам"""
    # char_index=5
    tokens = tz.tokenize(txt)
    # tokens[0]='как'
    print(len(tokens), tokens)
    restored = tz.untokenize(tokens)
    self.assertEqual(txt, restored)


  def test_char_to_token(self):
    tz = SpmGTokenizer()
    txt = """
    
    
    
    как     ныне
    сбирается вещий чувак    ЧТОТОТОТОТТОТОТО
    
    
    
    отмстить
    ненасытным баранам
    
    ч
    
    """
    # char_index=5
    txt = txt.strip()
    tokens1 = tz.tokenize(txt)
    # print(tokens1)
    # print('txt[char_index]=', txt[char_index])
    # token_n = token_at_index(char_index, txt, tz)
    # print('token_n=', token_n, tokens1[token_n])

    for char_index in range(0, len(txt)):
      charr = txt[char_index]
      token_n = token_at_index_(char_index, tokens1, tz)
      token = tokens1[token_n].replace('▁', ' ')

      # try:
      token.index(charr)
      print(f'char_index={char_index} \t token=[{token}] \t char=[{charr}] \t token_n={token_n}')
      # except:
      #   print(f'ERROR: char_index={char_index} \t token=[{token}] \t char=[{charr}] \t token_n={token_n}')

  def test_remove_char(self):
    doc = MarkedDoc(['12345', '12345', '12345', '12345', '12345'], [1, 2, 3, 4, 5])

    augment_dropout_chars_d(doc, 0.5)
    print(doc.tokens)

  def test_remove_tokens(self):
    doc = MarkedDoc(['12345', '12345', '12345', '12345', '12345'], [1, 2, 3, 4, 5])

    augment_dropout_words_d(doc, 0.5)
    print(doc.tokens, doc.categories_vector)

  def test_alter_case(self):
    doc = MarkedDoc(['aaaa', 'AAA', 'bbb', 'BBB', 'cccc'], [1, 2, 3, 4, 5])

    augment_alter_case_d(doc, 1)
    print(doc.tokens, doc.categories_vector)

  def test_drop_punkt(self):
    doc = MarkedDoc([',', 'AAA', '"', 'BBB', '.'], [1, 2, 3, 4, 5])

    augment_dropout_punctuation_d(doc, 1)
    print(doc.tokens, doc.categories_vector)

  # def test_concat (self):
  #   doc = MarkedDoc([',', 'AAA', '"', 'BBB', '.'], [1, 2, 3, 4, 5])
  #   doc2 = MarkedDoc(['12345', '12345', '12345', '12345', '12345'], [1, 2, 3, 4, 5])
  #   doc.concat(doc2)
  #   print(doc.tokens, doc.categories_vector)

  def test_trim(self):
    doc = MarkedDoc(['12345', '12345', '12345', '12345', '12345'], [1, 2, 3, 4, 5])
    augment_trim(doc, 1)
    print('augment_trim', doc.tokens, doc.categories_vector)
    self.assertGreaterEqual(doc.get_len(), 3)

    for c in range(5):
      augment_trim(doc, 1)
      print('augment_trim', doc.tokens, doc.categories_vector)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
