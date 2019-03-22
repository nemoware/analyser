#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8



import unittest

from legal_docs import LegalDocument
from patterns import *


class LegalDocumentTestCase(unittest.TestCase):
    def test_normalize_sentences_bounds(self):
        d = LegalDocument()

        text = ""
        self.assertEqual(text, d.normalize_sentences_bounds(text))

        text="A"
        self.assertEqual(text, d.normalize_sentences_bounds(text))

        text = "A."
        self.assertEqual(text, d.normalize_sentences_bounds(text))

        text = "Ай да А.С. Пушкин! Ай да сукин сын!"
        print ( d.normalize_sentences_bounds(text))



    def test_parse(self):
        d = LegalDocument("a")
        d.parse()
        print(d.tokens)
        self.assertEqual(1 + TEXT_PADDING, len(d.tokens))


if __name__ == '__main__':
    unittest.main()
