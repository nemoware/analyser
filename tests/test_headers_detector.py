#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import os
import pickle
import unittest

from headers_detector import doc_features, load_model
from legal_docs import LegalDocument


class TestHeaderDetector(unittest.TestCase):

  def test_doc_features(self):
    with open(os.path.join(os.path.dirname(__file__), '2. Договор по благ-ти Радуга.docx.pickle'), 'rb') as handle:
      contract: LegalDocument = pickle.load(handle)

    features, body_lines_ranges = doc_features(contract.tokens_map)
    self.assertEqual(27, len(features))
    print(features[0])
    pass

  def test_doc_features_predict(self):
    with open(os.path.join(os.path.dirname(__file__), '2. Договор по благ-ти Радуга.docx.pickle'), 'rb') as handle:
      contract: LegalDocument = pickle.load(handle)

    features, body_lines_ranges = doc_features(contract.tokens_map)

    model = load_model()
    predictions = model.predict(features)

    headlines_cnt = 0
    for i in range(len(predictions)):
      if predictions[i]:
        headlines_cnt += 1
        print(f'{i}\t🎖{contract.tokens_map.text_range(body_lines_ranges[i])}❗')
    self.assertEqual(8, headlines_cnt)

  def test_doc_features_predict_2(self):

    with open(os.path.join(os.path.dirname(__file__), 'Договор 8.docx.pickle'), 'rb') as handle:
      contract: LegalDocument = pickle.load(handle)

    features, body_lines_ranges = doc_features(contract.tokens_map)

    model = load_model()
    predictions = model.predict(features)

    headlines_cnt = 0
    for i in range(len(predictions)):
      if predictions[i]:
        headlines_cnt += 1
        print(f'{i}\t🎖{contract.tokens_map.text_range(body_lines_ranges[i])}❗')
    self.assertEqual(8, headlines_cnt)

  def test_doc_features_predict_3(self):

    with open(os.path.join(os.path.dirname(__file__), 'Договор _2_.docx.pickle'), 'rb') as handle:
      contract: LegalDocument = pickle.load(handle)

    features, body_lines_ranges = doc_features(contract.tokens_map)

    model = load_model()
    predictions = model.predict(features)

    headlines_cnt = 0
    for i in range(len(predictions)):
      if predictions[i]:
        headlines_cnt += 1
        print(f'{i}\t🎖{contract.tokens_map.text_range(body_lines_ranges[i])}❗')
    self.assertEqual(9, headlines_cnt)

  def test_doc_features_predict_4(self):

    with open(os.path.join(os.path.dirname(__file__), 'Договор 2.docx.pickle'), 'rb') as handle:
      contract: LegalDocument = pickle.load(handle)

    features, body_lines_ranges = doc_features(contract.tokens_map)

    model = load_model()
    predictions = model.predict(features)

    headlines_cnt = 0
    for i in range(len(predictions)):
      if predictions[i]:
        headlines_cnt += 1
        print(f'{i}\t🎖{contract.tokens_map.text_range(body_lines_ranges[i])}❗')

    for x in contract.paragraphs:
      print(x.header)

    print (contract.text)
    self.assertEqual(8, headlines_cnt)

unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
