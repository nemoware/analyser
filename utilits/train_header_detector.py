#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from doc_structure import get_tokenized_line_number
from integration.word_document_parser import WordDocParser, join_paragraphs
from text_tools import Tokens

models_pth = os.path.join(Path(__file__).parents[1], 'vocab')
print(models_pth)

files_dir = '/Users/artem/Downloads/Telegram Desktop/X0/'

popular_headers = pd.DataFrame.from_csv(os.path.join(models_pth, 'headers_by_popularity.csv'))[2:50]
popular_headers = list(popular_headers['text'])

from joblib import dump


def _count_strange_symbols(txt, strange_symbols):
  res = 0
  for c in strange_symbols:
    res += txt.count(c)
  return res


def line_features(tokens: Tokens):
  features = {}
  txt = ' '.join(tokens)

  numbers, span, k, s = get_tokenized_line_number(tokens, 0)
  if not numbers:
    numbers = []
    number_minor = 0
  else:
    number_minor = numbers[-1]

  header_id = ' '.join(tokens[span[1]:])
  header_id = header_id.lower()
  # print (numbers, s, k)

  all_upper = header_id.upper() == header_id

  features['popular'] = header_id in popular_headers

  features['new_lines'] = txt.count('\n')
  features['has_contract'] = txt.lower().find('договор')
  features['all_uppercase'] = all_upper
  features['len_tokens'] = len(tokens)
  features['len_chars'] = len(txt)
  features['number_level'] = len(numbers)
  features['number_minor'] = number_minor
  features['number_roman'] = s
  features['dots'] = txt.count('.')
  features['commas'] = txt.count(',')
  features['brackets'] = txt.count(')')
  features['dashes'] = txt.count('-')
  features['strange_symbols'] = _count_strange_symbols(txt, '[_$@+]_?^&')

  return features


def doc_line_features(contract):
  tmap = contract.tokens_map
  features = []
  for p in contract.paragraphs:

    header_tokens = tmap[p.header.slice]
    header_features = line_features(header_tokens)
    header_features['actual'] = 1
    print('☢️', header_tokens)
    features.append(header_features)

    bodymap = tmap.slice(p.body.slice)
    body_lines_ranges = bodymap.split_spans('\n', add_delimiter=True)
    # line_number:int=0
    for line_span in body_lines_ranges:
      line_tokens = bodymap.tokens_by_range(line_span)
      # print('📃', line_tokens )
      body_features = line_features(line_tokens)
      body_features['actual'] = 0
      features.append(body_features)
      # print('📃-', body_features )

  return features


def read_all_contracts():
  client = MongoClient('mongodb://localhost:27017/')
  db = client['gpn']
  collection = db['legaldocs']

  wp = WordDocParser()
  filenames = wp.list_filenames(files_dir)

  cnt = 0
  failures = 0

  for fn in filenames:

    shortfn = fn.split('/')[-1]
    pth = '/'.join(fn.split('/')[5:-1])
    _doc_id = pth + '/' + shortfn

    cnt += 1
    print(cnt, fn)

    res = collection.find_one({"_id": _doc_id, 'version': wp.version})
    if res is None:
      try:
        # parse and save to DB
        res = wp.read_doc(fn)

        res['short_filename'] = shortfn
        res['path'] = pth
        res['_id'] = _doc_id
        res['version'] = wp.version

        collection.delete_one({'_id': _doc_id})
        collection.insert_one(res)

        yield res

      except Exception:
        print(f"{fn}\nException in WordDocParser code:")
        traceback.print_exc(file=sys.stdout)
        failures += 1

    else:
      yield res
      print(cnt, res["documentDate"], res["documentType"], res["documentNumber"])


"""
  ACHTUNG! ["] not found with text.find, next text is: ``
55о 05`00``
в.д.
Точка №11

  """

if __name__ == '__main__':

  features_dicts = []
  count = 0
  for c in read_all_contracts():

    contract = join_paragraphs(c, c['_id'])
    _doc_features = doc_line_features(contract)
    features_dicts += _doc_features

    count += 1
    if count > 125: break

  featuresX_data = pd.DataFrame.from_records(features_dicts)

  labels = np.array(featuresX_data['actual'])
  # Remove the labels from the features
  # axis 1 refers to the columns
  features = featuresX_data.drop('actual', axis=1)
  # Saving feature names for later use
  feature_list = list(features.columns)
  # Convert to numpy array
  features = np.array(features)

  # Split the data into training and testing sets
  train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                              random_state=42)
  print('Training Features Shape:', train_features.shape)
  print('Training Labels Shape:', train_labels.shape)
  print('Testing Features Shape:', test_features.shape)
  print('Testing Labels Shape:', test_labels.shape)

  rf = RandomForestClassifier(n_estimators=500, random_state=42)
  # Train the model
  rf.fit(train_features, train_labels)

  # Use the forest's predict method on the test data
  predictions = rf.predict(test_features)
  # # Calculate the absolute errors
  errors = abs(predictions - test_labels)
  # # Print out the mean absolute error (mae)
  print('Mean Absolute Error:', round(np.mean(errors), 4), 'degrees.')

  dump(rf, os.path.join(models_pth, 'rf_headers_detector_model.joblib'))
