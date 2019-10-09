import os

import numpy as np
import pandas as pd
from joblib import load
from sklearn.ensemble import RandomForestClassifier

from doc_structure import get_tokenized_line_number
from documents import TextMap
from hyperparams import models_path
from integration.word_document_parser import PARAGRAPH_DELIMITER
from text_tools import Tokens

popular_headers = pd.DataFrame.from_csv(os.path.join(models_path, 'headers_by_popularity.csv'))[2:50]
popular_headers = list(popular_headers['text'])


def doc_features(tokens_map: TextMap):
  body_lines_ranges = tokens_map.split_spans(PARAGRAPH_DELIMITER, add_delimiter=True)

  _doc_features = []
  _line_spans = []
  ln = 0
  _prev_features=None
  for line_span in body_lines_ranges:
    _line_spans.append(line_span)

    _features = line_features(tokens_map.tokens_by_range(line_span), ln, _prev_features)
    _doc_features.append(_features)
    _prev_features = _features
    ln+=1
  doc_featuresX_data = pd.DataFrame.from_records(_doc_features)
  doc_features_data = np.array(doc_featuresX_data)

  return doc_features_data, _line_spans


def load_model() -> RandomForestClassifier:
  loaded_model = load(os.path.join(models_path, 'rf_headers_detector_model.joblib'))
  return loaded_model


def line_features(tokens: Tokens, line_number, prev_features):
  # TODO: add previous and next lines features
  txt = ' '.join(tokens)

  numbers, span, k, s = get_tokenized_line_number(tokens, 0)
  if not numbers:
    numbers = []
    number_minor = 0
  else:
    number_minor = numbers[-1]

  header_id = ' '.join(tokens[span[1]:])
  header_id = header_id.lower()

  all_upper = header_id.upper() == header_id

  features = {
    'line_number':line_number,
    'popular': header_id in popular_headers,
    # 'cr_count': txt.count('\r'),

    'has_contract': txt.lower().find('договор'),
    'has_article': txt.lower().find('статья'),
    'all_uppercase': all_upper,
    'len_tokens': len(tokens),
    'len_chars': len(txt),
    'number_level': len(numbers),
    'number_minor': number_minor,
    'number_roman': s,
    'dots': header_id.count('.'),
    'commas': header_id.count(','),
    'brackets': _count_strange_symbols(txt, '(){}[]'),
    'dashes': header_id.count('-'),
    'colons': header_id.count(':'),
    'semicolons': header_id.count(';'),
    'strange_symbols': _count_strange_symbols(header_id, '[$@+]?^&'),
    'capitals': _count_capitals(txt),
    'digits': _count_digits(header_id),
    'quotes': _count_strange_symbols(txt, '«»"\'"'),
    'underscores': _count_strange_symbols(txt, '_')
  }

  # if prev_features is None:
  #   # features['prev-number_level'] = 0
  #   features['prev-len_chars']=-1
  # else:
  #   # features['prev-number_level'] = prev_features['number_level']
  #   features['prev-len_chars'] = prev_features['len_chars']



  return features

def _count_capitals(txt):
  s=0
  for c in txt:
    if c.isupper():
      s+=1
  return s

def _count_digits(txt):
  s=0
  for c in txt:
    if c.isdigit():
      s+=1
  return s

def _count_strange_symbols(txt, strange_symbols):
  res = 0
  for c in strange_symbols:
    res += txt.count(c)
  return res
