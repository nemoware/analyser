#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from bson import json_util
from pandas import DataFrame

from analyser.headers_detector import get_tokens_features
from analyser.hyperparams import work_dir
from analyser.legal_docs import LegalDocument
from integration.db import get_mongodb_connection
from integration.word_document_parser import join_paragraphs
from tf_support.embedder_elmo import ElmoEmbedder

SAVE_PICKLES = False
_DEV_MODE = False


def _get_semantic_map(doc, confidence_override=None) -> DataFrame:
  if 'user' in doc.__dict__:
    _tags = doc.__dict__['user']['attributes']
  else:
    _tags = doc.__dict__['analysis']['attributes']

  _tags = _tags
  _attention = np.zeros((len(_tags), doc.__len__()))

  df = DataFrame()
  for i, _kind in enumerate(_tags):
    t = _tags[_kind]
    df[_kind] = 0
    _conf = t['confidence']
    if confidence_override is not None:
      _conf = confidence_override

    _span = t['span']
    _attention[i][_span[0]:_span[1]] = _conf

  for i, _kind in enumerate(_tags):
    df[_kind] = 0
    df[_kind] = _attention[i]

  return df


def _get_subject(d):
  if 'user' in d and 'attributes' in d['user'] and 'subject' in d['user']['attributes']:
    return d['user']['attributes']['subject']['value']

  else:
    if 'attributes' in d['analysis'] and 'subject' in d['analysis']['attributes']:
      return d['analysis']['attributes']['subject']['value']


def save_contract_datapoint(d, stats):
  id = str(d['_id'])

  doc: LegalDocument = join_paragraphs(d['parse'], id)

  if not _DEV_MODE:
    embedder = ElmoEmbedder.get_instance('elmo')
    doc.embedd_tokens(embedder)
    fn = os.path.join(work_dir, f'{id}-datapoint-embeddings')
    np.save(fn, doc.embeddings)

  _dict = doc.__dict__
  _dict['analysis'] = d['analysis']
  if 'user' in d:
    _dict['user'] = d['user']
  _dict['semantic_map'] = _get_semantic_map(doc, 1.0)

  token_features = get_tokens_features(doc.tokens)
  token_features['h'] = 0

  _dict['token_features'] = token_features

  if SAVE_PICKLES:
    fn = os.path.join(work_dir, f'datapoint-{id}.pickle')
    with open(fn, 'wb') as f:
      pickle.dump(doc, f)
      print('PICKLED: ', fn)

  fn = os.path.join(work_dir, f'{id}-datapoint-token_features')
  np.save(fn, _dict['token_features'])

  fn = os.path.join(work_dir, f'{id}-datapoint-semantic_map')
  np.save(fn, _dict['semantic_map'])

  stats.at[id, 'checksum'] = doc.get_checksum()
  stats.at[id, 'version'] = d['analysis']['version']

  stats.at[id, 'export_date'] = datetime.now()
  stats.at[id, 'subject'] = _get_subject(d)
  stats.at[id, 'analyze_date'] = d['analysis']['analyze_timestamp']

  if 'user' in d:
    stats.at[id, 'user_correction_date'] = d['user']['updateDate']


def get_updated_contracts(lastdate):
  print('obtaining DB connection...')
  db = get_mongodb_connection()

  print('obtaining DB connection: DONE')
  documents_collection = db['documents']
  print('linking documents collection: DONE')

  query = {
    '$and': [
      {"parse.documentType": "CONTRACT"},
      # {"analysis.attributes": {"$ne": None}},
      {'$or': [{"analysis.attributes.subject": {"$ne": None}}, {"user.attributes.subject": {"$ne": None}}]},

      {'$or': [
        {'analysis.analyze_timestamp': {'$gt': lastdate}},
        {'user.updateDate': {'$gt': lastdate}}
      ]}
    ]
  }

  print('running DB query')
  res = documents_collection.find(query).limit(2)
  if _DEV_MODE:
    res.limit(5)

  print('running DB query: DONE')

  return res


def export_docs_to_single_json(documents):
  arr = {}
  for k, d in enumerate(documents):

    if '_id' not in d['user']['author']:
      print(f'error: user attributes doc {d["_id"]} is not linked to any user')

    if 'auditId' not in d:
      print(f'error: doc {d["_id"]} is not linked to any audit')

    arr[str(d['_id'])] = d
    # arr.append(d)
    print(k, d['_id'])

  with open(os.path.join(work_dir, 'exported_docs.json'), 'w', encoding='utf-8') as outfile:
    json.dump(arr, outfile, indent=2, ensure_ascii=False, default=json_util.default)


def export_recent_contracts():
  stats: DataFrame = _get_contract_trainset_meta()
  stats = stats.sort_values('export_date')
  lastdate = datetime(1900, 1, 1)

  if len(stats) > 0:
    lastdate = stats.iloc[0]['export_date']

  print('latest export_date:', lastdate)
  docs = get_updated_contracts(lastdate)  # Cursor, not list

  # export_docs_to_single_json(docs)

  for d in docs:
    if d['parse']['documentType'] == 'CONTRACT':
      save_contract_datapoint(d, stats)

  stats.sort_values(["user_correction_date", 'analyze_date'], inplace=True, ascending=False)
  stats.drop_duplicates(subset="checksum", keep='first', inplace=True)

  stats.to_csv(os.path.join(work_dir, 'contract_trainset_meta.csv'), index=True)


def _get_contract_trainset_meta():
  try:
    df = pd.read_csv(os.path.join(work_dir, 'contract_trainset_meta.csv'), index_col='_id')
  except FileNotFoundError:
    df = DataFrame(columns=['export_date'])

  df.index.name = '_id'
  return df


if __name__ == '__main__':
  '''
  0. Read 'contract_trainset_meta.csv CSV, find the last datetime of export
  1. Fetch recent docs from DB: update date > last datetime of export 
  2. Embedd them, save embeddings, save other features
  
  '''
  # os.environ['GPN_DB_NAME'] = 'gpn'
  # os.environ['GPN_DB_HOST'] = '192.168.10.36'
  # os.environ['GPN_DB_PORT'] = '27017'
  # db = get_mongodb_connection()
  #
  export_recent_contracts()
