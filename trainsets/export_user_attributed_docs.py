#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import json
import os
import pickle
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pymongo
from bson import json_util
from keras.preprocessing.sequence import pad_sequences
from pandas import DataFrame

from analyser.headers_detector import get_tokens_features
from analyser.hyperparams import work_dir, models_path
from analyser.legal_docs import LegalDocument, make_headline_attention_vector
from analyser.structures import ContractSubject
from integration.db import get_mongodb_connection
from integration.word_document_parser import join_paragraphs
from tf_support.embedder_elmo import ElmoEmbedder
from tf_support.super_contract_model import uber_detection_model_005_1_1, seq_labels_contract
from tf_support.tools import KerasTrainingContext

SAVE_PICKLES = False
_DEV_MODE = False
_EMBEDD = True


def pad_things(xx, maxlen, padding='post'):
  for x in xx:
    _v = x.mean()
    yield pad_sequences([x], maxlen=maxlen, padding=padding, truncating=padding, value=_v, dtype='float32')[0]


def _get_semantic_map(doc: LegalDocument, confidence_override=None) -> DataFrame:
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

  # add missing columns
  for sl in seq_labels_contract:
    if sl not in df:
      df[sl] = np.zeros(len(doc))

  df['headline_h1'] = make_headline_attention_vector(doc)  ##adding headers

  return df[seq_labels_contract]  # re-order columns


def _get_attribute_value(d, attr):
  if 'user' in d and 'attributes' in d['user'] and attr in d['user']['attributes']:
    return d['user']['attributes'][attr]['value']

  else:
    if 'attributes' in d['analysis'] and attr in d['analysis']['attributes']:
      return d['analysis']['attributes'][attr]['value']


def _get_subject(d):
  return _get_attribute_value(d, 'subject')


class UberModelTrainsetManager:

  def __init__(self, work_dir: str):

    self.work_dir: str = work_dir
    self.stats: DataFrame = self._get_contract_trainset_meta(work_dir)

    if len(self.stats) > 0:
      self.stats.sort_values(["user_correction_date", 'analyze_date', 'export_date'], inplace=True, ascending=False)
      self.lastdate = self.stats[["user_correction_date", 'analyze_date']].max().max()
    else:
      self.lastdate = datetime(1900, 1, 1)

    print(f'latest export_date: [{self.lastdate}]')

    self.embedder = None

  def save_contract_datapoint(self, d):
    id = str(d['_id'])

    doc: LegalDocument = join_paragraphs(d['parse'], id)

    if not _DEV_MODE and _EMBEDD:
      print(f'embedding doc {id}....')
      doc.embedd_tokens(self.embedder)
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

    stats = self.stats  # shortcut
    stats.at[id, 'checksum'] = doc.get_checksum()
    stats.at[id, 'version'] = d['analysis']['version']

    stats.at[id, 'export_date'] = datetime.now()
    stats.at[id, 'subject'] = _get_subject(d)
    stats.at[id, 'analyze_date'] = d['analysis']['analyze_timestamp']

    if 'user' in d:
      stats.at[id, 'user_correction_date'] = d['user']['updateDate']

  def get_updated_contracts(self):
    print('obtaining DB connection...')
    db = get_mongodb_connection()

    print('obtaining DB connection: DONE')
    documents_collection = db['documents']
    print('linking documents collection: DONE')

    # TODO: filter by version
    query = {
      '$and': [
        {"parse.documentType": "CONTRACT"},
        # {"analysis.attributes": {"$ne": None}},
        {'$or': [{"analysis.attributes.subject": {"$ne": None}}, {"user.attributes.subject": {"$ne": None}}]},

        {'$or': [
          {'analysis.analyze_timestamp': {'$gt': self.lastdate}},
          {'user.updateDate': {'$gt': self.lastdate}}
        ]}
      ]
    }

    print(f'running DB query {query}')
    res = documents_collection.find(filter=query, sort=[('analysis.analyze_timestamp', pymongo.ASCENDING),
                                                        ('user.updateDate', pymongo.ASCENDING)])

    if _DEV_MODE:
      res.limit(5)

    print('running DB query: DONE')

    return res

  def _get_contract_trainset_meta(self, work_dir):
    try:
      df = pd.read_csv(os.path.join(work_dir, 'contract_trainset_meta.csv'), index_col='_id')
      df['user_correction_date'] = pd.to_datetime(df['user_correction_date'])
      df['analyze_date'] = pd.to_datetime(df['analyze_date'])

    except FileNotFoundError:
      df = DataFrame(columns=['export_date'])
    df.index.name = '_id'
    return df

  def export_recent_contracts(self):
    if _EMBEDD:
      self.embedder = ElmoEmbedder.get_instance('elmo')

    docs = self.get_updated_contracts()  # Cursor, not list

    # export_docs_to_single_json(docs)

    for d in docs:
      self.save_contract_datapoint(d)

      self.stats.sort_values(["user_correction_date", 'analyze_date'], inplace=True, ascending=False)
      self.stats.drop_duplicates(subset="checksum", keep='first', inplace=True)

      self._save_stats()

  def _save_stats(self):
    self.stats.to_csv(os.path.join(work_dir, 'contract_trainset_meta.csv'), index=True)

  def get_xyw(self, id):

    row = self.stats.loc[id]

    try:

      weight = 0.5
      if row['user_correction_date'] is not None:  # more weight to user-corrected datapoints
        weight *= 10.0

      subj = row['subject']
      subject_one_hot = ContractSubject.encode_1_hot()[subj]

      fn = os.path.join(self.work_dir, f'{id}-datapoint-embeddings.npy')
      embeddings = np.load(fn)

      fn = os.path.join(self.work_dir, f'{id}-datapoint-token_features.npy')
      token_features = np.load(fn)

      fn = os.path.join(work_dir, f'{id}-datapoint-semantic_map.npy')
      semantic_map = np.load(fn)

      return ((embeddings, token_features), (semantic_map, subject_one_hot), weight)
    except:
      self.stats.at[id, 'valid'] = False
      self._save_stats()
      return ((None, None), (None, None), None)

  def get_indices_split(self, category_column_name: str = 'subject', test_proportion=0.25) -> (
          [int], [int]):
    np.random.seed(42)
    df = self.stats[self.stats['valid'] != False]
    print(df.head)
    cat_count = df[category_column_name].value_counts()  # distribution by category

    _bags = {key: [] for key in cat_count.index}

    for index, row in df.iterrows():
      subj_code = row[category_column_name]
      _bags[subj_code].append(index)

    train_indices = []
    test_indices = []

    for subj_code in _bags:
      bag = _bags[subj_code]
      split_index: int = int(len(bag) * test_proportion)

      train_indices += bag[split_index:]
      test_indices += bag[:split_index]

    # remove instesection
    intersection = np.intersect1d(test_indices, train_indices)
    test_indices = [e for e in test_indices if e not in intersection]

    # shuffle

    np.random.shuffle(test_indices)
    np.random.shuffle(train_indices)

    return train_indices, test_indices

  def init_model(self):
    ctx = KerasTrainingContext(work_dir)

    model_factory_fn = uber_detection_model_005_1_1
    model_name = model_factory_fn.__name__

    model = model_factory_fn(name=model_name, ctx=ctx, trained=True)
    model.name = model_name

    weights_file_old = os.path.join(models_path, model_name + ".weights")
    weights_file_new = os.path.join(work_dir, model_name + ".uptrained.weights")

    try:
      model.load_weights(weights_file_new)
      print(f'weights loaded: {weights_file_new}')

    except:
      msg = f'cannot load  {model_name} from  {weights_file_new}'
      warnings.warn(msg)
      model.load_weights(weights_file_old)
      print(f'weights loaded: {weights_file_old}')

    # freeze bottom 6 layers, including 'embedding_reduced'
    for l in model.layers[0:6]:
      # print('init_model: Layer:', l.name, l.trainable)
      l.trainable = False

    model.summary()
    return model, ctx

  def train(self):
    train_indices, test_indices = self.get_indices_split()
    model, ctx = self.init_model()

    test_gen = self.make_generator(test_indices, 3)
    train_gen = self.make_generator(train_indices, 3)
    # model.fit_generator()
    ctx.EVALUATE_ONLY = False
    ctx.EPOCHS = 20
    ctx.train_and_evaluate_model(model, train_gen, test_gen, retrain=True)

    # ctx = KerasTrainingContext(work_dir)
    # ctx.init_model(uber_detection_model_005_1_1)

  def make_generator(self, indices: [int], batch_size: int):

    # random.seed(42)
    np.random.seed(42)

    while True:
      maxlen = 128 * random.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
      cutoff = 16 * random.choice([0, 0, 0, 1, 1, 2, 3])

      # Select files (paths/indices) for the batch
      batch_indices = np.random.choice(a=indices, size=batch_size)

      batch_input_e = []
      batch_input_h = []
      batch_output_a = []
      batch_output_b = []

      weights = []

      # Read in each input, perform preprocessing and get labels
      for i in batch_indices:
        (emb, tok_f), (sm, subj), w = self.get_xyw(i)
        if emb is not None:
          _padded = list(pad_things([emb, tok_f, sm], maxlen))
          # if CUT_EMB_PRE:
          # padded = list(pad_things(padded, maxlen - cutoff, padding='pre'))
          emb = _padded[0]
          tok_f = _padded[1]
          sm = _padded[2]

          batch_input_e.append(emb)
          batch_input_h.append(tok_f)

          batch_output_a.append(sm)
          batch_output_b.append(subj)

          weights.append(w)

      batch_x_e = np.array(batch_input_e, dtype=np.float32)
      batch_x_h = np.array(batch_input_h)

      batch_y_1 = np.array(batch_output_a)
      batch_y_2 = np.array(batch_output_b)

      ww = np.array(weights)

      # Return a tuple of (input, output, weights) to feed the network

      yield ([batch_x_e, batch_x_h], [batch_y_1, batch_y_2], [ww, ww])


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

  umtm = UberModelTrainsetManager(work_dir)
  umtm.export_recent_contracts()

  umtm.train()

  #
  # if1 = umtm.stats.index[0]
  #
  # _xyw = umtm.get_xyw(if1)
  # print(_xyw[2])
