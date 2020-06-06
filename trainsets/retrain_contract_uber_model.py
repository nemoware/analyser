#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import json
import os
import random
import warnings
from datetime import datetime

import matplotlib
from sklearn.metrics import classification_report

from analyser.text_tools import split_version
from colab_support.renderer import plot_cm
from trainsets.trainset_tools import split_trainset_evenly, get_feature_log_weights

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymongo
from bson import json_util
from keras import Model
from keras.preprocessing.sequence import pad_sequences
from pandas import DataFrame

from analyser.headers_detector import get_tokens_features
from analyser.hyperparams import models_path

from analyser.hyperparams import work_dir as default_work_dir
from analyser.legal_docs import LegalDocument, make_headline_attention_vector
from analyser.structures import ContractSubject
from integration.db import get_mongodb_connection
from integration.word_document_parser import join_paragraphs
from tf_support import super_contract_model
from tf_support.embedder_elmo import ElmoEmbedder
from tf_support.super_contract_model import uber_detection_model_005_1_1, seq_labels_contract
from tf_support.tools import KerasTrainingContext

SAVE_PICKLES = False
_DEV_MODE = False
_EMBEDD = True


# TODO: sort org1 and org2 by span start
# TODO: use averaged tags confidence for sample weighting
# TODO: cache embeddings on analysis phase

def pad_things(xx, maxlen, padding='post'):
  for x in xx:
    _v = x.mean()
    yield pad_sequences([x], maxlen=maxlen, padding=padding, truncating=padding, value=_v, dtype='float32')[0]


def _get_semantic_map(doc: LegalDocument, confidence_override=None) -> DataFrame:
  if 'user' in doc.__dict__:
    _tags = doc.__dict__['user']['attributes']
  else:
    _tags = doc.__dict__['analysis']['attributes']

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
    return d['user']['attributes'][attr]

  else:
    if 'attributes' in d['analysis'] and attr in d['analysis']['attributes']:
      return d['analysis']['attributes'][attr]


def _get_subject(d):
  att = _get_attribute_value(d, 'subject')

  return att


class UberModelTrainsetManager:

  def __init__(self, work_dir: str, model_variant_fn=uber_detection_model_005_1_1):
    self.model_variant_fn = model_variant_fn
    self.work_dir: str = work_dir
    self.stats: DataFrame = self.load_contract_trainset_meta()

  def save_contract_data_arrays(self, doc: LegalDocument, id_override=None):
    id_ = doc._id
    if id_override is not None:
      id_ = id_override

    token_features = get_tokens_features(doc.tokens)
    semantic_map = _get_semantic_map(doc, 1.0)

    fn = os.path.join(self.work_dir, f'{id_}-datapoint-token_features')
    np.save(fn, token_features)

    fn = os.path.join(self.work_dir, f'{id_}-datapoint-semantic_map')
    np.save(fn, semantic_map)

    fn = os.path.join(self.work_dir, f'{id_}-datapoint-embeddings')
    np.save(fn, doc.embeddings)

  def save_contract_datapoint(self, d):
    _id = str(d['_id'])

    doc: LegalDocument = join_paragraphs(d['parse'], _id)

    if not _DEV_MODE and _EMBEDD:
      fn = os.path.join(self.work_dir, f'{_id}-datapoint-embeddings')
      if os.path.isfile(fn + '.npy'):
        print(f'skipping embedding doc {_id}...., {fn} exits')
        doc.embeddings = np.load(fn + '.npy')
      else:
        print(f'embedding doc {_id}....')
        embedder = ElmoEmbedder.get_instance('elmo')  # lazy init
        doc.embedd_tokens(embedder)

    _dict = doc.__dict__  # shortcut
    _dict['analysis'] = d['analysis']
    if 'user' in d:
      _dict['user'] = d['user']

    self.save_contract_data_arrays(doc)

    stats = self.stats  # shortcut
    stats.at[_id, 'checksum'] = doc.get_checksum()
    stats.at[_id, 'version'] = d['analysis']['version']

    stats.at[_id, 'export_date'] = datetime.now()
    stats.at[_id, 'analyze_date'] = d['analysis']['analyze_timestamp']

    subj_att = _get_subject(d)
    stats.at[_id, 'subject'] = subj_att['value']
    stats.at[_id, 'subject confidence'] = subj_att['confidence']

    if 'user' in d:
      stats.at[_id, 'user_correction_date'] = d['user']['updateDate']

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

  @staticmethod
  def _remove_obsolete_datapoints(df: DataFrame):
    if 'valid' not in df:
      df['valid'] = True

    for i, row in df.iterrows():

      int_v = split_version(row['version'])

      if pd.isna(row['user_correction_date']):
        if not (int_v[0] >= 1 and int_v[1] >= 6):
          df.at[i, 'valid'] = False

  def load_contract_trainset_meta(self):
    try:
      df = pd.read_csv(os.path.join(self.work_dir, 'contract_trainset_meta.csv'), index_col='_id')
      df['user_correction_date'] = pd.to_datetime(df['user_correction_date'])
      df['analyze_date'] = pd.to_datetime(df['analyze_date'])
      df.index.name = '_id'

      UberModelTrainsetManager._remove_obsolete_datapoints(df)

      if 'valid' in df:
        df = df[df['valid'] != False]

    except FileNotFoundError:
      df = DataFrame(columns=['export_date'])
      df.index.name = '_id'

    print(f'TOTAL DATAPOINTS IN TRAINSET: {len(df)}')
    return df

  def export_recent_contracts(self):
    self.stats: DataFrame = self.load_contract_trainset_meta()

    self.lastdate = datetime(1900, 1, 1)
    if len(self.stats) > 0:
      # self.stats.sort_values(["user_correction_date", 'analyze_date', 'export_date'], inplace=True, ascending=False)
      self.lastdate = self.stats[["user_correction_date", 'analyze_date']].max().max()
    print(f'latest export_date: [{self.lastdate}]')
    docs = self.get_updated_contracts()  # Cursor, not list

    # export_docs_to_single_json(docs)

    for d in docs:
      self.save_contract_datapoint(d)
      self._save_stats()

  def _save_stats(self):

    # TODO are you sure, you need to drop_duplicates on every step?
    # todo: might be .. move this code to self._save_stats()
    so = []
    if 'user_correction_date' in self.stats:
      so.append('user_correction_date')
    if 'analyze_date' in self.stats:
      so.append('analyze_date')
    if len(so) > 0:
      self.stats.sort_values(so, inplace=True, ascending=False)
      self.stats.drop_duplicates(subset="checksum", keep='first', inplace=True)

    self.stats.to_csv(os.path.join(self.work_dir, 'contract_trainset_meta.csv'), index=True)

  def get_xyw(self, doc_id):

    row = self.stats.loc[doc_id]

    try:

      _subj = row['subject']
      subject_one_hot = ContractSubject.encode_1_hot()[_subj]

      fn = os.path.join(self.work_dir, f'{doc_id}-datapoint-embeddings.npy')
      embeddings = np.load(fn)

      fn = os.path.join(self.work_dir, f'{doc_id}-datapoint-token_features.npy')
      token_features = np.load(fn)

      fn = os.path.join(self.work_dir, f'{doc_id}-datapoint-semantic_map.npy')
      semantic_map = np.load(fn)
      self.stats.at[doc_id, 'error'] = None
      return (
        (embeddings, token_features),
        (semantic_map, subject_one_hot),
        (row['sample_weight'], row['subject_weight']))

    except Exception as e:
      print(e)
      self.stats.at[doc_id, 'valid'] = False
      self.stats.at[doc_id, 'error'] = str(e)
      self._save_stats()
      return ((None, None), (None, None), (None, None))

  def init_model(self) -> (Model, KerasTrainingContext):
    ctx = KerasTrainingContext(self.work_dir)

    model_name = self.model_variant_fn.__name__

    model = self.model_variant_fn(name=model_name, ctx=ctx, trained=True)
    model.name = model_name

    weights_file_old = os.path.join(models_path, model_name + ".weights")
    weights_file_new = os.path.join(self.work_dir, model_name + ".weights")

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
      l.trainable = False

    model.compile(loss=super_contract_model.losses, optimizer='Nadam', metrics=super_contract_model.metrics)
    model.summary()

    return model, ctx

  def validate_trainset(self):
    self.stats: DataFrame = self.load_contract_trainset_meta()
    for i in self.stats.index:
      self.get_xyw(i)
    self._save_stats()

  def train(self, generator_factory_method):
    self.stats: DataFrame = self.load_contract_trainset_meta()
    '''
    Phase I: frozen bottom 6 common layers
    Phase 2: all unfrozen, entire trainset, low LR
    :return:
    '''

    batch_size = 24  # TODO: make a param
    train_indices, test_indices = split_trainset_evenly(self.stats, 'subject')
    model, ctx = self.init_model()
    ctx.EVALUATE_ONLY = False

    ######################
    ## Phase I retraining
    # frozen bottom layers
    ######################

    ctx.EPOCHS = 25
    ctx.set_batch_size_and_trainset_size(batch_size, len(test_indices), len(train_indices))

    test_gen = generator_factory_method(test_indices, batch_size)
    train_gen = generator_factory_method(train_indices, batch_size)

    ctx.train_and_evaluate_model(model, train_gen, test_gen, retrain=True)

    ######################
    ## Phase II finetuning
    #  all unfrozen, entire trainset, low LR
    ######################
    ctx.unfreezeModel(model)
    model.compile(loss=super_contract_model.losses, optimizer='Nadam', metrics=super_contract_model.metrics)
    model.summary()

    ctx.EPOCHS *= 2
    train_gen = generator_factory_method(train_indices + test_indices, batch_size)
    test_gen = generator_factory_method(test_indices, batch_size)
    ctx.train_and_evaluate_model(model, train_gen, test_generator=test_gen, retrain=False, lr=2e-5)

    self.make_training_report(ctx, model)

  def make_training_report(self, ctx: KerasTrainingContext, model: Model):
    ## plot results
    _log = ctx.get_log(model.name)
    if _log is not None:
      _metrics = _log.keys()
      plot_compare_models(ctx, [model.name], _metrics, self.work_dir)

    gen = self.make_generator(self.stats.index, 20)
    plot_subject_confusion_matrix(self.work_dir, model, steps=12, generator=gen)

  def calculate_samples_weights(self):
    self.stats: DataFrame = self.load_contract_trainset_meta()
    subject_weights = get_feature_log_weights(self.stats, 'subject')

    for i, row in self.stats.iterrows():
      subj_name = row['subject']

      sample_weight = row['subject confidence']
      if not pd.isna(row['user_correction_date']):  # more weight for user-corrected datapoints
        sample_weight = 10.0  # TODO: must be estimated anyhow smartly

      subject_weight = sample_weight * subject_weights[subj_name]
      self.stats.at[i, 'subject_weight'] = subject_weight
      self.stats.at[i, 'sample_weight'] = sample_weight

    # normalize weights, so the sum == Number of samples
    self.stats.sample_weight /= self.stats.sample_weight.mean()
    self.stats.subject_weight /= self.stats.subject_weight.mean()

    self._save_stats()

  def make_generator(self, indices: [int], batch_size: int):

    np.random.seed(42)

    while True:
      maxlen = 128 * random.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
      # cutoff = 16 * random.choice([0, 0, 0, 1, 1, 2, 3])

      # Select files (paths/indices) for the batch
      batch_indices = np.random.choice(a=indices, size=batch_size)

      batch_input_emb = []
      batch_input_token_f = []
      batch_output_sm = []
      batch_output_subj = []

      weights = []
      weights_subj = []

      # Read in each input, perform preprocessing and get labels
      for i in batch_indices:
        row = self.stats.loc[i]
        subj_name = row['subject']

        (emb, tok_f), (sm, subj), (sample_weight, subject_weight) = self.get_xyw(i)

        if emb is not None:  # paranoia, TODO: fail execution, because trainset mut be verifyded in advance

          _padded = list(pad_things([emb, tok_f, sm], maxlen))
          # _padded = list(pad_things(_padded, maxlen - cutoff, padding='pre'))

          emb = _padded[0]
          tok_f = _padded[1]
          sm = _padded[2]

          batch_input_emb.append(emb)
          batch_input_token_f.append(tok_f)

          batch_output_sm.append(sm)
          batch_output_subj.append(subj)

          weights.append(sample_weight)
          weights_subj.append(subject_weight)
        # end if emb
      # end for loop

      # Return a tuple of (input, output, weights) to feed the network
      yield ([np.array(batch_input_emb), np.array(batch_input_token_f)],
             [np.array(batch_output_sm), np.array(batch_output_subj)],
             [np.array(weights), np.array(weights_subj)])


def export_docs_to_single_json(documents, work_dir):
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


def onehots2labels(preds):
  _x = np.argmax(preds, axis=-1)
  return [ContractSubject(k).name for k in _x]


def plot_subject_confusion_matrix(image_save_path, model, steps=12, generator=None):
  all_predictions = []
  all_originals = []

  for _ in range(steps):
    x, y, _ = next(generator)

    orig_test_labels = onehots2labels(y[1])

    _preds = onehots2labels(model.predict(x)[1])
    # _labels = sorted(np.unique(orig_test_labels + _preds))

    all_predictions += _preds
    all_originals += orig_test_labels

  plot_cm(all_originals, all_predictions)

  img_path = os.path.join(image_save_path, f'subjects-confusion-matrix-{model.name}.png')
  plt.savefig(img_path, bbox_inches='tight')

  report = classification_report(all_originals, all_predictions, digits=3)

  print(report)
  with open(os.path.join(image_save_path, f'subjects-classification_report-{model.name}.txt'), "w") as text_file:
    text_file.write(report)


def plot_compare_models(ctx, models: [str], metrics, image_save_path):
  _metrics = [m for m in metrics if not m.startswith('val_')]

  for i, m in enumerate(models):

    data: pd.DataFrame = ctx.get_log(m)

    if data is not None:
      data.set_index('epoch')

      for metric in _metrics:
        plt.figure(figsize=(16, 6))
        plt.grid()
        plt.title(f'{metric}')
        for metric_variant in ['', 'val_']:
          key = metric_variant + metric
          if key in data:

            x = data['epoch'][-100:]
            y = data[key][-100:]

            c = 'red'  # plt.cm.jet_r(i * colorstep)
            if metric_variant == '':
              c = 'blue'
            plt.plot(x, y, label=f'{key}', alpha=0.2, color=c)

            y = y.rolling(4, win_type='gaussian').mean(std=4)
            plt.plot(x, y, label=f'{key} SMOOTH', color=c)

            plt.legend(loc='upper right')

        img_path = os.path.join(image_save_path, f'{m}-{metric}.png')
        plt.savefig(img_path, bbox_inches='tight')

    else:
      print('cannot plot')


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

  umtm = UberModelTrainsetManager(default_work_dir)

  umtm.export_recent_contracts()
  umtm.calculate_samples_weights()
  umtm.validate_trainset()
  umtm.train(umtm.make_generator)
