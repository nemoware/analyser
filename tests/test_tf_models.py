#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

import numpy as np

from analyser.ml_tools import Embeddings
from tests.test_utilits import load_json_sample
from tf_support.embedder_elmo import ElmoEmbedder
from tf_support.super_contract_model import uber_detection_model_001
from tf_support.tools import KerasTrainingContext
from trainsets.retrain_contract_uber_model import _get_semantic_map, DbJsonDoc


class TestLoadTfModel(unittest.TestCase):

  def test_embedder(self):
    embedder = ElmoEmbedder.get_instance()
    e: Embeddings = embedder.embedd_tokens([['мама', 'мыла', 'Раму', 'а', 'Рама', 'мыл', 'Шиву']])
    #just expecting no failure
    print(e)

  def test_get_semantic_map(self):
    json_dict = load_json_sample('contract_db_1.json')

    sm = _get_semantic_map(DbJsonDoc(json_dict))
    print(sm.shape)

  def test_load_uber_model_001(self):
    ctx = KerasTrainingContext()
    model = ctx.init_model(uber_detection_model_001, verbose=2)
    self.assertIsNotNone(model)

    fake_emb = np.zeros((1, 100, 1024))
    fake_lab = np.zeros((1, 100, 15))

    prediction = model.predict([fake_emb, fake_lab])
    self.assertEqual(2, len(prediction))


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
