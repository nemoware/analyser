#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

import numpy as np


from tf_support.super_contract_model import uber_detection_model_001
from tf_support.tf_subject_model import load_subject_detection_trained_model
from tf_support.tools import KerasTrainingContext


class TestLoadTfModel(unittest.TestCase):

  def test_load_model(self):
    model = load_subject_detection_trained_model()

    fake_emb = np.zeros((100, 1024))

    v = model.predict(np.array([fake_emb]))
    print(v)

  def test_load_uber_model_001(self):
    ctx = KerasTrainingContext()
    model = ctx.init_model(uber_detection_model_001, verbose=2)
    self.assertIsNotNone(model)

    fake_emb = np.zeros((1, 100, 1024))
    fake_lab = np.zeros((1, 100, 15))

    prediction = model.predict([fake_emb, fake_lab])
    self.assertEqual(2, len(prediction))


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
