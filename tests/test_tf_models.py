#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

import numpy as np

from tf_support.tf_subject_model import load_subject_detection_trained_model


class TestLoadTfModel(unittest.TestCase):

  def test_load_model(self):
    model = load_subject_detection_trained_model()

    fake_emb = np.zeros((100, 1024))

    v = model.predict(np.array([fake_emb]))
    print(v)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
