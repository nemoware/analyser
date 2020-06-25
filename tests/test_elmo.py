#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import unittest

from tf_support.embedder_elmo import ElmoEmbedder


class ElmoTestCase(unittest.TestCase):

  def test_embedd(self):
    ee = ElmoEmbedder.get_instance('default')
    r = ee.embedd_tokenized_text([['просто', 'одно', 'предложение']], [3])
    self.assertEqual(1024, r.shape[-1])
    self.assertEqual(3, r.shape[-2])


if __name__ == '__main__':
  unittest.main()
