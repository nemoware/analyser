#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from fuzzy_matcher import AttentionVectors, prepare_patters_for_embedding
from patterns import FuzzyPattern
from text_tools import Tokens, hot_punkt


def make_punkt_mask(tokens: Tokens) -> np.ndarray:
  return 1.0 - 0.999 * hot_punkt(tokens)


class PatternSearchModel:

  def __init__(self, tf, hub,
               module_url='https://storage.googleapis.com/az-nlp/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz'):
    self.tf = tf
    self.hub = hub
    self.module_url = module_url

    # cosine_similarities, cosine_similarities_improved
    self.embedding_session = None
    self.make_embedding_session_and_graph()

  def _center(self, x):
    return self.tf.reduce_mean(x, axis=0)

  def _pad_tensor(self, tensor, padding, el, name):
    tf = self.tf
    _mask_padding = tf.tile(el, padding, name='pad_' + name)
    return tf.concat([tensor, _mask_padding], axis=0, name='pad_tensor_' + name)

  def _build_graph(self) -> None:
    tf = self.tf  # hack for PyCharm because i don't want to download TF, it is provided by CoLab from UI

    # BUILD IT -----------------------------------------------------------------
    elmo = self.hub.Module(self.module_url, trainable=False)

    # inputs:--------------------------------------------------------------------
    self.text_input = tf.placeholder(dtype=tf.string, name="text_input")
    self.text_lengths = tf.placeholder(dtype=tf.int32, name='text_lengths')

    self.pattern_input = tf.placeholder(dtype=tf.string, name='pattern_input')
    self.pattern_lengths = tf.placeholder(dtype=tf.int32, name='pattern_lengths')
    self.pattern_slices = tf.placeholder(dtype=tf.int32, name='pattern_slices', shape=[None, 2])
    # ------------------------------------------------------------------- /inputs

    patterns_max_len = tf.math.reduce_max(self.pattern_lengths, keepdims=True, name='patterns_max_len')

    number_of_patterns = tf.shape(self.pattern_input)[0]

    text_input_padded = [
      self._pad_tensor(self.text_input[0], patterns_max_len, tf.constant(['\n']), name='text_input_ext')]

    # 1. text embedding---------------
    # TODO: try to deal with segmented text (this is about trailing index - [0] )
    _text_embedding = self._embed(elmo,
                                  text_input_padded,
                                  [self.text_lengths[0] + patterns_max_len[0]])[0]

    # 2. patterns embedding
    _patterns_embeddings = self._embed(elmo, self.pattern_input, self.pattern_lengths)

    # for looping
    text_range = tf.range(self.text_lengths[0], dtype=tf.int32, name='text_input_range')
    patterns_range = tf.range(number_of_patterns, dtype=tf.int32, name='patterns_range')

    self.cosine_similarities = tf.map_fn(
      lambda i: self.for_every_pattern((self.pattern_lengths[i], self.pattern_slices[i], _patterns_embeddings[i]),
                                       _text_embedding,
                                       text_range), patterns_range, dtype=tf.float32, name='cosine_similarities')

    def improve_dist(attention_vector, pattern_len):
      """
        finding closest point (aka 'best point')
      """
      max_i = tf.math.argmax(attention_vector, output_type=tf.dtypes.int32)
      best_embedding_range = _text_embedding[max_i:max_i + pattern_len]  # metapattern

      return self._convolve(text_range, _text_embedding, pattern_emb_sliced=best_embedding_range,
                            name='improving')

    def find_best_embeddings():
      return tf.map_fn(
        lambda pattern_i: improve_dist(self.cosine_similarities[pattern_i], self.pattern_lengths[pattern_i]),
        patterns_range, dtype=tf.float32, name="find_best_embeddings")

    self.cosine_similarities_improved = find_best_embeddings()

    unpadded_text_embedding_ = self._embed(elmo,
                                           self.text_input,
                                           [self.text_lengths[0]])[0]
    text_center = self._center(unpadded_text_embedding_)
    self.distances_to_center = tf.map_fn(
      lambda i: self.get_matrix_vector_similarity(unpadded_text_embedding_[i:i + 1],
                                                  text_center),
      text_range, dtype=tf.float32)
    word_text_center_i = tf.math.argmax(self.distances_to_center, output_type=tf.dtypes.int32)
    word_text_center = unpadded_text_embedding_[word_text_center_i]
    self.distances_to_local_center = tf.map_fn(
      lambda i: self.get_matrix_vector_similarity(unpadded_text_embedding_[i:i + 1],
                                                  word_text_center), text_range, dtype=tf.float32)

  @staticmethod
  def _embed(elmo, text_input_p, lengths_p):
    # 1. text embedding
    return elmo(
      inputs={
        "tokens": text_input_p,
        "sequence_len": lengths_p
      },
      signature="tokens",
      as_dict=True)["elmo"]

  def _normalize(self, x):
    #       _norm = tf.norm(x, keep_dims=True)
    #       return x/ (_norm + 1e-8)
    return self.tf.nn.l2_normalize(x, 0)  # TODO: try different norm

  def get_vector_similarity(self, a, b):
    a_norm = self._normalize(a)  # normalizing is kinda required if we want cosine return [0..1] range
    b_norm = self._normalize(b)  # DO WE? TODO: try different norm
    return 1.0 - self.tf.losses.cosine_distance(a_norm, b_norm, axis=0)  # TODO: how on Earth Cosine could be > 1????

  def get_matrix_vector_similarity(self, matrix, vector):
    m_center = self._center(matrix)
    return self.get_vector_similarity(vector, m_center)

  def for_every_pattern(self, pattern_info, _text_embedding, text_range):
    pattern_slice = pattern_info[1]

    _patterns_embeddings = pattern_info[2]
    pattern_emb_sliced = _patterns_embeddings[pattern_slice[0]: pattern_slice[1]]

    return self._convolve(text_range, _text_embedding, pattern_emb_sliced, name='p_match')

  def _convolve(self, text_range, _text_embedding, pattern_emb_sliced, name=''):
    tf = self.tf

    window_size = tf.shape(pattern_emb_sliced)[0]

    p_center = self._center(pattern_emb_sliced)

    _blurry = tf.map_fn(
      lambda i: self.get_matrix_vector_similarity(matrix=_text_embedding[i:i + window_size],
                                                  vector=p_center),
      text_range, dtype=tf.float32, name=name + '_sim_wnd')

    _sharp = tf.map_fn(
      lambda i: self.get_matrix_vector_similarity(matrix=_text_embedding[i:i + 1],
                                                  vector=p_center),
      text_range, dtype=tf.float32, name=name + '_sim_w1')

    return tf.math.maximum(_blurry, _sharp, name=name + '_merge')

  def make_embedding_session_and_graph(self):
    tf = self.tf  # hack for PyCharm because I don't want to download TF, it is provided by CoLab from UI
    embedding_graph = tf.Graph()

    with embedding_graph.as_default():
      self._build_graph()

      init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

      self.embedding_session = tf.Session(graph=embedding_graph)
      self.embedding_session.run(init_op)

    embedding_graph.finalize()

  # ------
  def get_distances_to_center(self, text_tokens: Tokens):
    runz = [self.distances_to_center, self.distances_to_local_center]
    d, dl = self.embedding_session.run(runz, feed_dict={
      self.text_input: [text_tokens],  # text_input
      self.text_lengths: [len(text_tokens)],  # text_lengths

      self.pattern_input: [['a']],
      self.pattern_lengths: [1],
      self.pattern_slices: [[0, 1]]

    })

    return d, dl

  # ------

  def _fill_dict(self, text_tokens: Tokens, patterns: List[FuzzyPattern]):
    patterns_tokens, patterns_lengths, pattern_slices, _ = prepare_patters_for_embedding(patterns)
    feed_dict = {
      self.text_input: [text_tokens],  # text_input
      self.text_lengths: [len(text_tokens)],  # text_lengths

      self.pattern_input: patterns_tokens,
      self.pattern_lengths: patterns_lengths,
      self.pattern_slices: pattern_slices

    }
    return feed_dict

  def find_patterns(self, text_tokens: Tokens, patterns: List[FuzzyPattern]) -> AttentionVectors:

    runz = [self.cosine_similarities]

    feed_dict = self._fill_dict(text_tokens, patterns)
    attentions = self.embedding_session.run(runz, feed_dict=feed_dict)[0]

    av = AttentionVectors()

    for i in range(len(patterns)):
      pattern = patterns[i]
      av.add(pattern.name, attentions[i])

    return av

  def find_patterns_and_improved(self, text_tokens: Tokens, patterns: List[FuzzyPattern]) -> AttentionVectors:

    runz = [self.cosine_similarities, self.cosine_similarities_improved]

    feed_dict = self._fill_dict(text_tokens, patterns)
    attentions, improved_attentions = self.embedding_session.run(runz, feed_dict=feed_dict)

    av = AttentionVectors()

    for i in range(len(patterns)):
      pattern = patterns[i]
      av.add(pattern.name, attentions[i], improved_attentions[i])

    return av


if __name__ == '__main__':
  from patterns import AbstractPatternFactory, FuzzyPattern

  from text_tools import tokenize_text
  import re
  from text_normalize import normalize_text, replacements_regex


  PM = PatternSearchModel(tf, hub)

  search_for = "Изящество стиля"
  from patterns import AbstractPatternFactory, FuzzyPattern

  from text_tools import tokenize_text
  import re
  from text_normalize import normalize_text, replacements_regex

  _sample_text = """
  Берлинская дада-ярмарка и Международная выставка сюрреализма в париж­ской галерее «Изящные искусства» в 1938 году стали высшими точками развития двух движений и подвели им итог. На «Сюрреалис­тической улице», за манекенами, выстроившимися в проходе в главный зал, располагались плакаты, приглашения, объявления и фотографии, отсылающие к ранним этапам сюрреализма. В главном зале, за оформление которого отвечал Марсель Дюшан 

  , а за освеще­ние — Ман Рэй 

  , картины 1920-х годов висели рядом с более ранними работами, что подчеркивало развити
  е сюрреалистического «интернаци­она­ла». Зародившись как литературное течение, к концу 1930-х годов сюрреализм уже около 15 лет господствовал в художественном авангарде Парижа. Прежде чем пойти на спад с началом
  Второй мировой войны, он стал частью светской культуры Парижа и даже до некотоРой
  степени присягнул высокой моде, подобно тому как русский авангард — пусть совсем иначе — присягну 
  в свое время революции. Изящество стиля, свойственное сюррелизму, способствоало этому 

  ­ сближению, которое, в свою очередь, упрочило положение многих представителей направления в обществе. 

  Однако поначалу для литераторов и художников-бунтарей, ничуть не стремившихся к социаль­ному успеху, 
  была куда более естественной связь с дадаизмом"""

  _regex_addon = [
    (re.compile(r'[­]'), '-'),
  ]
  TOKENS = tokenize_text(normalize_text(_sample_text, replacements_regex + _regex_addon))


  class PF(AbstractPatternFactory):
    def __init__(self):
      AbstractPatternFactory.__init__(self, None)
      self._build_ner_patterns()

    def _build_ner_patterns(self):
      def cp(name, tuples):
        return self.create_pattern(name, tuples)

      cp('_custom', ('', search_for, ''))


  # ---
  pf = PF()

  av = PM.find_patterns(text_tokens=TOKENS, patterns=pf.patterns)
  print(av)
