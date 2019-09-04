#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# legal_docs.py
import json
import time
from functools import wraps

from doc_structure import DocumentStructure
from documents import TextMap
from embedding_tools import AbstractEmbedder
from ml_tools import normalize, smooth, extremums, smooth_safe, ProbableValue, \
  max_exclusive_pattern, TokensWithAttention, SemanticTag, FixedVector
from parsing import print_prof_data, ParsingSimpleContext
from patterns import *
from patterns import AV_SOFT, AV_PREFIX, PatternSearchResult, PatternSearchResults
from structures import ORG_2_ORG
from text_normalize import *
from text_tools import *
from text_tools import untokenize, np
from transaction_values import extract_sum_from_tokens, split_by_number_2, extract_sum_and_sign_2, ValueConstraint, \
  VALUE_SIGN_MIN_TOKENS, detect_sign, extract_sum_from_tokens_2, currencly_map, extract_sum

REPORTED_DEPRECATED = {}

import gc

from ml_tools import put_if_better


def remove_sr_duplicates_conditionally(list: PatternSearchResults):
  ret = []
  dups = {}
  for r in list:
    put_if_better(dups, r.key_index, r, lambda a, b: a.confidence > b.confidence)

  for x in dups.values():
    ret.append(x)
  del dups
  return ret


def substract_search_results(a: PatternSearchResults, b: PatternSearchResults) -> PatternSearchResults:
  b_indexes = [x.key_index for x in b]
  result: PatternSearchResults = []
  for x in a:
    if x.key_index not in b_indexes:
      result.append(x)
  return result


class HeadlineMeta:
  def __init__(self, index, _type, confidence: float, subdoc):
    self.index: int = index
    self.confidence: float = confidence
    self.type: str = _type
    self.subdoc: LegalDocument = subdoc
    self.body: LegalDocument = None

    self.attention: List[float] = None  # optional


def deprecated(fn):
  @wraps(fn)
  @wraps(fn)
  def with_reporting(*args, **kwargs):
    if fn.__name__ not in REPORTED_DEPRECATED:
      REPORTED_DEPRECATED[fn.__name__] = 1
      print("----WARNING!: function {} is deprecated".format(fn.__name__))

    ret = fn(*args, **kwargs)
    return ret

  return with_reporting


class LegalDocument:

  def __init__(self, original_text=None, name="legal_doc"):

    self.ID = None  # TODO
    self.filename = None
    self._original_text = original_text

    self._normal_text = None

    # todo: use pandas' DataFrame
    self.distances_per_pattern_dict = {}

    self.tokens_map: TextMap = None
    self.tokens_map_norm: TextMap = None

    self.sections = None
    self.name = name

    # subdocs
    self.start = 0
    self.end = None  # TODO:

    # TODO: probably we don't have to keep embeddings, just distances_per_pattern_dict
    self.embeddings = None

  def get_tags(self):
    raise NotImplementedError()

  def to_json(self) -> str:
    j = DocumentJson(self)
    return json.dumps(j.__dict__, indent=4, ensure_ascii=False, default=lambda o: '<not serializable>')

  def get_tokens_cc(self):
    return self.tokens_map.tokens

  def get_tokens(self):
    return self.tokens_map_norm.tokens

  def get_original_text(self):
    return self._original_text

  def get_normal_text(self):
    return self._normal_text

  def get_text(self):
    return self.tokens_map.text

  tokens_cc = property(get_tokens_cc)
  tokens = property(get_tokens)
  original_text = property(get_original_text)
  normal_text = property(get_normal_text)
  text = property(get_text)

  def parse(self, txt=None) -> None:
    if txt is None:
      txt = self.original_text

    assert txt is not None

    self._normal_text = self.preprocess_text(txt)
    self.tokens_map = TextMap(self._normal_text)

    _case_normalizer = CaseNormalizer()
    self.tokens_map_norm = _case_normalizer.normalize_tokens_map_case(self.tokens_map)

    self.structure = DocumentStructure()
    self.structure.detect_document_structure(self.tokens_map)

  def preprocess_text(self, txt):
    if txt is None:
      txt = self.original_text
    return normalize_text(txt, replacements_regex)

  def find_sections_by_headlines_2(self, context: ParsingSimpleContext, head_types_list,
                                   embedded_headlines: List['LegalDocument'], pattern_prefix,
                                   threshold) -> dict:

    hl_meta_by_index = {}
    sections = {}

    for head_type in head_types_list:

      confidence_by_headline = self._find_best_headline_by_pattern_prefix_2(embedded_headlines,
                                                                            pattern_prefix + head_type)
      closest_headline_index = int(np.argmax(confidence_by_headline))

      if confidence_by_headline[closest_headline_index] > threshold:

        obj = HeadlineMeta(closest_headline_index,
                           head_type,
                           confidence=confidence_by_headline[closest_headline_index],
                           subdoc=embedded_headlines[closest_headline_index])

        if closest_headline_index in hl_meta_by_index:
          # replace
          e_obj = hl_meta_by_index[closest_headline_index]
          if e_obj.confidence < obj.confidence:
            # replace
            hl_meta_by_index[closest_headline_index] = obj
        else:
          hl_meta_by_index[closest_headline_index] = obj


      else:
        context.warning(f'Cannot find headline matching pattern "{pattern_prefix + head_type}"*')

    for hl in hl_meta_by_index.values():
      try:
        hl.body = self._doc_section_under_headline(hl, render=False)
        sections[hl.type] = hl

      except ValueError as error:
        context.warning(str(error))
        # print(error)

    return sections

  def _doc_section_under_headline(self, headline_info: HeadlineMeta, render=False):
    warnings.warn("deprecated", DeprecationWarning)
    if render:
      print('Searching for section:', headline_info.type)

    bi_next = headline_info.index + 1

    headline_indexes = self.structure.headline_indexes

    headline_index = self.structure.headline_indexes[headline_info.index]
    if bi_next < len(headline_indexes):
      headline_next_id = headline_indexes[headline_info.index + 1]
    else:
      headline_next_id = None

    subdoc = subdoc_between_lines(headline_index, headline_next_id, self)

    if len(subdoc.tokens) < 2:
      raise ValueError(
        'Empty "{}" section between detected headlines #{} and #{}'.format(headline_info.type, headline_index,
                                                                           headline_next_id))

    if render:
      print('=' * 100)
      print(headline_info.subdoc.text)
      print('-' * 100)
      print(subdoc.text)

    return subdoc

  @deprecated
  def _find_best_headline_by_pattern_prefix(self, embedded_headlines: List['LegalDocument'], pattern_prefix: str,
                                            threshold):
    warnings.warn("deprecated", DeprecationWarning)

    import math

    number_of_headlines = len(embedded_headlines)
    confidence_by_headline = np.zeros(number_of_headlines)

    attention_vectors_by_headline = {}

    for i in range(number_of_headlines):
      subdoc = embedded_headlines[i]

      headline_name_av, _c = rectifyed_sum_by_pattern_prefix(subdoc.distances_per_pattern_dict, pattern_prefix,
                                                             relu_th=0.6)
      headline_name_av = smooth_safe(headline_name_av, 4)

      _max_id = np.argmax(headline_name_av)
      _max = np.max(headline_name_av)
      _sum = math.log(1 + np.sum(headline_name_av[_max_id - 1:_max_id + 2]))

      confidence_by_headline[i] = _max + _sum
      attention_vectors_by_headline[i] = headline_name_av

    closest_headline_index = int(np.argmax(confidence_by_headline))

    if confidence_by_headline[closest_headline_index] < threshold:
      raise ValueError('Cannot find headline matching pattern "{}"'.format(pattern_prefix))

    return closest_headline_index, confidence_by_headline, attention_vectors_by_headline[closest_headline_index]

  def _find_best_headline_by_pattern_prefix_2(self, embedded_headlines: List['LegalDocument'], pattern_prefix: str):

    import math

    number_of_headlines = len(embedded_headlines)
    confidence_by_headline = np.zeros(number_of_headlines)

    attention_vectors_by_headline = {}

    for i in range(number_of_headlines):
      subdoc = embedded_headlines[i]

      headline_name_av, _c = rectifyed_sum_by_pattern_prefix(subdoc.distances_per_pattern_dict, pattern_prefix,
                                                             relu_th=0.6)
      headline_name_av = smooth_safe(headline_name_av, 4)

      _max_id = np.argmax(headline_name_av)
      _max = np.max(headline_name_av)
      _sum = math.log(1 + np.sum(headline_name_av[_max_id - 1:_max_id + 2]))

      confidence_by_headline[i] = _max + _sum
      attention_vectors_by_headline[i] = headline_name_av

    return confidence_by_headline

  def find_sentence_beginnings(self, indices):
    return [find_token_before_index(self.tokens, i, '\n', 0) for i in indices]

  # @profile
  def calculate_distances_per_pattern(self, pattern_factory: AbstractPatternFactory, dist_function=DIST_FUNC,
                                      verbosity=1, merge=False, pattern_prefix=None):
    assert self.embeddings is not None
    self.distances_per_pattern_dict = calculate_distances_per_pattern(self, pattern_factory, dist_function, merge=merge,
                                                                      verbosity=verbosity,
                                                                      pattern_prefix=pattern_prefix)

    return self.distances_per_pattern_dict

  def print_structured(self, numbered_only=False):
    self.structure.print_structured(self, numbered_only)

  def subdoc_slice(self, __s: slice, name='undef'):
    assert self.tokens_map is not None
    _s = slice(max((0, __s.start)), max((0, __s.stop)))

    klazz = self.__class__
    sub = klazz("REF")
    sub.start = _s.start
    sub.end = _s.stop

    if self.embeddings is not None:
      sub.embeddings = self.embeddings[_s]

    if self.distances_per_pattern_dict is not None:
      sub.distances_per_pattern_dict = {}
      for d in self.distances_per_pattern_dict:
        sub.distances_per_pattern_dict[d] = self.distances_per_pattern_dict[d][_s]

    sub.tokens_map = self.tokens_map.slice(_s)
    sub.tokens_map_norm = self.tokens_map_norm.slice(_s)

    sub.name = f'{self.name}.{name}'
    return sub

  @deprecated
  def subdoc(self, start, end):
    warnings.warn("use subdoc_slice", DeprecationWarning)
    _s = slice(max(0, start), end)
    return self.subdoc_slice(_s)

  def make_attention_vector(self, factory, pattern_prefix, recalc_distances=True) -> (List[float], str):
    # ---takes time
    if recalc_distances:
      calculate_distances_per_pattern(self, factory, merge=True, pattern_prefix=pattern_prefix)
    # ---
    vectors = filter_values_by_key_prefix(self.distances_per_pattern_dict, pattern_prefix)
    vectors_i = []

    attention_vector_name = AV_PREFIX + pattern_prefix
    attention_vector_name_soft = AV_SOFT + attention_vector_name

    for v in vectors:
      if max(v) > 0.6:
        vector_i, _ = improve_attention_vector(self.embeddings, v, relu_th=0.6, mix=0.9)
        vectors_i.append(vector_i)
      else:
        vectors_i.append(v)

    x = max_exclusive_pattern(vectors_i)
    self.distances_per_pattern_dict[attention_vector_name_soft] = x
    x = relu(x, 0.8)

    self.distances_per_pattern_dict[attention_vector_name] = x
    return x, attention_vector_name

  def find_sentences_by_attention_vector(self, attention: FixedVector) -> List[SemanticTag]:
    # results: PatternSearchResults = []
    #
    # for i in np.nonzero(attention)[0]:
    #   _b = self.tokens_map.sentence_at_index(i)
    #   _slice = slice(_b[0], _b[1])
    #
    #   if _slice.stop != _slice.start:
    #
    #     sum_ = sum(attention[_slice])
    #     #       confidence = np.mean( np.nonzero(x[sl]) )
    #     nonzeros_count = len(np.nonzero(attention[_slice])[0])
    #     confidence = 0
    #
    #     if nonzeros_count > 0:
    #       confidence = sum_ / nonzeros_count
    #
    #     if confidence > 0.8:
    #       r = SemanticTag(kind, value, span=_b)
    #       r.confidence = confidence
    #       r.parent = self
    #
    #       results.append(r)
    #
    # results = remove_sr_duplicates_conditionally(results)
    #
    # return results
    XXX
    pass

  def find_sentences_by_pattern_prefix(self, org_level, factory, pattern_prefix) -> PatternSearchResults:

    """

    :param factory:
    :param pattern_prefix:
    :return:
    """
    warnings.warn("use find_sentences_by_attention_vector", DeprecationWarning)
    attention, attention_vector_name = self.make_attention_vector(factory, pattern_prefix)

    results: PatternSearchResults = []

    for i in np.nonzero(attention)[0]:
      _b = self.tokens_map.sentence_at_index(i)
      _slice = slice(_b[0], _b[1])

      if _slice.stop != _slice.start:

        sum_ = sum(attention[_slice])
        #       confidence = np.mean( np.nonzero(x[sl]) )
        nonzeros_count = len(np.nonzero(attention[_slice])[0])
        confidence = 0

        if nonzeros_count > 0:
          confidence = sum_ / nonzeros_count

        if confidence > 0.8:
          r = PatternSearchResult(ORG_2_ORG[org_level], _slice)
          r.attention_vector_name = attention_vector_name
          r.pattern_prefix = pattern_prefix
          r.confidence = confidence
          r.parent = self

          results.append(r)

    results = remove_sr_duplicates_conditionally(results)

    return results

  def reset_embeddings(self):
    print('-----ARE YOU SURE YOU NEED TO DROP EMBEDDINGS NOW??---------')
    del self.embeddings
    self.embeddings = None
    gc.collect()

  @deprecated
  def embedd(self, pattern_factory):
    warnings.warn("use embedd_tokens, provide embedder", DeprecationWarning)
    self.embedd_tokens(pattern_factory.embedder)

  def embedd_tokens(self, embedder: AbstractEmbedder, verbosity=2):
    max_tokens = 7000
    if len(self.tokens_map_norm) > max_tokens:
      self._embedd_large(embedder, max_tokens, verbosity)
    else:
      self.embeddings = self._emb(self.tokens, embedder)

  # @profile
  def _emb(self, tokens, embedder):
    embeddings, _g = embedder.embedd_tokenized_text([tokens], [len(tokens)])
    embeddings = embeddings[0]
    return embeddings

  def _embedd_large(self, embedder, max_tokens=8000, verbosity=2):

    overlap = 100  # max_tokens // 5

    number_of_windows = 1 + len(self.tokens_map_norm) // max_tokens
    window = max_tokens

    if verbosity > 1:
      print(
        "WARNING: Document is too large for embedding: {} tokens. Splitting into {} windows overlapping with {} tokens ".format(
          len(self.tokens_map_norm), number_of_windows, overlap))

    start = 0
    embeddings = None
    # tokens = []
    while start < len(self.tokens_map_norm):

      subtokens = self.tokens_map_norm[start:start + window + overlap]
      if verbosity > 2:
        print("Embedding region:", start, len(subtokens))

      sub_embeddings = self._emb(subtokens, embedder)[0:window]

      # sub_embeddings = sub_embeddings[0:window]
      # subtokens = subtokens[0:window]

      if embeddings is None:
        embeddings = sub_embeddings
      else:
        embeddings = np.concatenate([embeddings, sub_embeddings])
      # tokens += subtokens

      start += window
      print_prof_data()

    self.embeddings = embeddings
    # self.tokens = tokens


class DocumentJson:

  def from_json(jsondata):
    c = DocumentJson(None)
    c.__dict__ = jsondata
    tags = []
    for t in c.tags:
      tag = SemanticTag(None, None, None)
      tag.__dict__ = t
      tags.append(tag)

    c.tags = tags
    return c

  def __init__(self, doc: LegalDocument):
    self.ID = None
    self.filename = None
    self.original_text = None
    self.normal_text = None

    self.import_timestamp = time.time()
    self.analyze_timestamp = time.time()
    self.tokenization_maps = {}

    if doc is None:
      return

    self.checksum = hash(doc.normal_text)
    self.tokenization_maps['$words'] = doc.tokens_map.map

    for field in doc.__dict__:
      print(field)
      if field in self.__dict__:
        self.__dict__[field] = doc.__dict__[field]

    self.original_text = doc.original_text
    self.normal_text = doc.normal_text

    _tags: [SemanticTag] = []

    for hi in doc.structure.headline_indexes:
      s = doc.structure.structure[hi]
      _t = SemanticTag('headline', doc.tokens_map.text_range(s.span), s.span)

      _tags.append(_t)

    _tags += doc.get_tags()
    self.tags = [tag.__dict__ for tag in _tags]


def rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th: float = 0.0):
  warnings.warn("rectifyed_sum_by_pattern_prefix is deprecated", DeprecationWarning)
  vectors = filter_values_by_key_prefix(distances_per_pattern_dict, prefix)
  vectors = [x for x in vectors]
  return rectifyed_sum(vectors, relu_th), len(vectors)


def mean_by_pattern_prefix(distances_per_pattern_dict, prefix):
  warnings.warn("deprecated", DeprecationWarning)
  #     print('mean_by_pattern_prefix', prefix, relu_th)
  _sum, c = rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0.0)
  return normalize(_sum)


def rectifyed_normalized_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0.5):
  return normalize(rectifyed_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th))


def rectifyed_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0.5):
  #     print('mean_by_pattern_prefix', prefix, relu_th)
  _sum, c = rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th)
  _sum /= c
  return _sum


class BasicContractDocument(LegalDocument):

  def __init__(self, original_text=None):
    LegalDocument.__init__(self, original_text)

  def get_subject_ranges(self, indexes_zipped, section_indexes: List):

    subj_range = None
    head_range = None
    for i in range(len(indexes_zipped) - 1):
      if indexes_zipped[i][0] == 1:
        subj_range = range(indexes_zipped[i][1], indexes_zipped[i + 1][1])
      if indexes_zipped[i][0] == 0:
        head_range = range(indexes_zipped[i][1], indexes_zipped[i + 1][1])
    if head_range is None:
      print("WARNING: Contract type might be not known!!")
      head_range = range(0, 0)
    if subj_range is None:
      print("WARNING: Contract subject might be not known!!")
      if len(self.tokens) < 80:
        _end = len(self.tokens)
      else:
        _end = 80
      subj_range = range(0, _end)
    return head_range, subj_range


# SUMS -----------------------------


class ProtocolDocument(LegalDocument):

  def __init__(self, original_text=None):
    LegalDocument.__init__(self, original_text)


# Support masking ==================

def find_section_by_caption(cap, subdocs):
  solution_section = None
  mx = 0
  for subdoc in subdocs:
    d = subdoc.distances_per_pattern_dict[cap]
    _mx = d.max()
    if _mx > mx:
      solution_section = subdoc
      mx = _mx
  return solution_section


def mask_sections(section_name_to_weight_dict, doc):
  mask = np.zeros(len(doc.tokens))

  for name in section_name_to_weight_dict:
    section = find_section_by_caption(name, doc.subdocs)
    #         print([section.start, section.end])
    mask[section.start:section.end] = section_name_to_weight_dict[name]
  return mask


# Charter Docs


class CharterDocument(LegalDocument):
  def __init__(self, original_text, name="charter"):
    LegalDocument.__init__(self, original_text, name)

    self._constraints: List[PatternSearchResult] = []
    self.value_constraints = {}

    self._org = None

    self.org_type_tag: SemanticTag = None
    self.org_name_tag: SemanticTag = None

    # TODO:remove it
    self._charity_constraints_old = {}
    self._value_constraints_old = {}

  def get_tags(self) -> [SemanticTag]:
    return [self.org_type_tag, self.org_name_tag]

  def get_org(self):
    warnings.warn("use org_type_tag and org_name_tag", DeprecationWarning)
    return self._org

  def set_org(self, org):
    warnings.warn("use org_type_tag and org_name_tag", DeprecationWarning)
    self._org = org

  def get_constraints_old(self):
    return self._value_constraints_old

  constraints_old = property(get_constraints_old)
  org = property(get_org, set_org)

  def constraints_by_org_level(self, org_level: OrgStructuralLevel, constraint_subj: ContractSubject = None) -> List[
    PatternSearchResult]:
    for p in self._constraints:
      if p.org_level is org_level and (constraint_subj is None or p.subject_mapping['subj'] == constraint_subj):
        yield p


def max_by_pattern_prefix(distances_per_pattern_dict, prefix, attention_vector=None):
  ret = {}

  for p in distances_per_pattern_dict:
    if p.startswith(prefix):
      x = distances_per_pattern_dict[p]

      if attention_vector is not None:
        x = np.array(x)
        x += attention_vector

      ret[p] = x.argmax()

  return ret


def split_into_sections(doc, caption_indexes):
  sorted_keys = sorted(caption_indexes, key=lambda s: caption_indexes[s])

  doc.subdocs = []
  for i in range(1, len(sorted_keys)):
    key = sorted_keys[i - 1]
    next_key = sorted_keys[i]
    start = caption_indexes[key]
    end = caption_indexes[next_key]
    print(key, [start, end])

    subdoc = doc.subdoc(start, end)
    subdoc.filename = key
    doc.subdocs.append(subdoc)


def extract_sum_from_doc(doc: LegalDocument, attention_mask=None, relu_th=0.5):
  sum_pos, _c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, 'sum_max', relu_th=relu_th)
  sum_neg, _c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, 'sum_max_neg', relu_th=relu_th)

  sum_pos -= sum_neg

  sum_pos = smooth(sum_pos, window_len=8)
  #     sum_pos = relu(sum_pos, 0.65)

  if attention_mask is not None:
    sum_pos *= attention_mask

  sum_pos = normalize(sum_pos)

  return _extract_sums_from_distances(doc, sum_pos), sum_pos


def _extract_sum_from_distances____(doc: LegalDocument, sums_no_padding):
  max_i = np.argmax(sums_no_padding)
  start, end = doc.tokens_map.sentence_at_index(max_i)
  sentence_tokens = doc.tokens[start + 1:end]

  f, sentence = extract_sum_from_tokens(sentence_tokens)

  return (f, (start, end), sentence)


def _extract_sums_from_distances(doc: LegalDocument, x):
  maximas = extremums(x)

  results = []
  for max_i in maximas:
    start, end = doc.tokens_map.sentence_at_index(max_i)
    sentence_tokens = doc.tokens[start + 1:end]

    f, sentence = extract_sum_from_tokens(sentence_tokens)

    if f is not None:
      result = {
        'sum': f,
        'region': (start, end),
        'sentence': sentence,
        'confidence': x[max_i]
      }
      results.append(result)

  return results


MIN_DOC_LEN = 5


@deprecated
def make_soft_attention_vector(doc: LegalDocument, pattern_prefix, relu_th=0.5, blur=60, norm=True):
  warnings.warn("make_soft_attention_vector is deprecated", DeprecationWarning)
  assert doc.distances_per_pattern_dict is not None

  if len(doc.tokens) < MIN_DOC_LEN:
    print("----ERROR: make_soft_attention_vector: too few tokens {} ".format(doc.text))
    return np.full(len(doc.tokens), 0.0001)

  attention_vector, _c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, pattern_prefix,
                                                         relu_th=relu_th)
  attention_vector = relu(attention_vector, relu_th=relu_th)

  attention_vector = smooth_safe(attention_vector, window_len=blur)
  attention_vector = smooth_safe(attention_vector, window_len=blur)
  try:
    if norm:
      attention_vector = normalize(attention_vector)
  except:
    print(
      "----ERROR: make_soft_attention_vector: attention_vector for pattern prefix {} is not contrast, len = {}".format(
        pattern_prefix, len(attention_vector)))
    attention_vector = np.full(len(attention_vector), attention_vector[0])

  return attention_vector


@deprecated
def soft_attention_vector(doc, pattern_prefix, relu_th=0.5, blur=60, norm=True):
  warnings.warn("deprecated", DeprecationWarning)
  assert doc.distances_per_pattern_dict is not None

  if len(doc.tokens) < MIN_DOC_LEN:
    print("----ERROR: soft_attention_vector: too few tokens {} ".format(doc.text))
    return np.full(len(doc.tokens), 0.0001)

  attention_vector, c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, pattern_prefix, relu_th=relu_th)
  assert c > 0
  attention_vector = relu(attention_vector, relu_th=relu_th)

  attention_vector = smooth_safe(attention_vector, window_len=blur)
  attention_vector = smooth_safe(attention_vector, window_len=blur)
  attention_vector /= c
  try:
    if norm:
      attention_vector = normalize(attention_vector)
  except:
    print("----ERROR: soft_attention_vector: attention_vector for pattern prefix {} is not contrast, len = {}".format(
      pattern_prefix, len(attention_vector)))

    attention_vector = np.full(len(attention_vector), attention_vector[0])
  return attention_vector


def _expand_slice(s: slice, exp):
  return slice(s.start - exp, s.stop + exp)


@deprecated
def extract_all_contraints_from_sentence(sentence_subdoc: LegalDocument, attention_vector: List[float]) -> List[
  ProbableValue]:
  warnings.warn("deprecated", DeprecationWarning)
  tokens = sentence_subdoc.tokens
  assert len(attention_vector) == len(tokens)

  text_fragments, indexes, ranges = split_by_number_2(tokens, attention_vector, 0.2)

  constraints: List[ProbableValue] = []
  if len(indexes) > 0:

    for region in ranges:
      vc = extract_sum_and_sign_2(sentence_subdoc, region)

      _e = _expand_slice(region, 10)
      vc.context = TokensWithAttention(tokens[_e], attention_vector[_e])
      confidence = attention_vector[region.start]
      pv = ProbableValue(vc, confidence)

      constraints.append(pv)

  return constraints


from transaction_values import complete_re


def extract_all_contraints_from_sr(search_result: PatternSearchResult, attention_vector: List[float]) -> List[
  ProbableValue]:
  warnings.warn("use extract_all_contraints_from_sr_2", DeprecationWarning)

  def __tokens_before_index(string, index):
    warnings.warn("deprecated", DeprecationWarning)
    return len(string[:index].split(' '))

  sentence = ' '.join(search_result.tokens)
  all_values = [slice(m.start(0), m.end(0)) for m in re.finditer(complete_re, sentence)]
  constraints: List[ProbableValue] = []

  for a in all_values:
    # print(tokens_before_index(sentence, a.start), 'from', sentence[a])
    token_index_s = __tokens_before_index(sentence, a.start) - 1
    token_index_e = __tokens_before_index(sentence, a.stop)

    region = slice(token_index_s, token_index_e)

    vc: ValueConstraint = extract_sum_and_sign_3(search_result, region)
    _e = _expand_slice(region, 10)
    vc.context = TokensWithAttention(search_result.tokens[_e], attention_vector[_e])
    confidence = attention_vector[region.start]
    pv = ProbableValue(vc, confidence)

    constraints.append(pv)

  return constraints


def subdoc_between_lines(line_a: int, line_b: int, doc):
  _str = doc.structure.structure
  start = _str[line_a].span[1]
  if line_b is not None:
    end = _str[line_b].span[0]
  else:
    end = len(doc.tokens)
  return doc.subdoc(start, end)


org_types = {
  'org_unknown': 'undefined',
  'org_ao': 'Акционерное общество',
  'org_zao': 'Закрытое акционерное общество',
  'org_oao': 'Открытое акционерное общество',
  'org_ooo': 'Общество с ограниченной ответственностью',
  'org_nc': 'Некоммерческая организация'
}


def calculate_distances_per_pattern(doc: LegalDocument, pattern_factory: AbstractPatternFactory,
                                    dist_function=DIST_FUNC, merge=False,
                                    pattern_prefix=None, verbosity=1):
  distances_per_pattern_dict = {}
  if merge:
    distances_per_pattern_dict = doc.distances_per_pattern_dict

  c = 0
  for pat in pattern_factory.patterns:
    if pattern_prefix is None or pat.name[:len(pattern_prefix)] == pattern_prefix:
      if verbosity > 1: print(f'estimating distances to pattern {pat.name}', pat)

      dists = make_pattern_attention_vector(pat, doc.embeddings, dist_function)
      distances_per_pattern_dict[pat.name] = dists
      c += 1

  # if verbosity > 0:
  #   print(distances_per_pattern_dict.keys())
  if (c == 0):
    raise ValueError('no pattern with prefix: ' + pattern_prefix)

  return distances_per_pattern_dict


from transaction_values import _re_greather_then, _re_less_then, _re_greather_then_1


def detect_sign_2(txt: TextMap) -> (int, (int, int)):
  """
  todo: rename to 'find_value_sign'
  :param txt:
  :return:
  """

  a = next(txt.finditer(_re_greather_then_1), None)  # не менее, не превышающую
  if a:
    return +1, a

  a = next(txt.finditer(_re_less_then), None)  # менее
  if a:
    return -1, a
  else:
    a = next(txt.finditer(_re_greather_then), None)
    if a:
      return +1, a

  return 0, None


find_value_sign = detect_sign_2


class ValueSemanticTags:
  def __init__(self, sign: SemanticTag, value: SemanticTag, currency: SemanticTag):
    self.value: SemanticTag = value
    self.sign: SemanticTag = sign
    self.currency: SemanticTag = currency

    self._group_name = ''

  def mult_confidence(self, confidence_k):
    self.value.confidence *= confidence_k
    self.sign.confidence *= confidence_k
    self.currency.confidence *= confidence_k

  def set_group_name(self, name):
    self._group_name = name
    self.value.parent = self._group_name
    self.sign.parent = self._group_name
    self.currency.parent = self._group_name

  def get_group_name(self):
    return self._group_name

  group_name = property(get_group_name, set_group_name)

  def as_asrray(self):
    return [self.sign, self.value, self.currency]

  def offset_spans(self, offset):
    self.value.offset(offset)
    self.sign.offset(offset)
    self.currency.offset(offset)


def extract_sum_sign_currency(doc: LegalDocument, region: (int, int)) -> ValueSemanticTags:
  _s = slice(-VALUE_SIGN_MIN_TOKENS + region[0], region[1])
  subdoc: LegalDocument = doc.subdoc_slice(_s)

  _sign, _sign_span = find_value_sign(subdoc.tokens_map)

  # ======================================
  value, currency = extract_sum(subdoc.text)
  # ======================================

  sign = SemanticTag('sign', _sign, _sign_span)
  sum = SemanticTag('value', value, region)
  currency = SemanticTag('currency', currency, region)

  sign.offset(subdoc.start)

  return ValueSemanticTags(sign, sum, currency)


def extract_sum_and_sign_3(sr: PatternMatch, region: slice) -> ValueConstraint:
  warnings.warn("use extract_sum_sign_currency", DeprecationWarning)

  _slice = slice(region.start - VALUE_SIGN_MIN_TOKENS, region.stop)
  subtokens = sr.tokens[_slice]
  _prefix_tokens = subtokens[0:VALUE_SIGN_MIN_TOKENS + 1]
  _prefix = untokenize(_prefix_tokens)
  _sign = detect_sign(_prefix)
  # ======================================
  _sum = extract_sum_from_tokens_2(subtokens)
  # ======================================

  currency = "UNDEF"
  value = np.nan
  if _sum is not None:
    currency = _sum[1]
    if _sum[1] in currencly_map:
      currency = currencly_map[_sum[1]]
    value = _sum[0]

  vc = ValueConstraint(value, currency, _sign, TokensWithAttention([], []))

  return vc

#
#
# if __name__ == '__main__':
#
#   ex="""
#   с учетом положений пункта 8.4.4 устава , одобрение заключения , изменения , продления , возобновления или расторжения обществом ( i ) любых договоров страхования , если стоимость соответствующего договора или нескольких взаимосвязанных договоров превышает 100 000 ( сто тысяч ) долларов сша ( или эквивалент этой суммы в рублях или иной валюте ) , либо ( ii ) договоров страхования , относящихся к операторскому договору 6к или связанных с операторским договором 6к до закрытия сделки 6к независимо от суммы ;
#   """
#   #
#   # self.pattern_prefix: str = None
#   # self.attention_vector_name: str = None
#   # self.parent: LegalDocument = None
#   # self.confidence: float = 0
#   # self.region: slice = None
#
#   def extract_sum_and_sign_3(tokens:Tokens, region: slice) -> ValueConstraint:
#     _slice = slice(region.start - VALUE_SIGN_MIN_TOKENS, region.stop)
#     subtokens = tokens[_slice]
#     _prefix_tokens = subtokens[0:VALUE_SIGN_MIN_TOKENS + 1]
#     _prefix = untokenize(_prefix_tokens)
#     _sign = detect_sign(_prefix)
#     # ======================================
#     _sum = extract_sum_from_tokens_2(subtokens)
#     # ======================================
#
#     currency = "UNDEF"
#     value = np.nan
#     if _sum is not None:
#       currency = _sum[1]
#       if _sum[1] in currencly_map:
#         currency = currencly_map[_sum[1]]
#       value = _sum[0]
#
#     vc = ValueConstraint(value, currency, _sign, TokensWithAttention([], []))
#
#     return vc
#
#
#   def extract_all_contraints_from_sr( sentence ) :
#
#     tokens = sentence.split(' ')
#     def tokens_before_index(string, index):
#       return len(string[:index].split(' '))
#
#
#     all = [slice(m.start(0), m.end(0)) for m in re.finditer(complete_re, sentence)]
#     constraints: List[ProbableValue] = []
#
#     for a in all:
#       print(tokens_before_index(sentence, a.start), 'from', sentence[a])
#       token_index_s = tokens_before_index(sentence, a.start) - 1
#       token_index_e = tokens_before_index(sentence, a.stop)
#
#       region = slice(token_index_s, token_index_e)
#       print('region=', region)
#
#
#       vc = extract_sum_and_sign_3(tokens, region)
#       print(vc.sign)
#       # _e = _expand_slice(region, 10)
#       # vc.context = TokensWithAttention(search_result.tokens[_e], attention_vector[_e])
#       # confidence = attention_vector[region.start]
#       # pv = ProbableValue(vc, confidence)
#       #
#       # constraints.append(pv)
#
#     # return constraints
#
#
#
#   extract_all_contraints_from_sr(ex)
