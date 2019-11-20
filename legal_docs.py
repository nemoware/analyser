#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# legal_docs.py
import datetime
import json
from functools import wraps

from bson import json_util

from doc_structure import get_tokenized_line_number
from documents import TextMap
from embedding_tools import AbstractEmbedder
from ml_tools import normalize, smooth_safe, max_exclusive_pattern, SemanticTag, conditional_p_sum, put_if_better, \
  calc_distances_per_pattern_dict
from patterns import *
from structures import ContractTags
from tests.test_text_tools import split_sentences_into_map
from text_normalize import *
from text_tools import *
from transaction_values import _re_greather_then, _re_less_then, _re_greather_then_1, VALUE_SIGN_MIN_TOKENS, \
  find_value_spans

REPORTED_DEPRECATED = {}


def remove_sr_duplicates_conditionally(list_: PatternSearchResults):
  ret = []
  dups = {}
  for r in list_:
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


class Paragraph:
  def __init__(self, header: SemanticTag, body: SemanticTag):
    self.header: SemanticTag = header
    self.body: SemanticTag = body


class LegalDocument:

  def __init__(self, original_text=None, name="legal_doc"):

    self._id = None  # TODO
    self.date = None

    self.filename = None
    self._original_text = original_text
    self._normal_text = None

    # todo: use pandas' DataFrame
    self.distances_per_pattern_dict = {}

    self.tokens_map: TextMap = None
    self.tokens_map_norm: TextMap or None = None

    self.sections = None  # TODO:deprecated
    self.paragraphs: List[Paragraph] = []
    self.name = name

    # subdocs
    self.start = 0
    self.end = None  # TODO:

    # TODO: probably we don't have to keep embeddings, just distances_per_pattern_dict
    self.embeddings = None

  def parse(self, txt=None):
    if txt is None:
      txt = self.original_text

    assert txt is not None

    self._normal_text = self.preprocess_text(txt)
    self.tokens_map = TextMap(self._normal_text)

    self.tokens_map_norm = CaseNormalizer().normalize_tokens_map_case(self.tokens_map)
    # body = SemanticTag(kind=None, value=None, span=(0, len(self.tokens_map)));
    # header = SemanticTag(kind=None, value=None, span=(0, 0));
    # self.paragraphs = [Paragraph(header, body)]
    return self

  def __len__(self):
    return self.tokens_map.get_len()

  def __add__(self, suffix: 'LegalDocument'):
    '''
    1) dont forget to add spaces between concatenated docs!!
    2) embeddings are lost
    3)
    :param suffix: doc to add
    :return: self + suffix
    '''
    assert self._normal_text is not None
    assert suffix._normal_text is not None

    self.distances_per_pattern_dict = {}
    self._normal_text += suffix.normal_text
    self._original_text += suffix.original_text

    self.tokens_map += suffix.tokens_map
    self.tokens_map_norm += suffix.tokens_map_norm

    self.sections = None

    self.paragraphs += suffix.paragraphs
    # subdocs
    self.end = suffix.end
    self.embeddings = None

    return self

  def get_tags(self) -> [SemanticTag]:
    return []

  def get_tags_attention(self):
    _attention = np.zeros(self.__len__())

    for t in self.get_tags():
      _attention[t.as_slice()] += t.confidence
    return _attention

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

  def preprocess_text(self, txt):
    assert txt is not None
    return normalize_text(txt, replacements_regex)

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

  def subdoc_slice(self, __s: slice, name='undef'):
    assert self.tokens_map is not None
    # TODO: support None in slice begin
    _s = slice(max((0, __s.start)), max((0, __s.stop)))

    klazz = self.__class__
    sub = klazz(None)
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

  def __getitem__(self, key):
    if isinstance(key, slice):
      # Get the start, stop, and step from the slice
      return self.subdoc_slice(key)
    else:
      raise TypeError("Invalid argument type.")

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
      if max(v) > 0.6:  # TODO: get rid of magic numbers
        vector_i, _ = improve_attention_vector(self.embeddings, v, relu_th=0.6, mix=0.9)
        vectors_i.append(vector_i)
      else:
        vectors_i.append(v)

    x = max_exclusive_pattern(vectors_i)
    self.distances_per_pattern_dict[attention_vector_name_soft] = x
    x = relu(x, 0.8)

    self.distances_per_pattern_dict[attention_vector_name] = x
    return x, attention_vector_name

  def embedd_tokens(self, embedder: AbstractEmbedder, verbosity=2, max_tokens=8000):
    warnings.warn("use embedd_words", DeprecationWarning)
    if self.tokens:
      max_tokens = max_tokens
      if len(self.tokens_map_norm) > max_tokens:
        self.embeddings = _embedd_large(self.tokens_map_norm, embedder, max_tokens, verbosity)
      else:
        self.embeddings = embedder.embedd_tokens(self.tokens)
    else:
      raise ValueError(f'cannot embed doc {self.filename}, no tokens')

  def get_tag_text(self, tag: SemanticTag):
    return self.tokens_map.text_range(tag.span)

  def substr(self, tag: SemanticTag) -> str:
    return self.tokens_map.text_range(tag.span)

  def tag_value(self, tagname):
    t = SemanticTag.find_by_kind(self.get_tags(), tagname)
    if t:
      return t.value
    else:
      return None


class DocumentJson:

  @staticmethod
  def from_json(json_string: str) -> 'DocumentJson':
    jsondata = json.loads(json_string, object_hook=json_util.object_hook)

    c = DocumentJson(None)
    c.__dict__ = jsondata

    return c

  def __init__(self, doc: LegalDocument):

    self._id: str = None
    self.original_text = None
    self.normal_text = None

    self.analyze_timestamp = datetime.datetime.now()
    self.tokenization_maps = {}

    if doc is None:
      return

    self.checksum = hash(doc.normal_text)
    self.tokenization_maps['words'] = doc.tokens_map.map

    for field in doc.__dict__:
      if field in self.__dict__:
        self.__dict__[field] = doc.__dict__[field]

    self.original_text = doc.original_text
    self.normal_text = doc.normal_text

    self.attributes = self.__tags_to_attributes_dict(doc.get_tags())
    self.headers = self.__tags_to_attributes_list([hi.header for hi in doc.paragraphs])

  def __tags_to_attributes_list(self, _tags):

    attributes = []
    for t in _tags:
      key, attr = self.__tag_to_attribute(t)
      attributes.append(attr)

    return attributes

  def __tag_to_attribute(self, t: SemanticTag):

    key = t.get_key()
    attribute = t.__dict__.copy()
    del attribute['kind']
    if '_parent_tag' in attribute:
      if t.parent is not None:
        attribute['parent'] = t.parent
      del attribute['_parent_tag']

    return key, attribute

  def __tags_to_attributes_dict(self, _tags: [SemanticTag]):

    attributes = {}
    for t in _tags:
      key, attr = self.__tag_to_attribute(t)

      if key in attributes:
        raise RuntimeError(key + ' duplicated key')

      attributes[key] = attr

    return attributes

  def dumps(self):
    return json.dumps(self.__dict__, indent=2, ensure_ascii=False, default=json_util.default)


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
    # print(key, [start, end])

    subdoc = doc.subdoc(start, end)
    subdoc.filename = key
    doc.subdocs.append(subdoc)


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
  if c == 0:
    raise ValueError('no pattern with prefix: ' + pattern_prefix)

  return distances_per_pattern_dict


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


class ContractValue:
  def __init__(self, sign: SemanticTag, value: SemanticTag, currency: SemanticTag, parent: SemanticTag = None):
    self.value = value
    self.sign = sign
    self.currency = currency
    self.parent = parent

  def as_list(self) -> [SemanticTag]:
    return [self.value, self.sign, self.currency, self.parent]

  def __add__(self, addon):
    for t in self.as_list():
      t.offset(addon)
    return self

  def span(self):
    left = min([tag.span[0] for tag in self.as_list()])
    right = max([tag.span[0] for tag in self.as_list()])
    return left, right

  def __mul__(self, confidence_k):
    for _r in self.as_list():
      _r.confidence *= confidence_k
    return self

  def integral_sorting_confidence(self) -> float:
    return conditional_p_sum(
      [self.parent.confidence, self.value.confidence, self.currency.confidence, self.sign.confidence])


def extract_sum_sign_currency(doc: LegalDocument, region: (int, int)) -> ContractValue or None:
  subdoc: LegalDocument = doc[region[0] - VALUE_SIGN_MIN_TOKENS: region[1]]

  _sign, _sign_span = find_value_sign(subdoc.tokens_map)

  # ======================================
  results = find_value_spans(subdoc.text)
  # ======================================

  if results:
    value_char_span, value, currency_char_span, currency, including_vat, original_value = results
    value_span = subdoc.tokens_map.token_indices_by_char_range_2(value_char_span)
    currency_span = subdoc.tokens_map.token_indices_by_char_range_2(currency_char_span)

    group = SemanticTag('sign_value_currency', None, region)

    sign = SemanticTag(ContractTags.Sign.display_string, _sign, _sign_span, parent=group)
    sign.offset(subdoc.start)

    value_tag = SemanticTag(ContractTags.Value.display_string, value, value_span, parent=group)
    value_tag.offset(subdoc.start)

    currency = SemanticTag(ContractTags.Currency.display_string, currency, currency_span, parent=group)
    currency.offset(subdoc.start)

    return ContractValue(sign, value_tag, currency, group)
  else:
    return None


def tokenize_doc_into_sentences_map(doc: LegalDocument, max_len_chars=150) -> TextMap:
  tm = TextMap('', [])

  # if doc.paragraphs:
  #   for p in doc.paragraphs:
  #     header_lines = doc.substr(p.header).splitlines(True)
  #     for line in header_lines:
  #       tm += split_sentences_into_map(line, max_len_chars)
  #
  #     body_lines = doc.substr(p.body).splitlines(True)
  #     for line in body_lines:
  #       tm += split_sentences_into_map(line, max_len_chars)
  # else:
  body_lines = doc.text.splitlines(True)
  for line in body_lines:
    tm += split_sentences_into_map(line, max_len_chars)

  return tm


PARAGRAPH_DELIMITER = '\n'


def _embedd_large(text_map, embedder, max_tokens=8000, verbosity=2):
  overlap = max_tokens // 20

  number_of_windows = 1 + len(text_map) // max_tokens
  window = max_tokens

  if verbosity > 1:
    msg = f"WARNING: Document is too large for embedding: {len(text_map)} tokens. Splitting into {number_of_windows} windows overlapping with {overlap} tokens "
    warnings.warn(msg)

  start = 0
  embeddings = None
  # tokens = []
  while start < len(text_map):

    subtokens: Tokens = text_map[start:start + window + overlap]
    if verbosity > 2:
      print("Embedding region:", start, len(subtokens))

    sub_embeddings = embedder.embedd_tokens(subtokens)[0:window]

    if embeddings is None:
      embeddings = sub_embeddings
    else:
      embeddings = np.concatenate([embeddings, sub_embeddings])

    start += window

  return embeddings
  # self.tokens = tokens


def embedd_sentences(text_map: TextMap, embedder: AbstractEmbedder, verbosity=2, max_tokens=100):
  warnings.warn("use embedd_words", DeprecationWarning)

  max_tokens = max_tokens
  if len(text_map) > max_tokens:
    return _embedd_large(text_map, embedder, max_tokens, verbosity)
  else:
    return embedder.embedd_tokens(text_map.tokens)


def make_headline_attention_vector(doc):
  parser_headline_attention_vector = np.zeros(len(doc.tokens_map))

  for p in doc.paragraphs:
    parser_headline_attention_vector[p.header.slice] = 1

  return parser_headline_attention_vector


def headers_as_sentences(doc: LegalDocument):
  numbered = [doc.tokens_map.slice(p.header.as_slice()) for p in doc.paragraphs]

  stripped = []

  for s in numbered:
    n, span, _, _ = get_tokenized_line_number(s.tokens, 0)
    line = s.text_range([span[1], None]).strip()

    stripped.append(line)

  return stripped


def map_headlines_to_patterns(charter, patterns_dict, patterns_embeddings, elmo_embedder_default, pattern_prefix: str,
                              pattern_suffixes: [str]):
  headers = headers_as_sentences(charter)
  headers_embedding = elmo_embedder_default.embedd_strings(headers)


  header_to_pattern_distances = calc_distances_per_pattern_dict(headers_embedding,
                                                                patterns_dict,
                                                                patterns_embeddings)

  patterns_by_headers = [()] * len(headers)
  for e in range(len(headers)):
    # for each header
    max_confidence = 0
    for pattern_suffix in pattern_suffixes:
      pattern_name = pattern_prefix + pattern_suffix
      # find best pattern
      confidence = header_to_pattern_distances[pattern_name][e]
      if confidence > max_confidence and confidence > 0.66:
        patterns_by_headers[e] = (pattern_name, pattern_suffix, confidence, headers[e], charter.paragraphs[e])
        max_confidence = confidence

  return patterns_by_headers, header_to_pattern_distances
