#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# legal_docs.py
import datetime
import json
from enum import Enum

from bson import json_util

import analyser
from analyser.doc_structure import get_tokenized_line_number
from analyser.documents import TextMap, split_sentences_into_map
from analyser.embedding_tools import AbstractEmbedder
from analyser.ml_tools import SemanticTag, conditional_p_sum, \
  FixedVector, attribute_patternmatch_to_index, calc_distances_per_pattern
from analyser.patterns import *
from analyser.structures import ContractTags
from analyser.text_normalize import *
from analyser.text_tools import *
from analyser.transaction_values import _re_greather_then, _re_less_then, _re_greather_then_1, VALUE_SIGN_MIN_TOKENS, \
  find_value_spans

REPORTED_DEPRECATED = {}


class ParserWarnings(Enum):
  org_name_not_found = 1,
  org_type_not_found = 2,
  org_struct_level_not_found = 3,
  date_not_found = 4
  number_not_found = 5,
  value_section_not_found = 7,
  contract_value_not_found = 8,
  subject_section_not_found = 6,
  contract_subject_not_found = 9,
  contract_subject_section_not_found = 12,
  protocol_agenda_not_found = 10,

  boring_agenda_questions = 11


class Paragraph:
  def __init__(self, header: SemanticTag, body: SemanticTag):
    self.header: SemanticTag = header
    self.body: SemanticTag = body

  def as_combination(self) -> SemanticTag:
    return SemanticTag(self.header.kind + '-' + self.body.kind, None, span=(self.header.span[0], self.body.span[1]))


class LegalDocument:

  def __init__(self, original_text=None, name="legal_doc"):

    self._id = None  # TODO
    self.date: SemanticTag or None = None
    self.number: SemanticTag or None = None

    self.filename = None
    self._original_text = original_text
    self._normal_text = None
    self.warnings: [str] = []

    # todo: use pandas' DataFrame
    self.distances_per_pattern_dict = {}

    self.tokens_map: TextMap or None = None
    self.tokens_map_norm: TextMap or None = None

    self.sections = None  # TODO:deprecated
    self.paragraphs: List[Paragraph] = []
    self.name = name

    # subdocs
    self.start = 0
    self.end = None  # TODO:

    # TODO: probably we don't have to keep embeddings, just distances_per_pattern_dict
    self.embeddings = None

  def warn(self, msg: ParserWarnings, comment: str = None):
    w = {}
    if comment:
      w['comment'] = comment
    w['code'] = msg.name
    self.warnings.append(w)

  def parse(self, txt=None):
    if txt is None:
      txt = self.original_text

    assert txt is not None

    self._normal_text = self.preprocess_text(txt)
    self.tokens_map = TextMap(self._normal_text)
    self.tokens_map_norm = CaseNormalizer().normalize_tokens_map_case(self.tokens_map)
    return self

  def sentence_at_index(self, i: int, return_delimiters=True) -> (int, int):
    return self.tokens_map.sentence_at_index(i, return_delimiters)

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

  def headers_as_sentences(self) -> [str]:
    return headers_as_sentences(self)

  def get_tags_attention(self) -> FixedVector:
    _attention = np.zeros(self.__len__())

    for t in self.get_tags():
      _attention[t.as_slice()] += t.confidence
    return _attention

  def to_json_obj(self):
    j = DocumentJson(self)
    return j.__dict__

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

  def get_checksum(self):
    return hash(self._normal_text)

  tokens_cc = property(get_tokens_cc)
  tokens = property(get_tokens)
  original_text = property(get_original_text)
  normal_text = property(get_normal_text, None)
  text = property(get_text)
  checksum = property(get_checksum, None)

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

  def subdoc(self, start, end):
    warnings.warn("use subdoc_slice", DeprecationWarning)
    _s = slice(max(0, start), end)
    return self.subdoc_slice(_s)

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

  def is_same_org(self, org_name: str) -> bool:
    tags: [SemanticTag] = self.get_tags()
    for t in tags:
      if t.kind in ['org-1-name', 'org-2-name', 'org-3-name']:
        if t.value == org_name:
          return True
    return False

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


class LegalDocumentExt(LegalDocument):

  def __init__(self, doc: LegalDocument):
    super().__init__('')
    if doc is not None:
      self.__dict__ = doc.__dict__

    self.sentence_map: TextMap or None = None
    self.sentences_embeddings: [] = None
    self.distances_per_sentence_pattern_dict = {}

  def parse(self, txt=None):
    super().parse(txt)
    self.sentence_map = tokenize_doc_into_sentences_map(self, 200)
    return self

  def subdoc_slice(self, __s: slice, name='undef'):
    sub = super().subdoc_slice(__s, name)
    span = [max((0, __s.start)), max((0, __s.stop))]

    if self.sentence_map:
      sentences_span = self.tokens_map.remap_span(span, self.sentence_map)
      _slice = slice(sentences_span[0], sentences_span[1])
      sub.sentence_map = self.sentence_map.slice(_slice)

      if self.sentences_embeddings is not None:
        sub.sentences_embeddings = self.sentences_embeddings[_slice]
    else:
      warnings.warn('split into sentences first')
    return sub


class DocumentJson:

  @staticmethod
  def from_json(json_string: str) -> 'DocumentJson':
    jsondata = json.loads(json_string, object_hook=json_util.object_hook)

    c = DocumentJson(None)
    c.__dict__ = jsondata

    return c

  def __init__(self, doc: LegalDocument):
    self.version = analyser.__version__

    self._id: str = None
    self.original_text = None
    self.normal_text = None
    self.warnings: [str] = []

    self.analyze_timestamp = datetime.datetime.now()
    self.tokenization_maps = {}

    if doc is None:
      return
    self.checksum = doc.get_checksum()
    self.warnings: [str] = list(doc.warnings)

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

    if isinstance(t.value, Enum):
      attribute['value'] = t.value.name

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


MIN_DOC_LEN = 5


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


def find_value_sign(txt: TextMap) -> (int, (int, int)):
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
    a = next(txt.finditer(_re_greather_then), None)  # более
    if a:
      return +1, a

  return 0, None


class ContractValue:
  def __init__(self, sign: SemanticTag, value: SemanticTag, currency: SemanticTag, parent: SemanticTag = None):
    self.value = value
    self.sign = sign
    self.currency = currency
    self.parent = parent

  def as_list(self) -> [SemanticTag]:
    if self.sign.value != 0:
      return [self.value, self.sign, self.currency, self.parent]
    else:
      return [self.value, self.currency, self.parent]

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
    value_span = subdoc.tokens_map.token_indices_by_char_range(value_char_span)
    currency_span = subdoc.tokens_map.token_indices_by_char_range(currency_char_span)

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


def headers_as_sentences(doc: LegalDocument, normal_case=True, strip_number=True) -> [str]:
  _map = doc.tokens_map
  if normal_case:
    _map = doc.tokens_map_norm

  numbered = [_map.slice(p.header.as_slice()) for p in doc.paragraphs]
  stripped: [str] = []

  for s in numbered:
    if strip_number:
      a = get_tokenized_line_number(s.tokens, 0)
      _, span, _, _ = a
      line = s.text_range([span[1], None]).strip()
    else:
      line = s.text
    stripped.append(line)

  return stripped


def map_headlines_to_patterns(doc: LegalDocument,
                              patterns_named_embeddings,
                              elmo_embedder_default: AbstractEmbedder):
  headers: [str] = doc.headers_as_sentences()

  if not headers:
    return []

  headers_embedding = elmo_embedder_default.embedd_strings(headers)

  header_to_pattern_distances = calc_distances_per_pattern(headers_embedding, patterns_named_embeddings)
  return attribute_patternmatch_to_index(header_to_pattern_distances)


def remap_attention_vector(v: FixedVector, source_map: TextMap, target_map: TextMap) -> FixedVector:
  av = np.zeros(len(target_map))

  for i in range(len(source_map)):
    span = i, i + 1

    t_span = source_map.remap_span(span, target_map)
    av[t_span[0]:t_span[1]] = v[i]
  return av
