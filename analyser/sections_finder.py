import warnings
from typing import List

import numpy as np
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity

from analyser.embedding_tools import AbstractEmbedder
from analyser.hyperparams import HyperParameters
from analyser.legal_docs import LegalDocument
from analyser.ml_tools import max_exclusive_pattern, softmax_rows
from analyser.ml_tools import put_if_better, SemanticTag, calc_distances_per_pattern, \
  attribute_patternmatch_to_index, rectifyed_sum
from analyser.parsing import ParsingSimpleContext
from analyser.patterns import AbstractPatternFactory


class HeadlineMeta:
  def __init__(self, index, _type, confidence: float, subdoc, body=None):
    warnings.warn("deprecated", DeprecationWarning)

    self.index: int = index
    self.confidence: float = confidence
    self.type: str = _type

    self.subdoc: LegalDocument = subdoc  # headline
    self.body: LegalDocument = body

    self.attention: List[float] = None  # optional

  def get_header(self):
    return self.subdoc.text

  def as_tag(self):
    st = SemanticTag(self.type, None, (self.subdoc.start, self.body.end))
    st.confidence = self.confidence
    return st

  header = property(get_header)


class FocusingSectionsFinder:
  def __init__(self, ctx: ParsingSimpleContext):
    self.ctx: ParsingSimpleContext = ctx

  def find_sections(self, doc: LegalDocument, factory: AbstractPatternFactory, headlines: List[str],
                    headline_patterns_prefix: str = 'headline.',
                    confidence_threshold=HyperParameters.header_topic_min_confidence) -> dict:

    sections_filtered = {}
    for section_type in headlines:
      # find best header for each section

      pattern_prefix = f'{headline_patterns_prefix}{section_type}'

      headers: [LegalDocument] = [doc.subdoc_slice(p.header.as_slice()) for p in doc.paragraphs]

      _max_confidence = 0
      _max_header_i = 0

      for header_index, header in enumerate(headers):

        if header.text and header.text.strip():
          vvs = header.calculate_distances_per_pattern(factory, pattern_prefix=pattern_prefix, merge=False)
          vv = max_exclusive_pattern(list(vvs.values()))
          _confidence = max(vv)

          if _confidence > _max_confidence:
            _max_confidence = _confidence
            _max_header_i = header_index

      if _max_confidence > confidence_threshold:
        # TODO: use one-hots, do not collapse to a single topic
        put_if_better(sections_filtered, _max_header_i, (_max_confidence, section_type), lambda a, b: a[1] < b[1])

    sections = {}
    for header_i in sections_filtered:
      confidence, section_type = sections_filtered[header_i]

      para = doc.paragraphs[header_i]
      body = doc.subdoc_slice(para.body.as_slice(), name=section_type + '.body')
      head = doc.subdoc_slice(para.header.as_slice(), name=section_type + ".head")
      hl_info = HeadlineMeta(header_i, section_type, confidence, head, body)

      sections[section_type] = hl_info

    doc.sections = sections  # TODO: keep doc immutable
    return sections


def map_headlines_to_patterns(doc: LegalDocument,
                              patterns_named_embeddings: DataFrame,
                              elmo_embedder_default: AbstractEmbedder):
  warnings.warn("consider using map_headers, it returns probalility distribution", DeprecationWarning)
  headers: [str] = doc.headers_as_sentences()

  if not headers:
    return []

  headers_embedding = elmo_embedder_default.embedd_strings(headers)

  header_to_pattern_distances = calc_distances_per_pattern(headers_embedding, patterns_named_embeddings)
  return attribute_patternmatch_to_index(header_to_pattern_distances)


def __make_headers_df(doc: LegalDocument):
  # doc.sections
  headers_tags: [SemanticTag] = [p.header for p in doc.paragraphs]
  headers: [LegalDocument] = [doc.subdoc_slice(h.as_slice()) for h in headers_tags]

  # headers
  headers_df = DataFrame()
  headers_df['text'] = [h.text for h in headers]

  headers_df['start'] = [h.start for h in headers]
  headers_df['end'] = [h.end for h in headers]

  headers_df['body_start'] = [p.body.span[0] for p in doc.paragraphs]
  headers_df['body_end'] = [p.body.span[1] for p in doc.paragraphs]

  return headers_df, headers


def map_headers(doc: LegalDocument, centroids: DataFrame, relu_threshold=0.45) -> DataFrame:
  headers_df, headers = __make_headers_df(doc)

  for section_type in centroids.key.unique():
    patters_emb = centroids[centroids.key == section_type][centroids.columns[0:1024]].values
    headers_df[section_type] = 0.0  # seed

    for header_index, header in enumerate(headers):
      _embedding = header.embeddings
      distances = np.array(cosine_similarity(patters_emb, _embedding)).T

      # distances = relu(distances,  0.4666)
      # vv = max_exclusive_pattern(distances)
      vv = rectifyed_sum(distances, relu_threshold)

      _confidence = max(vv)

      headers_df.at[header_index, section_type] = _confidence

  section_types = centroids.key.unique()
  headers_df = softmax_rows(headers_df, section_types)

  return headers_df
