import warnings
from typing import List

from hyperparams import HyperParameters
from legal_docs import LegalDocument
from ml_tools import put_if_better, max_exclusive_pattern, SemanticTag
from parsing import ParsingSimpleContext
from patterns import AbstractPatternFactory


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

      headers = [doc.subdoc_slice(p.header.as_slice()) for p in doc.paragraphs]

      _max_confidence = 0
      _max_header_i = 0
      for header_index in range(len(headers)):
        header = headers[header_index]
        if header.text and header.text.strip():
          vvs = header.calculate_distances_per_pattern(factory, pattern_prefix=pattern_prefix, merge=False)
          vv = max_exclusive_pattern(list(vvs.values()))
          _confidence = max(vv)

          if _confidence > _max_confidence:
            _max_confidence = _confidence
            _max_header_i = header_index

      if _max_confidence > confidence_threshold:
        put_if_better(sections_filtered, _max_header_i, (_max_confidence, section_type), lambda a, b: a[1] < b[1])

    sections = {}
    for header_i in sections_filtered:
      confidence, section_type = sections_filtered[header_i]

      para = doc.paragraphs[header_i]
      body = doc.subdoc_slice(para.body.as_slice(), name=section_type + '.body')
      head = doc.subdoc_slice(para.header.as_slice(), name=section_type + ".head")
      hl_info = HeadlineMeta(header_i, section_type, confidence, head, body)

      sections[section_type] = hl_info

    doc.sections = sections
    return sections
