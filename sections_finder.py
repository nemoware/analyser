import warnings
from typing import List

import numpy as np

from legal_docs import LegalDocument
from ml_tools import put_if_better, cut_above, relu, filter_values_by_key_prefix, \
  smooth_safe, max_exclusive_pattern, sum_probabilities
from parsing import ParsingSimpleContext
from patterns import AbstractPatternFactory, improve_attention_vector


class HeadlineMeta:
  def __init__(self, index, _type, confidence: float, subdoc, body=None):
    warnings.warn("deprecated", DeprecationWarning)

    self.index: int = index
    self.confidence: float = confidence
    self.type: str = _type

    self.subdoc: LegalDocument = subdoc
    self.body: LegalDocument = body

    self.attention: List[float] = None  # optional

class SectionsFinder:

  def __init__(self, ctx: ParsingSimpleContext):
    self.ctx: ParsingSimpleContext = ctx
    pass

  def find_sections(self, doc: LegalDocument, factory: AbstractPatternFactory, headlines: List[str],
                    headline_patterns_prefix: str = 'headline.', additional_attention: List[float] = None) -> dict:
    raise NotImplementedError()


class FocusingSectionsFinder(SectionsFinder):
  def __init__(self, ctx: ParsingSimpleContext):
    SectionsFinder.__init__(self, ctx)

  def find_sections(self, doc_: LegalDocument, factory: AbstractPatternFactory, headlines: List[str],
                    headline_patterns_prefix: str = 'headline.', additional_attention: List[float] = None) -> dict:

    headers = [doc_.subdoc_slice(p.header.as_slice()) for p in doc_.paragraphs]

    # section_by_index = {}
    header_index = 0
    types_by_header = {}
    headers_by_type = {}
    for header in headers:
      # find best section_type (pattern match) for each header
      for section_type in headlines:
        # header X type cartesian

        # like 'name.', 'head.all.', 'head.gen.', 'head.directors.'
        pattern_prefix = f'{headline_patterns_prefix}{section_type}'
        header.calculate_distances_per_pattern(factory, pattern_prefix=pattern_prefix, merge=True)
        vectors = filter_values_by_key_prefix(header.distances_per_pattern_dict, pattern_prefix)

        headline_correspondence_v = sum_probabilities(list(vectors))
        confidence = max(headline_correspondence_v)



        # key = header_index
        y=(section_type, confidence, header.text)
        put_if_better(types_by_header, header_index, y, lambda a, b: a[1]>b[1] )

        # if key in types_by_header:
        #   if confidence > types_by_header[key][1]:
        #     types_by_header[key] = (section_type, confidence, header.text)
        # else:
        #   types_by_header[key] = (section_type, confidence, header.text)

      header_index += 1

    headers_by_type = {}
    for header_index in types_by_header.keys():
      stats = types_by_header[header_index]
      key = stats[0]  #
      y=(header_index, stats[1], stats[2])
      put_if_better(headers_by_type, key, y, lambda a, b: a[1] > b[1])

    sections={}
    for header_index in types_by_header.keys():

      stats = types_by_header[header_index]
      type = stats[0]
      para = doc_.paragraphs[header_index]
      body = doc_.subdoc_slice(para.body.as_slice(), name=type+'.body')
      head = doc_.subdoc_slice(para.header.as_slice(), name=type+".head")
      hl_info = HeadlineMeta(header_index, type, stats[1], head, body)
      sections[type]=hl_info

    doc_.sections=sections
    return sections

  def ___find_sections(self, doc: LegalDocument, factory: AbstractPatternFactory, headlines: List[str],
                       headline_patterns_prefix: str = 'headline.', additional_attention: List[float] = None) -> dict:

    """
    Fuzziy Finds maps sections to known (interresting) ones
    TODO: try it on Contracts and Protocols as well
    TODO: if well, move from here
    🍄 🍄 🍄 🍄 🍄 Keep in in the dark and feed it sh**
    """
    warnings.warn("deprecated", DeprecationWarning)

    def is_hl_more_confident(a: HeadlineMeta, b: HeadlineMeta):
      return a.confidence > b.confidence

    #     assert do
    headlines_attention_vector = self.normalize_headline_attention_vector(self.make_headline_attention_vector(doc))

    section_by_index = {}
    for section_type in headlines:
      # like ['name.', 'head.all.', 'head.gen.', 'head.directors.']:
      pattern_prefix = f'{headline_patterns_prefix}{section_type}'
      doc.calculate_distances_per_pattern(factory, pattern_prefix=pattern_prefix, merge=True)

      # warning! these are the boundaries of the headline, not of the entire section
      bounds, confidence, attention = self._find_charter_section_start(doc, pattern_prefix, headlines_attention_vector,
                                                                       additional_attention)

      if confidence > 0.5:
        sl = slice(bounds[0], bounds[1])
        hl_info = HeadlineMeta(None, section_type, confidence, doc.subdoc_slice(sl, name=section_type))
        hl_info.attention = attention
        put_if_better(section_by_index, key=sl.start, x=hl_info, is_better=is_hl_more_confident)

    # end-for

    # now slicing the doc
    sorted_starts = [i for i in sorted(section_by_index.keys())]
    # // sorted_starts.append(len(doc.tokens))

    section_by_type = {}

    for i in range(len(sorted_starts)):
      index = sorted_starts[i]
      section: HeadlineMeta = section_by_index[index]
      start = index  # todo: probably take the end of the caption
      end = doc.structure.next_headline_after(start)

      # end_alt = sorted_starts[i + 1]
      #
      section_len = end - start

      sli = slice(start, start + section_len)
      section.body = doc.subdoc_slice(sli, name=section.type)
      section.attention = section.attention[sli]
      section_by_type[section.type] = section

    # end-for
    doc.sections = section_by_type

    self.ctx._logstep("Splitting Document into sections ✂️ 📃 -> 📄📄📄")
    return section_by_type

  """ ❤️ == GOOD HEART LINE ====================================================== """

  def make_headline_attention_vector(self, doc):

    headlines_attention_vector = np.zeros(len(doc.tokens))
    for p in doc.paragraphs:
      l = slice(p.header.span[0], p.header.span[1])
      headlines_attention_vector[l] = 1

    return np.array(headlines_attention_vector)

  """ ❤️ == GOOD HEART LINE ====================================================== """

  def normalize_headline_attention_vector(self, headline_attention_vector_pure):
    # XXX: test it
    #   _max_head_threshold = max(headline_attention_vector_pure) * 0.75
    _max_head_threshold = 1  # max(headline_attention_vector_pure) * 0.75
    # XXX: test it
    #   print(_max_head)
    headline_attention_vector = cut_above(headline_attention_vector_pure, _max_head_threshold)
    #   headline_attention_vector /= 2 # 5 is the maximum points a headline may gain during headlne detection : TODO:
    return relu(headline_attention_vector)

  """ ❤️ == GOOD HEART LINE ====================================================== """

  def _find_charter_section_start(self, doc, headline_pattern_prefix, headlines_attention_vector, additional_attention):

    assert headlines_attention_vector is not None

    vectors = filter_values_by_key_prefix(doc.distances_per_pattern_dict, headline_pattern_prefix)
    # sum_probabilities
    # v = rectifyed_sum(vectors, 0.3)
    v = max_exclusive_pattern(vectors)
    v = relu(v, 0.6)

    if additional_attention is not None:
      additional_attention_s = smooth_safe(additional_attention, 6)
      v += additional_attention_s

    #   v, _ = improve_attention_vector(doc.embeddings, v, relu_th=0.1)
    v *= (headlines_attention_vector + 0.1)
    if max(v) > 0.75:
      v, _ = improve_attention_vector(doc.embeddings, v, relu_th=0.0)

    doc.distances_per_pattern_dict["ha$." + headline_pattern_prefix] = v

    # span = 100
    best_id = np.argmax(v)
    # dia = slice(max(0, best_id - span), min(best_id + span, len(v)))

    bounds = doc.tokens_map.sentence_at_index(best_id)

    confidence = v[best_id]
    return bounds, confidence, v
