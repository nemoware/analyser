# -*- coding: utf-8 -*-
from contract_parser import extract_all_contraints_from_sr_2
from legal_docs import BasicContractDocument, LegalDocument, deprecated
from ml_tools import ProbableValue
from parsing import ParsingContext
from patterns import *
from protocol_parser import ProtocolPatternFactory
from renderer import AbstractRenderer
from transaction_values import extract_sum_from_tokens



class ProtocolRenderer(AbstractRenderer):
  pass

def select_most_confident_if_almost_equal( a: ProbableValue, alternative: ProbableValue,
                                          equality_range=0.0):

  try:
    if abs (a.value.value -  alternative.value.value) < equality_range:
      if a.confidence > alternative.confidence:
        return a
      else:
        return alternative
  except:
    return a

  return a

# from split import *
# ----------------------
class ProtocolDocument(BasicContractDocument):
  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

    self.values: List[ProbableValue] = []

  def get_found_sum(self):

    print(f'deprecated: {self.get_found_sum}, use  .values')
    best_value: ProbableValue = max(self.values,
                                    key=lambda item: item.value.value)

    most_confident_value = max(self.values, key=lambda item: item.confidence)
    best_value = select_most_confident_if_almost_equal(best_value, most_confident_value )
    return best_value

    # self.sums = sums_no_padding
    #
    # meta = {
    #   'tokens': len(sums),
    #   'index found': min_i,
    #   'd-range': (sums.min(), sums.max()),
    #   'confidence': confidence,
    #   'mean': sums.mean(),
    #   'std': np.std(sums),
    #   'min': sums[min_i],
    # }
    #
    # start, end = get_sentence_bounds_at_index(min_i, self.tokens)
    # sentence_tokens = self.tokens[start + 1:end]
    #
    # f, sentence = extract_sum_from_tokens(sentence_tokens)
    #
    # return (f, (start, end), sentence, meta)



  found_sum = property (get_found_sum)


  @deprecated
  def _find_values_OLD(self, pattern_factory):
    warnings.warn("deprecated", DeprecationWarning)
    min_i, sums_no_padding, confidence = pattern_factory.sum_pattern.find(self.embeddings)

    self.sums = sums_no_padding
    sums = sums_no_padding

    meta = {
      'tokens': len(sums),
      'index found': min_i,
      'd-range': (sums.min(), sums.max()),
      'confidence': confidence,
      'mean': sums.mean(),
      'std': np.std(sums),
      'min': sums[min_i],
    }

    start, end = self.tokens_map.sentence_at_index(min_i)


    sentence_tokens = self.tokens[start + 1:end]

    f, sentence = extract_sum_from_tokens(sentence_tokens)

    self.found_sum = (f, (start, end), sentence, meta)



  def find_sections_indexes(self, distances_per_section_pattern, min_section_size=20):
    x = distances_per_section_pattern
    pattern_to_best_index = np.array([[idx, np.argmin(ma.masked_invalid(row))] for idx, row in enumerate(x)])

    # replace best indices with sentence starts
    pattern_to_best_index[:, 1] = self.find_sentence_beginnings(pattern_to_best_index[:, 1])

    # sort by sentence start
    pattern_to_best_index = np.sort(pattern_to_best_index.view('i8,i8'), order=['f1'], axis=0).view(np.int)

    # remove "duplicated" indexes
    return self.remove_similar_indexes(pattern_to_best_index, 1, min_section_size)

  @deprecated
  def remove_similar_indexes(self, indexes, column, min_section_size=20):
    warnings.warn("deprecated", DeprecationWarning)
    indexes_zipped = [indexes[0]]

    for i in range(1, len(indexes)):
      if indexes[i][column] - indexes[i - 1][column] > min_section_size:
        pattern_to_token = indexes[i]
        indexes_zipped.append(pattern_to_token)
    return np.squeeze(indexes_zipped)

  def split_text_into_sections(self, paragraph_split_pattern: ExclusivePattern, min_section_size=10):

    distances_per_section_pattern, __ranges, __winning_patterns = \
      paragraph_split_pattern.calc_exclusive_distances(self.embeddings)

    # finding pattern positions
    x = distances_per_section_pattern
    indexes_zipped = self.find_sections_indexes(x, min_section_size)

    self.section_indexes = indexes_zipped

    return indexes_zipped, __ranges, __winning_patterns


class ProtocolAnlysingContext(ParsingContext):
  def __init__(self, embedder, renderer: ProtocolRenderer):
    ParsingContext.__init__(self, embedder)
    self.renderer: AbstractRenderer = renderer
    self.protocols_factory = None

    self.protocol: ProtocolDocument = None

  def process(self, text):
    self._reset_context()

    if self.protocols_factory is None:
      self.protocols_factory = ProtocolPatternFactory(self.embedder)
      self._logstep("Pattern factory created, patterns embedded into ELMO space")

    # ----
    pnames = [p.name[0:5] for p in self.protocols_factory.subject_pattern.patterns]
    c = Counter(pnames)
    # ----

    protocol = ProtocolDocument(text)
    print(f"ProtocolDocument text: len({len(text)})")
    protocol.parse()

    self.protocol = protocol
    protocol.embedd(self.protocols_factory)
    self._logstep("Document embedded into ELMO space")

    self.process_embedded_doc(protocol)

  def find_values_2(self, value_section: LegalDocument) -> List[ProbableValue]:

    value_attention_vector = 1.0 - self.protocols_factory.sum_pattern._find_patterns(value_section.embeddings)
    # GLOBALS__['renderer'].render_color_text(value_section.tokens, dists)

    value_atterntion_vector_name = 'value_attention_vector_tuned'
    value_section.distances_per_pattern_dict[value_atterntion_vector_name] = value_attention_vector

    values: List[ProbableValue] = extract_all_contraints_from_sr_2(value_section, value_attention_vector)

    return values

  def process_embedded_doc(self, doc: ProtocolDocument):

    section_indexes, __ranges, __winning_patterns = doc.split_text_into_sections(
      self.protocols_factory.paragraph_split_pattern)

    counter, ranges, winning_patterns = subject_weight_per_section(doc, self.protocols_factory.subject_pattern,
                                                                   self.protocols_factory.paragraph_split_pattern)

    section_names = [self.protocols_factory.paragraph_split_pattern.patterns[s[0]].name for s in doc.section_indexes]
    sidx = section_names.index('p_solution')
    if sidx < 0:
      sidx = section_names.index('p_agenda')
    if sidx < 0:
      sidx = section_names.index('p_question')

    if sidx < 0:
      sidx = 0

    #   html += winning_patterns_to_html(
    #       doc.tokens, ranges,
    #       winning_patterns,
    #       range(section_indexes[sidx][1], section_indexes[sidx+1][1]),
    #       colormaps=subject_colormaps )

    doc.values = self.find_values_2(doc)

    self._logstep("value found")

    doc.per_subject_distances = None  # Hack

    self.renderer.print_results(doc)
    self.renderer.render_subject(counter)

  #   print(protocol.normal_text)

  """### Upload file code"""

  def get_value(self):
    return self.protocol.values

  values = property(get_value)


"""### Define Patterns"""

"""### ProtocolDocument class"""

import numpy.ma as ma

"""## Processing"""

from collections import Counter


def subject_weight_per_section(doc, subj_pattern, paragraph_split_pattern):
  assert doc.section_indexes is not None

  distances_per_subj_pattern_, ranges_, winning_patterns = subj_pattern.calc_exclusive_distances(
    doc.embeddings)

  ranges_global = [
    np.nanmin(distances_per_subj_pattern_),
    np.nanmax(distances_per_subj_pattern_)]

  section_names = [[paragraph_split_pattern.patterns[s[0]].name, s[1]] for s in doc.section_indexes]
  voting: List[str] = []
  for i in range(1, len(section_names)):
    p1 = section_names[i - 1]
    p2 = section_names[i]

    distances_per_pattern_t = distances_per_subj_pattern_[:, p1[1]:p2[1]]

    dist_per_pat = []
    for row in distances_per_pattern_t:
      dist_per_pat.append(np.nanmin(row))

    patindex = np.nanargmin(dist_per_pat)
    pat_prefix = subj_pattern.patterns[patindex].name[:5]
    #         print(patindex, pat_prefix)

    voting.append(pat_prefix)

    ## HACK more attention to particular sections
    if p1[0] == 'p_agenda' or p1[0] == 'p_solution' or p1[0] == 'p_question':
      voting.append(pat_prefix)

  return Counter(voting), ranges_global, winning_patterns
