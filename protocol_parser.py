import warnings
from collections.__init__ import Counter
from typing import List

from numpy import ma as ma

from contract_agents import find_org_names, ORG_LEVELS_re
from contract_parser import extract_all_contraints_from_sr_2
from hyperparams import HyperParameters
from legal_docs import BasicContractDocument, deprecated
from legal_docs import LegalDocument
from ml_tools import ProbableValue, FixedVector, SemanticTag
from ml_tools import select_most_confident_if_almost_equal
from parsing import ParsingContext
from patterns import AbstractPatternFactory, FuzzyPattern, CoumpoundFuzzyPattern, ExclusivePattern, np
from structures import ORG_LEVELS_names
from text_tools import is_long_enough, span_len


class ProtocolDocument3(LegalDocument):
  '''

  '''

  # TODO: rename it

  def __init__(self, doc: LegalDocument):
    super().__init__('')
    if doc is not None:
      self.__dict__ = doc.__dict__

    # self.subjects = None
    # self.contract_values: List[ContractValue] = []

    self.agents_tags: [SemanticTag] = []
    self.org_level: [SemanticTag] = []

  def get_tags(self) -> [SemanticTag]:
    tags = []
    tags += self.agents_tags
    tags += self.org_level

    return tags


class ProtocolPatternFactory(AbstractPatternFactory):
  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)

    self._build_paragraph_split_pattern()
    self._build_subject_pattern()
    self._build_sum_margin_extraction_patterns()
    self.embedd()

  def _build_sum_margin_extraction_patterns(self):
    suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
    prefix = ''

    sum_comp_pat = CoumpoundFuzzyPattern()

    sum_comp_pat.add_pattern(self.create_pattern('sum_max1', (prefix + 'стоимость', 'не более 0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max2', (prefix + 'цена', 'не больше 0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max3', (prefix + 'стоимость <', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max4', (prefix + 'цена менее', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max5', (prefix + 'стоимость не может превышать', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max6', (prefix + 'общая сумма может составить', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max7', (prefix + 'лимит соглашения', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max8', (prefix + 'верхний лимит стоимости', '0', suffix)))
    sum_comp_pat.add_pattern(self.create_pattern('sum_max9', (prefix + 'максимальная сумма', '0', suffix)))

    sum_comp_pat.add_pattern(
      self.create_pattern('sum_max_neg1', ('ежемесячно не позднее', '0', 'числа каждого месяца')), -0.8)
    sum_comp_pat.add_pattern(self.create_pattern('sum_max_neg2', ('приняли участие в голосовании', '0', 'человек')),
                             -0.8)
    sum_comp_pat.add_pattern(
      self.create_pattern('sum_max_neg3', ('срок действия не должен превышать', '0', 'месяцев с даты выдачи')), -0.8)
    sum_comp_pat.add_pattern(
      self.create_pattern('sum_max_neg4', ('позднее чем за', '0', 'календарных дней до даты его окончания ')), -0.8)
    sum_comp_pat.add_pattern(self.create_pattern('sum_max_neg5', ('общая площадь', '0', 'кв . м.')), -0.8)

    self.sum_pattern = sum_comp_pat

  def _build_subject_pattern(self):
    ep = ExclusivePattern()

    PRFX = "Повестка дня заседания: \n"

    if True:
      ep.add_pattern(self.create_pattern('t_deal_1', (PRFX, 'Об одобрении сделки', 'связанной с продажей')))
      ep.add_pattern(self.create_pattern('t_deal_2', (
        PRFX + 'О согласии на', 'совершение сделки', 'связанной с заключением договора')))
      ep.add_pattern(self.create_pattern('t_deal_3', (
        PRFX + 'об одобрении', 'крупной сделки', 'связанной с продажей недвижимого имущества')))

      for p in ep.patterns:
        p.soft_sliding_window_borders = True

    if True:
      ep.add_pattern(self.create_pattern('t_org1', (PRFX, 'О создании филиала', 'Общества')))
      ep.add_pattern(self.create_pattern('t_org2', (PRFX, 'Об утверждении Положения', 'о филиале Общества')))
      ep.add_pattern(self.create_pattern('t_org3', (PRFX, 'О назначении руководителя', 'филиала')))
      ep.add_pattern(self.create_pattern('t_org4', (PRFX, 'О прекращении полномочий руководителя', 'филиала')))
      ep.add_pattern(self.create_pattern('t_org5', (PRFX, 'О внесении изменений', '')))

    if True:
      ep.add_pattern(
        self.create_pattern('t_charity_1', (PRFX + 'О предоставлении', 'безвозмездной', 'финансовой помощи')))
      ep.add_pattern(
        self.create_pattern('t_charity_2', (PRFX + 'О согласии на совершение сделки', 'пожертвования', '')))
      ep.add_pattern(self.create_pattern('t_charity_3', (PRFX + 'Об одобрении сделки', 'пожертвования', '')))

      t_char_mix = CoumpoundFuzzyPattern()
      t_char_mix.name = "t_charity_mixed"

      t_char_mix.add_pattern(
        self.create_pattern('tm_charity_1', (PRFX + 'О предоставлении', 'безвозмездной финансовой помощи', '')))
      t_char_mix.add_pattern(
        self.create_pattern('tm_charity_2', (PRFX + 'О согласии на совершение', 'сделки пожертвования', '')))
      t_char_mix.add_pattern(self.create_pattern('tm_charity_3', (PRFX + 'Об одобрении сделки', 'пожертвования', '')))

      ep.add_pattern(t_char_mix)

    self.subject_pattern = ep

  def _build_paragraph_split_pattern(self):
    PRFX = ". \n"
    PRFX1 = """
    кворум для проведения заседания и принятия решений имеется.

    """

    sect_pt = ExclusivePattern()

    if True:
      # IDX 0
      p_agenda = CoumpoundFuzzyPattern()
      p_agenda.name = "p_agenda"

      p_agenda.add_pattern(self.create_pattern('p_agenda_1', (PRFX1, 'Повестка', 'дня заседания:')))
      p_agenda.add_pattern(self.create_pattern('p_agenda_2', (PRFX1, 'Повестка', 'дня:')))

      sect_pt.add_pattern(p_agenda)

    if True:
      # IDX 1
      p_solution = CoumpoundFuzzyPattern()
      p_solution.name = "p_solution"
      p_solution.add_pattern(
        self.create_pattern('p_solution1', (PRFX, 'решение', 'принятое по вопросу повестки дня: одобрить')))
      p_solution.add_pattern(self.create_pattern('p_solution2', (PRFX + 'формулировка', 'решения', ':одобрить')))

      sect_pt.add_pattern(p_solution)

    sect_pt.add_pattern(self.create_pattern('p_head', (PRFX, 'Протокол \n ', 'заседания')))
    sect_pt.add_pattern(self.create_pattern('p_question', (
      PRFX + 'Первый', 'вопрос', 'повестки дня заседания поставленный на голосование')))
    sect_pt.add_pattern(self.create_pattern('p_votes', (PRFX, 'Результаты голосования', 'за против воздержаолось')))
    sect_pt.add_pattern(self.create_pattern('p_addons', (PRFX, 'Приложения', '')))

    self.paragraph_split_pattern = sect_pt


class ProtocolDocument(BasicContractDocument):
  # TODO: use anothwer parent

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

    self.values: List[ProbableValue] = []
    self.section_indices: [int] = None

  def subject_weight_per_section(self, subj_pattern, paragraph_split_pattern):
    assert self.section_indices is not None

    distances_per_subj_pattern_, ranges_, winning_patterns = subj_pattern.calc_exclusive_distances(self.embeddings)

    ranges_global = [
      np.nanmin(distances_per_subj_pattern_),
      np.nanmax(distances_per_subj_pattern_)]

    section_names = [[paragraph_split_pattern.patterns[s[0]].name, s[1]] for s in self.section_indices]
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

      # TODO: HACK more attention to particular sections
      if p1[0] == 'p_agenda' or p1[0] == 'p_solution' or p1[0] == 'p_question':
        voting.append(pat_prefix)

    return Counter(voting), ranges_global, winning_patterns

  def get_found_sum(self) -> ProbableValue:

    print(f'deprecated: {self.get_found_sum}, use  .values')
    best_value: ProbableValue = max(self.values, key=lambda item: item.value.value)

    most_confident_value = max(self.values, key=lambda item: item.confidence)
    best_value = select_most_confident_if_almost_equal(best_value, most_confident_value)

    return best_value

  found_sum: ProbableValue = property(get_found_sum)

  def find_sections_indices(self, distances_per_section_pattern: FixedVector, min_section_size=20) -> [int]:
    x: FixedVector = distances_per_section_pattern
    pattern_to_best_index = np.array([[idx, np.argmin(ma.masked_invalid(row))] for idx, row in enumerate(x)])

    # replace best indices with sentence starts
    pattern_to_best_index[:, 1] = self.find_sentence_beginnings(pattern_to_best_index[:, 1])

    # sort by sentence start
    pattern_to_best_index = np.sort(pattern_to_best_index.view('i8,i8'), order=['f1'], axis=0).view(np.int)

    # remove "duplicated" indexes
    return self.remove_similar_indexes(pattern_to_best_index, 1, min_section_size)

  @deprecated
  def remove_similar_indexes(self, indices: [int], column: int, min_section_size: int = 20) -> [int]:
    warnings.warn("deprecated", DeprecationWarning)
    indices_zipped = [indices[0]]

    for i in range(1, len(indices)):
      if indices[i][column] - indices[i - 1][column] > min_section_size:
        pattern_to_token = indices[i]
        indices_zipped.append(pattern_to_token)

    return np.squeeze(indices_zipped)

  def split_text_into_sections(self, paragraph_split_pattern: ExclusivePattern, min_section_size=10):

    distances_per_section_pattern, _, __ = paragraph_split_pattern.calc_exclusive_distances(self.embeddings)

    # finding pattern positions

    self.section_indices = self.find_sections_indices(distances_per_section_pattern, min_section_size)

    return self.section_indices


class ProtocolAnlysingContext(ParsingContext):

  def __init__(self, embedder):
    ParsingContext.__init__(self, embedder)

    self.protocols_factory: ProtocolPatternFactory = None

    self.protocol: ProtocolDocument = None

  def process(self, text) -> ProtocolDocument:
    self._reset_context()

    if self.protocols_factory is None:
      self.protocols_factory = ProtocolPatternFactory(self.embedder)
      self._logstep("Pattern factory created, patterns embedded into ELMO space")

    # # ----
    # pnames = [p.name[0:5] for p in self.protocols_factory.subject_pattern.patterns]
    # c = Counter(pnames)
    # # ----

    self.protocol = ProtocolDocument(text)
    self.protocol.parse()
    self.protocol.embedd_tokens(self.protocols_factory.embedder)

    self.process_embedded_doc(self.protocol)
    return self.protocol

  def process_embedded_doc(self, doc: ProtocolDocument):

    section_indices = doc.split_text_into_sections(
      self.protocols_factory.paragraph_split_pattern)

    counter, ranges, winning_patterns = doc.subject_weight_per_section(self.protocols_factory.subject_pattern,
                                                                       self.protocols_factory.paragraph_split_pattern)

    section_names = [self.protocols_factory.paragraph_split_pattern.patterns[s[0]].name for s in doc.section_indices]
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
    #       range(section_indices[sidx][1], section_indices[sidx+1][1]),
    #       colormaps=subject_colormaps )

    doc.values = self.find_values_2(doc)
    doc.per_subject_distances = counter  # Hack

    # self.renderer.print_results(doc)
    # self.renderer.render_subject(counter)

  def find_values_2(self, value_section: LegalDocument) -> List[ProbableValue]:

    value_attention_vector = 1.0 - self.protocols_factory.sum_pattern._find_patterns(value_section.embeddings)
    value_section.distances_per_pattern_dict['value_attention_vector_tuned'] = value_attention_vector
    values: List[ProbableValue] = extract_all_contraints_from_sr_2(value_section, value_attention_vector)
    return values

  def get_value(self):
    return self.protocol.values

  values = property(get_value)


def find_protocol_org(protocol: ProtocolDocument3) -> List[SemanticTag]:
  ret = []
  x: List[SemanticTag] = find_org_names(protocol[0:HyperParameters.protocol_caption_max_size_words])
  nm = SemanticTag.find_by_kind(x, 'org.1.name')
  if nm is not None:
    ret.append(nm)

  tp = SemanticTag.find_by_kind(x, 'org.1.type')
  if tp is not None:
    ret.append(tp)

  protocol.agents_tags = ret
  return ret


import re

from pyjarowinkler import distance


def closest_name(pattern: str, knowns: [str]) -> (str, int):
  #
  min_distance = 0
  found = None
  for b in knowns:
    d = distance.get_jaro_distance(pattern, b, winkler=True, scaling=0.1)
    if d > min_distance:
      found = b
      min_distance = d

  return found, min_distance


def find_org_structural_level(doc: LegalDocument):
  compiled_re = re.compile(ORG_LEVELS_re, re.MULTILINE | re.IGNORECASE | re.UNICODE)

  entity_type = 'org_structural_level'
  for m in re.finditer(compiled_re, doc.text):

    char_span = m.span(entity_type)
    span = doc.tokens_map.token_indices_by_char_range_2(char_span)
    val = doc.tokens_map.text_range(span)

    val, conf = closest_name(val, ORG_LEVELS_names)

    confidence = conf * (1.0 - (span[0] / len(doc)))  # relative distance from the beginning of the document
    if span_len(char_span) > 1 and is_long_enough(val):
      tag = SemanticTag(entity_type, val, span)
      tag.confidence = confidence
      if confidence > 0.2:
        yield tag
