from typing import List

from contract_patterns import ContractPatternFactory
from legal_docs import LegalDocument, HeadlineMeta, extract_all_contraints_from_sentence, deprecated, \
  extract_sum_and_sign_3, _expand_slice
from ml_tools import ProbableValue, max_exclusive_pattern_by_prefix, relu, np, filter_values_by_key_prefix, \
  rectifyed_sum, TokensWithAttention
from parsing import ParsingConfig, ParsingContext
from patterns import AV_SOFT, AV_PREFIX
from renderer import AbstractRenderer
from sections_finder import SectionsFinder, FocusingSectionsFinder
from structures import ContractSubject
from transaction_values import ValueConstraint

default_contract_parsing_config: ParsingConfig = ParsingConfig()
contract_subjects = [ContractSubject.RealEstate, ContractSubject.Charity, ContractSubject.Deal]


class ContractDocument3(LegalDocument):

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)
    self.subjects: List[ProbableValue] = [ProbableValue(ContractSubject.Other, 0.0)]
    self.contract_values: [ProbableValue] = []


class ContractAnlysingContext(ParsingContext):

  def __init__(self, embedder, renderer: AbstractRenderer):
    ParsingContext.__init__(self, embedder)
    self.renderer: AbstractRenderer = renderer
    self.pattern_factory = ContractPatternFactory(embedder)

    self.contract = None
    # self.contract_values = None

    self.config = default_contract_parsing_config

    # self.sections_finder: SectionsFinder = DefaultSectionsFinder(self)
    self.sections_finder: SectionsFinder = FocusingSectionsFinder(self)

  def _reset_context(self):
    super(ContractAnlysingContext, self)._reset_context()

    if self.contract is not None:
      del self.contract
      self.contract = None

    # if self.contract_values is not None:
    #   del self.contract_values
    #   self.contract_values = None

  def analyze_contract(self, contract_text):
    self._reset_context()
    """
    MAIN METHOD
    
    :param contract_text: 
    :return: 
    """
    doc = ContractDocument3(contract_text)
    doc.parse()
    self.contract = doc

    self._logstep("parsing document 👞 and detecting document high-level structure")

    self.contract.embedd(self.pattern_factory)
    self.sections_finder.find_sections(doc, self.pattern_factory, self.pattern_factory.headlines,
                                       headline_patterns_prefix='headline.')

    # -------------------------------values
    values = self.fetch_value_from_contract(doc)
    # -------------------------------subj
    doc.subjects = self.recognize_subject(doc)
    self._logstep("fetching transaction values")
    # -------------------------------values

    self.renderer.render_values(values)
    # self.contract_values = values
    doc.contract_values = values

    self.log_warnings()

    return doc, values

  def get_contract_values(self):
    return self.contract.contract_values

  contract_values = property(get_contract_values)

  def select_most_confident_if_almost_equal(self, a: ProbableValue, alternative: ProbableValue, m_convert,
                                            equality_range=0.0):

    if abs(m_convert(a.value).value - m_convert(alternative.value).value) < equality_range:
      if a.confidence > alternative.confidence:
        return a
      else:
        return alternative
    return a

  def find_contract_best_value(self, m_convert):
    best_value: ProbableValue = max(self.contract_values,
                                    key=lambda item: m_convert(item.value).value)

    most_confident_value = max(self.contract_values, key=lambda item: item.confidence)
    best_value = self.select_most_confident_if_almost_equal(best_value, most_confident_value, m_convert,
                                                            equality_range=20)

    return best_value

  @deprecated
  def make_subj_attention_vectors(self, subdoc, subj_types_prefixes):
    r = {}
    for subj_types_prefix in subj_types_prefixes:
      attention_vector = max_exclusive_pattern_by_prefix(subdoc.distances_per_pattern_dict, subj_types_prefix)
      attention_vector_l = relu(attention_vector, 0.6)

      r[subj_types_prefix + 'attention_vector'] = attention_vector
      r[subj_types_prefix + 'attention_vector_l'] = attention_vector_l

    return r

  def __sub_attention_names(self, subj: ContractSubject):
    a = f'x_{subj}'
    b = AV_PREFIX + f'x_{subj}'
    c = AV_SOFT + a
    return a, b, c

  def make_subject_attention_vector_3(self, section, subject_kind: ContractSubject, addon=None) -> List[float]:
    from ml_tools import max_exclusive_pattern
    pattern_prefix, attention_vector_name, attention_vector_name_soft = self.__sub_attention_names(subject_kind)

    vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, pattern_prefix)
    x = max_exclusive_pattern(vectors)

    section.distances_per_pattern_dict[attention_vector_name_soft] = x
    section.distances_per_pattern_dict[attention_vector_name] = x

    #   x = x-np.mean(x)
    x = relu(x, 0.6)

    return x

  def estimate_confidence_2(self, x):
    return np.mean(sorted(x)[-10:])

  def map_subject_to_type(self, section: LegalDocument, denominator: float = 1) -> List[ProbableValue]:
    """
    :param section:
    :param denominator: confidence multiplyer
    :return:
    """
    section.calculate_distances_per_pattern(self.pattern_factory, merge=True, pattern_prefix='x_ContractSubject')
    all_subjects_vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, 'x_ContractSubject')
    all_mean = rectifyed_sum(all_subjects_vectors)

    subjects_mapping = []
    for subject_kind in contract_subjects:
      x = self.make_subject_attention_vector_3(section, subject_kind, all_mean)
      # confidence, sum_, nonzeros_count, _max = estimate_confidence(x)
      confidence = self.estimate_confidence_2(x)
      confidence *= denominator
      pv = ProbableValue(subject_kind, confidence)
      subjects_mapping.append(pv)

    return subjects_mapping

  def recognize_subject(self, doc) -> List[ProbableValue]:

    if 'subj' in doc.sections:
      subj_section = doc.sections['subj']

      subj_ = subj_section.body

      return self.map_subject_to_type(subj_)

    else:
      self.warning('раздел о предмете договора не найден')
      # try:
      self.warning('ищем предмет договора в первых 1500 словах')

      return self.map_subject_to_type(doc.subdoc_slice(slice(0, 1500)), denominator=0.7)
      # except:
      #   self.warning('поиск предмета договора полностью провален!')
      #   return [ProbableValue(ContractSubject.Other, 0.0)]

  def fetch_value_from_contract(self, contract: LegalDocument) -> List[ProbableValue]:

    def filter_nans(vcs: List[ProbableValue]) -> List[ProbableValue]:
      r: List[ProbableValue] = []
      for vc in vcs:
        if vc.value is not None and not np.isnan(vc.value.value):
          r.append(vc)
      return r

    renderer = self.renderer

    price_factory = self.pattern_factory

    sections = contract.sections

    result: List[ValueConstraint] = []

    if 'price.' in sections:
      value_section_info: HeadlineMeta = sections['price.']
      value_section = value_section_info.body
      section_name = value_section_info.subdoc.untokenize_cc()
      result = filter_nans(_try_to_fetch_value_from_section_2(value_section, price_factory))
      if len(result) == 0:
        self.warning(f'В разделе "{ section_name }" стоимость сделки не найдена!')

      if self.verbosity_level > 1:
        renderer.render_value_section_details(value_section_info)
        self._logstep(f'searching for transaction values in section  "{ section_name }"')
        # ------------
        # value_section.reset_embeddings()  # careful with this. Hope, we will not be required to search here
    else:
      self.warning('Раздел про стоимость сделки не найден!')

    if len(result) == 0:
      if 'subj' in sections:

        # fallback
        value_section_info = sections['subj']
        value_section = value_section_info.body
        section_name = value_section_info.subdoc.untokenize_cc()
        print(f'- Ищем стоимость в разделе { section_name }')
        result: List[ProbableValue] = filter_nans(_try_to_fetch_value_from_section_2(value_section, price_factory))

        # decrease confidence:
        for _r in result:
          _r.confidence *= 0.7

        if self.verbosity_level > 0:
          print('alt price section DOC', '-' * 20)
          renderer.render_value_section_details(value_section_info)
          self._logstep(f'searching for transaction values in section  "{ section_name }"')

        if len(result) == 0:
          self.warning(f'В разделе "{ section_name }" стоимость сделки не найдена!')

    if len(result) == 0:
      if 'pricecond' in sections:

        # fallback
        value_section_info = sections['pricecond']
        value_section = value_section_info.body
        section_name = value_section_info.subdoc.untokenize_cc()
        print(f'-WARNING: Ищем стоимость в разделе { section_name }!')
        result: List[ProbableValue] = filter_nans(_try_to_fetch_value_from_section_2(value_section, price_factory))
        if self.verbosity_level > 0:
          print('alt price section DOC', '-' * 20)
          renderer.render_value_section_details(value_section_info)
          self._logstep(f'searching for transaction values in section  "{ section_name }"')
        # ------------
        for _r in result:
          _r.confidence *= 0.7
        # value_section.reset_embeddings()  # careful with this. Hope, we will not be required to search here
        if len(result) == 0:
          self.warning(f'В разделе "{ section_name }" стоимость сделки не найдена!')

    if len(result) == 0:
      self.warning('Ищем стоимость во всем документе!')

      #     trying to find sum in the entire doc
      value_section = contract
      result: List[ProbableValue] = filter_nans(_try_to_fetch_value_from_section_2(value_section, price_factory))
      if self.verbosity_level > 1:
        print('ENTIRE DOC', '--' * 70)
        self._logstep(f'searching for transaction values in the entire document')
      # ------------
      # decrease confidence:
      for _r in result:
        _r.confidence *= 0.6
      # value_section.reset_embeddings()  # careful with this. Hope, we will not be required to search here

    return result


def _try_to_fetch_value_from_section_2(value_section: LegalDocument, factory: ContractPatternFactory) -> List[
  ProbableValue]:
  value_section.calculate_distances_per_pattern(factory)

  vectors = factory.make_contract_value_attention_vectors(value_section)

  value_section.distances_per_pattern_dict = {**value_section.distances_per_pattern_dict, **vectors}

  values: List[ProbableValue] = extract_all_contraints_from_sr_2(value_section,
                                                                 value_section.distances_per_pattern_dict[
                                                                   'value_attention_vector_tuned'])

  return values


from transaction_values import complete_re
import re


def extract_all_contraints_from_sr_2(search_result: LegalDocument, attention_vector: List[float]) -> List[
  ProbableValue]:
  def __tokens_before_index(string, index):
    return len(string[:index].split(' '))

  sentence = ' '.join(search_result.tokens)
  # print("SENT:", sentence)
  all_values = [slice(m.start(0), m.end(0)) for m in re.finditer(complete_re, sentence)]
  constraints: List[ProbableValue] = []

  for a in all_values:
    # print(tokens_before_index(sentence, a.start), 'from', sentence[a])
    token_index_s = __tokens_before_index(sentence, a.start) - 1
    token_index_e = __tokens_before_index(sentence, a.stop)

    region = slice(token_index_s, token_index_e)

    print('REG:', ' '.join(search_result.tokens[region]))

    vc = extract_sum_and_sign_3(search_result, region)
    _e = _expand_slice(region, 10)
    vc.context = TokensWithAttention(search_result.tokens[_e], attention_vector[_e])
    confidence = attention_vector[region.start]
    pv = ProbableValue(vc, confidence)

    constraints.append(pv)

  return constraints


def _try_to_fetch_value_from_section___(value_section: LegalDocument, factory: ContractPatternFactory) -> List[
  ProbableValue]:
  # value_section.embedd(factory)
  value_section.calculate_distances_per_pattern(factory)

  # context._logstep(f'embedding for transaction values in section  "{ section_name }"')

  vectors = factory.make_contract_value_attention_vectors(value_section)

  value_section.distances_per_pattern_dict = {**value_section.distances_per_pattern_dict, **vectors}

  values: List[ProbableValue] = extract_all_contraints_from_sentence(value_section,
                                                                     value_section.distances_per_pattern_dict[
                                                                       'value_attention_vector_tuned'])

  return values
