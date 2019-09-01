import warnings
from typing import List

from contract_agents import agent_infos_to_tags, find_org_names_spans
from contract_patterns import ContractPatternFactory
from legal_docs import LegalDocument, HeadlineMeta, extract_sum_sign_currency, ValueSemanticTags
from ml_tools import ProbableValue, relu, np, filter_values_by_key_prefix, \
  rectifyed_sum, SemanticTag, FixedVector, estimate_confidence_by_mean_top
from parsing import ParsingConfig, ParsingContext
from patterns import AV_SOFT, AV_PREFIX
from renderer import AbstractRenderer
from sections_finder import SectionsFinder, FocusingSectionsFinder
from structures import ContractSubject
from transaction_values import ValueConstraint

default_contract_parsing_config: ParsingConfig = ParsingConfig()
contract_subjects = [ContractSubject.RealEstate, ContractSubject.Charity, ContractSubject.Deal]

from transaction_values import complete_re as transaction_values_re


class ContractDocument3(LegalDocument):
  '''

  '''

  # TODO: rename it

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

    self.subjects = None
    self.contract_values: List[ValueSemanticTags] = []

    self.agents_tags = None

  def get_tags(self) -> [SemanticTag]:
    tags = []
    tags += self.agents_tags
    tags.append(self.subjects)
    for contract_value in self.contract_values:
      tags += contract_value.as_asrray()

    return tags

  def parse(self, txt=None):
    super().parse()
    agent_infos = find_org_names_spans(self.tokens_map_norm)
    self.agents_tags = agent_infos_to_tags(agent_infos)


ContractDocument = ContractDocument3  # Alias!


def filter_nans(vcs: List[ProbableValue]) -> List[ProbableValue]:
  warnings.warn("use numpy built-in functions", DeprecationWarning)
  r: List[ProbableValue] = []
  for vc in vcs:
    if vc.value is not None and not np.isnan(vc.value.value):
      r.append(vc)
  return r


class ContractAnlysingContext(ParsingContext):

  def __init__(self, embedder, renderer: AbstractRenderer, pattern_factory=None):
    ParsingContext.__init__(self, embedder)
    self.renderer: AbstractRenderer = renderer
    if not pattern_factory:
      self.pattern_factory = ContractPatternFactory(embedder)
    else:
      self.pattern_factory = pattern_factory

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

  def analyze_contract(self, contract_text):
    """
    MAIN METHOD
    
    :param contract_text: 
    :return: 
    """

    self._reset_context()

    # create DOC
    self.contract = ContractDocument(contract_text)
    self.contract.parse()

    self._logstep("parsing document 👞 and detecting document high-level structure")

    self.contract.embedd_tokens(self.pattern_factory.embedder)
    self.sections_finder.find_sections(self.contract, self.pattern_factory, self.pattern_factory.headlines,
                                       headline_patterns_prefix='headline.')

    # -------------------------------values
    self.contract.contract_values = self.find_contract_value_NEW(self.contract)
    # -------------------------------subject
    self.contract.subjects = self.find_contract_subject_region(self.contract)
    # TODO: convert to semantic tags
    # --------------------------------------

    self._logstep("fetching transaction values")

    # self.renderer.render_values(self.contract.contract_values)
    self.log_warnings()

    return self.contract, self.contract.contract_values

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

  def __sub_attention_names(self, subj: ContractSubject):
    a = f'x_{subj}'
    b = AV_PREFIX + f'x_{subj}'
    c = AV_SOFT + a
    return a, b, c

  def make_subject_attention_vector_3(self, section, subject_kind: ContractSubject, addon=None) -> FixedVector:
    from ml_tools import max_exclusive_pattern
    pattern_prefix, attention_vector_name, attention_vector_name_soft = self.__sub_attention_names(subject_kind)

    vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, pattern_prefix)
    x = max_exclusive_pattern(vectors)

    section.distances_per_pattern_dict[attention_vector_name_soft] = x
    section.distances_per_pattern_dict[attention_vector_name] = x

    #   x = x-np.mean(x)
    x = relu(x, 0.6)

    return x

  def map_subject_to_type(self, section: LegalDocument, denominator: float = 1.0) -> List[ProbableValue]:
    """
    :param section:
    :param denominator: confidence multiplyer
    :return:
    """
    section.calculate_distances_per_pattern(self.pattern_factory, merge=True, pattern_prefix='x_ContractSubject')

    all_subjects_vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, 'x_ContractSubject')
    all_subjects_mean: FixedVector = rectifyed_sum(all_subjects_vectors)

    subjects_mapping: List[ProbableValue] = []
    for subject_kind in contract_subjects:  # like ContractSubject.RealEstate ..
      x: FixedVector = self.make_subject_attention_vector_3(section, subject_kind, all_subjects_mean)

      confidence = estimate_confidence_by_mean_top(x)
      confidence *= denominator
      pv = ProbableValue(subject_kind, confidence)
      subjects_mapping.append(pv)

    return subjects_mapping

  def find_contract_subject(self, doc) -> List[ProbableValue]:
    warnings.warn("use find_contract_subject_region", DeprecationWarning)
    if 'subj' in doc.sections:
      subj_section = doc.sections['subj']
      subject_subdoc = subj_section.body
      denominator = 1
    else:
      self.warning('раздел о предмете договора не найден, ищем предмет договора в первых 1500 словах')
      subject_subdoc = doc.subdoc_slice(slice(0, 1500))
      denominator = 0.7

    return self.map_subject_to_type(subject_subdoc, denominator=denominator)

  def find_contract_subject_region(self, doc) -> SemanticTag:

    if 'subj' in doc.sections:
      subj_section = doc.sections['subj']
      subject_subdoc = subj_section.body
      denominator = 1
    else:
      self.warning('раздел о предмете договора не найден, ищем предмет договора в первых 1500 словах')
      subject_subdoc = doc.subdoc_slice(slice(0, 1500))
      denominator = 0.7

    return self.find_contract_subject_regions(subject_subdoc, denominator=denominator)

  def _find_most_relevant_paragraph(self, section: LegalDocument, attention_vector: FixedVector):

    paragraph_attention_vector = np.zeros_like(attention_vector)
    top_index = 0
    for i in np.nonzero(attention_vector)[0]:
      par = section.tokens_map.sentence_at_index(i)
      paragraph_len = par[1] - par[0]
      if paragraph_len:
        paragraph_attention_vector[par[0]: par[1]] += attention_vector[i] + attention_vector[i] / paragraph_len
        if paragraph_attention_vector[par[0]] > paragraph_attention_vector[top_index]:
          top_index = par[0]

    par = section.tokens_map.sentence_at_index(top_index)
    return par, paragraph_attention_vector[top_index]

  def find_contract_subject_regions(self, section: LegalDocument, denominator: float = 1.0) -> SemanticTag:

    section.calculate_distances_per_pattern(self.pattern_factory, merge=True, pattern_prefix='x_ContractSubject')

    all_subjects_vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, 'x_ContractSubject')
    all_subjects_mean: FixedVector = rectifyed_sum(all_subjects_vectors)

    max_confidence = 0
    max_subject_kind = None
    max_paragraph_span = None
    for subject_kind in contract_subjects:  # like ContractSubject.RealEstate ..
      subject_attention_vector: FixedVector = self.make_subject_attention_vector_3(section, subject_kind,
                                                                                   all_subjects_mean)

      paragraph_span, confidence = self._find_most_relevant_paragraph(section, subject_attention_vector)
       
      if confidence > max_confidence:
        max_confidence = confidence
        max_subject_kind = subject_kind
        max_paragraph_span = paragraph_span

    result = SemanticTag('subject', max_subject_kind, max_paragraph_span)
    result.confidence = max_confidence * denominator
    result.offset(section.start)

    return result

  def find_contract_value_NEW(self, contract: ContractDocument) -> List[ValueSemanticTags]:
    # preconditions
    assert contract.sections is not None

    search_sections_order = [
      ['price.', 1], ['subj', 0.75], ['pricecond', 0.75], [None, 0.5]  # todo: check 'price', not 'price.'
    ]

    for section, confidence_k in search_sections_order:
      if section in contract.sections or section is None:
        if section in contract.sections:
          value_section = contract.sections[section].body
          _section_name = contract.sections[section].subdoc.text.strip()
        else:
          value_section = contract
          _section_name = 'entire contract'

        if self.verbosity_level > 1:
          self._logstep(f'searching for transaction values in section ["{section}"] "{_section_name}"')

        result: List[ValueSemanticTags] = find_value_sign_currency(value_section, self.pattern_factory)
        if not result:
          self.warning(f'В разделе "{_section_name}" стоимость сделки не найдена!')
        else:
          for _r in result:
            # decrease confidence:
            _r.mult_confidence(confidence_k)
            _r.offset_spans(value_section.start)

          return result

      else:
        self.warning('Раздел про стоимость сделки не найден!')

  def find_contract_value(self, contract: ContractDocument) -> List[ProbableValue]:
    # preconditions
    warnings.warn("use find_contract_value_NEW", DeprecationWarning)
    assert contract.sections is not None

    price_factory = self.pattern_factory
    sections = contract.sections
    result: List[ValueConstraint] = []

    # TODO iterate over section names
    if 'price.' in sections:  # todo: check 'price', not 'price.'

      value_section_info: HeadlineMeta = sections['price.']
      value_section = value_section_info.body
      section_name = value_section_info.subdoc.text
      result = filter_nans(_try_to_fetch_value_from_section_2(value_section, price_factory))
      if len(result) == 0:
        self.warning(f'В разделе "{section_name}" стоимость сделки не найдена!')

      if self.verbosity_level > 1:
        self._logstep(f'searching for transaction values in section  "{section_name}"')
        # ------------
        # value_section.reset_embeddings()  # careful with this. Hope, we will not be required to search here
    else:
      self.warning('Раздел про стоимость сделки не найден!')

    if len(result) == 0:
      if 'subj' in sections:

        # fallback
        value_section_info = sections['subj']
        value_section = value_section_info.body
        section_name = value_section_info.subdoc.text
        print(f'- Ищем стоимость в разделе {section_name}')
        result: List[ProbableValue] = filter_nans(_try_to_fetch_value_from_section_2(value_section, price_factory))

        # decrease confidence:
        for _r in result:
          _r.confidence *= 0.7

        if self.verbosity_level > 0:
          print('alt price section DOC', '-' * 20)
          self._logstep(f'searching for transaction values in section  "{section_name}"')

        if len(result) == 0:
          self.warning(f'В разделе "{section_name}" стоимость сделки не найдена!')

    if len(result) == 0:
      if 'pricecond' in sections:

        # fallback
        value_section_info = sections['pricecond']
        value_section = value_section_info.body
        section_name = value_section_info.subdoc.text
        print(f'-WARNING: Ищем стоимость в разделе {section_name}!')
        result: List[ProbableValue] = filter_nans(_try_to_fetch_value_from_section_2(value_section, price_factory))
        if self.verbosity_level > 0:
          print('alt price section DOC', '-' * 20)
          self._logstep(f'searching for transaction values in section  "{section_name}"')
        # ------------
        for _r in result:
          _r.confidence *= 0.7
        # value_section.reset_embeddings()  # careful with this. Hope, we will not be required to search here
        if len(result) == 0:
          self.warning(f'В разделе "{section_name}" стоимость сделки не найдена!')

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


def find_value_sign_currency(value_section_subdoc: LegalDocument, factory: ContractPatternFactory) -> List[
  ValueSemanticTags]:
  ''' merge dictionaries of attention vectors '''

  value_section_subdoc.calculate_distances_per_pattern(factory)
  vectors = factory.make_contract_value_attention_vectors(value_section_subdoc)
  value_section_subdoc.distances_per_pattern_dict = {**value_section_subdoc.distances_per_pattern_dict, **vectors}

  v = value_section_subdoc.distances_per_pattern_dict['value_attention_vector_tuned']

  # TODO: apply confidence to semantic tags

  spans = [m for m in value_section_subdoc.tokens_map.finditer(transaction_values_re)]
  values_list = [extract_sum_sign_currency(value_section_subdoc, span) for span in spans]

  return values_list


def _try_to_fetch_value_from_section_2(value_section_subdoc: LegalDocument, factory: ContractPatternFactory) -> List[
  ProbableValue]:
  warnings.warn("use find_value_sign_currency ", DeprecationWarning)
  ''' merge dictionaries of attention vectors '''

  value_section_subdoc.calculate_distances_per_pattern(factory)
  vectors = factory.make_contract_value_attention_vectors(value_section_subdoc)
  value_section_subdoc.distances_per_pattern_dict = {**value_section_subdoc.distances_per_pattern_dict, **vectors}

  v = value_section_subdoc.distances_per_pattern_dict['value_attention_vector_tuned']

  values: List[ProbableValue] = find_all_value_sign_currency(value_section_subdoc)

  # TODO: apply confidence to semantic tags

  return values


def find_all_value_sign_currency(doc: LegalDocument) -> List:
  warnings.warn("use find_value_sign_currency ", DeprecationWarning)
  """
  TODO: rename
  :param doc: LegalDocument
  :param attention_vector: List[float]
  :return: List[ProbableValue]
  """
  spans = [m for m in doc.tokens_map.finditer(transaction_values_re)]
  return [extract_sum_sign_currency(doc, span) for span in spans]


extract_all_contraints_from_sr_2 = find_all_value_sign_currency  # alias for compatibility, todo: remove it
