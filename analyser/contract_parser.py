from analyser.contract_agents import find_org_names
from analyser.contract_patterns import ContractPatternFactory
from analyser.legal_docs import LegalDocument, extract_sum_sign_currency, ContractValue
from analyser.ml_tools import *

from analyser.parsing import ParsingContext
from analyser.patterns import AV_SOFT, AV_PREFIX

from analyser.sections_finder import FocusingSectionsFinder
from analyser.structures import ContractSubject

contract_subjects = [ContractSubject.RealEstate, ContractSubject.Charity, ContractSubject.Deal]

from analyser.transaction_values import complete_re as transaction_values_re

from analyser.hyperparams import HyperParameters


class ContractDocument3(LegalDocument):
  '''

  '''

  # TODO: rename it

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

    self.subjects = []
    self.contract_values: List[ContractValue] = []

    self.agents_tags = []

  def get_tags(self) -> [SemanticTag]:
    tags = []
    if self.date is not None:
      tags.append(self.date)

    if self.number is not None:
      tags.append(self.number)

    if self.agents_tags:
      tags += self.agents_tags

    if self.subjects:
      tags.append(self.subjects)

    if self.contract_values:
      for contract_value in self.contract_values:
        tags += contract_value.as_list()

    # TODO: filter tags if _t.isNotEmpty():
    return tags


ContractDocument = ContractDocument3  # Alias!


class ContractAnlysingContext(ParsingContext):
  # TODO: rename this class

  def __init__(self, embedder=None, pattern_factory: ContractPatternFactory = None):
    ParsingContext.__init__(self, embedder)

    self.pattern_factory: ContractPatternFactory or None = pattern_factory
    if embedder is not None:
      self.init_embedders(embedder, None)

    self.sections_finder = FocusingSectionsFinder(self)

  def init_embedders(self, embedder, elmo_embedder_default):
    self.embedder = embedder
    if self.pattern_factory is None:
      self.pattern_factory = ContractPatternFactory(embedder)

  def analyze_contract(self, contract_text):
    warnings.warn("call 1) find_org_date_number 2) find_attributes", DeprecationWarning)

    self._reset_context()
    # create DOC
    contract = ContractDocument(contract_text)
    contract.parse()

    self._logstep("parsing document 👞 and detecting document high-level structure")
    contract.embedd_tokens(self.pattern_factory.embedder)

    return self.find_attributes(contract)

  def find_org_date_number(self, contract: ContractDocument) -> ContractDocument:
    """
    phase 1, before embedding TF, GPU, and things
    searching for attributes required for filtering
    :param charter:
    :return:
    """
    contract.agents_tags = find_org_names(contract)
    return contract

  def find_attributes(self, contract: ContractDocument) -> ContractDocument:
    assert self.embedder is not None, 'call `init_embedders` first'
    """
    this analyser should care about embedding, because it decides wheater it needs (NN) embeddings or not  
    """

    self._reset_context()

    # ------ lazy embedding
    if contract.embeddings is None:
      contract.embedd_tokens(self.embedder)

    # ------ agents
    contract.agents_tags = find_org_names(contract)
    self._logstep("parsing document 👞 and detecting document high-level structure")

    # ------ structure
    self.sections_finder.find_sections(contract, self.pattern_factory, self.pattern_factory.headlines,
                                       headline_patterns_prefix='headline.')

    # -------------------------------values
    contract.contract_values = self.find_contract_value_NEW(contract)
    self._logstep("finding contract values")

    # -------------------------------subject
    contract.subjects = self.find_contract_subject_region(contract)
    self._logstep("detecting contract subject")
    # --------------------------------------

    self.log_warnings()

    return contract
    # , self.contract.contract_values

  def select_most_confident_if_almost_equal(self, a: ProbableValue, alternative: ProbableValue, m_convert,
                                            equality_range=0.0):

    if abs(m_convert(a.value).value - m_convert(alternative.value).value) < equality_range:
      if a.confidence > alternative.confidence:
        return a
      else:
        return alternative
    return a

  def __sub_attention_names(self, subj: ContractSubject):
    a = f'x_{subj}'
    b = AV_PREFIX + f'x_{subj}'
    c = AV_SOFT + a
    return a, b, c

  def make_subject_attention_vector_3(self, section, subject_kind: ContractSubject, addon=None) -> FixedVector:

    pattern_prefix, attention_vector_name, attention_vector_name_soft = self.__sub_attention_names(subject_kind)

    vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, pattern_prefix)
    if addon is not None:
      vectors = list(vectors)
      vectors.append(addon)
    x = max_exclusive_pattern(vectors)
    assert x is not None, f'no patterns for {subject_kind}'

    section.distances_per_pattern_dict[attention_vector_name_soft] = x
    section.distances_per_pattern_dict[attention_vector_name] = x

    #   x = x-np.mean(x)
    x = relu(x, 0.6)

    return x

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

  def find_contract_subject_regions(self, section: LegalDocument, denominator: float = 1.0) -> SemanticTag:
    # TODO: build trainset on contracts, train simple model for detectin start and end of contract subject region
    # TODO: const(loss) function should measure distance from actual span to expected span

    section.calculate_distances_per_pattern(self.pattern_factory, merge=True, pattern_prefix='x_ContractSubject')
    section.calculate_distances_per_pattern(self.pattern_factory, merge=True, pattern_prefix='headline.subj')

    all_subjects_vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, 'headline.subj')
    subject_headline_attention: FixedVector = rectifyed_sum(all_subjects_vectors) / 2

    max_confidence = 0
    max_subject_kind = None
    max_paragraph_span = None

    for subject_kind in contract_subjects:  # like ContractSubject.RealEstate ..
      subject_attention_vector: FixedVector = self.make_subject_attention_vector_3(section, subject_kind,
                                                                                   subject_headline_attention)

      paragraph_span, confidence, paragraph_attention_vector = _find_most_relevant_paragraph(section,
                                                                                             subject_attention_vector,
                                                                                             min_len=20,
                                                                                             return_delimiters=False)

      if self.verbosity_level > 2:
        print(f'--------------------confidence {subject_kind}=', confidence)
      if confidence > max_confidence:
        max_confidence = confidence
        max_subject_kind = subject_kind
        max_paragraph_span = paragraph_span

    if max_subject_kind:
      subject_tag = SemanticTag('subject', max_subject_kind.name, max_paragraph_span)
      subject_tag.confidence = max_confidence * denominator
      subject_tag.offset(section.start)

      return subject_tag

  def find_contract_value_NEW(self, contract: ContractDocument) -> List[ContractValue]:
    # preconditions
    assert contract.sections is not None, 'find sections first'

    search_sections_order = [
      ['cvalue', 1], ['pricecond', 0.75], ['subj', 0.75], [None, 0.5]  # todo: check 'price', not 'price.'
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

        values_list: List[ContractValue] = find_value_sign_currency(value_section, self.pattern_factory)

        if not values_list:
          # search in next section
          self.warning(f'В разделе "{_section_name}" ["{section}"] стоимость сделки не найдена!')

        else:
          # decrease confidence:
          for g in values_list:
            g *= confidence_k

          # ------
          # reduce number of found values
          # take only max value and most confident ones (we hope, it is the same finding)

          max_confident_cv = max_confident(values_list)
          max_valued_cv = max_value(values_list)
          if max_confident_cv == max_valued_cv:
            return [max_confident_cv]
          else:
            # TODO:
            max_valued_cv *= 0.5
            return [max_valued_cv]


      else:
        self.warning(f'Раздел [{section}]  не обнаружен')


def find_value_sign_currency(value_section_subdoc: LegalDocument, factory: ContractPatternFactory = None) -> List[
  ContractValue]:
  if factory is not None:
    value_section_subdoc.calculate_distances_per_pattern(factory)
    vectors = factory.make_contract_value_attention_vectors(value_section_subdoc)
    # merge dictionaries of attention vectors
    value_section_subdoc.distances_per_pattern_dict = {**value_section_subdoc.distances_per_pattern_dict, **vectors}

    attention_vector_tuned = value_section_subdoc.distances_per_pattern_dict['value_attention_vector_tuned']
  else:
    # HATI-HATI: this case is for Unit Testing only
    attention_vector_tuned = None

  return find_value_sign_currency_attention(value_section_subdoc, attention_vector_tuned)


def find_value_sign_currency_attention(value_section_subdoc: LegalDocument, attention_vector_tuned=None,
                                       parent_tag=None) -> List[
  ContractValue]:
  spans = [m for m in value_section_subdoc.tokens_map.finditer(transaction_values_re)]
  values_list = []

  for span in spans:
    value_sign_currency = extract_sum_sign_currency(value_section_subdoc, span)
    if value_sign_currency is not None:

      # Estimating confidence by looking at attention vector
      if attention_vector_tuned is not None:
        # offsetting spans
        value_sign_currency += value_section_subdoc.start

        for t in value_sign_currency.as_list():
          t.confidence *= (HyperParameters.confidence_epsilon + estimate_confidence_by_mean_top_non_zeros(
            attention_vector_tuned[t.slice]))

        value_sign_currency.parent.set_parent_tag(parent_tag)

      values_list.append(value_sign_currency)

  return values_list


def max_confident(vals: List[ContractValue]) -> ContractValue:
  return max(vals, key=lambda a: a.integral_sorting_confidence())


def max_confident_tag(vals: List[SemanticTag]) -> SemanticTag:
  warnings.warn("use max_confident_tags", DeprecationWarning)
  return max(vals, key=lambda a: a.confidence)


def max_value(vals: List[ContractValue]) -> ContractValue:
  return max(vals, key=lambda a: a.value.value)


def _find_most_relevant_paragraph(section: LegalDocument, subject_attention_vector: FixedVector, min_len: int,
                                  return_delimiters=True):
  # paragraph_attention_vector = smooth(attention_vector, 6)

  _blur = HyperParameters.subject_paragraph_attention_blur
  _padding = _blur * 2 + 1

  paragraph_attention_vector = smooth_safe(np.pad(subject_attention_vector, _padding, mode='constant'), _blur)[
                               _padding:-_padding]

  top_index = int(np.argmax(paragraph_attention_vector))
  span = section.tokens_map.sentence_at_index(top_index)
  if min_len is not None and span[1] - span[0] < min_len:
    next_span = section.tokens_map.sentence_at_index(span[1] + 1, return_delimiters)
    span = (span[0], next_span[1])

  # confidence = paragraph_attention_vector[top_index]
  confidence_region = subject_attention_vector[span[0]:span[1]]
  confidence = estimate_confidence_by_mean_top_non_zeros(confidence_region)
  return span, confidence, paragraph_attention_vector


def find_all_value_sign_currency(doc: LegalDocument) -> List[ContractValue]:
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
