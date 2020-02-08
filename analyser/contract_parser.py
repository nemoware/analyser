from analyser.contract_agents import find_org_names
from analyser.contract_patterns import ContractPatternFactory, head_subject_patterns_prefix, contract_headlines_patterns
from analyser.doc_dates import find_document_date
from analyser.doc_numbers import find_document_number
from analyser.legal_docs import LegalDocument, ContractValue, ParserWarnings
from analyser.ml_tools import *
from analyser.parsing import ParsingContext, AuditContext, find_value_sign_currency, _find_most_relevant_paragraph
from analyser.patterns import AV_SOFT, AV_PREFIX, AbstractPatternFactory
from analyser.sections_finder import FocusingSectionsFinder
from analyser.structures import ContractSubject

contract_subjects = [
  ContractSubject.Charity,
  ContractSubject.RealEstate,
  ContractSubject.Renting,
  ContractSubject.Deal,
  ContractSubject.Loans,
  ContractSubject.PledgeEncumbrance]


class ContractDocument(LegalDocument):

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


ContractDocument3 = ContractDocument


class ContractParser(ParsingContext):
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

  def find_org_date_number(self, contract: ContractDocument, ctx: AuditContext) -> ContractDocument:
    """
    phase 1, before embedding TF, GPU, and things
    searching for attributes required for filtering
    :param charter:
    :return:
    """
    contract.agents_tags = find_org_names(contract[0:2000], max_names=2,
                                          audit_subsidiary_name=ctx.audit_subsidiary_name)
    contract.date = find_document_date(contract)
    contract.number = find_document_number(contract)

    if not contract.number:
      contract.warn(ParserWarnings.number_not_found)
    if not contract.date:
      contract.warn(ParserWarnings.date_not_found)

    return contract

  def find_attributes(self, contract: ContractDocument, ctx: AuditContext) -> ContractDocument:
    assert self.embedder is not None, 'call `init_embedders` first'
    """
    this analyser should care about embedding, because it decides wheater it needs (NN) embeddings or not  
    """

    self._reset_context()

    # ------ lazy embedding
    if contract.embeddings is None:
      contract.embedd_tokens(self.embedder)

    self._logstep("parsing document 👞 and detecting document high-level structure")

    # ------ structure
    self.sections_finder.find_sections(contract, self.pattern_factory, self.pattern_factory.headlines,
                                       headline_patterns_prefix='headline.')

    # -------------------------------values
    contract.contract_values = self.find_contract_value_NEW(contract)
    if not contract.contract_values:
      contract.warn(ParserWarnings.contract_value_not_found)
    self._logstep("finding contract values")

    # -------------------------------subject
    contract.subjects = self.find_contract_subject_region(contract)
    if not contract.subjects:
      contract.warn(ParserWarnings.contract_subject_not_found)
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

  def __sub_attention_names(self, subj: Enum):
    a = f'x_{subj}'
    b = AV_PREFIX + f'x_{subj}'
    c = AV_SOFT + a
    return a, b, c

  def make_subject_attention_vector_3(self, section: LegalDocument, subject_kind: ContractSubject,
                                      addon=None) -> FixedVector:

    pattern_prefix, attention_vector_name, attention_vector_name_soft = self.__sub_attention_names(subject_kind)

    _vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, pattern_prefix)
    _vectors = list(_vectors)

    if not _vectors:
      _vectors = []
      _vectors.append(np.zeros(len(section.tokens_map)))
      warnings.warn(f'no patterns for {subject_kind}')

    if addon is not None:
      _vectors.append(addon)

    vectors = []
    for v in _vectors:
      vectors.append(best_above(v, 0.4))

    # assert len(vectors) > 0, f'no vectors for {pattern_prefix} {attention_vector_name} {attention_vector_name_soft}'

    x = max_exclusive_pattern(vectors)
    x = relu(x, 0.6)
    section.distances_per_pattern_dict[attention_vector_name_soft] = x
    section.distances_per_pattern_dict[attention_vector_name] = x

    return x

  def find_contract_subject_region(self, doc) -> SemanticTag:
    if 'subj' in doc.sections:
      subj_section = doc.sections['subj']
      subject_subdoc = subj_section.body
      denominator = 1
    else:
      doc.warn(ParserWarnings.subject_section_not_found)
      self.warning('раздел о предмете договора не найден, ищем предмет договора в первых 1500 словах')
      doc.warn(ParserWarnings.contract_subject_section_not_found)
      subject_subdoc = doc[0:1500]
      denominator = 0.7

    a: SemanticTag = self.find_contract_subject_regions(subject_subdoc, denominator=denominator)

    header_subject, conf = find_headline_subject_match(doc, self.pattern_factory)
    if conf >= a.confidence or conf > 0.7:
      a.value = header_subject.name  # override subject kind detected in text by subject detected in 1st headline
      a.confidence = (a.confidence + conf) / 2.0
    return a

  def find_contract_subject_regions(self, section: LegalDocument, denominator: float = 1.0) -> SemanticTag:
    # TODO: build trainset on contracts, train simple model for detectin start and end of contract subject region
    # TODO: const(loss) function should measure distance from actual span to expected span

    section.calculate_distances_per_pattern(self.pattern_factory, merge=True, pattern_prefix='x_ContractSubject')
    section.calculate_distances_per_pattern(self.pattern_factory, merge=True, pattern_prefix='headline.subj')

    all_subjects_headlines_vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, 'headline.subj')

    subject_headline_attention: FixedVector = max_exclusive_pattern(all_subjects_headlines_vectors)
    subject_headline_attention = best_above(subject_headline_attention, 0.5)
    subject_headline_attention = momentum_t(subject_headline_attention, half_decay=120)
    subject_headline_attention_max = max(subject_headline_attention)

    section.distances_per_pattern_dict['subject_headline_attention'] = subject_headline_attention  # for debug

    max_confidence = 0
    max_subject_kind = None
    max_paragraph_span = None

    for subject_kind in contract_subjects:  # like ContractSubject.RealEstate ..
      subject_attention_vector: FixedVector = self.make_subject_attention_vector_3(section, subject_kind, None)
      if subject_headline_attention_max > 0.2:
        subject_attention_vector *= subject_headline_attention

      paragraph_span, confidence, paragraph_attention_vector = _find_most_relevant_paragraph(section,
                                                                                             subject_attention_vector,
                                                                                             min_len=20,
                                                                                             return_delimiters=False)
      if len(subject_attention_vector) < 400:
        confidence = estimate_confidence_by_mean_top_non_zeros(subject_attention_vector)

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
          msg = f'В разделе "{_section_name}" ["{section}"] стоимость сделки не найдена!'
          contract.warn(ParserWarnings.value_section_not_found,
                        f'В разделе "{_section_name}" стоимость сделки не найдена')
          self.warning(msg)

        else:
          # decrease confidence:
          for g in values_list:
            g *= confidence_k

          # ------
          # reduce number of found values
          # take only max value and most confident ones (we hope, it is the same finding)

          max_confident_cv: ContractValue = max_confident(values_list)
          max_valued_cv: ContractValue = max_value(values_list)
          if max_confident_cv == max_valued_cv:
            return [max_confident_cv]
          else:
            # TODO: Insurance docs have big value, its not what we're looking for. Biggest is not the best see https://github.com/nemoware/analyser/issues/55
            max_valued_cv *= 0.5
            return [max_valued_cv]


      else:
        self.warning(f'Раздел [{section}]  не обнаружен')


# --------------- END of CLASS


def match_headline_to_subject(section: LegalDocument, subject_kind: ContractSubject) -> FixedVector:
  pattern_prefix = f'{head_subject_patterns_prefix}{subject_kind}'

  _vectors = list(filter_values_by_key_prefix(section.distances_per_pattern_dict, pattern_prefix))

  if not _vectors:
    warnings.warn(f'no patterns for {subject_kind}')
    return np.zeros(len(section.tokens_map))

  vectors = [best_above(v, 0.4) for v in _vectors]

  x = max_exclusive_pattern(vectors)
  x = relu(x, 0.6)

  return x


def find_headline_subject_match(doc: LegalDocument, factory: AbstractPatternFactory) -> (ContractSubject, float):
  headers = [doc.subdoc_slice(p.header.as_slice()) for p in doc.paragraphs]

  max_confidence = 0
  best_subj = None
  for header_index, header in enumerate(
          headers[0:3]):  # take only 3 fist headlines; normally contract type is known by the 1st one.

    if header.text and header.text.strip():

      # TODO: must be pre-calculated
      header.calculate_distances_per_pattern(factory, pattern_prefix=head_subject_patterns_prefix, merge=False)

      for subject_kind in contract_headlines_patterns.values():  # like ContractSubject.RealEstate ..
        subject_attention_vector: FixedVector = match_headline_to_subject(header, subject_kind)
        _confidence = estimate_confidence_by_mean_top_non_zeros(subject_attention_vector)
        if _confidence > max_confidence:
          max_confidence = _confidence
          best_subj = subject_kind

        # print (subject_kind, _confidence)

  return best_subj, max_confidence


ContractAnlysingContext = ContractParser  ##just alias, for ipnb compatibility. TODO: remove


def max_confident(vals: List[ContractValue]) -> ContractValue:
  return max(vals, key=lambda a: a.integral_sorting_confidence())


def max_value(vals: List[ContractValue]) -> ContractValue:
  return max(vals, key=lambda a: a.value.value)
