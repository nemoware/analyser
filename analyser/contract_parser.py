from overrides import overrides

from analyser.contract_agents import ContractAgent, normalize_contract_agent
from analyser.contract_patterns import ContractPatternFactory, head_subject_patterns_prefix, contract_headlines_patterns
from analyser.doc_dates import find_date
from analyser.documents import TextMap
from analyser.legal_docs import LegalDocument, ContractValue, ParserWarnings
from analyser.ml_tools import *
from analyser.parsing import ParsingContext, AuditContext, _find_most_relevant_paragraph, \
  find_value_sign_currency_attention
from analyser.patterns import AV_SOFT, AV_PREFIX, AbstractPatternFactory
from analyser.structures import ContractSubject, contract_subjects
from analyser.text_tools import find_top_spans
from tf_support.embedder_elmo import ElmoEmbedder
from tf_support.tf_subject_model import load_subject_detection_trained_model, decode_subj_prediction, \
  nn_predict


class ContractDocument(LegalDocument):

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

    self.subjects: SemanticTag or None = None
    self.contract_values: List[ContractValue] = []

    self.agents_tags: [SemanticTag] = []

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

    # self.pattern_factory: ContractPatternFactory or None = pattern_factory
    if embedder is not None:
      self.init_embedders(embedder, None)

    self.subject_prediction_model = load_subject_detection_trained_model()

  def init_embedders(self, embedder, elmo_embedder_default):
    self.embedder = embedder
    if self.pattern_factory is None:
      self.pattern_factory = ContractPatternFactory(embedder)

  def find_org_date_number(self, contract_full: ContractDocument, ctx: AuditContext) -> ContractDocument:


    if self.embedder is None:
      self.embedder = ElmoEmbedder.get_instance()

    contract = contract_full[0:300] #warning, trimming doc for analysis phase 1
    if contract.embeddings is None:
      contract.embedd_tokens(self.embedder)

    semantic_map, subj_1hot = nn_predict(self.subject_prediction_model, contract)

    contract_full.agents_tags = nn_find_org_names(contract.tokens_map, semantic_map,
                                                  audit_subsidiary_name=ctx.audit_subsidiary_name)

    # TODO: maybe move contract.tokens_map into text map
    contract_full.number = nn_get_contract_number(contract.tokens_map, semantic_map)
    contract_full.date = nn_get_contract_date(contract.tokens_map, semantic_map)

    if not contract_full.date:
      contract_full.warn(ParserWarnings.date_not_found)

    if not contract_full.number:
      contract_full.warn(ParserWarnings.number_not_found)

    #
    #
    # # validating date & number position, date must go before any agents
    #
    # if contract.date is not None:
    #   date_start = contract.date.span[0]
    #   for at in contract.agents_tags:
    #     if at.span[0] < date_start:
    #       # date must go before companies names
    #       contract.date = None
    #
    # if contract.number is not None:
    #   number_start = contract.number.span[0]
    #   for at in contract.agents_tags:
    #     if at.span[0] < number_start:
    #       # doc number must go before companies names
    #       contract.number = None
    #
    #
    return contract_full

  @overrides
  def find_attributes(self, contract: ContractDocument, ctx: AuditContext) -> ContractDocument:
    """
    this analyser should care about embedding, because it decides wheater it needs (NN) embeddings or not  
    """
    if self.embedder is None:
      self.embedder = ElmoEmbedder.get_instance()

    self._reset_context()

    # ------ lazy embedding
    if contract.embeddings is None:
      contract.embedd_tokens(self.embedder)

    semantic_map, subj_1hot = nn_predict(self.subject_prediction_model, contract)

    # -------------------------------subject
    contract.subjects = nn_get_subject(contract.tokens_map, semantic_map, subj_1hot)
    if not contract.subjects:
      contract.warn(ParserWarnings.contract_subject_not_found)

    # -------------------------------values
    contract.contract_values = nn_find_contract_value(contract, semantic_map)
    if not contract.contract_values:
      contract.warn(ParserWarnings.contract_value_not_found)
    self._logstep("finding contract values")
    # --------------------------------------

    self.log_warnings()

    return contract

  def select_most_confident_if_almost_equal(self, a: ProbableValue, alternative: ProbableValue, m_convert,
                                            equality_range=0.0):

    if abs(m_convert(a.value).value - m_convert(alternative.value).value) < equality_range:
      if a.confidence > alternative.confidence:
        return a
      else:
        return alternative
    return a

  @staticmethod
  def make_subject_attention_vector_3(section: LegalDocument, subject_kind: ContractSubject,
                                      addon=None) -> FixedVector:

    pattern_prefix, attention_vector_name, attention_vector_name_soft = _sub_attention_names(subject_kind)

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

    header_subject, header_subject_conf, header_subject_subdoc = find_headline_subject_match(doc, self.pattern_factory)

    if header_subject is not None:
      if a is None:
        a = SemanticTag('subject', header_subject.name, (header_subject_subdoc.start, header_subject_subdoc.end))
        a.confidence = header_subject_conf

      if header_subject_conf >= a.confidence or header_subject_conf > 0.7:
        a.value = header_subject.name  # override subject kind detected in text by subject detected in 1st headline
        a.confidence = (a.confidence + header_subject_conf) / 2.0

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

    max_confidence: float = 0.
    max_subject_kind: ContractSubject or None = None
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
      subject_tag = SemanticTag('subject', max_subject_kind.name,
                                max_paragraph_span)  # TODO: check if it is OK to use enum value instead of just name
      subject_tag.confidence = max_confidence * denominator
      subject_tag.offset(section.start)

      return subject_tag


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


def find_headline_subject_match(doc: LegalDocument, factory: AbstractPatternFactory) -> (
        ContractSubject, float, LegalDocument):
  headers: [LegalDocument] = [doc.subdoc_slice(p.header.as_slice()) for p in doc.paragraphs]

  max_confidence = 0
  best_subj = None
  subj_header = None
  for header in headers[:3]:  # take only 3 fist headlines; normally contract type is known by the 1st one.

    if header.text and header.text.strip():

      # TODO: must be pre-calculated
      header.calculate_distances_per_pattern(factory, pattern_prefix=head_subject_patterns_prefix, merge=False)

      for subject_kind in contract_headlines_patterns.values():  # like ContractSubject.RealEstate ..
        subject_attention_vector: FixedVector = match_headline_to_subject(header, subject_kind)
        _confidence = estimate_confidence_by_mean_top_non_zeros(subject_attention_vector)
        if _confidence > max_confidence:
          max_confidence = _confidence
          best_subj = subject_kind
          subj_header = header

  return best_subj, max_confidence, subj_header


ContractAnlysingContext = ContractParser  ##just alias, for ipnb compatibility. TODO: remove


def max_confident(vals: List[ContractValue]) -> ContractValue:
  return max(vals, key=lambda a: a.integral_sorting_confidence())


def max_value(vals: List[ContractValue]) -> ContractValue:
  return max(vals, key=lambda a: a.value.value)


def _sub_attention_names(subj: Enum):
  a = f'x_{subj}'
  b = AV_PREFIX + f'x_{subj}'
  c = AV_SOFT + a
  return a, b, c


def nn_find_org_names(textmap: TextMap, semantic_map: DataFrame,
                      audit_subsidiary_name=None) -> [SemanticTag]:
  cas = []
  for o in [1, 2]:
    ca = ContractAgent()
    for n in ['name', 'alias', 'type']:
      tagname = f'org-{o}-{n}'
      tag = nn_get_tag_value(tagname, textmap, semantic_map)
      setattr(ca, n, tag)
    normalize_contract_agent(ca)
    cas.append(ca)

  if audit_subsidiary_name:
    # known subsidiary goes first
    cas = sorted(cas, key=lambda a: a.name.value != audit_subsidiary_name)
  else:
    cas = sorted(cas, key=lambda a: a.name.value)

  return _swap_org_tags(cas)


def _swap_org_tags(all_: [ContractAgent]) -> [SemanticTag]:
  tags = []
  for n, agent in enumerate(all_):
    for tag in agent.as_list():
      tagname = f"org-{n + 1}-{tag.kind.split('-')[-1]}"
      tag.kind = tagname
      tags.append(tag)

  return tags


def nn_find_contract_value(contract: ContractDocument, semantic_map: DataFrame) -> [ContractValue]:
  _keys = ['sign_value_currency/value', 'sign_value_currency/currency', 'sign_value_currency/sign']
  attention_vector = semantic_map[_keys].values.sum(axis=-1)

  values_list: [ContractValue] = find_value_sign_currency_attention(contract, attention_vector)

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


def nn_get_subject(textmap: TextMap, semantic_map: DataFrame, subj_1hot) -> SemanticTag:
  predicted_subj_name, confidence, _ = decode_subj_prediction(subj_1hot)

  tag = SemanticTag('subject', predicted_subj_name.name, span=None)
  tag.confidence = confidence

  tag_ = nn_get_tag_value('subject', textmap, semantic_map)
  if tag_ is not None:
    tag.span = tag_.span

  return tag


def nn_get_contract_number(textmap: TextMap, semantic_map: DataFrame) -> SemanticTag:
  tag = nn_get_tag_value('number', textmap, semantic_map)
  if tag is not None:
    tag.value = tag.value.strip().lstrip('№').lstrip('N ').lstrip()
  return tag


def nn_get_contract_date(textmap: TextMap, semantic_map: DataFrame) -> SemanticTag:
  tag = nn_get_tag_value('date', textmap, semantic_map)
  if tag is not None:
    _, dt = find_date(tag.value)
    tag.value = dt
    if dt is not None:
      return tag


def nn_get_tag_value(tagname: str, textmap: TextMap, semantic_map: DataFrame, threshold=0.3) -> SemanticTag or None:
  att = semantic_map[tagname].values
  slices = find_top_spans(att, threshold=threshold, limit=1)  # TODO: estimate per-tag thresholds

  if len(slices) > 0:
    span = slices[0].start, slices[0].stop
    value = textmap.text_range(span)
    tag = SemanticTag(tagname, value, span)
    tag.confidence = float(att[slices[0]].mean())
    return tag
  return None
