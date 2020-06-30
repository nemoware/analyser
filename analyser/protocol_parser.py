import re
from typing import Iterator

from analyser.contract_agents import complete_re as agents_re, find_org_names, ORG_LEVELS_re, find_org_names_raw, \
  ContractAgent, _rename_org_tags, protocol_caption_complete_re, protocol_caption_complete_re_ignore_case
from analyser.doc_dates import find_document_date
from analyser.doc_numbers import document_number_c, find_document_number_in_subdoc
from analyser.documents import sentences_attention_to_words
from analyser.embedding_tools import AbstractEmbedder
from analyser.legal_docs import LegalDocument, tokenize_doc_into_sentences_map, ContractValue, ParserWarnings, \
  LegalDocumentExt
from analyser.ml_tools import *
from analyser.parsing import ParsingContext, AuditContext, find_value_sign_currency_attention
from analyser.patterns import *
from analyser.structures import ORG_LEVELS_names
from analyser.text_normalize import r_group, r_quoted
from analyser.text_tools import is_long_enough, span_len
from analyser.transaction_values import complete_re as values_re

something = r'(\s*.{1,100}\s*)'
itog1 = r_group(r'\n' + r_group('итоги\s*голосования' + '|' + 'результаты\s*голосования') + r"[:\n]?")

_number_of_votes = r'(\s*[-: ]\s*)([0-9]|(нет)|[_]{1,10})[.;]*\s*'
r_votes_za = r_group(r_quoted('за') + _number_of_votes)
r_votes_pr = r_group(r_quoted('против') + _number_of_votes)
r_votes_vo = r_group(r_quoted('воздержался') + _number_of_votes)

protocol_votes_ = r_group(itog1 + something) + r_group(r_votes_za + r_votes_pr + r_votes_vo)
protocol_votes_re = re.compile(protocol_votes_, re.IGNORECASE | re.UNICODE)


class ProtocolAV(Enum):
  '''AV fo Attention Vecotrs'''
  bin_votes_attention = 1,
  relu_deal_approval = 2,
  digits_attention = 3,
  relu_value_attention_vector = 4


class ProtocolDocument(LegalDocumentExt):

  def __init__(self, doc: LegalDocument = None):
    super().__init__(doc)

    if doc is not None:
      self.__dict__ = {**super().__dict__, **doc.__dict__}

    self.agents_tags: [SemanticTag] = []
    self.org_level: [SemanticTag] = []
    self.org_tags: [SemanticTag] = []
    self.agenda_questions: [SemanticTag] = []
    self.margin_values: [ContractValue] = []
    self.contract_numbers: [SemanticTag] = []

  def get_tags(self) -> [SemanticTag]:
    tags = []
    if self.date is not None:
      tags.append(self.date)

    if self.number is not None:
      tags.append(self.number)

    tags += self.org_tags
    tags += self.org_level
    tags += self.agents_tags
    tags += self.agenda_questions
    tags += self.contract_numbers
    for mv in self.margin_values:
      tags += mv.as_list()

    return tags


ProtocolDocument4 = ProtocolDocument  # aliasing #todo: remove it


class ProtocolParser(ParsingContext):
  patterns_dict = [
    ['sum_max1', 'стоимость не более 0 млн. тыс. миллионов тысяч рублей долларов копеек евро'],

    # ['solution_1','решение, принятое по вопросу повестки дня:'],
    # ['solution_2','по вопросам повестки дня приняты следующие решения:'],

    ['not_value_1', 'размер уставного капитала 0 рублей'],
    ['not_value_2', 'принятие решения о назначении секретаря'],

    ['agenda_end_1', 'кворум для проведения заседания и принятия решений имеется'],
    ['agenda_end_2', 'Вопрос повестки дня заседания'],
    ['agenda_end_3', 'Формулировка решения по вопросу повестки дня заседания:'],

    ['agenda_start_1', 'повестка дня заседания'],
    ['agenda_start_2', 'Повестка дня'],

    ['deal_approval_1', 'одобрить совершение сделки'],
    ['deal_approval_2', 'одобрить сделку'],
    ['deal_approval_3', 'дать согласие на заключение договора'],
    ['deal_approval_4', 'принять решение о совершении сделки'],
    ['deal_approval_5', 'принять решение о совершении крупной сделки'],
    ['deal_approval_6', 'заключить договор аренды'],
    ['deal_approval_7', 'Одобрить сделку, связанную с заключением Дополнительного соглашения'],

    ['question_1', 'По вопросу № 0'],
    ['question_2', 'Первый вопрос повестки дня заседания'],
    ['question_3', 'Решение, принятое по вопросу повестки дня:'],
    ['question_4', 'Решение, принятое по 1 вопросу повестки дня:'],

    ['footers_1', 'Время подведения итогов голосования'],
    ['footers_2', 'Список приложений:'],
    ['footers_3', 'Подсчет голосов производил Секретарь Совета директоров'],
    ['footers_4', 'Протокол составлен в 2-х экземплярах']

  ]

  def __init__(self, embedder=None, elmo_embedder_default: AbstractEmbedder = None):
    ParsingContext.__init__(self, embedder)
    self.embedder = embedder
    self.elmo_embedder_default = elmo_embedder_default
    self.protocols_factory: ProtocolPatternFactory = None
    self.patterns_embeddings = None

    if embedder is not None and elmo_embedder_default is not None:
      self.init_embedders(embedder, elmo_embedder_default)

  def init_embedders(self, embedder, elmo_embedder_default):
    self.embedder = embedder
    self.elmo_embedder_default = elmo_embedder_default

    self.protocols_factory: ProtocolPatternFactory = ProtocolPatternFactory(embedder)
    patterns_te = [p[1] for p in ProtocolParser.patterns_dict]
    self.patterns_embeddings = elmo_embedder_default.embedd_strings(patterns_te)

  def ebmedd(self, doc: ProtocolDocument):
    assert self.embedder is not None, 'call init_embedders first'
    assert self.elmo_embedder_default is not None, 'call init_embedders first'

    ### ⚙️🔮 SENTENCES embedding
    doc.sentences_embeddings = self.elmo_embedder_default.embedd_strings(doc.sentence_map.tokens)

    ### ⚙️🔮 WORDS Ebmedding
    doc.embedd_tokens(self.embedder)

    doc.calculate_distances_per_pattern(self.protocols_factory)
    doc.distances_per_sentence_pattern_dict = calc_distances_per_pattern_dict(doc.sentences_embeddings,
                                                                              self.patterns_dict,
                                                                              self.patterns_embeddings)

  def find_org_date_number(self, doc: ProtocolDocument, ctx: AuditContext) -> ProtocolDocument:
    """
    phase 1, before embedding TF, GPU, and things
    searching for attributes required for filtering
    :param charter:
    :return:
    """
    doc.sentence_map = tokenize_doc_into_sentences_map(doc, 250)

    doc.org_level = max_confident_tags(list(find_org_structural_level(doc)))

    doc.org_tags = list(find_protocol_org(doc))
    doc.date = find_document_date(doc)

    if not doc.date:
      doc.warn(ParserWarnings.date_not_found)

    if not doc.org_level:
      doc.warn(ParserWarnings.org_struct_level_not_found)
    return doc

  def find_attributes(self, doc: ProtocolDocument, ctx: AuditContext = None) -> ProtocolDocument:

    if doc.sentences_embeddings is None or doc.embeddings is None:
      self.ebmedd(doc)  # lazy embedding

    doc.agenda_questions = self.find_question_decision_sections(doc)
    doc.margin_values = self.find_margin_values(doc)
    doc.contract_numbers = self.find_contract_numbers(doc)
    doc.agents_tags = list(self.find_agents_in_all_sections(doc, doc.agenda_questions, ctx.audit_subsidiary_name))

    self.validate(doc)
    return doc

  def validate(self, doc):
    if not doc.agenda_questions:
      doc.warn(ParserWarnings.protocol_agenda_not_found)

    if not doc.margin_values and not doc.agents_tags and not doc.contract_numbers:
      doc.warn(ParserWarnings.boring_agenda_questions)

  def find_agents_in_all_sections(self,
                                  doc: LegalDocument,
                                  agenda_questions: [SemanticTag],
                                  audit_subsidiary_name: str) -> [SemanticTag]:
    ret = []
    for parent in agenda_questions:
      x: [SemanticTag] = self._find_agents_in_section(doc, parent, audit_subsidiary_name)
      if x:
        ret += x
    return ret

  def _find_agents_in_section(self, protocol: LegalDocument, parent: SemanticTag,
                              audit_subsidiary_name) -> [SemanticTag]:

    span = parent.span
    doc = protocol[span[0]:span[1]]

    all: [ContractAgent] = find_org_names_raw(doc, max_names=10, parent=parent, decay_confidence=False)
    all: [ContractAgent] = sorted(all, key=lambda a: a.name.value != audit_subsidiary_name)

    start_from = 2
    if all and all[0].name.value == audit_subsidiary_name:
      start_from = 1

    return _rename_org_tags(all, 'contract_agent_', start_from=start_from)

  def find_margin_values(self, doc) -> [ContractValue]:
    assert ProtocolAV.relu_value_attention_vector.name in doc.distances_per_pattern_dict, 'call find_question_decision_sections first'

    values: [ContractValue] = []
    for agenda_question_tag in doc.agenda_questions:
      subdoc = doc[agenda_question_tag.as_slice()]
      subdoc_values: [ContractValue] = find_value_sign_currency_attention(subdoc,
                                                                          ProtocolAV.relu_value_attention_vector.name,
                                                                          parent_tag=agenda_question_tag,
                                                                          absolute_spans=True)
      values += subdoc_values
      if len(subdoc_values) > 1:
        confidence = 1.0 / len(subdoc_values)

        for k, v in enumerate(subdoc_values):
          v *= confidence  # decrease confidence
          v.parent.kind = SemanticTag.number_key(v.parent.kind, k)

    return values

  def find_contract_numbers(self, doc) -> [ContractValue]:

    values = []
    for agenda_question_tag in doc.agenda_questions:
      subdoc = doc[agenda_question_tag.as_slice()]

      numbers = find_document_number_in_subdoc(subdoc, tagname='number', parent=agenda_question_tag)

      for k, v in enumerate(numbers):
        # v.parent = agenda_question_tag
        v.kind = SemanticTag.number_key(v.kind, k)
      values += numbers
    return values

  def collect_spans_having_votes(self, segments, textmap):
    warnings.warn("use TextMap.regex_attention", DeprecationWarning)
    """
    search for votes in each document segment
    collect only
    :param segments:
    :param textmap:
    :return:  segments with votes
    """
    for span in segments:
      subdoc = textmap.slice(span)
      protocol_votes = list(subdoc.finditer(protocol_votes_re))
      if protocol_votes:
        yield span

  def find_protocol_sections_edges(self, distances_per_pattern_dict):

    patterns = ['footers_', 'deal_approval_', 'question_']
    vv_ = []
    for p in patterns:
      v_ = max_exclusive_pattern_by_prefix(distances_per_pattern_dict, p)
      v_ = relu(v_, 0.55)
      vv_.append(v_)

    v_sections_attention = sum_probabilities(vv_)
    v_sections_attention = best_above(v_sections_attention, 0.55)
    return v_sections_attention

  def _get_value_attention_vector(self, doc: LegalDocument):
    s_value_attention_vector = max_exclusive_pattern_by_prefix(doc.distances_per_pattern_dict, 'sum_max_p_')

    doc.distances_per_pattern_dict['__max_value_av'] = s_value_attention_vector  # just for debugging
    not_value_av = max_exclusive_pattern_by_prefix(doc.distances_per_pattern_dict, 'not_sum_')

    not_value_av = smooth_safe(not_value_av, window_len=5)

    not_value_av = relu(not_value_av, 0.5)
    doc.distances_per_pattern_dict['__not_value_av'] = not_value_av  # just for debugging

    s_value_attention_vector -= not_value_av * 0.8
    s_value_attention_vector = relu(s_value_attention_vector, 0.3)
    return s_value_attention_vector

  def find_question_decision_sections(self, doc: ProtocolDocument):
    wa = doc.distances_per_pattern_dict  # words attention #shortcut
    # -----

    # DEAL APPROVAL SENTENCES
    v_deal_approval = max_exclusive_pattern_by_prefix(doc.distances_per_sentence_pattern_dict, 'deal_approval_')
    _spans, deal_approval_av = sentences_attention_to_words(v_deal_approval, doc.sentence_map,
                                                            doc.tokens_map)
    deal_approval_relu_av = best_above(deal_approval_av, 0.5)

    # VOTES
    votes_av = doc.tokens_map.regex_attention(protocol_votes_re)

    # DOC NUMBERS
    numbers_av = doc.tokens_map.regex_attention(document_number_c)

    # DOC AGENTS orgs
    agents_av = doc.tokens_map.regex_attention(agents_re)

    # DOC MARGIN VALUES
    margin_values_av = self._get_value_attention_vector(doc)
    margin_values_v = doc.tokens_map.regex_attention(values_re)
    margin_values_v *= margin_values_av
    wa[ProtocolAV.relu_value_attention_vector.name] = margin_values_av

    # -----
    combined_av = sum_probabilities([
      deal_approval_relu_av,
      margin_values_v,
      agents_av / 2,
      votes_av / 2,
      numbers_av / 2])

    combined_av_norm = combined_av = best_above(combined_av, 0.2)
    # --------------

    protocol_sections_edges = self.find_protocol_sections_edges(doc.distances_per_sentence_pattern_dict)
    _question_spans_sent = spans_between_non_zero_attention(protocol_sections_edges)
    question_spans_words = doc.sentence_map.remap_slices(_question_spans_sent, doc.tokens_map)
    agenda_questions = list(find_confident_spans(question_spans_words, combined_av_norm, 'agenda_item', 0.5))

    return agenda_questions


class ProtocolPatternFactory(AbstractPatternFactory):

  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self)

    create_value_negation_patterns(self)
    create_value_patterns(self)

    self.embedd(embedder)


def find_protocol_org(protocol: ProtocolDocument) -> List[SemanticTag]:
  ret = []
  x: List[SemanticTag] = find_org_names(protocol[0:HyperParameters.protocol_caption_max_size_words], max_names=1,
                                        regex=protocol_caption_complete_re,
                                        re_ignore_case=protocol_caption_complete_re_ignore_case)

  nm = SemanticTag.find_by_kind(x, 'org-1-name')
  if nm is not None:
    ret.append(nm)
  else:
    protocol.warn(ParserWarnings.org_name_not_found)

  tp = SemanticTag.find_by_kind(x, 'org-1-type')
  if tp is not None:
    ret.append(tp)
  else:
    protocol.warn(ParserWarnings.org_type_not_found)
  return ret


def closest_name(pattern: str, knowns: [str]) -> (str, int):
  #
  min_distance = 0
  found = None
  for b in knowns:
    d = jaro.get_jaro_distance(pattern, b, winkler=True, scaling=0.1)
    if d > min_distance:
      found = b
      min_distance = d

  return found, min_distance


def find_org_structural_level(doc: LegalDocument) -> Iterator[SemanticTag]:
  compiled_re = re.compile(ORG_LEVELS_re, re.MULTILINE | re.IGNORECASE | re.UNICODE)

  entity_type = 'org_structural_level'
  for m in re.finditer(compiled_re, doc.text):

    char_span = m.span(entity_type)
    span = doc.tokens_map.token_indices_by_char_range(char_span)
    val = doc.tokens_map.text_range(span)

    val, conf = closest_name(val, ORG_LEVELS_names)

    confidence = conf * (1.0 - (span[0] / len(doc)))  # relative distance from the beginning of the document
    if span_len(char_span) > 1 and is_long_enough(val):

      if confidence > HyperParameters.org_level_min_confidence:
        tag = SemanticTag(entity_type, OrgStructuralLevel.find_by_display_string(val), span)
        tag.confidence = confidence

        yield tag


def find_confident_spans(slices: [int], block_confidence: FixedVector, tag_name: str, threshold: float) -> Iterator[
  SemanticTag]:
  count = 0
  for _slice in slices:
    pv = block_confidence[_slice[0]:_slice[1]]
    confidence = estimate_confidence_by_mean_top_non_zeros(pv, 5)

    if confidence > threshold:
      count += 1
      st = SemanticTag(SemanticTag.number_key(tag_name, count), None, _slice)
      st.confidence = confidence
      yield (st)
