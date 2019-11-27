import re
from typing import Iterator

from contract_agents import find_org_names, ORG_LEVELS_re
from contract_agents import find_org_names_in_tag
from contract_parser import find_value_sign_currency_attention
from hyperparams import HyperParameters
from legal_docs import LegalDocument, tokenize_doc_into_sentences_map, ContractValue
from ml_tools import *
from parsing import ParsingContext
from patterns import *
from structures import ORG_LEVELS_names
from text_normalize import r_group, ru_cap, r_quoted
from text_tools import is_long_enough, span_len
from tf_support.embedder_elmo import ElmoEmbedder

VALUE_ATTENTION_VECTOR_NAME = 'relu_value_attention_vector'

something = r'(\s*.{1,100}\s*)'
itog1 = r_group(r_group(ru_cap('итоги голосования') + '|' + ru_cap('результаты голосования')) + r"[:\n]?")

r_votes_za = r_group(r_quoted('за'))
r_votes_pr = r_group(r_quoted('против') + something)
r_votes_vo = r_group(r_quoted('воздержался') + something)

protocol_votes_ = r_group(itog1 + something) + r_group(r_votes_za + something + r_votes_pr + something + r_votes_vo)
protocol_votes_re = re.compile(protocol_votes_, re.IGNORECASE | re.UNICODE)


class ProtocolDocument4(LegalDocument):

  def __init__(self, doc: LegalDocument or None):
    super().__init__('')
    if doc is not None:
      self.__dict__ = doc.__dict__

    self.sentence_map: TextMap = None
    self.sentences_embeddings = None

    self.distances_per_sentence_pattern_dict = {}

    self.agents_tags: [SemanticTag] = []
    self.org_level: [SemanticTag] = []
    self.agenda_questions: [SemanticTag] = []
    self.margin_values: [ContractValue] = []

  def get_tags(self) -> [SemanticTag]:
    tags = []
    tags += self.agents_tags
    tags += self.org_level
    tags += self.agenda_questions
    for mv in self.margin_values:
      tags += mv.as_list()

    return tags


ProtocolDocument = ProtocolDocument4  # aliasing


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
    ['deal_approval_1.1', 'одобрить сделку'],
    ['deal_approval_2', 'дать согласие на заключение договора'],
    ['deal_approval_3', 'принять решение о совершении сделки'],
    ['deal_approval_3.1', 'принять решение о совершении крупной сделки'],
    ['deal_approval_4', 'заключить договор аренды'],

    ['question_1', 'По вопросу № 0'],
    ['question_2', 'Первый вопрос повестки дня заседания'],
    ['question_3', 'Решение, принятое по вопросу повестки дня:'],
    ['question_4', 'Решение, принятое по 1 вопросу повестки дня:'],

    ['footers_1', 'Время подведения итогов голосования'],
    ['footers_2', 'Список приложений:'],
    ['footers_3', 'Подсчет голосов производил Секретарь Совета директоров'],
    ['footers_4', 'Протокол составлен в 2-х экземплярах']

  ]

  def __init__(self, embedder, elmo_embedder_default: ElmoEmbedder):
    ParsingContext.__init__(self, embedder)
    self.elmo_embedder_default = elmo_embedder_default
    self.protocols_factory: ProtocolPatternFactory = ProtocolPatternFactory(embedder)

    patterns_te = [p[1] for p in ProtocolParser.patterns_dict]
    self.patterns_embeddings = elmo_embedder_default.embedd_strings(patterns_te)

  def ebmedd(self, doc: ProtocolDocument):
    doc.sentence_map = tokenize_doc_into_sentences_map(doc, 250)

    ### ⚙️🔮 SENTENCES embedding
    doc.sentences_embeddings = self.elmo_embedder_default.embedd_strings(doc.sentence_map.tokens)

    ### ⚙️🔮 WORDS Ebmedding
    doc.embedd_tokens(self.embedder)

    doc.calculate_distances_per_pattern(self.protocols_factory)
    doc.distances_per_sentence_pattern_dict = calc_distances_per_pattern_dict(doc.sentences_embeddings,
                                                                              self.patterns_dict,
                                                                              self.patterns_embeddings)

  def analyse(self, doc: ProtocolDocument):
    self.ebmedd(doc)
    self._analyse_embedded(doc)

  def _analyse_embedded(self, doc: ProtocolDocument):
    doc.org_level = max_confident_tags(list(find_org_structural_level(doc)))
    doc.agents_tags = list(find_protocol_org(doc))

    doc.agenda_questions = self.find_question_decision_sections(doc)
    doc.margin_values = self.find_values(doc)

    doc.agents_tags += list(self.find_agents_in_all_sections(doc, doc.agenda_questions))

  def find_agents_in_all_sections(self, doc: LegalDocument, agenda_questions: List[SemanticTag]) -> List[SemanticTag]:
    ret = []
    for parent in agenda_questions:
      x: List[SemanticTag] = self._find_agents_in_section(doc, parent, tag_kind_prefix='contract_agent_',
                                                          decay_confidence=False)
      if x:
        ret += x
    return ret

  def _find_agents_in_section(self, protocol: LegalDocument, parent: SemanticTag, tag_kind_prefix: str,
                              decay_confidence=False) -> List[
    SemanticTag]:
    x: List[SemanticTag] = find_org_names_in_tag(protocol, parent, max_names=10, tag_kind_prefix=tag_kind_prefix,
                                                 decay_confidence=decay_confidence)
    return x

  def find_values(self, doc) -> [ContractValue]:
    value_attention_vector = doc.distances_per_pattern_dict[VALUE_ATTENTION_VECTOR_NAME]

    # values: [ContractValue] = find_value_sign_currency_attention(doc, value_attention_vector)

    values = []
    for agenda_question_tag in doc.agenda_questions:
      subdoc = doc[agenda_question_tag.as_slice()]
      subdoc_values: [ContractValue] = find_value_sign_currency_attention(subdoc, value_attention_vector,
                                                                          parent_tag=agenda_question_tag)
      values += subdoc_values
      if len(subdoc_values) > 1:
        confidence = 1.0 / len(subdoc_values)
        k = 0
        for v in subdoc_values:
          k += 1
          v *= confidence  # decrease confidence
          v.parent.kind = f'{v.parent.kind}-{k}'

    # # set parents for values
    # for tag in doc.agenda_questions:
    #   # subdoc = doc[tag.as_slice()]
    #   for v in values:
    #     if tag.is_nested(v.span()):
    #       v.parent.set_parent_tag(tag)
    #       # v.parent.parent = tag.kind

    return values

  def collect_spans_having_votes(self, segments, textmap):
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

    patterns = ['deal_approval_', 'footers_', 'question_']
    vv_ = []
    for p in patterns:
      v_ = max_exclusive_pattern_by_prefix(distances_per_pattern_dict, p)
      v_ = relu(v_, 0.5)
      vv_.append(v_)

    v_sections_attention = sum_probabilities(vv_)

    v_sections_attention = relu(v_sections_attention, 0.7)
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
    wa = doc.distances_per_pattern_dict  # words attention
    v_sections_attention = self.find_protocol_sections_edges(doc.distances_per_sentence_pattern_dict)

    # --------------
    question_spans_sent = spans_between_non_zero_attention(v_sections_attention)
    question_spans_words = doc.sentence_map.remap_slices(question_spans_sent, doc.tokens_map)
    # --------------

    # *More* attention to spans having votes
    spans_having_votes = list(self.collect_spans_having_votes(question_spans_sent, doc.sentence_map))

    spans_having_votes_words = doc.sentence_map.remap_slices(spans_having_votes, doc.tokens_map)
    # questions_attention =  spans_to_attention(question_spans_words, len(doc))
    wa['bin_votes_attention'] = spans_to_attention(spans_having_votes_words, len(doc))

    # v_deal_approval_words = sentence_map.remap_spans(v_deal_approval,  doc.tokens_map )
    v_deal_approval = max_exclusive_pattern_by_prefix(doc.distances_per_sentence_pattern_dict, 'deal_approval_')
    _spans, v_deal_approval_words_attention = sentences_attention_to_words(v_deal_approval, doc.sentence_map,
                                                                           doc.tokens_map)

    ## value attention

    wa[VALUE_ATTENTION_VECTOR_NAME] = self._get_value_attention_vector(doc)
    wa['relu_deal_approval'] = relu(v_deal_approval_words_attention, 0.5)

    _value_attention_vector = sum_probabilities(
      [wa[VALUE_ATTENTION_VECTOR_NAME],
       wa['relu_deal_approval'],
       wa['bin_votes_attention'] / 3.0])

    wa[VALUE_ATTENTION_VECTOR_NAME] = relu(_value_attention_vector, 0.5)
    # // words_spans_having_votes = doc.sentence_map.remap_slices(spans_having_votes, doc.tokens_map)

    values: List[ContractValue] = find_value_sign_currency_attention(doc, wa[VALUE_ATTENTION_VECTOR_NAME])

    numbers_attention = np.zeros(len(doc.tokens_map))
    numbers_confidence = np.zeros(len(doc.tokens_map))
    for v in values:
      numbers_confidence[v.value.as_slice()] += v.value.confidence
      numbers_attention[v.value.as_slice()] = 1
      numbers_attention[v.currency.as_slice()] = 1
      numbers_attention[v.sign.as_slice()] = 1

    block_confidence = sum_probabilities([numbers_attention, wa['relu_deal_approval'], wa['bin_votes_attention'] / 5])

    return list(find_confident_spans(question_spans_words, block_confidence, 'agenda_item', 0.6))


class ProtocolPatternFactory(AbstractPatternFactory):
  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)

    self._build_subject_pattern()

    create_value_negation_patterns(self)
    create_value_patterns(self)

    self.embedd()

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
      ep.add_pattern(self.create_pattern('t_org_1', (PRFX, 'О создании филиала', 'Общества')))
      ep.add_pattern(self.create_pattern('t_org_2', (PRFX, 'Об утверждении Положения', 'о филиале Общества')))
      ep.add_pattern(self.create_pattern('t_org_3', (PRFX, 'О назначении руководителя', 'филиала')))
      ep.add_pattern(self.create_pattern('t_org_4', (PRFX, 'О прекращении полномочий руководителя', 'филиала')))
      ep.add_pattern(self.create_pattern('t_org_5', (PRFX, 'О внесении изменений', '')))

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


def find_protocol_org(protocol: ProtocolDocument) -> List[SemanticTag]:
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
        tag = SemanticTag(entity_type, val, span)
        tag.confidence = confidence

        yield tag


def find_confident_spans(slices: [int], block_confidence: FixedVector, tag_name: str, threshold: float) -> Iterator[
  SemanticTag]:
  k = 0
  for _slice in slices:
    k += 1
    pv = block_confidence[_slice[0]:_slice[1]]
    confidence = estimate_confidence_by_mean_top_non_zeros(pv, 5)

    if confidence > threshold:
      st = SemanticTag(f"{tag_name}_{k}", None, _slice)
      st.confidence = confidence
      yield (st)
