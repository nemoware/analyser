# origin: charter_parser.py
from analyser.contract_agents import find_org_names
from analyser.contract_parser import _find_most_relevant_paragraph, find_value_sign_currency_attention
from analyser.doc_dates import find_document_date

from analyser.embedding_tools import AbstractEmbedder
from analyser.legal_docs import LegalDocument, LegalDocumentExt, remap_attention_vector, ContractValue, \
  tokenize_doc_into_sentences_map, embedd_sentences, map_headlines_to_patterns, ParserWarnings
from analyser.ml_tools import *
from analyser.parsing import ParsingContext, AuditContext
from analyser.patterns import build_sentence_patterns, PATTERN_DELIMITER
from analyser.structures import *
from analyser.transaction_values import number_re

WARN = '\033[1;31m'

competence_headline_pattern_prefix = 'headline'

number_key = SemanticTag.number_key


class CharterDocument(LegalDocumentExt):

  def __init__(self, doc: LegalDocument = None):
    super().__init__(doc)
    if doc is not None:
      self.__dict__ = {**super().__dict__, **doc.__dict__}
    self.org_tags = []

    # TODO: remove this
    self.charity_tags = []
    self.org_levels = []
    self.constraint_tags = []
    self.org_level_tags = []

    self.margin_values: [ContractValue] = []

  def reset_attributes(self):
    # reset for preventing doubling tags
    self.margin_values = []
    self.constraint_tags = []
    self.charity_tags = []
    self.org_levels = []
    self.org_level_tags = []

  # def sentence_at_index(self, i: int, return_delimiters=True) -> (int, int):
  #
  #   char_range = self.tokens_map.char_range((i, i + 1))
  #   sentences_range = self.sentence_map.token_indices_by_char_range(char_range)
  #
  #   char_range = self.sentence_map.char_range(sentences_range)
  #   words_range = self.tokens_map.token_indices_by_char_range(char_range)
  #
  #   return words_range

  def get_tags(self) -> [SemanticTag]:
    tags = []

    if self.date is not None:
      tags.append(self.date)

    if self.number is not None:
      tags.append(self.number)

    tags += self.org_tags
    tags += self.charity_tags
    tags += self.org_levels
    tags += self.org_level_tags
    tags += self.constraint_tags

    for mv in self.margin_values:
      tags += mv.as_list()

    return tags


""" ❤️ == GOOD CharterDocumentParser  ====================================== """


def _make_org_level_patterns() -> pd.DataFrame:
  p = competence_headline_pattern_prefix  # just shortcut
  comp_str_pat = pd.DataFrame()
  for ol in OrgStructuralLevel:
    comp_str_pat[PATTERN_DELIMITER.join([p, ol.name])] = [ol.display_string.lower()]
    comp_str_pat[PATTERN_DELIMITER.join([p, 'comp', 'q', ol.name])] = [
      f'к компетенции {ol.display_string} относятся следующие вопросы'.lower()]
    comp_str_pat[PATTERN_DELIMITER.join([p, 'comp', ol.name])] = f"компетенции {ol.display_string}".lower()

  _key = PATTERN_DELIMITER.join([p, 'comp', 'qr', OrgStructuralLevel.ShareholdersGeneralMeeting.name])
  comp_str_pat[_key] = ['Компетенция Общего собрания акционеров Общества'.lower()]

  _key = PATTERN_DELIMITER.join([p, 'comp', 'qr', OrgStructuralLevel.BoardOfDirectors.name])
  comp_str_pat[_key] = ['Компетенция Совета директоров Общества'.lower()]

  _key = PATTERN_DELIMITER.join([p, 'comp', 'qr', OrgStructuralLevel.CEO.name])
  comp_str_pat[_key] = ['Единоличный исполнительный орган Общества'.lower()]

  return comp_str_pat.astype('str')


class CharterParser(ParsingContext):
  strs_subjects_patterns = {

    CharterSubject.Deal: [
      'принятие решений о совершении сделок'
    ],

    CharterSubject.BigDeal: [
      'принятие решений о совершении крупных сделок',
      'принятие решений о согласии на совершение или последующее одобрение крупных сделок'
    ],

    CharterSubject.Charity: [
      "оплата (встречное предоставление) в неденежной форме",
      "пожертвования на политические или благотворительные цели",
      "предоставление безвозмездной финансовой помощи",
      "сделки дарения",
      'безвозмездное отчуждение имущества',
      "договоры спонсорского и благотворительного характера",
      "передача в безвозмездное пользование",
      "мена, дарение, безвозмездное отчуждение",
      'внесение вкладов или пожертвований на политические или благотворительные цели'
    ],

    CharterSubject.Lawsuit: [
      'урегулирование любых судебных споров и разбирательств',
      'заключение мирового соглашения по судебному делу с ценой иска '
    ],

    CharterSubject.RealEstate: [
      'сделки с имуществом Общества',
      'стоимость отчуждаемого имущества',
      'сделок ( в том числе нескольких взаимосвязанных сделок ) с имуществом Общества'
    ],

    CharterSubject.Insurance: [
      'заключение договоров страхования',
      'возобновления договоров страхования'
      'совершение сделок страхования'
    ],

    CharterSubject.Consulting: [
      'заключение договора оказания консультационных услуг',
      'заключение агентского договора',
      'согласование договора оказания консультационных услуг или агентского договора',
      'оказание обществу информационных юридических услуг'
    ],

    # CharterSubject.Other: [
    #   'решения о взыскании с Генерального директора убытков',
    #   'заключение договоров об отступном, новации или прощении долга, договоров об уступке права требования и переводе долга',
    #   'нецелевое расходование Обществом денежных средств'
    # ],

    CharterSubject.Loans: [
      'получение или предоставление займов, кредитов (в том числе вексельных)',
      'предоставление гарантий и поручительств по обязательствам',
      'предоставление займа или получения заимствования, кредита, финансирования, выплаты или отсрочки по займу, кредиту, финансированию или задолженности',
      'предоставление обеспечений исполнения обязательств',
      'получение банковских гарантий'
      # 'о выдаче или получении Обществом векселей, производстве по ним передаточных надписей, авалей, платежей',
    ],

    CharterSubject.Renting: [
      'получение в аренду или субаренду недвижимого имущества'
    ],

    CharterSubject.RentingOut: [
      'передача в аренду или субаренду недвижимого имущества'
    ]

  }

  def __init__(self, embedder: AbstractEmbedder = None, elmo_embedder_default: AbstractEmbedder = None):
    ParsingContext.__init__(self, embedder)

    self.embedder = embedder
    self.elmo_embedder_default: AbstractEmbedder = elmo_embedder_default

    self.patterns_dict: DataFrame = _make_org_level_patterns()
    self.patterns_named_embeddings: DataFrame = None

    if embedder is not None and elmo_embedder_default is not None:
      self.init_embedders(embedder, elmo_embedder_default)

  def init_embedders(self, embedder, elmo_embedder_default):
    self.embedder = embedder
    self.elmo_embedder_default: AbstractEmbedder = elmo_embedder_default

    self.subj_patterns_embeddings = embedd_charter_subject_patterns(CharterParser.strs_subjects_patterns,
                                                                    elmo_embedder_default)

    __patterns_embeddings = elmo_embedder_default.embedd_strings(self.patterns_dict.values[0])
    self.patterns_named_embeddings = pd.DataFrame(__patterns_embeddings.T, columns=self.patterns_dict.columns)

  def _ebmedd(self, doc: CharterDocument):
    assert self.elmo_embedder_default is not None, 'call init_embedders first'
    ### ⚙️🔮 SENTENCES embedding
    doc.sentences_embeddings = embedd_sentences(doc.sentence_map, self.elmo_embedder_default)
    doc.distances_per_sentence_pattern_dict = calc_distances_per_pattern(doc.sentences_embeddings,
                                                                         self.patterns_named_embeddings)

  def find_org_date_number(self, charter: LegalDocumentExt, ctx: AuditContext) -> LegalDocument:
    """
    phase 1, before embedding
    searching for attributes required for filtering
    :param charter:
    :return:
    """

    # TODO move this call from here to CharterDoc
    charter.sentence_map = tokenize_doc_into_sentences_map(charter, HyperParameters.charter_sentence_max_len)
    charter.org_tags = find_charter_org(charter)

    charter.date = find_document_date(charter)

    return charter

  def find_attributes(self, _charter: CharterDocument, ctx: AuditContext) -> CharterDocument:
    margin_values = []
    org_levels = []
    constraint_tags = []
    if _charter.sentences_embeddings is None:
      # lazy embedding
      self._ebmedd(_charter)

    # reset for preventing doubling tags
    _charter.reset_attributes()

    # --------------
    # (('Pattern name', 16), 0.8978644013404846),
    patterns_by_headers = map_headlines_to_patterns(_charter,
                                                    self.patterns_named_embeddings,
                                                    self.elmo_embedder_default)

    _parent_org_level_tag_keys = []
    for p_mapping in patterns_by_headers:
      # for each 'competence' article
      _pattern_name = p_mapping[0][0]
      _paragraph_id = p_mapping[0][1]

      paragraph_body = _charter.paragraphs[_paragraph_id].body
      confidence = p_mapping[1]
      _org_level_name = _pattern_name.split('/')[-1]
      org_level: OrgStructuralLevel = OrgStructuralLevel[_org_level_name]
      subdoc = _charter.subdoc_slice(paragraph_body.as_slice())
      # --
      parent_org_level_tag = SemanticTag(org_level.name, org_level, paragraph_body.span)
      parent_org_level_tag.confidence = confidence
      # -------
      # constraint_tags, values, subject_attentions_map = self.attribute_charter_subjects(subdoc,
      #                                                                                   parent_org_level_tag)

      _constraint_tags, _margin_values = self.find_attributes_in_sections(subdoc, parent_org_level_tag)
      margin_values += _margin_values
      constraint_tags += _constraint_tags

      if _constraint_tags:
        # _key = parent_org_level_tag.get_key()
        #   if _key in _parent_org_level_tag_keys:  # number keys to avoid duplicates
        #     parent_org_level_tag.kind = number_key(_key, len(_parent_org_level_tag_keys))
        org_levels.append(parent_org_level_tag)  # TODO: collect all, then assign to charter
        # _parent_org_level_tag_keys.append(_key)

    # --------------- populate charter

    _charter.org_levels = org_levels
    _charter.constraint_tags = constraint_tags
    _charter.margin_values = margin_values
    return _charter

  def find_attributes_in_sections(self, subdoc, parent_org_level_tag):

    subject_attentions_map = get_charter_subj_attentions(subdoc, self.subj_patterns_embeddings)
    subject_spans = collect_subjects_spans(subdoc, subject_attentions_map)

    values: [ContractValue] = find_value_sign_currency_attention(subdoc, None, absolute_spans=False)
    self._rename_margin_values_tags(values)
    valued_sentence_spans = collect_sentences_having_constraint_values(subdoc, values, merge_spans=True)

    united_spans = []
    for c in valued_sentence_spans:
      united_spans.append(c)
    for c in subject_spans:
      united_spans.append(c)

    united_spans = merge_colliding_spans(united_spans)

    constraint_tags, subject_attentions_map = self.attribute_spans_to_subjects(united_spans, subdoc,
                                                                               parent_org_level_tag,
                                                                               absolute_spans=False)

    # nesting values
    for parent_tag in constraint_tags:
      for value in values:
        v_group = value.parent
        if parent_tag.contains(v_group.span):
          v_group.set_parent_tag(parent_tag)

    # if values:
    #   _key = parent_org_level_tag.get_key()
    #   #   if _key in _parent_org_level_tag_keys:  # number keys to avoid duplicates
    #   #     parent_org_level_tag.kind = number_key(_key, len(_parent_org_level_tag_keys))
    #   org_levels.append(parent_org_level_tag)  # TODO: collect all, then assign to charter
    #   _parent_org_level_tag_keys.append(_key)


    # offsetting tags to absolute values
    for value in values: value += subdoc.start
    for constraint_tag in constraint_tags: constraint_tag += subdoc.start

    return constraint_tags, values

  def _rename_margin_values_tags(self, values):

    for value in values:
      if value.sign.value < 0:
        sfx = '-max'
      elif value.sign.value > 0:
        sfx = '-min'
      else:
        sfx = ''

      value.parent.kind = f"constraint{sfx}"

    known_keys = []
    k = 0  # constraints numbering
    for value in values:
      k += 1
      if value.parent.get_key() in known_keys:
        value.parent.kind = f"{value.parent.kind}{TAG_KEY_DELIMITER}{k}"

      known_keys.append(value.parent.get_key())

  def attribute_spans_to_subjects(self,
                                  unique_sentence_spans: Spans,
                                  subdoc: LegalDocumentExt,
                                  parent_org_level_tag: SemanticTag,
                                  absolute_spans=True):

    subject_attentions_map = get_charter_subj_attentions(subdoc, self.subj_patterns_embeddings)
    all_subjects = [k for k in subject_attentions_map.keys()]
    constraint_tags = []
    # attribute sentences to subject
    for sentence_number, contract_value_sentence_span in enumerate(unique_sentence_spans, start=1):

      max_confidence = 0
      best_subject = None

      for subj in all_subjects:
        av: FixedVector = subject_attentions_map[subj]['words']

        confidence_region: FixedVector = av[span_to_slice(contract_value_sentence_span)]
        confidence = estimate_confidence_by_mean_top_non_zeros(confidence_region)

        if confidence > max_confidence:
          max_confidence = confidence
          best_subject = subj
      # end for

      if best_subject is not None:
        constraint_tag = SemanticTag(number_key(best_subject, sentence_number),
                                     best_subject,
                                     contract_value_sentence_span,
                                     parent=parent_org_level_tag)
        constraint_tag.confidence = max_confidence
        constraint_tags.append(constraint_tag)

        all_subjects.remove(best_subject)  # taken: avoid duplicates

    # ofsetting
    if absolute_spans:
      for constraint_tag in constraint_tags:
        constraint_tag.offset(subdoc.start)

    return constraint_tags, subject_attentions_map


def collect_sentences_having_constraint_values(subdoc: LegalDocumentExt, contract_values: [ContractValue],
                                               merge_spans=True) -> Spans:
  # collect sentences having constraint values
  unique_sentence_spans: Spans = []
  for contract_value in contract_values:
    contract_value_sentence_span = subdoc.sentence_at_index(contract_value.parent.span[0], return_delimiters=False)
    if contract_value_sentence_span not in unique_sentence_spans:
      unique_sentence_spans.append(contract_value_sentence_span)
    contract_value_sentence_span = subdoc.sentence_at_index(contract_value.parent.span[1], return_delimiters=False)
    if contract_value_sentence_span not in unique_sentence_spans:
      unique_sentence_spans.append(contract_value_sentence_span)
  # --
  # TODO: do not join here, join by subject
  if merge_spans:
    unique_sentence_spans = merge_colliding_spans(unique_sentence_spans, eps=1)
  return unique_sentence_spans


def put_if_better(destination: dict, key, x, is_better: staticmethod):
  if key in destination:
    if is_better(x, destination[key]):
      destination[key] = x
  else:
    destination[key] = x


def split_by_number_2(tokens: List[str], attention: FixedVector, threshold) -> (
        List[List[str]], List[int], List[slice]):
  indexes = []
  last_token_is_number = False
  for i in range(len(tokens)):

    if attention[i] > threshold and len(number_re.findall(tokens[i])) > 0:
      if not last_token_is_number:
        indexes.append(i)
      last_token_is_number = True
    else:
      last_token_is_number = False

  text_fragments = []
  ranges: List[slice] = []
  if len(indexes) > 0:
    for i in range(1, len(indexes)):
      _slice = slice(indexes[i - 1], indexes[i])
      text_fragments.append(tokens[_slice])
      ranges.append(_slice)

    text_fragments.append(tokens[indexes[-1]:])
    ranges.append(slice(indexes[-1], len(tokens)))
  return text_fragments, indexes, ranges


def embedd_charter_subject_patterns(patterns_dict, embedder: AbstractEmbedder):
  emb_subj_patterns = {}
  for subj in patterns_dict.keys():
    strings = patterns_dict[subj]
    prefix = PATTERN_DELIMITER.join(['subject', subj.name])

    emb_subj_patterns[subj] = {
      'patterns': build_sentence_patterns(strings, prefix, subj),
      'embedding': embedder.embedd_strings(strings)
    }

  return emb_subj_patterns


def get_charter_subj_attentions(subdoc: LegalDocumentExt, emb_subj_patterns):
  _distances_per_subj = {}

  for subj in emb_subj_patterns.keys():
    patterns_distances = calc_distances_per_pattern_dict(subdoc.sentences_embeddings,
                                                         emb_subj_patterns[subj]['patterns'],
                                                         emb_subj_patterns[subj]['embedding'])

    prefix = PATTERN_DELIMITER.join(['subject', subj.name])

    subj_av = relu(max_exclusive_pattern_by_prefix(patterns_distances, prefix), 0.6)  # TODO: use hyper parameter
    subj_av_words = remap_attention_vector(subj_av, subdoc.sentence_map, subdoc.tokens_map)

    _distances_per_subj[subj] = {
      'words': subj_av_words,
      'sentences': subj_av,  ## TODO: this is not in use
    }
  return _distances_per_subj


def find_subject_paragraphs(parent_org_level_tag: SemanticTag,
                            subdoc: LegalDocument,
                            subj: CharterSubject,
                            subject_attention: FixedVector) -> SemanticTag:
  warnings.warn("use collect_subjects_spans and attribute_spans_to_subjects", DeprecationWarning)
  paragraph_span, confidence, paragraph_attention_vector = _find_most_relevant_paragraph(subdoc,
                                                                                         subject_attention,
                                                                                         min_len=20,
                                                                                         return_delimiters=False)

  if confidence > HyperParameters.charter_subject_attention_confidence:
    subject_tag = SemanticTag(subj.name, subj, paragraph_span, parent=parent_org_level_tag)
    subject_tag.offset(subdoc.start)
    subject_tag.confidence = confidence
    return subject_tag


def collect_subjects_spans(subdoc, subject_attentions_map, min_len=20):
  spans = []
  for subj in CharterSubject:

    subject_attention = subject_attentions_map[subj]['words']
    paragraph_span, confidence, paragraph_attention_vector = _find_most_relevant_paragraph(subdoc,
                                                                                           subject_attention,
                                                                                           min_len=min_len,
                                                                                           return_delimiters=False)
    if confidence > HyperParameters.charter_subject_attention_confidence:
      if paragraph_span not in spans:
        spans.append(paragraph_span)

  unique_sentence_spans = merge_colliding_spans(spans, eps=1)
  # for span in unique_sentence_spans:
  #   print(span, subdoc.tokens_map.text_range(span) )

  return unique_sentence_spans


def find_charter_org(charter: LegalDocument) -> [SemanticTag]:
  """
  TODO: see also find_protocol_org
  :param charter:
  :return:
  """
  ret = []
  x: List[SemanticTag] = find_org_names(charter[0:HyperParameters.protocol_caption_max_size_words], max_names=1)
  nm = SemanticTag.find_by_kind(x, 'org-1-name')
  if nm is not None:
    ret.append(nm)
  else:
    charter.warn(ParserWarnings.org_name_not_found)

  tp = SemanticTag.find_by_kind(x, 'org-1-type')
  if tp is not None:
    ret.append(tp)
  else:
    charter.warn(ParserWarnings.org_type_not_found)

  return ret
