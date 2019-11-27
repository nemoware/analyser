# origin: charter_parser.py

from contract_parser import _find_most_relevant_paragraph, find_value_sign_currency_attention
from embedding_tools import AbstractEmbedder
from legal_docs import LegalDocument, LegalDocumentExt, remap_attention_vector, ContractValue, \
  tokenize_doc_into_sentences_map, embedd_sentences, map_headlines_to_patterns
from ml_tools import *
from parsing import ParsingContext
from patterns import build_sentence_patterns, PATTERN_DELIMITER
from structures import *
from transaction_values import number_re

WARN = '\033[1;31m'

competence_headline_pattern_prefix = 'headline'


class CharterDocument(LegalDocumentExt):

  def __init__(self, doc: LegalDocument):
    super().__init__(doc)
    if doc is not None:
      self.__dict__ = doc.__dict__

    self.sentence_map: TextMap = None
    self.sentences_embeddings = None

    self.distances_per_sentence_pattern_dict = {}

    self.charity_tags = []
    self.org_levels = []
    self.constraint_tags = []
    self.org_level_tags = []

    self.margin_values: [ContractValue] = []

  def get_tags(self) -> [SemanticTag]:
    tags = []
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

    CharterSubject.Charity: [
      'пожертвований на политические или благотворительные цели',
      'предоставление безвозмездной финансовой помощи',
      'сделок дарения',
      'договоров спонсорского и благотворительного характера',
      'передача в безвозмездное пользование',
      'мены, дарения, безвозмездное отчуждение '
    ],

    CharterSubject.Lawsuit: [
      'о начале/урегулировании любых судебных споров и разбирательств',
      'заключении Обществом мирового соглашения по судебному делу с ценой иска '
    ],

    CharterSubject.RealEstate: [
      'стоимость отчуждаемого имущества',
      'сделки с имуществом Общества',
      'сделок ( в том числе нескольких взаимосвязанных сделок ) с имуществом Общества'
    ],

    CharterSubject.Insurance: [
      'заключение договоров страхования',
      'возобновления договоров страхования'
      'совершение сделок страхования'
    ],

    CharterSubject.Consulting: [
      'договора оказания консультационных услуг',
      'заключения агентского договора',
      'согласование договора оказания консультационных услуг или агентского договора',
      'оказания обществу информационных юридических услуг '
    ],

    CharterSubject.Other: [
      'решения о взыскании с Генерального директора убытков',
      'заключение договоров об отступном , новации и/или прощении долга , договоров об уступке права требования и переводе долга',
      'о выдаче или получении Обществом векселей , производстве по ним передаточных надписей , авалей , платежей',
      'нецелевое расходование Обществом денежных средств'
    ]

  }

  def __init__(self, embedder: AbstractEmbedder, elmo_embedder_default: AbstractEmbedder):
    ParsingContext.__init__(self, embedder)
    self.elmo_embedder_default: AbstractEmbedder = elmo_embedder_default

    self.patterns_dict: DataFrame = _make_org_level_patterns()
    self.subj_patterns_embeddings = embedd_charter_subject_patterns(CharterParser.strs_subjects_patterns,
                                                                    elmo_embedder_default)

    __patterns_embeddings = elmo_embedder_default.embedd_strings(self.patterns_dict.values[0])
    self.patterns_named_embeddings = pd.DataFrame(__patterns_embeddings.T, columns=self.patterns_dict.columns)

  def ebmedd(self, doc: CharterDocument):
    doc.sentence_map = tokenize_doc_into_sentences_map(doc, 200)

    ### ⚙️🔮 SENTENCES embedding
    doc.sentences_embeddings = embedd_sentences(doc.sentence_map, self.elmo_embedder_default)
    doc.distances_per_sentence_pattern_dict = calc_distances_per_pattern(doc.sentences_embeddings,
                                                                         self.patterns_named_embeddings)

  def analyse(self, charter: CharterDocument):

    charter.margin_values = []
    charter.constraint_tags = []
    charter.charity_tags = []
    charter.org_levels = []
    charter.org_level_tags = []
    # --------------
    # (('Pattern name', 16), 0.8978644013404846),
    patterns_by_headers = map_headlines_to_patterns(charter,
                                                    self.patterns_named_embeddings,
                                                    self.elmo_embedder_default)

    _parent_org_level_tag_keys = []
    for p_mapping in patterns_by_headers:
      # kkk += 1

      _paragraph_id = p_mapping[0][1]
      _pattern_name = p_mapping[0][0]

      paragraph = charter.paragraphs[_paragraph_id]
      confidence = p_mapping[1]
      _org_level_name = _pattern_name.split('/')[-1]
      org_level: OrgStructuralLevel = OrgStructuralLevel[_org_level_name]
      subdoc = charter.subdoc_slice(paragraph.body.as_slice())

      parent_org_level_tag = SemanticTag(f"{org_level.name}", org_level.name, paragraph.body.span)
      parent_org_level_tag.confidence = confidence

      constraint_tags, values = self.attribute_charter_subjects(subdoc, self.subj_patterns_embeddings,
                                                                parent_org_level_tag)
      for value in values:
        value += subdoc.start

      for constraint_tag in constraint_tags:
        constraint_tag.offset(subdoc.start)

      charter.margin_values += values
      charter.constraint_tags += constraint_tags

      if values:
        _key = parent_org_level_tag.get_key()
        if _key in _parent_org_level_tag_keys:  # avoid duplicates
          parent_org_level_tag.kind = _key + f"-{len(_parent_org_level_tag_keys)}"
        charter.org_levels.append(parent_org_level_tag)
        _parent_org_level_tag_keys.append(_key)

      # charity_subj_av_words = subject_attentions_map[CharterSubject.Charity]['words']
      # charity_tag = find_charity_paragraphs(parent_org_level_tag, subdoc, (charity_subj_av_words + consent_words) / 2)
      # # print(charity_tag)
      # if charity_tag is not None:
      #   charter.charity_tags.append(charity_tag)

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

  def attribute_charter_subjects(self, subdoc: LegalDocumentExt, emb_subj_patterns, parent_org_level_tag: SemanticTag):
    """
    :param subdoc:
    :param emb_subj_patterns:

          emb_subj_patterns[subj] = {
            'patterns':patterns,
            'embedding':patterns_emb
          }

    :return:
    """

    # ---------------
    subject_attentions_map = get_charter_subj_attentions(subdoc, emb_subj_patterns)
    values: List[ContractValue] = find_value_sign_currency_attention(subdoc, None)
    # -------------------

    # collect sentences having constraint values
    sentence_spans = []
    for value in values:
      sentence_span = subdoc.tokens_map.sentence_at_index(value.parent.span[0], return_delimiters=False)
      if sentence_span not in sentence_spans:
        sentence_spans.append(sentence_span)
    sentence_spans = merge_colliding_spans(sentence_spans, eps=1)

    # ---
    # attribute sentences to subject
    constraint_tags = []

    i = 0
    for span in sentence_spans:
      i += 1
      max_confidence = 0
      best_subject: CharterSubject = CharterSubject.Other

      for subj in subject_attentions_map.keys():
        av = subject_attentions_map[subj]['words']

        confidence_region = av[span[0]:span[1]]
        confidence = estimate_confidence_by_mean_top_non_zeros(confidence_region)

        if confidence > max_confidence:
          max_confidence = confidence
          best_subject = subj

      #
      constraint_tag = SemanticTag(f'{best_subject.name}-{i}', best_subject.name, span, parent=parent_org_level_tag)
      constraint_tags.append(constraint_tag)

      # nest values
      for value in values:
        if constraint_tag.is_nested(value.parent.span):
          value.parent.set_parent_tag(constraint_tag)

      self._rename_margin_values_tags(values)

    return constraint_tags, values


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


def find_charity_paragraphs(parent_org_level_tag: SemanticTag, subdoc: LegalDocument,
                            charity_subject_attention: FixedVector):
  paragraph_span, confidence, paragraph_attention_vector = _find_most_relevant_paragraph(subdoc,
                                                                                         charity_subject_attention,
                                                                                         min_len=20,
                                                                                         return_delimiters=False)

  if confidence > HyperParameters.charter_charity_attention_confidence:
    subject_tag = SemanticTag(CharterSubject.Charity.name, CharterSubject.Charity.name, paragraph_span,
                              parent=parent_org_level_tag)
    subject_tag.offset(subdoc.start)
    subject_tag.confidence = confidence
    return subject_tag
