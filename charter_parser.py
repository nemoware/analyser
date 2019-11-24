# origin: charter_parser.py

from contract_parser import _find_most_relevant_paragraph, find_value_sign_currency_attention
from embedding_tools import AbstractEmbedder
from hyperparams import HyperParameters
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
      'сделки с имуществом Общества'
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

  def _make_patterns(self):

    comp_str_pat = []
    comp_str_pat += [[PATTERN_DELIMITER.join([competence_headline_pattern_prefix, ol.name]), ol.display_string] for ol
                     in OrgStructuralLevel]
    comp_str_pat += [[PATTERN_DELIMITER.join([competence_headline_pattern_prefix, 'comp', 'q', ol.name]),
                      "к компетенции " + ol.display_string + ' относятся следующие вопросы'] for ol in
                     OrgStructuralLevel]
    comp_str_pat += [[PATTERN_DELIMITER.join([competence_headline_pattern_prefix, 'comp', ol.name]),
                      "компетенции " + ol.display_string] for ol in OrgStructuralLevel]

    self.patterns_dict = comp_str_pat

  def __init__(self, embedder: AbstractEmbedder, elmo_embedder_default: AbstractEmbedder):
    ParsingContext.__init__(self, embedder)

    self.patterns_dict = []

    self.elmo_embedder_default: AbstractEmbedder = elmo_embedder_default

    self._make_patterns()
    patterns_te = [p[1] for p in self.patterns_dict]

    self.patterns_embeddings = elmo_embedder_default.embedd_strings(patterns_te)
    self.subj_patterns_embeddings = embedd_charter_subject_patterns(CharterParser.strs_subjects_patterns,
                                                                    elmo_embedder_default)

  def ebmedd(self, doc: CharterDocument):
    doc.sentence_map = tokenize_doc_into_sentences_map(doc, 200)

    ### ⚙️🔮 SENTENCES embedding
    doc.sentences_embeddings = embedd_sentences(doc.sentence_map, self.elmo_embedder_default)

    doc.distances_per_sentence_pattern_dict = calc_distances_per_pattern_dict(doc.sentences_embeddings,
                                                                              self.patterns_dict,
                                                                              self.patterns_embeddings)

  def analyse(self, charter: CharterDocument):
    patterns_by_headers = self.map_charter_headlines_to_patterns(charter)

    charter.margin_values = []
    charter.constraint_tags = []
    charter.charity_tags = []
    # --------------
    filtered = [p_mapping for p_mapping in patterns_by_headers if p_mapping]
    for p_mapping in filtered:
      paragraph = p_mapping[4]
      org_level_name = p_mapping[1].split('/')[-1]
      org_level = OrgStructuralLevel[org_level_name]
      subdoc = charter.subdoc_slice(paragraph.body.as_slice())

      parent_org_level_tag = SemanticTag(org_level.name, org_level, paragraph.body.span)
      charter.org_levels.append(parent_org_level_tag)

      constraint_tags, values = self.attribute_charter_subjects(subdoc, self.subj_patterns_embeddings,
                                                                parent_org_level_tag)

      for value in values:
        value += subdoc.start

      for constraint_tag in constraint_tags:
        constraint_tag.offset(subdoc.start)

      charter.margin_values += values
      charter.constraint_tags += constraint_tags

      # charity_subj_av_words = subject_attentions_map[CharterSubject.Charity]['words']
      # charity_tag = find_charity_paragraphs(parent_org_level_tag, subdoc, (charity_subj_av_words + consent_words) / 2)
      # # print(charity_tag)
      # if charity_tag is not None:
      #   charter.charity_tags.append(charity_tag)

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
      sentence_span = subdoc.tokens_map.sentence_at_index(value.parent.span[0], return_delimiters=True)
      if sentence_span not in sentence_spans:
        sentence_spans.append(sentence_span)
    sentence_spans = merge_colliding_spans(sentence_spans, eps=1)

    # ---
    # attribute sentences to subject
    constraint_tags = []

    for span in sentence_spans:

      max_confidence = 0
      best_subject = CharterSubject.Other

      for subj in subject_attentions_map.keys():
        av = subject_attentions_map[subj]['words']

        confidence_region = av[span[0]:span[1]]
        confidence = estimate_confidence_by_mean_top_non_zeros(confidence_region)

        if confidence > max_confidence:
          max_confidence = confidence
          best_subject = subj

      #
      constraint_tag = SemanticTag(f'{best_subject.name}', best_subject, span, parent=parent_org_level_tag)
      # constraint_tag.offset(subdoc.start)
      constraint_tags.append(constraint_tag)

      # nest values
      for value in values:
        # value+=subdoc.start
        if constraint_tag.is_nested(value.parent.span):
          value.parent.set_parent_tag(constraint_tag)

    return constraint_tags, values

  def map_charter_headlines_to_patterns(self, charter: LegalDocument):
    charter_parser = self

    p_suffices = [ol.name for ol in OrgStructuralLevel]
    p_suffices += [PATTERN_DELIMITER.join(['comp', ol.name]) for ol in OrgStructuralLevel]
    p_suffices += [PATTERN_DELIMITER.join(['comp', 'q', ol.name]) for ol in OrgStructuralLevel]

    map_, distances = map_headlines_to_patterns(charter,
                                                charter_parser.patterns_dict,
                                                charter_parser.patterns_embeddings,
                                                charter_parser.elmo_embedder_default,
                                                competence_headline_pattern_prefix,
                                                p_suffices)

    return map_


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

    subj_av = relu(max_exclusive_pattern_by_prefix(patterns_distances, prefix), 0.6)
    subj_av_words = remap_attention_vector(subj_av, subdoc.sentence_map, subdoc.tokens_map)

    _distances_per_subj[subj] = {
      'words': subj_av_words,
      'sentences': subj_av,
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
