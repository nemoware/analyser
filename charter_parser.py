# origin: charter_parser.py
from legal_docs import HeadlineMeta, LegalDocument, org_types, CharterDocument, \
  make_constraints_attention_vectors, extract_all_contraints_from_sentence
from ml_tools import *
from parsing import ParsingSimpleContext, head_types_dict
from patterns import FuzzyPattern, find_ner_end
from sections_finder import SectionsFinder, FocusingSectionsFinder
from text_tools import untokenize
from transaction_values import extract_sum, ValueConstraint


class CharterConstraintsParser(ParsingSimpleContext):

  def __init__(self, pattern_factory):
    ParsingSimpleContext.__init__(self)
    self.pattern_factory = pattern_factory
    pass

  ##---------------------------------------
  def extract_constraint_values_from_sections(self, sections):
    rez = {}

    for head_type in sections:
      section = sections[head_type]
      rez[head_type] = self.extract_constraint_values_from_section(section)

    return rez

  ##---------------------------------------
  def extract_constraint_values_from_section(self, section: HeadlineMeta):

    if self.verbosity_level > 1:
      print('extract_constraint_values_from_section', section.type)

    body = section.body

    body.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix='sum_max', merge=True)
    body.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix='sum__', merge=True)
    body.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix='d_order_', merge=True)

    a_vectors = make_constraints_attention_vectors(body)
    body.distances_per_pattern_dict = {**body.distances_per_pattern_dict, **a_vectors}

    if self.verbosity_level > 1:
      print('extract_constraint_values_from_section', 'embedding....')

    sentenses_having_values: List[LegalDocument] = []
    # senetences = split_by_token(body.tokens, '\n')

    ranges = split_by_token_into_ranges(body.tokens, '\n')

    for r in ranges:

      __line = untokenize(body.tokens[r])
      sum = extract_sum(__line)

      if sum is not None:
        ss_subdoc = body.subdoc_slice(r, name=f'sentence:{r.start}')
        sentenses_having_values.append(ss_subdoc)

      if self.verbosity_level > 2:
        print('-', sum, __line)

    r_by_head_type = {
      'section': head_types_dict[section.type],
      'caption': untokenize(section.subdoc.tokens_cc),
      'sentences': self.__extract_constraint_values_from_region(sentenses_having_values)
    }
    self._logstep(f"Finding margin transaction values in section {untokenize(section.subdoc.tokens_cc)}")
    return r_by_head_type

  ##---------------------------------------
  def __extract_constraint_values_from_region(self, sentenses_i: List[LegalDocument]):
    if sentenses_i is None or len(sentenses_i) == 0:
      return []

    sentences = []
    for sentence_subdoc in sentenses_i:
      constraints: List[ValueConstraint] = extract_all_contraints_from_sentence(sentence_subdoc,
                                                                                sentence_subdoc.distances_per_pattern_dict[
                                                                                  'deal_value_attention_vector'])

      sentence = {
        'subdoc': sentence_subdoc,
        'constraints': constraints
      }

      sentences.append(sentence)
    return sentences


""" ❤️ == GOOD CharterDocumentParser  ====================================== """
""" ❤️ == GOOD CharterDocumentParser  ====================================== """


class CharterDocumentParser(CharterConstraintsParser):
  def __init__(self, pattern_factory):
    CharterConstraintsParser.__init__(self, pattern_factory)

    self.sections_finder: SectionsFinder = FocusingSectionsFinder(self)

  def analyze_charter(self, txt, verbose=False):
    """
    🚀
    :param txt:
    """

    self._reset_context()

    # 0. parse
    _charter_doc = CharterDocument(txt)

    # 1. find top level structure
    _charter_doc.parse()
    _charter_doc.embedd(self.pattern_factory)
    self.doc: CharterDocument = _charter_doc

    # self.find_charter_sections_starts(self.pattern_factory.headlines)

    self.sections_finder.find_sections(self.doc, self.pattern_factory, self.pattern_factory.headlines,
                                       headline_patterns_prefix='headline.')

    # 2. NERS
    self.org = self.ners()

    # 3. constraints
    self.constraints = self.find_contraints()

    self.verbosity_level = 1
    self.log_warnings()

    return self.org, self.constraints

  def ners(self):
    if 'name' in self.doc.sections:
      section: HeadlineMeta = self.doc.sections['name']
      org = self.detect_ners(section.body)
      self._logstep("extracting NERs (named entities)")
    else:
      self.warning('Секция наименования компнании не найдена')
      self.warning('Попытаемся искать просто в начале документа')
      org = self.detect_ners(self.doc.subdoc_slice(slice(0, 3000), name='name_section'))

    """ 🚀️ = 🍄 🍄 🍄 🍄 🍄   TODO: ============================ """
    return org

  """ 🚀️ == GOOD CharterDocumentParser  ====================================================== """

  def parse(self, doc: CharterDocument):
    self.doc: CharterDocument = doc

    # TODO: move to doc.dict
    self.deal_attention = None  # make_improved_attention_vector(self.doc, 'd_order_')
    # 💵 💵 💰
    # TODO: move to doc.dict
    self.value_attention = None  # make_improved_attention_vector(self.doc, 'sum__')
    # 💰
    # TODO: move to doc.dict
    self.currency_attention_vector = None  # make_improved_attention_vector(self.doc, 'currency')
    self.competence_v = None

  # ---------------------------------------
  def find_contraints(self):
    # 5. extract constraint values
    sections_filtered = {}
    prefix = 'head.'
    for k in self.doc.sections:
      if k[:len(prefix)] == prefix:
        sections_filtered[k] = self.doc.sections[k]

    self._logstep(f'detecting sections: "{sections_filtered}" ')
    rz = self.extract_constraint_values_from_sections(sections_filtered)
    return rz

  def _do_nothing(self, head, a, b):
    pass  #

  """ 📃️ 🐌 == find_charter_sections_starts ====================================================== """

  # =======================

  def detect_ners(self, section):
    """
    XXX: TODO: 🚷🔥 moved from demo_charter.py
    :param section:
    :return:
    """
    assert section is not None
    factory = self.pattern_factory

    org_by_type_dict, org_type = self._detect_org_type_and_name(section)

    start = org_by_type_dict[org_type][0]
    start = start + len(factory.patterns_dict[org_type].embeddings)
    end = 1 + find_ner_end(section.tokens, start)

    orgname_sub_section: LegalDocument = section.subdoc(start, end)
    org_name = orgname_sub_section.untokenize_cc()

    rez = {
      'type': org_type,
      'name': org_name,
      'type_name': org_types[org_type],
      'tokens': section.tokens_cc,
      'attention_vector': section.distances_per_pattern_dict[org_type]
    }

    return rez

  def _detect_org_type_and_name(self, section: LegalDocument):
    """
        XXX: TODO: 🚷🔥 moved from demo_charter.py

    """
    factory = self.pattern_factory
    vectors = section.distances_per_pattern_dict  # shortcut

    section.calculate_distances_per_pattern(factory, pattern_prefix='org_', merge=True)
    section.calculate_distances_per_pattern(factory, pattern_prefix='ner_org', merge=True)
    section.calculate_distances_per_pattern(factory, pattern_prefix='nerneg_', merge=True)

    vectors['s_attention_vector_neg'] = factory._build_org_type_attention_vector(section)

    org_by_type = {}
    best_org_type = None
    _max = 0
    for org_type in org_types.keys():

      vector = vectors[org_type] * vectors['s_attention_vector_neg']
      if self.verbosity_level > 2:
        print('_detect_org_type_and_name, org_type=', org_type, vectors[org_type][0:10])

      idx = np.argmax(vector)
      val = vectors[org_type][idx]
      if val > _max:
        _max = val
        best_org_type = org_type

      org_by_type[org_type] = [idx, val]

    if self.verbosity_level > 2:
      print('_detect_org_type_and_name', org_by_type)

    return org_by_type, best_org_type


# ---


def put_if_better(dict: dict, key, x, is_better: staticmethod):
  if key in dict:
    if is_better(x, dict[key]):
      dict[key] = x
  else:
    dict[key] = x


# ❤️ == GOOD HEART LINE ========================================================

def make_smart_meta_click_pattern(attention_vector, embeddings, name=None):
  assert attention_vector is not None
  if name is None:
    import random
    name = 's-meta-na-' + str(random.random())

  best_id = np.argmax(attention_vector)
  confidence = attention_vector[best_id]
  best_embedding_v = embeddings[best_id]
  meta_pattern = FuzzyPattern(None, _name=name)
  meta_pattern.embeddings = np.array([best_embedding_v])

  return meta_pattern, confidence, best_id
