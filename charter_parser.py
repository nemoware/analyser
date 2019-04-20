# origin: charter_parser.py
from legal_docs import get_sentence_bounds_at_index, HeadlineMeta, LegalDocument, org_types, CharterDocument, \
  make_constraints_attention_vectors, extract_all_contraints_from_sentence
from ml_tools import *
from parsing import ParsingSimpleContext, head_types_dict
from patterns import FuzzyPattern, find_ner_end
from patterns import make_pattern_attention_vector
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

    body.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix='sum_max.', merge=True)
    body.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix='sum__', merge=True)
    body.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix='d_order.', merge=True)

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
        ss_subdoc = body.subdoc_slice(r)
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
    # self.pattern_factory = pattern_factory
    pass

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

    rz = self.extract_constraint_values_from_sections(sections_filtered)
    return rz

  def ners(self):
    if 'name' in self.doc.sections:
      section: HeadlineMeta = self.doc.sections['name']
      org = self.detect_ners(section.body)
      self._logstep("extracting NERs (named entities)")
    else:
      self.warning('Секция наименования компнании не найдена')
      self.warning('Попытаемся искать просто в начале документа')
      org = self.detect_ners(self.doc.subdoc(0, 3000))

    """ 🚀️ = 🍄 🍄 🍄 🍄 🍄   TODO: ============================ """
    return org

  def _do_nothing(self, head, a, b):
    pass  #

  """ 📃️ 🐌 == find_charter_sections_starts ====================================================== """

  def find_charter_sections_starts(self, section_types: str, headlines_patterns_prefix='headline.',
                                   debug_renderer=None):
    """
    Fuzziy Finds sections in the doc
    TODO: try it on Contracts and Protocols as well
    TODO: if well, move from here

    🍄 🍄 🍄 🍄 🍄 Keep in in the dark and feed it sh**

    :param section_types:
    :param headlines_patterns_prefix:
    :param debug_renderer: a method for displaying results, default is None (do_nothing)
    :return:

    """
    if debug_renderer == None:
      debug_renderer = self._do_nothing

    def is_hl_more_confident(a: HeadlineMeta, b: HeadlineMeta):
      return a.confidence > b.confidence

    doc = self.doc

    #     assert do
    self.headlines_attention_vector = self.normalize_headline_attention_vector(self.make_headline_attention_vector())

    doc.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix='competence', merge=True)
    self.competence_v, c__ = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, 'competence', 0.3)
    self.competence_v, c = improve_attention_vector(doc.embeddings, self.competence_v, mix=1)

    section_by_index = {}
    for section_type in section_types:
      # like ['name.', 'head.all.', 'head.gen.', 'head.directors.']:
      pattern_prefix = f'{headlines_patterns_prefix}{section_type}'
      print('ddd', pattern_prefix)
      doc.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix=pattern_prefix, merge=True)

      # warning! these are the boundaries of the headline, not of the entire section
      bounds, confidence, attention = self._find_charter_section_start(pattern_prefix, debug_renderer=debug_renderer)

      hl_info = HeadlineMeta(None, section_type, confidence, doc.subdoc(bounds[0], bounds[1]))
      hl_info.attention = attention
      put_if_better(section_by_index, bounds[0], hl_info, is_hl_more_confident)
    # end-for
    # s = slice(bounds[0], bounds[1])

    sorted_starts = [i for i in sorted(section_by_index.keys())]
    sorted_starts.append(len(doc.tokens))
    section_by_type = {}
    for i in range(len(sorted_starts) - 1):
      index = sorted_starts[i]
      section: HeadlineMeta = section_by_index[index]
      start = index  # todo: probably take the end of the caption
      end = sorted_starts[i + 1]

      section_len = min(end - start, 5000)  #

      section.body = doc.subdoc(start, start + section_len)
      section.attention = section.attention[start: start + section_len]
      section_by_type[section.type] = section

    # end-for
    doc.sections = section_by_type
    return section_by_type

  """ ❤️ == GOOD HEART LINE ====================================================== """

  def _find_charter_section_start(self, headline_pattern_prefix, debug_renderer):
    assert self.competence_v is not None
    assert self.headlines_attention_vector is not None

    competence_s = smooth(self.competence_v, 6)

    v, c__ = rectifyed_sum_by_pattern_prefix(self.doc.distances_per_pattern_dict, headline_pattern_prefix, 0.3)
    v += competence_s

    v *= self.headlines_attention_vector

    span = 100
    best_id = np.argmax(v)
    dia = slice(max(0, best_id - span), min(best_id + span, len(v)))
    debug_renderer(headline_pattern_prefix, self.doc.tokens_cc[dia], normalize(v[dia]))

    bounds = get_sentence_bounds_at_index(best_id, self.doc.tokens)
    confidence = v[best_id]
    return bounds, confidence, v

  """ ❤️ == GOOD HEART LINE ====================================================== """

  def make_headline_attention_vector(self):
    level_by_line = [max(i._possible_levels) for i in self.doc.structure.structure]

    headlines_attention_vector = []
    for i in self.doc.structure.structure:
      l = i.span[1] - i.span[0]
      headlines_attention_vector += [level_by_line[i.line_number]] * l

    return np.array(headlines_attention_vector)

    """ ❤️ == GOOD HEART LINE ====================================================== """

  def normalize_headline_attention_vector(self, headline_attention_vector_pure):
    # XXX: test it
    #   _max_head_threshold = max(headline_attention_vector_pure) * 0.75
    _max_head_threshold = 1  # max(headline_attention_vector_pure) * 0.75
    # XXX: test it
    #   print(_max_head)
    headline_attention_vector = cut_above(headline_attention_vector_pure, _max_head_threshold)
    #   headline_attention_vector /= 2 # 5 is the maximum points a headline may gain during headlne detection : TODO:
    return relu(headline_attention_vector)

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

  def _detect_org_type_and_name(self, section):
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


""" 💔🛐  ===========================📈=================================  ✂️ """


def improve_attention_vector(embeddings, vv, relu_th=0.5, mix=1):
  assert vv is not None
  meta_pattern, meta_pattern_confidence, best_id = make_smart_meta_click_pattern(vv, embeddings)
  meta_pattern_attention_v = make_pattern_attention_vector(meta_pattern, embeddings)
  meta_pattern_attention_v = relu(meta_pattern_attention_v, relu_th)

  meta_pattern_attention_v = meta_pattern_attention_v * mix + vv * (1.0 - mix)
  return meta_pattern_attention_v, best_id


""" ❤️  =============================📈=================================  ✂️ """

from legal_docs import rectifyed_sum_by_pattern_prefix


def make_improved_attention_vector(doc, pattern_prefix):
  _max_hit_attention, _ = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, pattern_prefix)
  improved = improve_attention_vector(doc.embeddings, _max_hit_attention, mix=1)
  return improved
