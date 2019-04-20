
# origin: charter_parser.py

from legal_docs import get_sentence_bounds_at_index, HeadlineMeta
from ml_tools import *
from patterns import FuzzyPattern
from patterns import make_pattern_attention_vector


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


# ❤️ == GOOD HEART LINE ========================================================

def improve_attention_vector(embeddings, vv, relu_th=0.5, mix=1):
  assert vv is not None
  meta_pattern, meta_pattern_confidence, best_id = make_smart_meta_click_pattern(vv, embeddings)
  meta_pattern_attention_v = make_pattern_attention_vector(meta_pattern, embeddings)
  meta_pattern_attention_v = relu(meta_pattern_attention_v, relu_th)

  meta_pattern_attention_v = meta_pattern_attention_v * mix + vv * (1.0 - mix)
  return meta_pattern_attention_v, best_id


""" ❤️  == GOOD HEART LINE ==========📈============================================  ✂️ """

from legal_docs import rectifyed_sum_by_pattern_prefix

def make_improved_attention_vector(doc, pattern_prefix):
  _max_hit_attention, _ = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, pattern_prefix)
  improved = improve_attention_vector(doc.embeddings, _max_hit_attention, mix=1)
  return improved


""" ❤️ == GOOD CharterDocumentParser  ====================================================== """

class CharterDocumentParser:
  def __init__(self, pattern_factory):
    self.pattern_factory = pattern_factory
    pass


  """ 🚀️ == GOOD CharterDocumentParser  ====================================================== """
  def parse(self, doc):
    self.doc = doc

    self.deal_attention = None# make_improved_attention_vector(self.doc, 'd_order_')
    # 💵 💵 💰
    self.value_attention = None# make_improved_attention_vector(self.doc, 'sum__')
    # 💰
    self.currency_attention_vector = None# make_improved_attention_vector(self.doc, 'currency')
    self.competence_v = None

  def _do_nothing(self, a, b):
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
      bounds, confidence = self._find_charter_section_start(pattern_prefix, debug_renderer=debug_renderer)

      hl_info = HeadlineMeta(None, section_type, confidence, doc.subdoc(bounds[0], bounds[1]))

      put_if_better(section_by_index, bounds[0], hl_info, is_hl_more_confident)
    # end-for
    # s = slice(bounds[0], bounds[1])

    sorted_starts = [i for i in sorted(section_by_index.keys())]
    sorted_starts.append(len(doc.tokens))
    for i in range(len(sorted_starts) - 1):
      index = sorted_starts[i]
      section: HeadlineMeta = section_by_index[index]
      start = index  # todo: probably take the end of the caption
      end = sorted_starts[i + 1]

      section_len = min(end - start, 5000)  #

      section.subdoc = doc.subdoc(start, start + section_len)

    # end-for

    return section_by_index

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
    return bounds, confidence


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
