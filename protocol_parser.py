from legal_docs import deprecated
from ml_tools import *
from patterns import FuzzyPattern
from patterns import make_pattern_attention_vector


# ❤️ == GOOD HEART LINE ========================================================

def make_smart_meta_click_pattern(attention_vector, embeddings, name=None):
  if name is None:
    import random
    name = 's-meta-na-' + str(random.random())

  best_id = np.argmax(attention_vector)
  print('make_smart_meta_click_pattern best_id', best_id)

  confidence = attention_vector[best_id]
  print('make_smart_meta_click_pattern confidence', confidence)

  best_embedding_v = embeddings[best_id]
  print('make_smart_meta_click_pattern best_embedding_v', best_embedding_v.shape)

  meta_pattern = FuzzyPattern(None, _name=name)
  meta_pattern.embeddings = np.array([best_embedding_v])

  return meta_pattern, confidence, best_id


@deprecated
def make_smart_pattern_attention_vector(doc, vv, relu_th=0.8):
  return improve_attention_vector(doc.embeddings, vv, relu_th=relu_th)[1]


# ❤️ == GOOD HEART LINE ========================================================

def improve_attention_vector(embeddings, vv, relu_th=0.5, mix=1):
  meta_pattern, meta_pattern_confidence, best_id = make_smart_meta_click_pattern(vv, embeddings)
  meta_pattern_attention_v = make_pattern_attention_vector(meta_pattern, embeddings)
  meta_pattern_attention_v = relu(meta_pattern_attention_v, relu_th)

  meta_pattern_attention_v = meta_pattern_attention_v * mix + vv * (1.0 - mix)
  return meta_pattern_attention_v, best_id


# ❤️ == GOOD HEART LINE ========================================================


from legal_docs import rectifyed_sum_by_pattern_prefix


def make_improved_attention_vector(doc, pattern_prefix):
  _max_hit_attention, _ = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, pattern_prefix)
  improved = improve_attention_vector(doc.embeddings, _max_hit_attention, mix=1)
  return improved


class CharterDocumentParser:
  def __init__(self, doc):
    self.doc = doc

    self.headlines_attention_vector = self.normalize_headline_attention_vector(self.make_headline_attention_vector(doc))

    self.competence_v, c__ = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, 'competence', 0.3)
    self.competence_v, c = improve_attention_vector(doc.embeddings, self.competence_v, mix=1)

    self.deal_attention = make_improved_attention_vector(self.doc, 'd_order_')
    # 💵 💵 💰
    self.value_attention = make_improved_attention_vector(self.doc, 'sum__')
    # 💰
    self.currency_attention_vector = make_improved_attention_vector(self.doc, 'currency')

  def _do_nothing(self, a, b):
    pass  #

  def find_charter_section_start(self, headline_pattern_prefix, debug_renderer):
    assert self.competence_v is not None
    assert self.headlines_attention_vector is not None

    if debug_renderer == None:
      debug_renderer = self._do_nothing

    competence_s = smooth(self.competence_v, 6)

    v, c__ = rectifyed_sum_by_pattern_prefix(self.doc.distances_per_pattern_dict, headline_pattern_prefix, 0.3)
    v += competence_s

    v *= self.headlines_attention_vector

    span = 100
    best_id = np.argmax(v)
    dia = slice(max(0, best_id - span), min(best_id + span, len(v)))
    debug_renderer(headline_pattern_prefix, self.doc.tokens_cc[dia], normalize(v[dia]))

    return best_id

  # ❤️ == GOOD HEART LINE ========================================================

  def make_headline_attention_vector(self):
    level_by_line = [max(i._possible_levels) for i in self.doc.structure.structure]

    headlines_attention_vector = []
    for i in self.doc.structure.structure:
      l = i.span[1] - i.span[0]
      headlines_attention_vector += [level_by_line[i.line_number]] * l

    return np.array(headlines_attention_vector)

  # ❤️ == GOOD HEART LINE ========================================================

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
