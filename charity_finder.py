def build_charity_patterns(factory):
  def cp(name, tuples):
    return factory.create_pattern(name, tuples)

  cp('x_charity_1', ('договор',
                     'благотворительного',
                     'пожертвования'))

  cp('x_charity_1.1', ('одобрение внесения Обществом каких-либо вкладов или пожертвований на политические или',
                       'благотворительные',
                       'цели'))

  cp('x_charity_1.2', ('одобрение внесения Обществом каких-либо вкладов или',
                       'пожертвований',
                       'на политические или благотворительные цели '))

  cp('x_charity_2', ('предоставление',
                     'безвозмездной',
                     'помощи финансовой')),

  cp('x_charity_3', ('согласование сделок',
                     ' дарения ',
                     ' '))


from typing import List

import numpy as np

from legal_docs import calculate_distances_per_pattern
from ml_tools import filter_values_by_key_prefix, max_exclusive_pattern, relu
from patterns import improve_attention_vector
from text_tools import get_sentence_bounds_at_index


def find_charity_constraints(doc, factory, head_sections:dict ) -> dict:
  charity_quotes_by_head_type = {}
  for section_name in head_sections:

    # section_name = section_prefix + head
    # print(head, '->', section_name)
    if section_name in doc.sections:
      subdoc = doc.sections[section_name].body
      # subdoc.calculate_distances_per_pattern(TFAA)

      print(section_name)
      bounds = find_charity_sentences(subdoc, factory)
      charity_quotes_by_head_type[section_name] = bounds

      # print('ok')
  return charity_quotes_by_head_type


def find_charity_sentences(subdoc, factory) -> List:
  """
    returns list of tuples (slice, confidence, summa-of-attention-)
  """

  calculate_distances_per_pattern(subdoc, factory, merge=True, pattern_prefix='x_charity_')

  slices = []
  vectors = filter_values_by_key_prefix(subdoc.distances_per_pattern_dict, 'x_charity_')
  vectors_i = []
  for v in vectors:
    if max(v) > 0.6:
      vector_i, _ = improve_attention_vector(subdoc.embeddings, v, relu_th=0.6, mix=0.9)
      vectors_i.append(vector_i)
    else:
      vectors_i.append(v)

  x = max_exclusive_pattern(vectors_i)
  x = relu(x, 0.8)
  subdoc.distances_per_pattern_dict['$at_x_charity_'] = x

  dups = {}
  for i in np.nonzero(x)[0]:
    bounds = get_sentence_bounds_at_index(i, subdoc.tokens)

    if bounds[0] not in dups:
      sl = slice(bounds[0], bounds[1])
      sum_ = sum(x[sl])
      confidence = 'x'
      #       confidence = np.mean( np.nonzero(x[sl]) )
      nonzeros_count = len(np.nonzero(x[sl])[0])
      print('nonzeros_count=', nonzeros_count)
      confidence = 0

      if nonzeros_count > 0:
        confidence = sum_ / nonzeros_count
      print('confidence=', confidence)
      if confidence > 0.8:
        # GLOBALS__['renderer'].render_color_text(subdoc.tokens_cc[sl],
        #                                         subdoc.distances_per_pattern_dict['$at_x_charity_'][sl], _range=(0, 1))
        print(i, sum_)

        slices.append((sl, confidence, sum_))

      dups[bounds[0]] = True

  return slices
