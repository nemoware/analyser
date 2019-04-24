import html
from typing import List

from legal_docs import LegalDocument
from legal_docs import calculate_distances_per_pattern
from ml_tools import filter_values_by_key_prefix, max_exclusive_pattern, relu
from patterns import AV_PREFIX, AV_SOFT
from patterns import improve_attention_vector
from structures import ContractSubject

# from renderer import v_color_map
interesting_subjects = [ContractSubject.RealEstate, ContractSubject.Charity]

doc.calculate_distances_per_pattern(CPF, merge=True, pattern_prefix='x_ContractSubject')


def sub_attention_names(subj):
  a = AV_PREFIX + f'x_{subj}'
  b = AV_SOFT + a
  return a, b


def render_doc(doc: LegalDocument):
  tokens = [html.escape(token) for token in doc.tokens_cc]
  attention_vectors = {}
  for subj in interesting_subjects:
    key1, key2 = sub_attention_names(subj)
    attention_vectors[key1] = doc.distances_per_pattern_dict[key1]
  #     attention_vectors[key2] = doc.distances_per_pattern_dict[key2]

  GLOBALS__['renderer'].render_multicolor_text(doc.tokens_cc, attention_vectors, v_color_map_2, min_color=None,
                                               _slice=None)


# for doc in docs:
#   render_doc(doc)
#   print()

# docs[0].distances_per_pattern_dict.keys()


def make_subj_attention_vector(self, factory, pattern_prefix, relu_th=0.8, recalc_distances=True) -> (List[float], str):
  # ---takes time
  if recalc_distances:
    calculate_distances_per_pattern(self, factory, merge=True, pattern_prefix=pattern_prefix)

  # ---
  vectors = filter_values_by_key_prefix(self.distances_per_pattern_dict, pattern_prefix)

  attention_vector_name = AV_PREFIX + pattern_prefix
  attention_vector_name_soft = AV_SOFT + attention_vector_name

  vectors_i = []
  for v in vectors:
    if max(v) > 0.6:
      print('click ' + pattern_prefix)
      #       v = v - np.mean(v)
      vector_i, _ = improve_attention_vector(self.embeddings, v, relu_th=0.8, mix=1)
      vectors_i.append(vector_i)
    else:
      vectors_i.append(v)

  vectors = vectors_i

  x = max_exclusive_pattern(vectors)
  self.distances_per_pattern_dict[attention_vector_name_soft] = x

  x = relu(x, relu_th)

  self.distances_per_pattern_dict[attention_vector_name] = x
  return x, attention_vector_name


def make_subj_attention_vector_2(doc, factory, pattern_prefix, relu_th=0.8, recalc_distances=True) -> (
        List[float], str):
  if recalc_distances:
    calculate_distances_per_pattern(doc, factory, merge=True, pattern_prefix=pattern_prefix)

  # ---
  attention_vector_name = AV_PREFIX + pattern_prefix
  attention_vector_name_soft = AV_SOFT + attention_vector_name

  vectors = filter_values_by_key_prefix(doc.distances_per_pattern_dict, pattern_prefix)
  x = max_exclusive_pattern(vectors)

  doc.distances_per_pattern_dict[attention_vector_name_soft] = x
  doc.distances_per_pattern_dict[attention_vector_name] = x

  #   x = x-np.mean(x)
  x = relu(x, relu_th)

  return x, attention_vector_name


def make_subj_attention_vector_3(doc, subj, additional_attention, k, relu_th=0.2):
  pattern_p = f'x_{subj}'

  key1, key2 = sub_attention_names(subj)

  subj_vectors = filter_values_by_key_prefix(doc.distances_per_pattern_dict, pattern_p)
  x = max_exclusive_pattern(subj_vectors)

  x += additional_attention
  x *= k
  x = relu(x, 0.5)

  # print(attention_vector_name_soft)
  doc.distances_per_pattern_dict[key1] = x
  doc.distances_per_pattern_dict[key2] = x


for doc in docs:

  vectors = filter_values_by_key_prefix(doc.distances_per_pattern_dict, 'x_ContractSubject')
  #   all_mean = rectifyed_sum(vectors)
  #   all_mean /=len(vectors)
  all_mean = max_exclusive_pattern(vectors)
  #   all_mean = relu(all_mean, 0.5)
  all_mean /= 2
  #   all_mean, _ = make_subj_attention_vector_2(doc, CPF, 'x_ContractSubject', relu_th=0.1)

  for subj in interesting_subjects:
    x = make_subj_attention_vector_3(doc, subj, -all_mean, 1, relu_th=0.2)

    #     confidence, sum_, nonzeros_count, _max = estimate_confidence (x)
    #     meanx = np.mean(x)
    #     print(f'{confidence:.2f} {meanx:.2f} \t {nonzeros_count} \t {subj}\t {_max:.3f} ')

    GLOBALS__['renderer'].render_color_text(doc.tokens, x, _range=(0, 1))
#   render_doc(doc)
