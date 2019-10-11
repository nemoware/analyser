from legal_docs import LegalDocument
from ml_tools import *


def find_sections_spans_by_indices(_doc: LegalDocument,
                                   headline_pattern_names,
                                   all_headlines_indices,
                                   all_headlines_attention_vector, topic,
                                   relu_threshold=0.6):

  _headlines_of_interest_attention_vector = relu(
    combined_attention_vectors(_doc.distances_per_pattern_dict, headline_pattern_names) * all_headlines_attention_vector,
    relu_threshold)
  _doc.distances_per_pattern_dict[topic+'_headlines_attention_vector']=_headlines_of_interest_attention_vector
  _headlines_of_interest_spans = find_non_zero_spans(_doc.tokens_map, _headlines_of_interest_attention_vector)
  _headlines_of_interest_indices = _headlines_of_interest_spans[:, 0]

  for i in _headlines_of_interest_indices:
    last = find_first_gt(i, all_headlines_indices)
    if last is None:
      last = len(_doc.tokens_map) - 1
    yield [i, last]
