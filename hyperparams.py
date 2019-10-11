import os
from pathlib import Path
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
models_path = os.path.join(__location__, 'vocab')


class HyperParameters:
  subsidiary_name_match_min_jaro_similarity = 0.8

  confidence_epsilon = 0.001
  parser_headline_attention_vector_denominator = 0.75