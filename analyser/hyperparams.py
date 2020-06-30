import os
import warnings
from pathlib import Path

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__location__path = Path(__location__)

models_path = os.path.join(__location__, 'vocab')

if 'GPN_WORK_DIR' in os.environ:
  work_dir = os.environ['GPN_WORK_DIR']
else:
  work_dir = os.path.join(__location__path.parent, 'work')
  warnings.warn('please set GPN_WORK_DIR environment variable')

print(f'USING WORKDIR: [{work_dir}]\n set ENV GPN_WORK_DIR to override')


class HyperParameters:
  protocol_caption_max_size_words = 200

  sentence_max_len = 200
  charter_sentence_max_len = sentence_max_len

  subsidiary_name_match_min_jaro_similarity = 0.9

  confidence_epsilon = 0.001

  parser_headline_attention_vector_denominator = 0.75

  header_topic_min_confidence = 0.7

  org_level_min_confidence = 0.8

  subject_paragraph_attention_blur = 10

  charter_charity_attention_confidence = 0.6
  charter_subject_attention_confidence = 0.66

  obligations_date_pattern_threshold = 0.4
  hdbscan_cluster_proximity = 0.8

  headers_detector_use_regressor = False  ## regressor vs classifyer
