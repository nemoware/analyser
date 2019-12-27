import time
from functools import wraps
from typing import List

import numpy as np

from analyser.contract_patterns import ContractPatternFactory
from analyser.hyperparams import HyperParameters
from analyser.legal_docs import LegalDocument, ContractValue, extract_sum_sign_currency
from analyser.ml_tools import estimate_confidence_by_mean_top_non_zeros, FixedVector, smooth_safe
from analyser.structures import ContractSubject
from analyser.transaction_values import complete_re as transaction_values_re

PROF_DATA = {}


class ParsingSimpleContext:
  def __init__(self):

    # ---
    self.verbosity_level = 2
    self.__step = 0

    self.warnings = []

  def _reset_context(self):
    self.warnings = []
    self.__step = 0

  def _logstep(self, name: str) -> None:
    s = self.__step
    print(f'❤️ ACCOMPLISHED: \t {s}.\t {name}')
    self.__step += 1

  def warning(self, text):
    t_ = '⚠️ WARNING: - ' + text
    self.warnings.append(t_)
    print(t_)

  def get_warings(self):
    return '\n'.join(self.warnings)

  def log_warnings(self):
    if len(self.warnings) > 0:
      print("Recent parsing warnings:")

      for w in self.warnings:
        print('\t\t', w)


class AuditContext:

  def __init__(self):
    self.audit_subsidiary_name: str = None


class ParsingContext(ParsingSimpleContext):
  def __init__(self, embedder=None):
    ParsingSimpleContext.__init__(self)
    self.embedder = embedder

  def init_embedders(self, embedder, elmo_embedder_default):
    raise NotImplementedError()

  def find_org_date_number(self, document: LegalDocument, ctx: AuditContext) -> LegalDocument:
    """
    phase 1, before embedding TF, GPU, and things
    searching for attributes required for filtering
    :param charter:
    :return:
    """
    raise NotImplementedError()

  def find_attributes(self, document: LegalDocument, ctx: AuditContext):
    raise NotImplementedError()


def profile(fn):
  @wraps(fn)
  @wraps(fn)
  def with_profiling(*args, **kwargs):
    start_time = time.time()

    ret = fn(*args, **kwargs)

    elapsed_time = time.time() - start_time

    if fn.__name__ not in PROF_DATA:
      PROF_DATA[fn.__name__] = [0, []]
    PROF_DATA[fn.__name__][0] += 1
    PROF_DATA[fn.__name__][1].append(elapsed_time)

    return ret

  return with_profiling


def print_prof_data():
  for fname, data in PROF_DATA.items():
    max_time = max(data[1])
    avg_time = sum(data[1]) / len(data[1])
    print("Function {} called {} times. ".format(fname, data[0]))
    print('Execution time max: {:.4f}, average: {:.4f}'.format(max_time, avg_time))


def clear_prof_data():
  global PROF_DATA
  PROF_DATA = {}


head_types_dict = {'head.directors': 'Совет директоров',
                   'head.all': 'Общее собрание участников/акционеров',
                   'head.gen': 'Генеральный директор',
                   #                      'shareholders':'Общее собрание акционеров',
                   'head.pravlenie': 'Правление общества',
                   'head.unknown': '*Неизвестный орган управления*'}
head_types = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie']

known_subjects = [
  ContractSubject.Charity,
  ContractSubject.RealEstate,
  ContractSubject.Lawsuit]


def find_value_sign_currency(value_section_subdoc: LegalDocument,
                             factory: ContractPatternFactory = None) -> List[ContractValue]:
  if factory is not None:
    value_section_subdoc.calculate_distances_per_pattern(factory)
    vectors = factory.make_contract_value_attention_vectors(value_section_subdoc)
    # merge dictionaries of attention vectors
    value_section_subdoc.distances_per_pattern_dict = {**value_section_subdoc.distances_per_pattern_dict, **vectors}

    attention_vector_tuned = 'value_attention_vector_tuned'
  else:
    # HATI-HATI: this case is for Unit Testing only
    attention_vector_tuned = None

  return find_value_sign_currency_attention(value_section_subdoc, attention_vector_tuned, absolute_spans=True)


def find_value_sign_currency_attention(value_section_subdoc: LegalDocument,
                                       attention_vector_name: str = None,
                                       parent_tag=None,
                                       absolute_spans=False) -> List[ContractValue]:

  attention_vector_tuned = None
  if attention_vector_name is not None:
    attention_vector_tuned = value_section_subdoc.distances_per_pattern_dict[attention_vector_name]

  spans = [m for m in value_section_subdoc.tokens_map.finditer(transaction_values_re)]
  values_list = []

  for span in spans:
    value_sign_currency = extract_sum_sign_currency(value_section_subdoc, span)
    if value_sign_currency is not None:

      # Estimating confidence by looking at attention vector
      if attention_vector_tuned is not None:

        for t in value_sign_currency.as_list():
          t.confidence *= (HyperParameters.confidence_epsilon + estimate_confidence_by_mean_top_non_zeros(
            attention_vector_tuned[t.slice]))
      # ---end if

      value_sign_currency.parent.set_parent_tag(parent_tag)
      value_sign_currency.parent.span = value_sign_currency.span()  ##fix span
      values_list.append(value_sign_currency)

  # offsetting
  if absolute_spans:  # TODO: do not offset here!!!!
    for value in values_list:
      value += value_section_subdoc.start

  return values_list


def _find_most_relevant_paragraph(section: LegalDocument,
                                  attention_vector: FixedVector,
                                  min_len: int,
                                  return_delimiters=True):
  _blur = HyperParameters.subject_paragraph_attention_blur
  _padding = _blur * 2 + 1

  paragraph_attention_vector = smooth_safe(np.pad(attention_vector, _padding, mode='constant'), _blur)[
                               _padding:-_padding]

  top_index = int(np.argmax(paragraph_attention_vector))
  span = section.sentence_at_index(top_index)
  if min_len is not None and span[1] - span[0] < min_len:
    next_span = section.sentence_at_index(span[1] + 1, return_delimiters)
    span = (span[0], next_span[1])

  # confidence = paragraph_attention_vector[top_index]
  confidence_region = attention_vector[span[0]:span[1]]
  confidence = estimate_confidence_by_mean_top_non_zeros(confidence_region)
  return span, confidence, paragraph_attention_vector
