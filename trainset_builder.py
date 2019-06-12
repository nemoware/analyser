from typing import List

import numpy as np

from contract_agents import entities_types, find_org_names_spans
from contract_augmentation import *
from contract_parser import ContractDocument3


def categories_vector_to_onehot_matrix(vector, height=10, add_none_category=True):
  if add_none_category:
    m = np.zeros((len(vector), 1 + int(height)))

    for i in range(len(vector)):
      val = vector[i]
      #     if val != 0:
      m[i][int(val)] = 1.0


  else:
    m = np.zeros((len(vector), int(height)))

    for i in range(len(vector)):
      val = vector[i]
      if val != 0:
        m[i][int(val - 1)] = 1.0

  return m


def _extend_trainset_with_obfuscated_contracts(parsed: List[ContractDocument3], n=10):
  _parsed = []

  for _pdoc in parsed:

    _parsed.append(_pdoc)  # take original

    for i in range(n):
      try:
        o_doc = obfuscate_contract(_pdoc)
        _parsed.append(o_doc)
      except:
        pass

  return _parsed


def validate_find_patterns_in_contract_results(d):
  assert len(d) == 2


def preprocess_contract(txt) -> ContractDocument3:
  trimmed = txt  # txt[0:1500]
  # normalized_contract = normalize_contract(trimmed)
  # normalized_contract = re.sub(r'\n{2,}', '\n\n', normalized_contract)
  _pdoc = ContractDocument3(trimmed)
  _pdoc.parse()
  return _pdoc


def parse_contracts(contracts, data) -> (List[ContractDocument3], List[ContractDocument3]):
  _parsed = []
  _failed = []
  for fn in data:
    _doc: ContractDocument3 = preprocess_contract(contracts[fn])
    _doc.filename = fn
    try:
      find_org_names_spans(_doc)
      # doc.agent_infos = find_org_names(doc.normal_text)
      validate_find_patterns_in_contract_results(_doc.agent_infos)
      _parsed.append(_doc)

    except:
      print(f'failed parsing: {fn}')
      _failed.append(_doc)

  return _parsed, _failed


# def _convert_char_slices_to_tokens(doc: ContractDocument3):
#   for org in doc.agent_infos:
#     for ent in org:
#       span = org[ent][1]
#
#       if span[0] > 0:
#         tokens_slice = tokens_in_range(span, doc.tokens_cc, doc.normal_text)
#         org[ent] = (org[ent][0], org[ent][1], tokens_slice)
#       else:
#         org[ent] = (org[ent][0], None, None)


def make_categories_vector(_pdoc:ContractDocument3) -> np.ndarray:
  vector = np.zeros(_pdoc.get_len())

  e = 1
  for agent_n in range(2):
    org = _pdoc.agent_infos[agent_n]
    for entity_type in entities_types:
      if entity_type in org:
        text_slice = org[entity_type][2]

        if text_slice is not None:
          vector[text_slice] = e

      e += 1

  return vector


def mark_headlines(_pdoc, destination, value):
  # headlines_markup_vector = np.zeros( doc.get_len()  )

  for hi in _pdoc.structure.headline_indexes:
    s = _pdoc.structure.structure[hi]
    sl = slice(s.span[0], s.span[1])
    destination[sl] = value

  # return headlines_markup_vector


def to_categories_vector(_pdoc: ContractDocument3):
  categories_vector = make_categories_vector(_pdoc)
  mx = categories_vector.max()
  mark_headlines(_pdoc, categories_vector, mx + 1)

  return categories_vector


NUM_AUGMENTED = 20


def prepare_train_data(contracts, add_none_category=False):
  data = list(contracts.keys())

  # 1. parse available docs with regex
  _parsed, _failed = parse_contracts(contracts, data)

  print(f'Extending trainset with obfuscated contracts;  docs: {len(_parsed)}')
  _parsed: List[ContractDocument3] = _extend_trainset_with_obfuscated_contracts(_parsed)

  TOKENS = []
  vectors = []

  print(f'Augmenting trainset; docs: {len(_parsed)}')
  for pdoc in _parsed:

    categories_vector = to_categories_vector(pdoc)
    vectors.append(categories_vector)
    TOKENS.append(pdoc.tokens_cc)

    for i in range(NUM_AUGMENTED):
      new_tokens, new_categories_vector = augment_contract(pdoc.tokens_cc, categories_vector)
      vectors.append(new_categories_vector)
      TOKENS.append(new_tokens)

  LABELS = []
  LENS = [len(x) for x in TOKENS]
  _maxlen = max(LENS)

  for i in range(len(TOKENS)):
    padding = (_maxlen - len(TOKENS[i]))
    TOKENS[i] = TOKENS[i] + ['PAD'] * padding

    v_padded = np.concatenate([vectors[i], [0] * padding])

    m = categories_vector_to_onehot_matrix(v_padded, height=v_padded.max() + 1, add_none_category=add_none_category)
    LABELS.append(m)

  return TOKENS, np.array(LABELS), LENS
