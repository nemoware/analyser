from typing import List

import numpy as np

from contract_agents import entities_types
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


def parse_contracts(contracts, filenames) -> (List[ContractDocument3], List[ContractDocument3]):
  _parsed = []
  _failed = []
  for fn in filenames:
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


def _make_categories_vector(_doc: ContractDocument3) -> np.ndarray:
  vector = np.zeros(_doc.get_len())

  e = 1
  for agent_n in range(2):
    org = _doc.agent_infos[agent_n]
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


def _to_categories_vector(_pdoc: ContractDocument3, headlines_index=11):
  categories_vector = _make_categories_vector(_pdoc)
  #   mx = categories_vector.max()
  mark_headlines(_pdoc, categories_vector, headlines_index)

  return categories_vector


def prepare_train_data__(contracts, augmenented_n=5, obfuscated_n=3):
  data = list(contracts.keys())

  # 1. parse available docs with regex
  _parsed, _failed = parse_contracts(contracts, data)

  print(f'Extending trainset with obfuscated contracts;  docs: {len(_parsed)}')
  _parsed: List[ContractDocument3] = _extend_trainset_with_obfuscated_contracts(_parsed, n=obfuscated_n)

  _tokenized_texts = []
  vectors = []

  print(f'Augmenting trainset; docs: {len(_parsed)}')
  for pdoc in _parsed:

    categories_vector = _to_categories_vector(pdoc)
    vectors.append(categories_vector)
    _tokenized_texts.append(pdoc.tokens_cc)

    for i in range(augmenented_n):
      new_tokens, new_categories_vector = augment_contract(pdoc.tokens_cc, categories_vector)
      vectors.append(new_categories_vector)
      _tokenized_texts.append(new_tokens)

  n_items = len(_tokenized_texts)

  _lengths = [len(x) for x in _tokenized_texts]
  _maxlen = max(_lengths)
  cat_height = 12

  _labels = np.zeros(shape=(n_items, _maxlen, cat_height), dtype=np.uint8)

  for i in range(n_items):
    padding = (_maxlen - len(_tokenized_texts[i]))
    _tokenized_texts[i] = _tokenized_texts[i] + ['PAD'] * padding

    v_padded = np.concatenate([vectors[i], [0] * padding])

    m = categories_vector_to_onehot_matrix(v_padded, height=11, add_none_category=True)

    _labels[i, :, :] = m

  return _tokenized_texts, _labels, _lengths, _failed



prepare_train_data = prepare_train_data__