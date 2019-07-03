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


def _extend_trainset_with_obfuscated_contracts(parsed: List[ContractDocument3], n=10, include_originals=True):
  _parsed = []

  for _pdoc in parsed:
    if include_originals:
      _parsed.append(_pdoc)  # take original

    for i in range(n):
      try:
        o_doc = obfuscate_contract(_pdoc, rate=1)
        _parsed.append(o_doc)
      except:
        pass

  return _parsed


def validate_find_patterns_in_contract_results(d):
  assert len(d) == 2


def preprocess_contract(txt, trim=0) -> ContractDocument3:
  trimmed = txt  # txt[0:1500]
  if trim > 0:
    trimmed = txt[0:trim]
  # normalized_contract = normalize_contract(trimmed)
  # normalized_contract = re.sub(r'\n{2,}', '\n\n', normalized_contract)
  _pdoc = ContractDocument3(trimmed)
  _pdoc.parse()
  return _pdoc


def parse_contracts(contracts, filenames, trim=0) -> (List[ContractDocument3], List[ContractDocument3]):
  _parsed = []
  _failed = []

  for fn in filenames:
    _doc: ContractDocument3 = preprocess_contract(contracts[fn], trim)
    _doc.filename = fn
    try:
      find_org_names_spans(_doc)
      # doc.agent_infos = find_org_names(doc.normal_text)
      validate_find_patterns_in_contract_results(_doc.agent_infos)
      _parsed.append(_doc)

    except:
      print(f'failed parsing: {fn}')
      # _doc: ContractDocument3 = preprocess_contract(contracts[fn], 0)
      _failed.append(_doc)

  return _parsed, _failed


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


def prepare_train_data(contracts, augmenented_n=5, obfuscated_n=3, trim=0, include_originals=True):
  vectors, _tokenized_texts, _failed = mark_contracts(contracts, augmenented_n, obfuscated_n, trim, include_originals)
  _tokenized_texts, _lengths, _padded_vectors = add_padding_to_max(_tokenized_texts, vectors)
  _labels = categories_vectors_to_onehot_matrices(_padded_vectors, 12)

  return _tokenized_texts, _labels, _lengths, _failed


def categories_vectors_to_onehot_matrices(vectors, matrix_heights):
  n_items = len(vectors)
  _labels = np.zeros(shape=(n_items, len(vectors[0]), matrix_heights), dtype=np.uint8)
  for i in range(n_items):
    _labels[i, :, :] = categories_vector_to_onehot_matrix(vectors[i], height=matrix_heights - 1,
                                                          add_none_category=True)
  return _labels


def add_padding_to_max(_tokenized_texts, mark_up_vectors=None, pad_symbol='PAD'):
  if mark_up_vectors is not None:
    assert len(_tokenized_texts) == len(mark_up_vectors)

  _lengths = [len(x) for x in _tokenized_texts]
  print('_lengths=', _lengths)
  _maxlen = max(_lengths)
  _padded_vectors = []

  for i in range(len(_tokenized_texts)):
    padding = (_maxlen - len(_tokenized_texts[i]))

    _tokenized_texts[i] = _tokenized_texts[i] + [pad_symbol] * padding

    if mark_up_vectors is not None:
      v_padded = np.concatenate([mark_up_vectors[i], [0] * padding])
      _padded_vectors.append(v_padded)

  return _tokenized_texts, _lengths, _padded_vectors


def random_widow(windowlen, textlen) -> slice:
  start = random.randint(0, textlen - windowlen)
  return slice(start, start + windowlen)


# ////


def mark_contracts(contracts, augmenented_n=5, obfuscated_n=3, trim=0, include_originals=True):
  data = list(contracts.keys())

  # 1. parse available docs with regex
  _parsed, _failed = parse_contracts(contracts, data, trim)

  print(f'Extending trainset with obfuscated contracts;  docs: {len(_parsed)}')
  _parsed: List[ContractDocument3] = _extend_trainset_with_obfuscated_contracts(_parsed, n=obfuscated_n,
                                                                                include_originals=include_originals)

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

  return vectors, _tokenized_texts, _failed



def random_samples(num, size, __vectors, __tokenized_texts):
  well_sized = []
  well_sized_vectors = []
  for n in range(len(__tokenized_texts)):
    if len(__tokenized_texts[n]) >= size:
      well_sized.append(__tokenized_texts[n])
      well_sized_vectors.append(__vectors[n])

  pieces = []
  vectors = []

  for n in range(num):
    random_index = random.randint(0, len(well_sized) - 1)
    txt = well_sized[random_index]
    vec = well_sized_vectors[random_index]

    wnd = random_widow(size, len(txt) - 1)

    piece = txt[wnd]
    vector = vec[wnd]

    vectors.append(vector)
    pieces.append(piece)

  pieces, _lengths, _padded_vectors = add_padding_to_max(pieces, vectors)
  _labels = categories_vectors_to_onehot_matrices(_padded_vectors, 12)

  return pieces, _labels, _lengths


def trim_vectors_to_size(_vectors: List, size: int):
  _vectors_trimmed = []

  for n in range(len(_vectors)):
    _vectors_trimmed.append(_vectors[n][0:size])
  return _vectors_trimmed


def prepare_train_data_pieces(num, size, contracts, augmenented_n=5, obfuscated_n=3, include_originals=True):
  _vectors, _tokenized_texts, _failed = mark_contracts(contracts, augmenented_n, obfuscated_n, trim=0,
                                                       include_originals=include_originals)

  print('number of augmened contracts =', len(_tokenized_texts))

  _vectors_trimmed = trim_vectors_to_size(_vectors, size * 2)
  _tokenized_texts_trimmed = trim_vectors_to_size(_tokenized_texts, size * 2)

  pieces, _labels, _lengths = random_samples(num, size, _vectors_trimmed, _tokenized_texts_trimmed)

  return pieces, _labels, _lengths, _failed
