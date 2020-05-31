# from tensorflow_docs import plots

import pickle

import pandas as pd

from analyser.legal_docs import LegalDocument


def load_doc_row(row: pd.Series) -> LegalDocument:
  _fn = row['pickle']
  with open(_fn, "rb") as pickle_in:
    _doc: LegalDocument = pickle.load(pickle_in)
    return _doc


def load_doc(index_in_trainset: int):
  row = dataset_manager.trainset_rows.iloc[index_in_trainset]
  return load_doc_row(row)


def load_doc(row: pd.Series) -> LegalDocument:
  _fn = row['pickle']
  with open(_fn, "rb") as pickle_in:
    _doc: LegalDocument = pickle.load(pickle_in)
    return _doc


# plotter = plots.HistoryPlotter(smoothing_std=2)
def make_xy_by_row(row: pd.Series) -> ():
  d = load_doc_row(row)
  subj_name = row['subject']

  subj = dataset_manager.subject_name_1hot_map[subj_name]

  semantic_map = get_semantic_labels(d)
  semantic_map = np.clip(semantic_map, 0, 1)

  token_features = get_tokens_features(d.tokens)
  token_features['h'] = make_predicted_headline_attention_vector(d, False)

  return ((d.embeddings, token_features.values), (semantic_map, subj))


def make_xy_subj(i: int):
  row = dataset_manager.trainset_rows.iloc[i]
  return make_xy_by_row(row)
