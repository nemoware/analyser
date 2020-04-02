import pickle
import random

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from pandas import DataFrame

from analyser.legal_docs import LegalDocument
from analyser.structures import ContractSubject

VALIDATION_SET_PROPORTION = 0.25
import urllib.request

from keras.models import load_model, Model

import os


def load_subject_detection_trained_model() -> Model:
  cp_name = 'conv_bi_LSTM_dropouts_rev.checkpoint'
  url = 'https://github.com/nemoware/analyser/releases/download/checkpoint.0.0.1/' + cp_name

  file_name = cp_name
  if not os.path.exists(file_name):
    print(f'downloading trained NN model from {url}')

    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
      data = response.read()  # a `bytes` object
      out_file.write(data)
      print(f'NN model saved as {cp_name}')

  print(f'loading NN model from {cp_name}')

  model: Model = load_model(file_name)
  model.summary()

  return model


class TrainsetBalancer:

  def __init__(self):
    pass

  def get_indices_split(self, df: DataFrame, category_column_name: str, test_proportion=VALIDATION_SET_PROPORTION) -> (
          [int], [int]):
    cat_count = df[category_column_name].value_counts()

    subject_bags = {key: [] for key in cat_count.index}

    _idx: int = 0
    for index, row in df.iterrows():
      subj_code = row[category_column_name]
      subject_bags[subj_code].append(_idx)

      _idx += 1

    _desired_number_of_samples = max(cat_count.values)
    for subj_code in subject_bags:
      bag = subject_bags[subj_code]
      if len(bag) < _desired_number_of_samples:
        repeats = int(_desired_number_of_samples / len(bag))
        bag = sorted(np.tile(bag, repeats))
        subject_bags[subj_code] = bag

    train_indices = []
    test_indices = []

    for subj_code in subject_bags:
      bag = subject_bags[subj_code]
      split_index: int = int(len(bag) * test_proportion)
      # print(split_index)

      train_indices += bag[split_index:]
      test_indices += bag[:split_index]

    return train_indices, test_indices


class SubjectTrainsetManager:
  EMB_NOISE_AMOUNT = 0.05
  OUTLIERS_PERCENT = 0.05

  def __init__(self, trainset_description_csv: str):
    self.outliers_percent = SubjectTrainsetManager.OUTLIERS_PERCENT

    self.embeddings_cache = {}
    # self.subj_count = {}

    self.csv_path = trainset_description_csv

    self.picle_relosver = None  # hack for tests

    # -------
    self.trainset_rows: DataFrame = self._read_trainset_meta()
    self.train_indices, self.test_indices = self.balance_trainset(self.trainset_rows)
    self.subject_name_1hot_map = self._encode_1_hot()

    self.number_of_classes = len(list(self.subject_name_1hot_map.values())[0])

  @staticmethod
  def balance_trainset(trainset_dataframe: DataFrame):

    tb = TrainsetBalancer()
    return tb.get_indices_split(trainset_dataframe, 'subject')

  def _read_trainset_meta(self) -> DataFrame:
    _trainset_meta = pd.read_csv(self.csv_path, encoding='utf-8')
    self._trainset_meta = _trainset_meta
    trainset_rows: DataFrame = _trainset_meta[_trainset_meta['valid']]

    self.subj_count = trainset_rows['subject'].value_counts()

    print(trainset_rows.info())
    return trainset_rows

  @staticmethod
  def _encode_1_hot():
    '''
    bit of paranoya to reserve order
    :return:
    '''
    all_subjects_map = ContractSubject.as_matrix()
    values = all_subjects_map[:, 1]

    # encoding integer subject codes in one-hot vectors
    _cats = to_categorical(values)

    subject_name_1hot_map = {all_subjects_map[i][0]: _cats[i] for i, k in enumerate(all_subjects_map)}

    return subject_name_1hot_map

  def noise_embedding(self, emb, var=0.1):
    _mean = 0
    sigma = var ** 0.5
    gauss = np.random.normal(_mean, sigma, emb.shape)
    return emb + gauss

  def _noise_amount(self, subj):
    subj_popularity = self.subj_count[subj]
    max_pop = max(self.subj_count.values)
    return 1 - subj_popularity / max_pop

  def get_embeddings_raw(self, _filename):
    # TODO:::
    filename = _filename
    if self.picle_relosver:
      filename = self.picle_relosver(_filename)

    if filename not in self.embeddings_cache:
      with open(filename, "rb") as pickle_in:
        doc: LegalDocument = pickle.load(pickle_in)
        self.embeddings_cache[filename] = doc.embeddings
        # todo: do not overload!!

    return self.embeddings_cache[filename]

  def get_embeddings(self, filename: str, subj: str, randomize=False):
    embedding = self.get_embeddings_raw(filename)

    if randomize:
      _var = SubjectTrainsetManager.EMB_NOISE_AMOUNT * self._noise_amount(subj)
      embedding = self.noise_embedding(embedding, var=_var)

    return embedding

  def make_fake_outlier(self, emb):
    label = self.subject_name_1hot_map['Other']
    _mode = np.random.choice([1, 2, 3, 4])

    # print('make_fake_outlier mode', _mode)

    if _mode == 1:
      return emb * -1, label

    elif _mode == 2:
      return self.noise_embedding(emb, 3), label

    elif _mode == 3:
      return emb * -0.5, label

    elif _mode == 4:
      return np.zeros_like(emb), label

    # return np.zeros_like(emb)

  def get_generator(self, batch_size, all_indices, randomize=False):
    while True:
      # Select files (paths/indices) for the batch
      batch_indices = np.random.choice(a=all_indices, size=batch_size)
      __split = int(self.outliers_percent * batch_size)

      batch_input = []
      batch_output = []

      # Read in each input, perform preprocessing and get labels
      for pos, i in enumerate(batch_indices):
        _row = self.trainset_rows.iloc[i]
        _subj = _row['subject']
        _filename = _row['pickle']

        _emb = self.get_embeddings(_filename, _subj, randomize=randomize)
        label = self.subject_name_1hot_map[_subj]

        if pos < __split:
          _emb, outlier_label = self.make_fake_outlier(_emb)

        batch_input.append(_emb)
        batch_output.append(label)

      # Return a tuple of (input, output) to feed the network
      # batch_x = np.array(batch_input)
      # TODO: "randomize" MAX_SEQUENCE_LENGTH
      maxlen = random.choice([700, 800, 900, 1000, 1100])
      batch_x = np.array(pad_sequences(batch_input, maxlen=maxlen, padding='post', truncating='post')).reshape(
        (batch_size, maxlen, 1024))
      batch_y = np.array(batch_output)

      yield (batch_x, batch_y)


if __name__ == '__main__':
  # print(sys.version)

  pickle_resolver = lambda f: '/Users/artem/work/nemo/goil/nlp_tools/tests/Договор _2_.docx.pickle'

  fn = '/Users/artem/Google Drive/GazpromOil/trainsets/meta_info/contracts.subjects-manually-filtered.csv'
  tsm: SubjectTrainsetManager = SubjectTrainsetManager(fn)
  tsm.picle_relosver = pickle_resolver  # hack for tests
  # for k, v in tsm.subject_name_1hot_map.items():
  #   print(k, v)
  gen = tsm.get_generator(batch_size=50, all_indices=[0, 1], randomize=True)
  x, y = next(gen)
  print('X->Y :', x.shape, '-->', y.shape)
  print("subject_bags", len(tsm.train_indices), len(tsm.test_indices))
  print("number_of_classes", tsm.number_of_classes)

  # model = load_subject_detection_trained_model()
