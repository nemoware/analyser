# see  /notebooks/TF_subjects.ipynb


from keras.layers import Conv1D, LSTM, Dense, Bidirectional, Input, Dropout
from keras.layers import MaxPooling1D

from analyser.hyperparams import models_path
from analyser.ml_tools import FixedVector
from analyser.structures import ContractSubject
from trainsets.trainset_tools import SubjectTrainsetManager

VALIDATION_SET_PROPORTION = 0.25

import urllib.request

from keras.models import Model

import os

EMB = 1024  # embedding dimentionality


def decode_subj_prediction(result: FixedVector) -> (ContractSubject, float, int):
  max_i = result.argmax()
  print(max_i)
  predicted_subj_name = ContractSubject(max_i)
  confidence = result[max_i]
  return predicted_subj_name, confidence, max_i


def predict_subject(model, doc) -> FixedVector:
  _embeddings_head = doc.embeddings[:1000]
  # reshaping to batch-like shape (1, ?, 1024)
  _embeddings_head = _embeddings_head.reshape((-1, _embeddings_head.shape[-2], EMB))
  print(_embeddings_head.shape)
  _predictions = model.predict(_embeddings_head)

  return _predictions[0]


def set_conv_bi_LSTM_dropouts_training_params(dataset_manager: SubjectTrainsetManager):
  dataset_manager.noisy_samples_amount = 0.75
  dataset_manager.outliers_percent = 0.1
  dataset_manager.noise_amount = 0.05


def conv_bi_LSTM_dropouts_binary(name="new_model"):
  CLASSES = 43
  input_text = Input(shape=[None, EMB], dtype='float32', name="input_text_emb")

  _out = input_text
  _out = Dropout(0.1, name="drops")(_out)
  _out = Conv1D(filters=16, kernel_size=(8), padding='same', activation='relu')(_out)
  _out = MaxPooling1D(pool_size=2)(_out)
  _out = Bidirectional(LSTM(16, return_sequences=False))(_out)
  _out = Dense(CLASSES, activation='softmax')(_out)

  model = Model(inputs=[input_text], outputs=_out, name=name)
  model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy', 'categorical_accuracy'])

  return model


def conv_biLSTM_binary_dropouts05(name="new_model"):
  CLASSES = 43
  input_text = Input(shape=[None, EMB], dtype='float32', name="input_text_emb")

  _out = input_text
  _out = Dropout(0.5, name="drops")(_out)
  _out = Conv1D(filters=16, kernel_size=(8), padding='same', activation='relu')(_out)
  _out = MaxPooling1D(pool_size=2)(_out)
  _out = Bidirectional(LSTM(16, return_sequences=False))(_out)
  _out = Dense(CLASSES, activation='softmax')(_out)

  model = Model(inputs=[input_text], outputs=_out, name=name)
  model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy', 'categorical_accuracy'])

  return model


def conv_bi_LSTM_dropouts(name="new_model") -> Model:
  '''
  Epoch 20
  loss: 0.0206 - acc: 1.0000 - val_loss: 0.3490 - val_acc: 0.9417


  see checkpoints/conv_bi_LSTM_dropouts
  :param name: name of the model
  :return: compiled TF model
  '''

  CLASSES = 43
  input_text = Input(shape=[None, EMB], dtype='float32', name="input_text_emb")

  _out = input_text
  _out = Dropout(0.1, name="drops")(_out)
  _out = Conv1D(filters=16, kernel_size=(8), padding='same', activation='relu')(_out)
  _out = MaxPooling1D(pool_size=2)(_out)
  _out = Bidirectional(LSTM(16, return_sequences=False))(_out)
  _out = Dense(CLASSES, activation='softmax')(_out)

  model = Model(inputs=[input_text], outputs=_out, name=name)
  model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy', 'categorical_accuracy'])

  return model


BEST_MODEL = conv_biLSTM_binary_dropouts05


def load_subject_detection_trained_model() -> Model:
  final_model = BEST_MODEL(BEST_MODEL.__name__ + '_final')
  cp_name = final_model.name + '.weights'
  url = f'https://github.com/nemoware/analyser/releases/download/checkpoint.0.0.1/{cp_name}'

  file_name = os.path.join(models_path, cp_name)
  if not os.path.exists(file_name):
    print(f'downloading trained NN model from {url}')

    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
      data = response.read()  # a `bytes` object
      out_file.write(data)
      print(f'NN model saved as {file_name}')

  print(f'loading NN model from {file_name}')

  final_model.load_weights(file_name)
  print(final_model.name)
  final_model.summary()

  return final_model
