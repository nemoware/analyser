# see  /notebooks/TF_subjects.ipynb


from keras.layers import Conv1D, LSTM, Dense, Bidirectional, Input, Dropout
from keras.layers import MaxPooling1D
from keras.models import Model

from analyser.structures import ContractSubject

EMB = 1024  # embedding dimentionality
CLASSES = len(ContractSubject)


def conv_bi_LSTM_dropouts(name="new_model"):
  '''
  Epoch 20
  loss: 0.0206 - acc: 1.0000 - val_loss: 0.3490 - val_acc: 0.9417


  see checkpoints/conv_bi_LSTM_dropouts
  :param name: name of the model
  :return: compiled TF model
  '''
  input_text = Input(shape=[None, EMB], dtype='float32', name="input_text_emb")

  _out = input_text
  _out = Dropout(0.1, name="drops")(_out)
  _out = Conv1D(filters=16, kernel_size=(8), padding='same', activation='relu')(_out)
  _out = MaxPooling1D(pool_size=2)(_out)
  _out = Bidirectional(LSTM(16, return_sequences=False))(_out)

  _out = Dense(CLASSES, activation='softmax')(_out)

  model = Model(inputs=[input_text], outputs=_out, name=name)
  model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

  return model
