from keras import Model
from keras.layers import Input, Conv1D, Dropout, LSTM, ThresholdedReLU
from keras.layers import concatenate

from analyser.headers_detector import TOKEN_FEATURES
from tf_support.addons import sigmoid_focal_crossentropy


def structure_detection_model_001(name, features=14):
  EMB = 1024
  FEATURES = features

  input_text_emb = Input(shape=[None, EMB], dtype='float32', name="input_text_emb")
  input_headlines = Input(shape=[None, TOKEN_FEATURES], dtype='float32', name="input_headlines_att")

  _out = Dropout(0.45, name="drops")(input_text_emb)  # small_drops_of_poison
  _out = concatenate([_out, input_headlines], axis=-1)
  _out = Conv1D(filters=FEATURES * 4, kernel_size=(2), padding='same', activation=None)(_out)
  _out = Conv1D(filters=FEATURES * 4, kernel_size=(4), padding='same', activation='relu', name='embedding_reduced')(
    _out)

  _out = Dropout(0.2)(_out)

  _out = LSTM(FEATURES * 4, return_sequences=True, activation="sigmoid")(_out)
  _out = LSTM(FEATURES, return_sequences=True, activation='sigmoid')(_out)
  

  model = Model(inputs=[input_text_emb, input_headlines], outputs=_out, name=name)

  model.compile(loss=sigmoid_focal_crossentropy, optimizer='Nadam', metrics=['mse', 'kullback_leibler_divergence', 'acc'])
  return model
