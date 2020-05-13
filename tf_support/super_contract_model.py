from keras import Model
from keras.layers import Input, Conv1D, Dropout, LSTM, Bidirectional, Dense, MaxPooling1D
from keras.layers import concatenate

from analyser.headers_detector import TOKEN_FEATURES
from tf_support.addons import sigmoid_focal_crossentropy, init_model


def structure_detection_model_001(name, features=14, embeddings_dim=1024):
  EMB = embeddings_dim
  FEATURES = features

  input_text_emb = Input(shape=[None, EMB], dtype='float32', name="input_text_emb")
  input_headlines = Input(shape=[None, TOKEN_FEATURES], dtype='float32', name="input_headlines_att")

  _out = Dropout(0.45, name="drops")(input_text_emb)  # small_drops_of_poison
  _out = concatenate([_out, input_headlines], axis=-1)
  _out = Conv1D(filters=FEATURES * 4, kernel_size=(2), padding='same', activation=None)(_out)
  _out = Conv1D(filters=FEATURES * 4, kernel_size=(4), padding='same', activation='relu', name='embedding_reduced')(
    _out)

  _out = Dropout(0.15)(_out)

  _out = LSTM(FEATURES * 4, return_sequences=True, activation="sigmoid")(_out)
  _out = LSTM(FEATURES, return_sequences=True, activation='sigmoid')(_out)

  model = Model(inputs=[input_text_emb, input_headlines], outputs=_out, name=name)

  model.compile(loss=sigmoid_focal_crossentropy, optimizer='Nadam',
                metrics=['mse', 'kullback_leibler_divergence', 'acc'])
  return model


def uber_detection_model_001(name):
  CLASSES = 43
  FEATURES = 14

  model_001 = init_model(structure_detection_model_001)

  # BASE
  base_model = model_001.get_layer(name='embedding_reduced').output
  in1 = model_001.get_layer(name='input_text_emb').input
  in2 = model_001.get_layer(name='input_headlines_att').input
  # ---------------------

  _out_d = Dropout(0.1, name='alzheimer')(base_model)  # small_drops_of_poison
  _out = LSTM(FEATURES * 4, return_sequences=True, activation="sigmoid", name='paranoia')(_out_d)
  _out = LSTM(FEATURES, return_sequences=True, activation='sigmoid', name='O1_tagging')(_out)

  # OUT 2: subject detection
  #
  pool_size = 2
  _out2 = MaxPooling1D(pool_size=pool_size, name='emotions')(_out_d)
  _out_mp = MaxPooling1D(pool_size=pool_size, name='insights')(_out)
  _out2 = concatenate([_out2, _out_mp], axis=-1, name='bipolar_disorder')
  _out2 = Bidirectional(LSTM(16, return_sequences=False, name='narcissism'), name='self_reflection')(_out2)

  _out2 = Dense(CLASSES, activation='softmax', name='O2_subject')(_out2)

  losses = {
    "O1_tagging": sigmoid_focal_crossentropy,
    "O2_subject": "binary_crossentropy",
  }
  model = Model(inputs=[in1, in2], outputs=[_out, _out2], name=name)
  model.compile(loss=losses, optimizer='adam', metrics=['kullback_leibler_divergence', 'mse', 'binary_crossentropy'])
  return model


if __name__ == '__main__':
  init_model(uber_detection_model_001, verbose=2)
