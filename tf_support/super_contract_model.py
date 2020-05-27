from keras import Model
from keras.layers import Input, Conv1D, Dropout, LSTM, Bidirectional, Dense, MaxPooling1D, Activation
from keras.layers import concatenate

from analyser.headers_detector import TOKEN_FEATURES
from tf_support.addons import sigmoid_focal_crossentropy
from tf_support.tools import KerasTrainingContext

seq_labels_contract_level_0 = [
  'date', 'number',
  'org-1-name', 'org-1-type', 'org-1-alias',
  'org-2-name', 'org-2-type', 'org-2-alias',
  'value', 'currency', 'sign'
]

seq_labels_contract_level_1 = [
  'headline_h1',
  'subject',
  '_reserved'
]
metrics = ['kullback_leibler_divergence', 'mse', 'binary_crossentropy']
seq_labels_contract = seq_labels_contract_level_1 + seq_labels_contract_level_0

DEFAULT_TRAIN_CTX = KerasTrainingContext()


def structure_detection_model_001(name, features=14, embeddings_dim=1024,
                                  ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX):
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


def get_base_model(factory, ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX):
  model_001 = ctx.init_model(factory, trained=True, verbose=1)

  # BASE
  base_model = model_001.get_layer(name='embedding_reduced').output
  in1 = model_001.get_layer(name='input_text_emb').input
  in2 = model_001.get_layer(name='input_headlines_att').input

  return base_model, [in1, in2]


def uber_detection_model_001(name, ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX):
  """
  Evaluation:
  > 0.0030140 	loss
  > 0.0100294 	O1_tagging_loss
  > 0.0059756 	O2_subject_loss
  > 0.0309147 	O1_tagging_kullback_leibler_divergence

  :param name:
  :return:
  """

  base_model, base_model_inputs = get_base_model(structure_detection_model_001, ctx=ctx)

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
  model = Model(inputs=base_model_inputs, outputs=[_out, _out2], name=name)
  model.compile(loss=losses, optimizer='adam', metrics=metrics)
  return model


def uber_detection_model_003(name, ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX) -> Model:
  # BASE

  base_model, base_model_inputs = get_base_model(structure_detection_model_001, ctx=ctx)
  # ---------------------

  _out_d = Dropout(0.5, name='alzheimer')(base_model)  # small_drops_of_poison
  _out = LSTM(FEATURES * 4, return_sequences=True, activation="sigmoid", name='paranoia')(_out_d)
  _out = LSTM(FEATURES, return_sequences=True, activation='sigmoid', name='O1_tagging')(_out)

  # OUT 2: subject detection
  #
  pool_size = 2
  _out2 = MaxPooling1D(pool_size=pool_size, name='emotions')(_out_d)
  _out_mp = MaxPooling1D(pool_size=pool_size, name='insights')(_out)
  _out2 = concatenate([_out2, _out_mp], axis=-1, name='bipolar_disorder')
  _out2 = Dropout(0.3, name='alzheimer_3')(_out2)
  _out2 = Bidirectional(LSTM(16, return_sequences=False, name='narcissisism'), name='self_reflection')(_out2)

  _out2 = Dense(CLASSES, activation='softmax', name='O2_subject')(_out2)

  losses = {
    "O1_tagging": sigmoid_focal_crossentropy,
    "O2_subject": "binary_crossentropy",
  }
  model = Model(inputs=base_model_inputs, outputs=[_out, _out2], name=name)
  model.compile(loss=losses, optimizer='adam', metrics=metrics)
  return model


CLASSES = 43
FEATURES = 14


def uber_detection_model_005_1_1(name, ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX):
  base_model, base_model_inputs = get_base_model(uber_detection_model_003, ctx=ctx)

  # ---------------------

  _out_d = Dropout(0.15, name='alzheimer')(base_model)  # small_drops_of_poison
  _out = Bidirectional(LSTM(FEATURES * 2, return_sequences=True, name='paranoia'), name='self_reflection_1')(_out_d)
  _out = Dropout(0.1, name='alzheimer_11')(_out)
  _out = LSTM(FEATURES, return_sequences=True, activation='sigmoid', name='O1_tagging')(_out)

  # OUT 2: subject detection
  #
  pool_size = 2
  _out2 = MaxPooling1D(pool_size=pool_size, name='emotions')(_out_d)
  _out_mp = MaxPooling1D(pool_size=pool_size, name='insights')(_out)
  _out2 = concatenate([_out2, _out_mp], axis=-1, name='bipolar_disorder')
  _out2 = Dropout(0.3, name='alzheimer_3')(_out2)
  _out2 = Bidirectional(LSTM(16, return_sequences=False, name='narcissisism'), name='self_reflection_2')(_out2)
  _out2 = Dropout(0.1, name='alzheimer_1')(_out2)

  _out2 = Dense(CLASSES, activation='softmax', name='O2_subject')(_out2)

  losses = {
    "O1_tagging": 'binary_crossentropy',
    "O2_subject": "binary_crossentropy",
  }
  model = Model(inputs=base_model_inputs, outputs=[_out, _out2], name=name)
  model.compile(loss=losses, optimizer='Nadam', metrics=metrics)
  return model


def uber_detection_model_005_3(name, ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX):
  base_model, base_model_inputs = get_base_model(uber_detection_model_003, ctx=ctx)

  # ---------------------

  _out_d = Dropout(0.15, name='alzheimer')(base_model)  # small_drops_of_poison

  _out = Bidirectional(LSTM(FEATURES * 2, return_sequences=True, name='paranoia'), name='self_reflection_1')(_out_d)
  _out = Dropout(0.1, name='alzheimer_11')(_out)
  _out = LSTM(FEATURES, return_sequences=True, activation='sigmoid', name='O1_tagging')(_out)

  # OUT 2: subject detection
  #
  pool_size = 2
  _base_emb_act = Activation('tanh')(_out_d)
  emotions = MaxPooling1D(pool_size=pool_size, name='emotions')(_base_emb_act)
  insights = MaxPooling1D(pool_size=pool_size, name='insights')(_out)
  _out2 = concatenate([emotions, insights], axis=-1, name='bipolar_disorder')

  _out2 = Dropout(0.3, name='alzheimer_3')(_out2)
  _out2 = Bidirectional(LSTM(16, return_sequences=False, name='narcissisism'), name='self_reflection_2')(_out2)
  _out2 = Dropout(0.1, name='alzheimer_1')(_out2)

  _out2 = Dense(CLASSES, activation='softmax', name='O2_subject')(_out2)

  losses = {
    "O1_tagging": "binary_crossentropy",
    "O2_subject": "binary_crossentropy",
  }
  model = Model(inputs=base_model_inputs, outputs=[_out, _out2], name=name)
  model.compile(loss=losses, optimizer='adam', metrics=metrics)
  return model


if __name__ == '__main__':
  ctx = KerasTrainingContext()
  ctx.init_model(uber_detection_model_005_1_1, verbose=2, trained=True, trainable=False)
