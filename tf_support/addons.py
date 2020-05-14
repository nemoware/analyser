import os

import keras.backend as K
import tensorflow as tf
from keras import activations, Model
from keras.engine import Layer

from analyser.hyperparams import models_path


class Mag(Layer):
  """
  computes magnitudes of channels vectors. Output shape is (b, x, e) --> (b, x, 1)
  """

  def __init__(self, **kwargs):
    super(Mag, self).__init__(**kwargs)
    self.supports_masking = False

  def call(self, inputs, **kwargs):
    _sum = K.sum(K.square(inputs), axis=-1)
    return K.expand_dims(K.sqrt(_sum), -1)

  def compute_output_shape(self, input_shape):
    sh = list(input_shape)[:-1] + [1]
    print('input_shape', input_shape)
    print('sh', sh)
    return tuple(sh)


def sigmoid_focal_crossentropy(
        y_true,
        y_pred,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
):
  """
  Args
      y_true: true targets tensor.
      y_pred: predictions tensor.
      alpha: balancing factor.
      gamma: modulating factor.
  Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
      same shape as `y_true`; otherwise, it is scalar.
  """
  if gamma and gamma < 0:
    raise ValueError("Value of gamma should be greater than or equal to zero")

  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

  # Get the cross_entropy for each entry
  ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

  # If logits are provided then convert the predictions into probabilities
  if from_logits:
    pred_prob = tf.sigmoid(y_pred)
  else:
    pred_prob = y_pred

  p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
  alpha_factor = 1.0
  modulating_factor = 1.0

  if alpha:
    alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

  if gamma:
    gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
    modulating_factor = tf.pow((1.0 - p_t), gamma)

  # compute the final loss and return
  return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)


def crf_nll(y_true, y_pred):
  """
  https://github.com/keras-team/keras-contrib/blob/3fc5ef709e061416f4bc8a92ca3750c824b5d2b0/keras_contrib/losses/crf_losses.py#L6
  """

  crf, idx = y_pred._keras_history[:2]
  if crf._outbound_nodes:
    raise TypeError('When learn_model="join", CRF must be the last layer.')
  if crf.sparse_target:
    y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), crf.units)
  X = crf._inbound_nodes[idx].input_tensors[0]
  mask = crf._inbound_nodes[idx].input_masks[0]
  nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
  return activations.relu(nloglik)


def init_model(model_factory_fn, model_name_override=None, weights_file_override=None, verbose=0) -> Model:
  model_name = model_factory_fn.__name__
  if model_name_override is not None:
    model_name = model_name_override

  model = model_factory_fn(model_name)
  model.name = model_name
  if verbose > 1:
    model.summary()

  ch_fn = os.path.join(models_path, model_name + ".weights")
  if weights_file_override is not None:
    ch_fn = os.path.join(models_path, weights_file_override + ".weights")

  model.load_weights(ch_fn)

  return model
