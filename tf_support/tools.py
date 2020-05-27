import os
import warnings

import keras.backend as K
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from analyser.hyperparams import models_path


class KerasTrainingContext:

  def __init__(self, checkpoints_path=models_path, session_index=0):
    self.session_index = session_index
    self.HISTORIES = {}
    self.model_checkpoint_path = checkpoints_path
    self.EVALUATE_ONLY = True
    self.EPOCHS = 18
    self.trained_models = {}
    self.validation_steps = 1
    self.steps_per_epoch = 1
    self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1E-6, verbose=1)

  def set_batch_size_and_trainset_size(self, batch_size, test_samples, train_samples):
    self.validation_steps = 1 + int(test_samples / batch_size)
    self.steps_per_epoch = 1 + int(train_samples / batch_size / 2)

  def get_stats_df(self):
    stats_path = os.path.join(self.model_checkpoint_path, 'train_statistics0.csv')

    try:
      stats = pd.read_csv(stats_path, index_col='model_name')
    except:
      print(f'🦖🦖🦖cannot read {stats_path}')
      stats = pd.DataFrame(columns=['model_name', 'epoch', 'val_acc', 'val_loss', 'loss', 'acc']).set_index(
        'model_name')

    stats.to_csv(stats_path)
    return stats, stats_path

  def save_stats(self, model_name):
    h = self.HISTORIES[model_name]
    stats, stats_path = self.get_stats_df()

    for m in h.params['metrics']:
      if m not in stats:
        stats[m] = float('nan')
    for m in h.params['metrics']:
      stats.at[model_name, m] = h.history[m][-1]

    stats.at[model_name, 'epoch'] = h.epoch[-1]

    stats.to_csv(stats_path)
    stats.to_csv('stats.csv')
    return stats

  def get_log(self, model_name) -> pd.DataFrame:
    _log_fn = f'{model_name}.{self.session_index}.log.csv'
    log_csv = os.path.join(self.model_checkpoint_path, _log_fn)
    try:
      return pd.read_csv(log_csv)
    except:
      print(f'log is not available {log_csv}')

  def get_lr_epoch_from_log(self, model_name) -> (float, int):
    _log = self.get_log(model_name)
    if _log is not None:
      lr = float(_log.iloc[-1]['lr'])
      epoch = int(_log.iloc[-1]['epoch'])
      epoch = max(epoch, len(_log)) + 1
      return lr, epoch
    else:
      return None, 0

  def init_model(self, model_factory_fn, model_name_override=None, weights_file_override=None,
                 verbose=0,
                 trainable=True, trained=False):
    model_name = model_factory_fn.__name__
    if model_name_override is not None:
      model_name = model_name_override

    model = model_factory_fn(name=model_name, ctx=self)
    model.name = model_name
    if verbose > 1:
      model.summary()

    ch_fn = os.path.join(self.model_checkpoint_path, model_name + ".weights")
    if weights_file_override is not None:
      ch_fn = os.path.join(self.model_checkpoint_path, weights_file_override + ".weights")

    try:
      model.load_weights(ch_fn)
    except:
      msg=f'cannot load  {model_name} from  {ch_fn}'
      warnings.warn(msg)
      if trained:
        raise FileExistsError(msg)

    if not trainable:
      model.trainable = False
      for l in model.layers:
        l.trainable = False

    return model

  def train_and_evaluate_model(self, model, generator, test_generator):
    print(f'model.name == {model.name}')
    self.trained_models[model.name] = model.name
    if self.EVALUATE_ONLY:
      print(f'training skipped EVALUATE_ONLY = {self.EVALUATE_ONLY}')
      return



    _log_fn = f'{model.name}.{self.session_index}.log.csv'
    _logger1 = CSVLogger(os.path.join(self.model_checkpoint_path, _log_fn), separator=',', append=True)
    _logger2 = CSVLogger(_log_fn, separator=',', append=True)

    # checkpoint = ModelCheckpoint(os.path.join(self.model_checkpoint_path, model.name),
    #                              monitor='val_loss', mode='min', save_best_only=True,
    #                              verbose=1)

    checkpoint_weights = ModelCheckpoint(os.path.join(self.model_checkpoint_path, model.name + ".weights"),
                                         monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True,
                                         verbose=1)

    lr, epoch = self.get_lr_epoch_from_log(model.name)
    print(f'continue: lr:{lr}, epoch:{epoch}')

    if lr is not None:
      K.set_value(model.optimizer.lr, lr)


    history = model.fit_generator(
      generator=generator,
      epochs=self.EPOCHS,
      callbacks=[self.reduce_lr, checkpoint_weights, _logger2, _logger1],  # , _logger1, _logger2
      steps_per_epoch=self.steps_per_epoch,
      validation_data=test_generator,
      validation_steps=self.validation_steps,
      initial_epoch=epoch)

    self.HISTORIES[model.name] = history
    self.save_stats(model.name)

    return history

    # plot_training_history(history)
    # plot_compare_models()
