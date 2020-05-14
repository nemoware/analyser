import os

import keras.backend as K
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau


def get_lr_epoch_from_log(model_name, log_path) -> (float, int):
  log_csv = os.path.join(log_path, model_name + '.log.csv')
  try:
    _log = pd.read_csv(log_csv)
    lr = float(_log.iloc[-1]['lr'])
    epoch = int(_log.iloc[-1]['epoch'])
    epoch = max(epoch, len(_log))
  except:
    print(f'log is not available {log_csv}')
    lr = None
    epoch = 0

  return lr, epoch


class KerasTrainingContext:

  def __init__(self, checkpoints_path):
    self.HISTORIES = []
    self.model_checkpoint_path = checkpoints_path
    self.EVALUATE_ONLY = True
    self.EPOCHS = 18

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

  def train_and_evaluate_model(self, model, generator, test_generator):
    if self.EVALUATE_ONLY:
      print(f'training skipped EVALUATE_ONLY = {self.EVALUATE_ONLY}')
      return

    print(f'model.name == {model.name}')

    _logger1 = CSVLogger(os.path.join(self.model_checkpoint_path, model.name + '.log.csv'), separator=',', append=True)
    _logger2 = CSVLogger(model.name + '.log.csv', separator=',', append=True)

    # checkpoint = ModelCheckpoint(os.path.join(self.model_checkpoint_path, model.name),
    #                              monitor='val_loss', mode='min', save_best_only=True,
    #                              verbose=1)

    checkpoint_weights = ModelCheckpoint(os.path.join(self.model_checkpoint_path, model.name + ".weights"),
                                         monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True,
                                         verbose=1)

    lr, epoch = get_lr_epoch_from_log(model.name, self.model_checkpoint_path)
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
