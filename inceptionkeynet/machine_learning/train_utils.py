import logging
from typing import Callable

import tensorflow as tf


def sparse_categorical_crossentropy_with_label_smoothing(n_classes: int, label_smoothing: float = 0.0) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    if label_smoothing == 0:
        logging.debug('Using default sparse crossentropy')
        return tf.losses.sparse_categorical_crossentropy
    if label_smoothing < 0 or label_smoothing > 1:
        raise ValueError(f'The amount of label smoothing must be in range [0, 1]; received {label_smoothing}.')
    if int(n_classes) != n_classes or n_classes <= 0:
        raise ValueError(f'The class count must be an integer greater than 0; received {n_classes}.')
    logging.debug('Using custom sparse crossentropy')
    def sparse_categorical_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_one_hot = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), n_classes, axis=-1)[:,0,:], tf.float32)
        return tf.losses.categorical_crossentropy(y_true_one_hot, y_pred, label_smoothing=label_smoothing)
    return sparse_categorical_crossentropy

class EarlyStoppingAtThreshold(tf.keras.callbacks.Callback):
    def __init__(self, monitor: str, threshold: float, mode: str = 'auto'):
        """mode is either "auto", "greater" or "smaller"."""
        super().__init__()
        self.monitor = monitor
        self.threshold = threshold

        if mode == 'auto':
            if 'acc' in self.monitor:
                self.mode = 'greater'
            else:
                self.mode = 'smaller'
        elif not mode in ['greater', 'smaller']:
            raise ValueError(f'"{mode}" is not a valid early stopping threshold mode. Only "auto", "greater" and "smaller" are valid values.')
        else:
            self.mode = mode

    def on_epoch_end(self, epoch, logs={}):
        current = logs[self.monitor]

        if (self.mode == 'greater' and current > self.threshold) or (self.mode == 'smaller' and current < self.threshold):
            logging.getLogger(__name__).info(f'Early stopping after epoch {epoch + 1}.')
            self.model.stop_training = True

class EarlyStoppingAfterThreshold(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', threshold=.5, min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False):
        super().__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)
        self.monitor = monitor
        self.threshold = threshold

        self.threshold_reached = False

    def on_epoch_end(self, epoch, logs):
        if not self.threshold_reached:
            self.threshold_reached = self.monitor_op(self.get_monitor_value(logs), self.threshold)

        if self.threshold_reached:
            return super().on_epoch_end(epoch, logs=logs)

class ThresholdedModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, threshold=None, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', options=None, **kwargs):
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, options, **kwargs)

        self.threshold = threshold
        self.best_weights = None

    def on_train_begin(self, logs):
        self.best_weights = None
        self.best_weights_filepath = None

        return super().on_train_begin(logs=logs)

    def on_epoch_end(self, epoch, logs):
        if epoch == 0:
            logging.getLogger(__name__).info(f'Saving initial model weights for {self.monitor} checkpoint...')
            return super().on_epoch_end(epoch, logs=logs)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Can save best model only with %s available, skipping.', self.monitor)
        else:
            if self.monitor_op(current, self.threshold):
                if self.save_best_only:
                    if self.monitor_op(current, self.best):
                        logging.getLogger(__name__).info(f'New best value for metric "{self.monitor}" ({current}), storing weights...')
                        self.best_weights = self.model.get_weights()
                        self.best_weights_filepath = self._get_file_path(epoch, logs)
                        # logging.getLogger(__name__).info(f'Done.')
                        self.best = current
                else:
                    return super().on_epoch_end(epoch, logs=logs)

    
    def on_train_end(self, logs):
        if self.save_best_only and not self.best_weights is None:
            logging.getLogger(__name__).info(f'Saving best model weights for {self.monitor} checkpoint...')
            final_weights = self.model.get_weights()
            self.model.set_weights(self.best_weights)
            if self.save_weights_only:
                self.model.save_weights(self.best_weights_filepath, overwrite=True, options=self._options)
            else:
                self.model.save(self.best_weights_filepath, overwrite=True, options=self._options)
            self.model.set_weights(final_weights)
            logging.getLogger(__name__).info(f'Done.')

        return super().on_train_end(logs=logs)