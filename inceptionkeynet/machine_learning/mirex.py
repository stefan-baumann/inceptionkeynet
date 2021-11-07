from __future__ import annotations
import inceptionkeynet # Enable postponed evaluation of annotations to allow for annotating methods returning their enclosing type

from typing import Any, List, Optional, Tuple
from enum import Enum, auto
import logging
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mir_eval.key

from inceptionkeynet.data import KeyMode, KeyRoot



class MirexClassificationClass(Enum):
    SAME = auto()
    FIFTH = auto()
    RELATIVE_MINOR = auto()
    RELATIVE_MAJOR = auto()
    PARALLEL_MINOR = auto()
    PARALLEL_MAJOR = auto()
    OTHER = auto()

    def get_multiplier(self) -> float:
        return {
            MirexClassificationClass.SAME: 1.0,
            MirexClassificationClass.FIFTH: 0.5,
            MirexClassificationClass.RELATIVE_MINOR: 0.3,
            MirexClassificationClass.RELATIVE_MAJOR: 0.3,
            MirexClassificationClass.PARALLEL_MINOR: 0.2,
            MirexClassificationClass.PARALLEL_MAJOR: 0.2,
            MirexClassificationClass.OTHER: 0.0,
        }[self]

    def __mul__(self, other):
        return other * self.get_multiplier()

    def __rmul__(self, other):
        return self.get_multiplier() * other
    
    @classmethod
    def compute(cls, prediction: Tuple[KeyRoot, KeyMode], target: Tuple[KeyRoot, KeyMode]) -> MirexClassificationClass:
        root_p, root_t = prediction[0].value, target[0].value # Process as their corresponding ints
        mode_p, mode_t = prediction[1], target[1]
        score = mir_eval.key.weighted_score(f'{KeyRoot(root_t).name} {KeyMode(mode_t).name.lower()}', f'{KeyRoot(root_p).name} {KeyMode(mode_p).name.lower()}')
        # Hotfix around wrong (non-compliant with standard MIREX specifications) behaviour by mir_eval
        # Will become obsolete when I finally get around to finishing this PR: https://github.com/craffel/mir_eval/pull/339
        if inceptionkeynet.MIREX_ALLOW_DESCENDING_FIFTHS:
            score_reverse = mir_eval.key.weighted_score(f'{KeyRoot(root_p).name} {KeyMode(mode_p).name.lower()}', f'{KeyRoot(root_t).name} {KeyMode(mode_t).name.lower()}')
            if score_reverse > score and score_reverse == .5:
                score = score_reverse
        if score == 1:
            return MirexClassificationClass.SAME
        elif score == .5:
            return MirexClassificationClass.FIFTH
        elif score == 0:
            if inceptionkeynet.MIREX_IGNORE_FIFTH_MODE and ((root_p + 7) % 12 == root_t or (root_t + 7) % 12 == root_p):
                return MirexClassificationClass.FIFTH
            return MirexClassificationClass.OTHER
        elif score == .3:
            if mode_p == KeyMode.MAJOR:
                return MirexClassificationClass.RELATIVE_MAJOR
            else:
                return MirexClassificationClass.RELATIVE_MINOR
        elif score == .2:
            if mode_p == KeyMode.MINOR:
                return MirexClassificationClass.PARALLEL_MINOR
            else:
                return MirexClassificationClass.PARALLEL_MAJOR
        else:
            raise ValueError('Unknown mir_eval score ' + str(score))
            



def __mirex_score_sparse_24_output(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_n: np.ndarray = y_true.numpy()
    y_pred_n: np.ndarray = y_pred.numpy()
    y_pred_sparse = np.argmax(y_pred_n, axis=-1)

    sum = 0
    for i in range(len(y_true_n)):
        i_p, i_t = y_pred_sparse[i], y_true_n[i]
        root_p, root_t = KeyRoot(int(i_p % 12)), KeyRoot(int(i_t % 12))
        mode_p, mode_t = KeyMode.from_int(int(i_p // 12)), KeyMode.from_int(int(i_t // 12))
        sum += MirexClassificationClass.compute((root_p, mode_p), (root_t, mode_t)).get_multiplier()

    return tf.constant(sum / len(y_true_n), dtype=tf.float32)

def __mirex_score_14_output(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_n: np.ndarray = y_true.numpy()
    y_pred_n: np.ndarray = y_pred.numpy()
    roots_p = np.argmax(y_pred_n[:,:11], axis=-1)
    roots_t = np.argmax(y_true_n[:,:11], axis=-1)
    modes_p = np.argmax(y_pred_n[:,12:], axis=-1)
    modes_t = np.argmax(y_true_n[:,12:], axis=-1)

    sum = 0
    for i in range(len(y_true_n)):
        root_p, root_t = KeyRoot(int(roots_p[i])), KeyRoot(int(roots_t[i]))
        mode_p, mode_t = KeyMode.from_int(int(modes_p[i])), KeyMode.from_int(int(modes_t[i]))
        sum += MirexClassificationClass.compute((root_p, mode_p), (root_t, mode_t)).get_multiplier()

    return tf.constant(sum / len(y_true_n), dtype=tf.float32)

def get_mirex_score(length: int = 24):
    if length == 24: # root + mode * 12
        def mirex_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return tf.py_function(__mirex_score_sparse_24_output, (y_true, y_pred), tf.float32)
        return mirex_score
    elif length == 14: # [root] + [mode]
        def mirex_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return tf.py_function(__mirex_score_14_output, (y_true, y_pred), tf.float32)
        return mirex_score
    else:
        raise ValueError('Invalid mirex score length')



class MirexScoreSplitModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data: Optional[tf.data.Dataset] = None, validation_data: Optional[tf.data.Dataset] = None, checkpoint_output_path: Optional[str] = None):
        super().__init__()
        self.train_data = train_data
        self.validation_data = validation_data
        self.checkpoint_output_path = checkpoint_output_path

        self.__best_score_train = None
        self.__best_score_val = None

    def calculate_mirex_scores(self, dataset):
        class_counts = dict(((key, 0) for key in MirexClassificationClass))
        for batch_X, batch_Y in dataset:
            pred_Y_root, pred_Y_mode = self.model.predict(batch_X)
            pred_Y_root_sparse, pred_Y_mode_sparse = np.argmax(pred_Y_root, axis=-1), np.argmax(pred_Y_mode, axis=-1)
            target_Y_root_sparse, target_Y_mode_sparse = batch_Y.values()
            
            for i in range(len(batch_X)):
                class_counts[MirexClassificationClass.compute((KeyRoot(pred_Y_root_sparse[i]), KeyMode.from_int(pred_Y_mode_sparse[i])), (KeyRoot(target_Y_root_sparse[i]), KeyMode.from_int(target_Y_mode_sparse[i])))] += 1
        
        return class_counts

    def on_epoch_end(self, epoch, logs={}):
        if not self.train_data is None:
            class_counts = self.calculate_mirex_scores(self.train_data)
            count_total = sum(class_counts.values())
            
            for key in MirexClassificationClass:
                logs[f'mirex_{key.name.lower()}'] = float(class_counts[key] / count_total)
            logs['mirex_score'] = sum((class_counts[key] * key for key in MirexClassificationClass))

            if not self.checkpoint_output_path is None and (self.__best_score_train is None or logs['mirex_score'] > self.__best_score_train):
                logging.getLogger(__name__).info(f'Saving model checkpoint with best train mirex score ({logs["mirex_score"]})...')
                self.model.save(os.path.join(self.checkpoint_output_path, 'train'), options=tf.saved_model.SaveOptions())
                self.__best_score_train = logs['mirex_score']
        
        if not self.validation_data is None:
            class_counts = self.calculate_mirex_scores(self.validation_data)
            count_total = sum(class_counts.values())
            
            for key in MirexClassificationClass:
                logs[f'val_mirex_{key.name.lower()}'] = float(class_counts[key] / count_total)
            logs['val_mirex_score'] = sum((class_counts[key] * key for key in MirexClassificationClass))

            if not self.checkpoint_output_path is None and (self.__best_score_val is None or logs['val_mirex_score'] > self.__best_score_val):
                self.model.save(os.path.join(self.checkpoint_output_path, 'val'), options=tf.saved_model.SaveOptions())
                logging.getLogger(__name__).info(f'Saving model checkpoint with best validation mirex score ({logs["val_mirex_score"]})...')
                self.__best_score_val = logs['val_mirex_score']
