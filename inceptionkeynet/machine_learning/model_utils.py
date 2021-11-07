from typing import Callable

import tensorflow as tf
import numpy as np



def add_sequential_output_to_split_model(split_model: tf.keras.Model) -> tf.keras.Model:
    """Adds a third output in sequential format to a model with two separate outputs for root and mode"""
    output_root = split_model.get_layer(name='output_root').output
    output_mode = split_model.get_layer(name='output_mode').output
    output_sequential = tf.keras.layers.Concatenate(axis=-1, name='output_sequential')([output_root, output_mode])
    return tf.keras.Model(split_model.inputs, outputs=[output_root, output_mode, output_sequential])

def remove_sequential_output_from_split_model(split_sequential_model: tf.keras.Model) -> tf.keras.Model:
    """Removes the additional sequential output output from a model with initially two separate outputs for root and mode"""
    output_root = split_sequential_model.get_layer(name='output_root')
    output_mode = split_sequential_model.get_layer(name='output_mode')
    return tf.keras.Model(split_sequential_model.inputs, outputs=[output_root, output_mode])

def split_model_to_sequential_model(split_model: tf.keras.Model) -> tf.keras.Model:
    """Adds a combined output in sequential format to a model with two separate outputs for root and mode to make it into a sequential-output model"""
    output_root = split_model.get_layer(name='output_root')
    output_mode = split_model.get_layer(name='output_mode')
    output_sequential = tf.keras.layers.Concatenate(axis=-1, name='output_sequential')([output_root, output_mode])
    return tf.keras.Model(split_model.inputs, outputs=[output_sequential])



def create_sequential_output_double_categorical_crossentropy_loss(root_weight: float = 1.0, mode_weight: float = 1.0) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    def sequential_double_categorical_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        root_loss = tf.keras.losses.categorical_crossentropy(y_true[:, :11], y_pred[:, :11])
        mode_loss = tf.keras.losses.categorical_crossentropy(y_true[:, 12:], y_pred[:, 12:])
        return root_weight * root_loss + mode_weight * mode_loss
    return sequential_double_categorical_crossentropy

def create_sequential_output_combined_accuracy_metric() -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    def __accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_n: np.ndarray = y_true.numpy()
        y_pred_n: np.ndarray = y_pred.numpy()
        roots_p = np.argmax(y_pred_n[:,:11], axis=-1)
        roots_t = np.argmax(y_true_n[:,:11], axis=-1)
        modes_p = np.argmax(y_pred_n[:,12:], axis=-1)
        modes_t = np.argmax(y_true_n[:,12:], axis=-1)

        sum = 0
        for i in range(len(y_true_n)):
            if roots_p[i] == roots_t[i] and modes_p[i] == modes_t[i]:
                sum += 1

        return tf.constant(sum / len(y_true_n), dtype=tf.float32)

    def combined_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.py_function(__accuracy, (y_true, y_pred), tf.float32)
    
    return combined_accuracy
        