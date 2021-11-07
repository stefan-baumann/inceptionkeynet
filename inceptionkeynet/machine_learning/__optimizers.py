# Import tensorflow with correct log level
import inceptionkeynet
__log_levels = { 'DEBUG': '0', 'INFO': '1', 'WARNING': '2' }
if inceptionkeynet.TERMINAL_LOG_LEVEL in __log_levels:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = __log_levels[inceptionkeynet.TERMINAL_LOG_LEVEL]

import tensorflow as tf

from inceptionkeynet.machine_learning.__hyperparameters import Hyperparameters


def get_optimizer(hyperparameters: Hyperparameters) -> tf.keras.optimizers.Optimizer:
    optimizer = hyperparameters[Hyperparameters.OPTIMIZER].value
    if optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=.9)
    optimizer = tf.keras.optimizers.get(optimizer)

    if not hyperparameters[Hyperparameters.LEARNING_RATE].value is None:
        optimizer.learning_rate = hyperparameters[Hyperparameters.LEARNING_RATE].value
    
    return optimizer