import inspect
from typing import Iterator, List, Type
from abc import ABCMeta, abstractclassmethod, abstractmethod, abstractproperty, ABC

import inceptionkeynet

import tensorflow as tf
from tensorflow.keras.layers import *

import inceptionkeynet.utils
from inceptionkeynet.machine_learning.__hyperparameters import *
from inceptionkeynet.data import KeyRoot, KeyMode



class ModelWrapper(ABC):
    @abstractclassmethod
    def get_name(cls):
        return 'Unnamed model'
    
    @classmethod
    def get_snake_case_name(cls):
        return inceptionkeynet.utils.name_to_path_snake_case(cls.get_name())

    @classmethod
    @abstractmethod
    def create_model(cls, hyperparameters: Hyperparameters) -> tf.keras.Model:
        raise NotImplementedError('This abstrat method has not been implemented.')



# Adapted from: https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py#L40
# License of original code:
# COPYRIGHT
# 
# Copyright (c) 2016 - 2018, the respective contributors.
# All rights reserved.
# 
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.
# The initial code of this repository came from https://github.com/keras-team/keras
# (the Keras repository), hence, for author information regarding commits
# that occured earlier than the first commit in the present repository,
# please see the original Keras repository.
# 
# LICENSE
# 
# The MIT License (MIT)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    bn_axis = 3
    x = tf.keras.layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = tf.keras.layers.Activation('relu', name=name)(x)
    return x

class InceptionKeyNet(ModelWrapper):
    @classmethod
    def get_name(cls):
        return 'InceptionKeyNet'

    @classmethod
    def create_model(cls, hyperparameters: Hyperparameters) -> tf.keras.Model:
        if hyperparameters[Hyperparameters.MODEL_INPUT_TYPE].value != ModelInputType.SPECTROGRAM or bool(hyperparameters[Hyperparameters.TRANSFORMATION_TYPE].value & TimeFrequencyTransformationType.TIME_AVERAGED):
            raise ValueError('This model can only be used on spectrograms.')
        
        size_reduction_factor = hyperparameters[Hyperparameters.INCEPTION_MODEL_SIZE_REDUCTION_FACTOR].value # 5
        enable_pooling = False
        pooling = 'avg'
        p = hyperparameters[Hyperparameters.DROPOUT_RATE].value

        input_shape = (hyperparameters[Hyperparameters.TRANSFORMATION_TYPE].value.get_spectrogram_width(), None, 1)

        input_layer = tf.keras.layers.Input(shape=input_shape)

        x = conv2d_bn(input_layer, 32 // size_reduction_factor, 3, 3, strides=(2, 2), padding='valid')
        x = conv2d_bn(x, 32 // size_reduction_factor, 3, 3, padding='valid')
        x = conv2d_bn(x, 64 // size_reduction_factor, 3, 3)
        if enable_pooling:
            x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv2d_bn(x, 80 // size_reduction_factor, 1, 1, padding='valid')
        x = conv2d_bn(x, 192 // size_reduction_factor, 3, 3, padding='valid')
        if enable_pooling:
            x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = Dropout(p, noise_shape=(None, 1, 1, None))(x)

        # mixed 0
        branch1x1 = conv2d_bn(x, 64 // size_reduction_factor, 1, 1)

        branch5x5 = conv2d_bn(x, 48 // size_reduction_factor, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64 // size_reduction_factor, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64 // size_reduction_factor, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96 // size_reduction_factor, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96 // size_reduction_factor, 3, 3)

        if enable_pooling:
            branch_pool = tf.keras.layers.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(x)
        else:
            branch_pool = x
        branch_pool = conv2d_bn(branch_pool, 32 // size_reduction_factor, 1, 1)
        x = tf.keras.layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed0')
        x = Dropout(p, noise_shape=(None, 1, 1, None))(x)

        # mixed 1
        branch1x1 = conv2d_bn(x, 64 // size_reduction_factor, 1, 1)

        branch5x5 = conv2d_bn(x, 48 // size_reduction_factor, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64 // size_reduction_factor, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64 // size_reduction_factor, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96 // size_reduction_factor, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96 // size_reduction_factor, 3, 3)

        if enable_pooling:
            branch_pool = tf.keras.layers.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(x)
        else:
            branch_pool = x
        branch_pool = conv2d_bn(branch_pool, 64 // size_reduction_factor, 1, 1)
        x = tf.keras.layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed1')
        x = Dropout(p, noise_shape=(None, 1, 1, None))(x)

        # mixed 2
        branch1x1 = conv2d_bn(x, 64 // size_reduction_factor, 1, 1)

        branch5x5 = conv2d_bn(x, 48 // size_reduction_factor, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64 // size_reduction_factor, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64 // size_reduction_factor, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96 // size_reduction_factor, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96 // size_reduction_factor, 3, 3)

        if enable_pooling:
            branch_pool = tf.keras.layers.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(x)
        else:
            branch_pool = x
        branch_pool = conv2d_bn(branch_pool, 64 // size_reduction_factor, 1, 1)
        x = tf.keras.layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed2')
        x = Dropout(p, noise_shape=(None, 1, 1, None))(x)

        # mixed 3
        branch3x3 = conv2d_bn(x, 384 // size_reduction_factor, 3, 3)

        branch3x3dbl = conv2d_bn(x, 64 // size_reduction_factor, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96 // size_reduction_factor, 3, 3)
        branch3x3dbl = conv2d_bn(
            branch3x3dbl, 96 // size_reduction_factor, 3, 3)

        if enable_pooling:
            branch_pool = tf.keras.layers.MaxPooling2D((3, 3))(x)
        else:
            branch_pool = x
        x = tf.keras.layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed3')
        x = Dropout(p, noise_shape=(None, 1, 1, None))(x)

        # mixed 4
        branch1x1 = conv2d_bn(x, 192 // size_reduction_factor, 1, 1)

        branch7x7 = conv2d_bn(x, 128 // size_reduction_factor, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 128 // size_reduction_factor, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192 // size_reduction_factor, 7, 1)

        branch7x7dbl = conv2d_bn(x, 128 // size_reduction_factor, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128 // size_reduction_factor, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128 // size_reduction_factor, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128 // size_reduction_factor, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192 // size_reduction_factor, 1, 7)

        if enable_pooling:
            branch_pool = tf.keras.layers.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(x)
        else:
            branch_pool = x
        branch_pool = conv2d_bn(branch_pool, 192 // size_reduction_factor, 1, 1)
        x = tf.keras.layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed4')
        x = Dropout(p, noise_shape=(None, 1, 1, None))(x)

        # mixed 5, 6
        for i in range(2):
            branch1x1 = conv2d_bn(x, 192 // size_reduction_factor, 1, 1)

            branch7x7 = conv2d_bn(x, 160 // size_reduction_factor, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 160 // size_reduction_factor, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192 // size_reduction_factor, 7, 1)

            branch7x7dbl = conv2d_bn(x, 160 // size_reduction_factor, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160 // size_reduction_factor, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160 // size_reduction_factor, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160 // size_reduction_factor, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192 // size_reduction_factor, 1, 7)

            if enable_pooling:
                branch_pool = tf.keras.layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            else:
                branch_pool = x
            branch_pool = conv2d_bn(branch_pool, 192 // size_reduction_factor, 1, 1)
            x = tf.keras.layers.concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=3,
                name='mixed' + str(5 + i))
            x = Dropout(p, noise_shape=(None, 1, 1, None))(x)

        # mixed 7
        branch1x1 = conv2d_bn(x, 192 // size_reduction_factor, 1, 1)

        branch7x7 = conv2d_bn(x, 192 // size_reduction_factor, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 192 // size_reduction_factor, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192 // size_reduction_factor, 7, 1)

        branch7x7dbl = conv2d_bn(x, 192 // size_reduction_factor, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192 // size_reduction_factor, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192 // size_reduction_factor, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192 // size_reduction_factor, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192 // size_reduction_factor, 1, 7)

        if enable_pooling:
            branch_pool = tf.keras.layers.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(x)
        else:
            branch_pool = x
        branch_pool = conv2d_bn(branch_pool, 192 // size_reduction_factor, 1, 1)
        x = tf.keras.layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed7')
        x = Dropout(p, noise_shape=(None, 1, 1, None))(x)

        # mixed 8
        branch3x3 = conv2d_bn(x, 192 // size_reduction_factor, 1, 1)
        branch3x3 = conv2d_bn(branch3x3, 320 // size_reduction_factor, 3, 3)

        branch7x7x3 = conv2d_bn(x, 192 // size_reduction_factor, 1, 1)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192 // size_reduction_factor, 1, 7)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192 // size_reduction_factor, 7, 1)
        branch7x7x3 = conv2d_bn(
            branch7x7x3, 192 // size_reduction_factor, 3, 3)

        if enable_pooling:
            branch_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        else:
            branch_pool = x
        x = tf.keras.layers.concatenate(
            [branch3x3, branch7x7x3, branch_pool],
            axis=3,
            name='mixed8')
        x = Dropout(p, noise_shape=(None, 1, 1, None))(x)

        # mixed 9
        for i in range(2):
            branch1x1 = conv2d_bn(x, 320 // size_reduction_factor, 1, 1)

            branch3x3 = conv2d_bn(x, 384 // size_reduction_factor, 1, 1)
            branch3x3_1 = conv2d_bn(branch3x3, 384 // size_reduction_factor, 1, 3)
            branch3x3_2 = conv2d_bn(branch3x3, 384 // size_reduction_factor, 3, 1)
            branch3x3 = tf.keras.layers.concatenate(
                [branch3x3_1, branch3x3_2],
                axis=3,
                name='mixed9_' + str(i))

            branch3x3dbl = conv2d_bn(x, 448 // size_reduction_factor, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 384 // size_reduction_factor, 3, 3)
            branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384 // size_reduction_factor, 1, 3)
            branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384 // size_reduction_factor, 3, 1)
            branch3x3dbl = tf.keras.layers.concatenate(
                [branch3x3dbl_1, branch3x3dbl_2], axis=3)

            if enable_pooling:
                branch_pool = tf.keras.layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            else:
                branch_pool = x
            branch_pool = conv2d_bn(branch_pool, 192 // size_reduction_factor, 1, 1)
            x = tf.keras.layers.concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=3,
                name='mixed' + str(9 + i))
            x = Dropout(p, noise_shape=(None, 1, 1, None))(x)
        
        x = tf.keras.layers.Conv2D(24, (79 if hyperparameters[Hyperparameters.TRANSFORMATION_TYPE].value == TimeFrequencyTransformationType.CONSTANT_Q or hyperparameters[Hyperparameters.TRANSFORMATION_TYPE].value == TimeFrequencyTransformationType.MADMOM_WIDE else (48 if hyperparameters[Hyperparameters.TRANSFORMATION_TYPE].value == TimeFrequencyTransformationType.MADMOM else -1), 1))(x)

        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Softmax(name='predictions')(x)

        return tf.keras.Model(inputs=input_layer, outputs=x)



class __ModelsMeta(type):
    def __iter__(self) -> Iterator[ModelWrapper]:
        return iter(self.list_models())

class Models(metaclass=__ModelsMeta):
    INCEPTIONKEYNET = InceptionKeyNet



    @classmethod
    def list_models(cls) -> List[Type[ModelWrapper]]:
        return [member[1] for member in inspect.getmembers(Models) if not member[0].startswith('_') and inspect.isclass(member[1])]
    
    @classmethod
    def get_by_name(cls, name: str) -> Type[ModelWrapper]:
        models = cls.list_models()
        model = next((model for model in models if model.get_snake_case_name() == name), None)
        if not model is None:
            return model
        model = next((model for model in models if model.get_name() == name), None)
        if not model is None:
            return model
        
        raise ValueError(f'No model with name "{name}" could be found.')