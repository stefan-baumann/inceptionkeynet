from __future__ import annotations # Enable postponed evaluation of annotations to allow for annotating methods returning their enclosing type

from typing import List, Dict, Iterable, Union, Tuple, Any, Optional, Generic, TypeVar, AnyStr, Iterator
import enum
import inspect

import inceptionkeynet.data as data



THyperparameter = TypeVar('THyperparameter')
class Hyperparameter(Generic[THyperparameter], data.DictSerializable):
    def __init__(self, name: str, default_value: THyperparameter, expects_list: bool = False):
        self.name = name
        self.default_value = default_value
        self.expects_list = expects_list

    def __call__(self, value: Union[THyperparameter, Iterable[THyperparameter]]) -> Union[HyperparameterValue[THyperparameter], Iterable[HyperparameterValue[THyperparameter]]]:
        if not self.expects_list:
            if not isinstance(value, type(self.default_value)) and (isinstance(value, Iterable) and not isinstance(value, str)):
                return [HyperparameterValue(self, v) for v in value]
        return HyperparameterValue(self, value)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Hyperparameter) and self.name == o.name and self.default_value == o.default_value and self.expects_list == o.expects_list

    def __repr__(self) -> str:
        return f'Hyperparameter(name=\'{self.name}\', default_value={repr(self.default_value)})'
    
    def to_dict(self):
        return {
            'name': self.name,
            'default_value': self.default_value,
            'expects_list': self.expects_list
        }

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> Hyperparameter:
        return Hyperparameter(dict['name'], dict['default_value'], dict['expects_list'])



class HyperparameterValue(Generic[THyperparameter], data.DictSerializable):
    def __init__(self, hyperparameter: Hyperparameter[THyperparameter], value: THyperparameter):
        self.hyperparameter = hyperparameter
        self.value = value

    def __repr__(self) -> str:
        return f'HyperparameterValue({repr(self.hyperparameter)}, value={repr(self.value)})'
    
    def to_dict(self):
        return {
            'name': self.hyperparameter.name,
            'value': self.value,
            'default_value': self.hyperparameter.default_value
        }

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> HyperparameterValue:
        return HyperparameterValue(Hyperparameter(dict['name'], dict['default_value']), dict['value'])



class __HyperparametersMeta(type(data.DictSerializable)):
    def __iter__(self) -> Iterator[Hyperparameter]:
        return iter(self.list_hyperparameters())



class ModelClassificationType(str, enum.Enum):
    ROOT = 'root' # [root]
    MODE = 'mode' # [mode]
    ROOT_MODE_COMBINED = 'root_mode_combined' # [root + 12*mode]
    ROOT_MODE_SEPARATE = 'root_mode_separate' # [[root], [mode]]
    ROOT_MODE_SEQUENTIAL = 'root_mode_sequential' # [root, mode]

class ModelInputType(str, enum.Enum):
    SPECTROGRAM = 'spectrogram'
    SPECTRUM = 'spectrum'

class TimeFrequencyTransformationType(enum.Flag):
    TIME_AVERAGED = enum.auto()
    
    CONSTANT_Q = enum.auto()

    def get_spectrogram_width(self) -> int:
        if bool(self & TimeFrequencyTransformationType.CONSTANT_Q):
            return 168
        raise ValueError(f'Unknown transformation type "{self}".')



class Hyperparameters(data.DictSerializable, metaclass=__HyperparametersMeta):
    APPLY_HARMONIC_SEPARATION_PREPROCESSING = Hyperparameter('hpss', False)
    
    TRANSFORMATION_TYPE = Hyperparameter('transformation_type', TimeFrequencyTransformationType.CONSTANT_Q | TimeFrequencyTransformationType.TIME_AVERAGED)

    SPECTROGRAM_MAX_LENGTH_FRAMES = Hyperparameter('spectrogram_max_length_frames', None)
    USE_FULL_TRACKS_FOR_VALIDATION = Hyperparameter('use_full_tracks_for_validation', True)
    """This has the side effect of forcing the validation batch size to be 1"""
    USE_FULL_TRACKS_FOR_TESTING = Hyperparameter('use_full_tracks_for_testing', True)
    """This has the side effect of forcing the testing batch size to be 1"""
    # SPECTROGRAM_TIME_LENGTH_RANGE_S = Hyperparameter('spectrogram_time_length_range_s', [None, None], expects_list=True)
    SPECTROGRAM_TIME_DOWNSCALING_FACTOR = Hyperparameter('spectrogram_time_downscaling_factor', 1)

    MODEL_NAME = Hyperparameter('model_name', None) # Specifies which model was used, has no default value
    MODEL_CLASSIFICATION_TYPE = Hyperparameter('classification_type', ModelClassificationType.ROOT_MODE_COMBINED)
    MODEL_INPUT_TYPE = Hyperparameter('input_type', ModelInputType.SPECTROGRAM)
    
    DROPOUT_RATE = Hyperparameter('dropout_rate', 0)

    INCEPTION_MODEL_SIZE_REDUCTION_FACTOR = Hyperparameter('inception_model_size_reduction_factor', 5)

    TRAINABLE_PARAMETER_COUNT = Hyperparameter('trainable_parameter_count', None)

    CROSS_VALIDATION_FOLD_COUNT = Hyperparameter('cross_validation_fold_count', 5)
    VALIDATE_ON_TEST_DATASET = Hyperparameter('validate_on_test_dataset', False)
    BATCH_SIZE = Hyperparameter('batch_size', 32)
    EPOCHS = Hyperparameter('epochs', 250)
    OPTIMIZER = Hyperparameter('optimizer', 'rmsprop')
    LABEL_SMOOTHING = Hyperparameter('label_smoothing', 0.0)
    LEARNING_RATE = Hyperparameter('learning_rate', None)
    EARLY_STOPPING = Hyperparameter('early_stopping', None)
    REDUCE_LR_ON_PLATEAU = Hyperparameter('reduce_lr_on_plateau', ['val_accuracy', None, .5], expects_list=True)
    """Order of values is [metric, epochs, factor]"""
    
    PITCH_SHIFT_AUGMENTATION_RANGE = Hyperparameter('pitch_shift', 0)
    GAUSSIAN_NOISE_AUGMENTATION_MAX = Hyperparameter('gaussian_noise', 0.0)
    RANDOM_EQ_AUGMENTATION_I_FREQ_MIN = Hyperparameter('ramdom_eq_i_freq_min', 0)
    RANDOM_EQ_AUGMENTATION_I_FREQ_MAX = Hyperparameter('ramdom_eq_i_freq_max', -1)
    RANDOM_EQ_AUGMENTATION_SIGMA_MAX = Hyperparameter('ramdom_eq_sigma_max', 0.0)
    RANDOM_EQ_AUGMENTATION_S_MAX = Hyperparameter('ramdom_eq_s_max', 0.0)
    LOUDNESS_AUGMENTATION_FACTOR_RANGE = Hyperparameter('loudness_augmentation', 1.0)
    TIME_WARPING_DISTANCE_RANGE = Hyperparameter('time_warping_distance', 0)
    TIME_MASKING_LENGTH_RANGE = Hyperparameter('time_masking_length', 0)
    FREQUENCY_MASKING_LENGTH_RANGE = Hyperparameter('frequency_masking_length', 0)

    MINIMUM_CONFIDENCE = Hyperparameter('min_confidence', 0.0)
    USE_SAMPLE_WEIGHT = Hyperparameter('use_sample_weight', False)

    TRAIN_DATASETS = Hyperparameter('train_datasets', None, expects_list=True)
    TEST_DATASETS = Hyperparameter('test_datasets', None, expects_list=True)
    AUDIO_SOURCE_WHITELIST = Hyperparameter('audio_source_whitelist', None, expects_list=True)
    AUDIO_SOURCE_WHITELIST_STRICT = Hyperparameter('audio_source_whitelist_strict', True)

    CHECKPOINT_CHOICE = Hyperparameter('checkpoint_choice', 'mirex')
    
    
    
    def __init__(self, values: Optional[Iterable[HyperparameterValue]] = None):
        self.values = sorted(values, key=lambda hv: hv.hyperparameter.name if not isinstance(hv, Iterable) else hv[0].hyperparameter.name) if not values is None else []
    
    def __iter__(self) -> Iterator[HyperparameterValue]:
        return iter(self.values)
    
    @classmethod
    def list_hyperparameters(cls) -> List[Hyperparameter]:
        return [member[1] for member in inspect.getmembers(Hyperparameters) if not member[0].startswith('_') and isinstance(member[1], Hyperparameter)]

    def __getitem__(self, index: Union[Hyperparameter[THyperparameter], str]) -> HyperparameterValue[THyperparameter]:
        if isinstance(index, str):
            hyperparameter = next((h for h in Hyperparameters.list_hyperparameters() if h.name == index), None)
        else:
            hyperparameter = index
        
        value = next((v for v in self.values if v.hyperparameter == hyperparameter), None)
        return value if not value is None else hyperparameter(hyperparameter.default_value)

    def __setitem__(self, key: Union[Hyperparameter[THyperparameter], str], value: Union[HyperparameterValue[THyperparameter], THyperparameter]):
        if isinstance(key, str):
            hyperparameter = next((h for h in Hyperparameters.list_hyperparameters() if h.name == key), None)
        else:
            hyperparameter = key
        
        if value is HyperparameterValue and value.hyperparameter != hyperparameter:
            raise ValueError('Incompatible hyperparameters: A hyperparameter cannot be set to an instance of a HyperparameterValue belonging to a different hyperparameter.')

        existing_value = next((v for v in self.values if v.hyperparameter == hyperparameter), None)
        value = value if value is HyperparameterValue else hyperparameter(value)
        if not existing_value is None:
            self.values[self.values.index(existing_value)] = value
        else:
            self.values.append(value)

    def __repr__(self) -> str:
        return f'Hyperparameters([{", ".join(repr(value) for value in self.values)}])'
    
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Hyperparameters):
            return False
        else:
            for hv in self:
                if (not hv.value == hv.hyperparameter.default_value) and o[hv.hyperparameter].value != hv.value:
                    return False
            for hv in o:
                if (not hv.value == hv.hyperparameter.default_value) and self[hv.hyperparameter].value != hv.value:
                    return False
            return True


    @classmethod
    def combinations(cls, hyperparameters: Iterable[Union[Iterable[HyperparameterValue], HyperparameterValue]]) -> Iterable[Hyperparameters]:
        hyperparameters = list(list(hs) if (isinstance(hs, Iterable) and not isinstance(hs, str)) else [hs] for hs in hyperparameters)
        if len(hyperparameters) > len(set(hs[0].hyperparameter.name for hs in hyperparameters)):
            raise ValueError('Cannot generate a list of hyperparameter combinations from a list containing duplicate hyperparameters.')
        
        if len(hyperparameters) == 0:
            return [Hyperparameters()]
        
        def recursive_combinator(base_combination: List[HyperparameterValue], remaining_parameters: List[List[HyperparameterValue]]) -> Iterable[List[HyperparameterValue]]:
            if len(remaining_parameters) == 1:
                for value in remaining_parameters[0]:
                    yield base_combination + [value]
            else:
                for value in remaining_parameters[0]:
                    yield from recursive_combinator(base_combination + [value], remaining_parameters[1:])

        return (Hyperparameters(values) for values in recursive_combinator([], hyperparameters))

    def to_dict(self):
        return {
            'values': list(value.to_dict() for value in self.values)
        }

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> Hyperparameters:
        values = []
        available_hyperparameters = cls.list_hyperparameters()
        for value in dict['values']:
            hyperparameter = next((h for h in available_hyperparameters if h.name == value['name']), None)
            if hyperparameter is None:
                raise ValueError(f'The hyperparameter "{value["name"]}" is unknown and can thus not be parsed.')
            if value['default_value'] != hyperparameter.default_value:
                raise ValueError(f'The default value of hyperparameter "{value["name"]}" has been read as {value["default_value"]} while {hyperparameter.default_value} was expected.')
            values.append(hyperparameter(value['value']))
        return Hyperparameters(values)
        


if __name__ == '__main__':
    h = Hyperparameters.PITCH_SHIFT_AUGMENTATION_RANGE(3)
    print(h.to_dict())
    print(HyperparameterValue.from_dict(h.to_dict()).to_dict())

    # import time
    # start_time = time.time()
    # for i in range(0, 1000):
    #     a = [member for member in inspect.getmembers(Hyperparameters) if not member[0].startswith('_') and isinstance(member[1], Hyperparameter)]
    # print(f'Time per iteration: {(time.time() - start_time) / 1000}s')
    # print([member[1] for member in inspect.getmembers(Hyperparameters) if not member[0].startswith('_') and isinstance(member[1], Hyperparameter)])

    print(list(Hyperparameters))

    hs = Hyperparameters([
        Hyperparameters.PITCH_SHIFT_AUGMENTATION_RANGE(3)
    ])
    import json
    print(hs.dumps(json))
    print(Hyperparameters.from_dict(hs.to_dict()).to_dict())
    print(hs[Hyperparameters.PITCH_SHIFT_AUGMENTATION_RANGE])
    print(hs[Hyperparameters.GAUSSIAN_NOISE_AUGMENTATION_MAX])
    print(hs.values)
    hs[Hyperparameters.GAUSSIAN_NOISE_AUGMENTATION_MAX] = .5
    print(hs[Hyperparameters.GAUSSIAN_NOISE_AUGMENTATION_MAX])
    print(hs.values)
    hs['gaussian_noise'] = .75
    print(hs[Hyperparameters.GAUSSIAN_NOISE_AUGMENTATION_MAX])
    
    # print('This is expected to fail')
    # print(Hyperparameters.loads('{"values": [{"name": "gaussian_noise", "value": 0.5, "default_value": 1.0}]}', json))

    print('\n'.join(repr(h) for h in Hyperparameters.combinations([Hyperparameters.GAUSSIAN_NOISE_AUGMENTATION_MAX([0, .25, .5]), Hyperparameters.PITCH_SHIFT_AUGMENTATION_RANGE([-1, 0, 1])])))
    print('\n'.join(repr(h) for h in Hyperparameters.combinations([Hyperparameters.GAUSSIAN_NOISE_AUGMENTATION_MAX([0, .25, .5])])))
    print('\n'.join(repr(h) for h in Hyperparameters.combinations([])))