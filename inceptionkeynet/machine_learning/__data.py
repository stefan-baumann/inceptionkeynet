from typing import *

# Import tensorflow with correct log level
import inceptionkeynet
__log_levels = { 'DEBUG': '0', 'INFO': '1', 'WARNING': '2' }
if inceptionkeynet.TERMINAL_LOG_LEVEL in __log_levels:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = __log_levels[inceptionkeynet.TERMINAL_LOG_LEVEL]

import numpy as np
import tensorflow as tf

from inceptionkeynet.machine_learning.__hyperparameters import * # Hyperparameter, HyperparameterValue, Hyperparameters
from inceptionkeynet.data import *
from inceptionkeynet.data_mining import AudioMiners
from inceptionkeynet.processing import *
from inceptionkeynet.processing.transformers import AbsoluteConstantQTransformer, AdditiveGaussianNoiseAugmentationTransformer, AdditiveGaussianNoiseTransformer, FrequencyMaskingAugmentationTransformer, FrequencyMaskingTransformer, LoudnessAugmentationTransformer, PitchShiftTransformer, SpectrogramTimeDownsamplingTransformer, TimeFrequencyDomainTimeWarpAugmentationTransformer, TimeLogFrequencyDomainBellCurveEQAugmentationTransformer, TimeMaskingAugmentationTransformer
import inceptionkeynet.utils
from inceptionkeynet.machine_learning.mirex import *
from inceptionkeynet.datasets import Datasets



class ModelExecutionType(Enum):
    TRAINING = auto(),
    VALIDATING = auto(),
    TESTING = auto(),



class DataPipeline:
    def __init__(self, hyperparameters: Hyperparameters):
        self.hyperparameters = hyperparameters

        self.miner_whitelist_strict = self.hyperparameters[Hyperparameters.AUDIO_SOURCE_WHITELIST_STRICT].value
        self.additional_miner_names = []
        if self.hyperparameters[Hyperparameters.AUDIO_SOURCE_WHITELIST].value is None:
            self.miner_whitelist = False
            self.miner_names = [miner.name for miner in AudioMiners.list_audio_miners()]
        else:
            self.miner_whitelist = True
            self.miner_names = self.hyperparameters[Hyperparameters.AUDIO_SOURCE_WHITELIST].value
            if not self.miner_whitelist_strict:
                self.additional_miner_names = [miner.name for miner in AudioMiners.list_audio_miners() if not miner.name in self.miner_names]

        self.apply_sample_weight = self.hyperparameters[Hyperparameters.USE_SAMPLE_WEIGHT].value

        self.__random = random.Random()

        self.__get_target_internal = self.__create_get_target_method()
        self.__buffered_preprocessing_chains = self.__create_preprocessing_chains()
        self.__buffered_processing_chains = self.__create_processing_chains()

        self.__output_shapes, self.__output_types = self.__get_dataset_output_info()

    def __create_get_target_method(self) -> Callable[[KeyRoot, KeyMode], Union[tf.Tensor, Dict[str, tf.Tensor]]]:
        if self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT:
            def get_target(root: KeyRoot, mode: AnnotatedKeyMode):
                return tf.constant([root.value], dtype=tf.int32)
        elif self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.MODE:
            def get_target(root: KeyRoot, mode: AnnotatedKeyMode):
                return tf.constant([mode.to_int()], dtype=tf.int32)
        elif self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_COMBINED:
            def get_target(root: KeyRoot, mode: AnnotatedKeyMode):
                return tf.constant([root.value + (mode.to_int() * len(KeyRoot))], dtype=tf.int32)
        elif self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_SEQUENTIAL:
            def get_target(root: KeyRoot, mode: AnnotatedKeyMode):
                target = np.zeros(shape=(len(KeyRoot) + len(KeyMode),))
                target[root.value] = 1
                target[mode.to_int() + len(KeyRoot)] = 1
                return tf.constant(target, dtype=tf.int32)
        elif self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_SEPARATE:
            def get_target(root: KeyRoot, mode: AnnotatedKeyMode):
                target = np.zeros(shape=(len(KeyRoot) + len(KeyMode),))
                target[root.value] = 1
                target[mode.to_int() + len(KeyRoot)] = 1
                return { 'output_root': [root.value], 'output_mode': [mode.to_int()], 'output_sequential': tf.constant(target, dtype=tf.float32) }
        else:
            raise ValueError('Unknown model classification type.')

        return get_target
    
    def __create_preprocessing_chains(self) -> List[Tuple[str, TransformerChain, List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]]]:
        default_chain: List[Transformer] = []
        default_label_ops: List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]] = []
        augmentation_chains: List[Tuple[List[Transformer], List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]]] = [([], [])]

        def append_transformer(transformer: Transformer, label_op: Optional[Callable[[Tuple[KeyRoot, KeyMode]], Tuple[KeyRoot, KeyMode]]] = None):
            default_chain.append(transformer)
            if not label_op is None:
                default_label_ops.append(label_op)
            for chain, label_ops in augmentation_chains:
                chain.append(transformer)
                if not label_op is None:
                    label_ops.appennd(label_op)
        
        def append_augmenting_transformer(augmenting_transformer: Union[Transformer, List[Transformer]], label_op: Optional[Union[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]], List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]]] = None, default_transformer: Optional[Transformer] = None, default_label_op: Optional[Callable[[Tuple[KeyRoot, KeyMode]], Tuple[KeyRoot, KeyMode]]] = None):
            nonlocal augmentation_chains
            if not default_transformer is None:
                default_chain.append(default_transformer)
                if not default_label_op is None:
                    default_label_ops.append(default_label_op)
            if isinstance(augmenting_transformer, list):
                old_chains = augmentation_chains
                augmentation_chains = []
                for old_chain, old_label_ops in old_chains:
                    for i_t, transformer in enumerate(augmenting_transformer):
                        if not label_op is None:
                            if isinstance(label_op, list):
                                new_label_ops = old_label_ops + [label_op[i_t]]
                            else:
                                new_label_ops = old_label_ops + [label_op]
                        else:
                            new_label_ops = old_label_ops
                        augmentation_chains.append((old_chain + [transformer], new_label_ops))
            else:
                for chain, label_ops in augmentation_chains:
                    chain.append(augmenting_transformer)
                    if not label_op is None:
                        label_ops.append(label_op)

        if self.hyperparameters[Hyperparameters.PITCH_SHIFT_AUGMENTATION_RANGE].value != 0:
            shift_range = self.hyperparameters[Hyperparameters.PITCH_SHIFT_AUGMENTATION_RANGE].value
            transformers = []
            label_ops = []
            def create_shift_label_op(shift):
                def label_op(root, mode):
                    return (KeyRoot((root.value + shift) % len(KeyRoot)), mode)
                return label_op
            for shift in [s for s in range(-shift_range, shift_range + 1) if s != 0]:
                transformers.append(PitchShiftTransformer(shift_semitones=shift))
                label_ops.append(create_shift_label_op(shift))
            append_augmenting_transformer(transformers, label_ops, default_transformer=None, default_label_op=None)

        if bool(self.hyperparameters[Hyperparameters.TRANSFORMATION_TYPE].value & TimeFrequencyTransformationType.CONSTANT_Q):
            append_transformer(AbsoluteConstantQTransformer())
        else:
            raise ValueError('Unknown transformer chain base type.')

        if self.hyperparameters[Hyperparameters.SPECTROGRAM_TIME_DOWNSCALING_FACTOR].value != 1:
            append_transformer(SpectrogramTimeDownsamplingTransformer(self.hyperparameters[Hyperparameters.SPECTROGRAM_TIME_DOWNSCALING_FACTOR].value))
                


        def create_label_op_applicator(label_ops: List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]) -> Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]:
            if len(label_ops) > 0:
                def apply_label_ops(root: KeyRoot, mode: KeyMode) -> Tuple[KeyRoot, KeyMode]:
                    for label_op in label_ops:
                        root, mode = label_op(root, mode)
                    return root, mode
            else:
                def apply_label_ops(root: KeyRoot, mode: KeyMode) -> Tuple[KeyRoot, KeyMode]:
                    return root, mode
            return apply_label_ops

        default_chain = TransformerChain(default_chain)
        buffered_chains: List[Tuple[str, TransformerChain, List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]]] = [(default_chain.get_disk_name(), default_chain, create_label_op_applicator(default_label_ops))] # Index 0 is default, all others are with augmentation
        for chain, label_ops in augmentation_chains:
            chain = TransformerChain(chain)
            buffered_chains.append((chain.get_disk_name(), chain, create_label_op_applicator(label_ops)))

        return buffered_chains
        
    def __create_processing_chains(self) -> List[Tuple[str, TransformerChain, List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]]]:
        default_chain: List[Transformer] = []
        default_label_ops: List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]] = []
        augmentation_chains: List[Tuple[List[Transformer], List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]]] = [([], [])]

        def append_transformer(transformer: Transformer, label_op: Optional[Callable[[Tuple[KeyRoot, KeyMode]], Tuple[KeyRoot, KeyMode]]] = None):
            default_chain.append(transformer)
            if not label_op is None:
                default_label_ops.append(label_op)
            for chain, label_ops in augmentation_chains:
                chain.append(transformer)
                if not label_op is None:
                    label_ops.appennd(label_op)
        
        def append_augmenting_transformer(augmenting_transformer: Union[Transformer, List[Transformer]], label_op: Optional[Union[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]], List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]]] = None, default_transformer: Optional[Transformer] = None, default_label_op: Optional[Callable[[Tuple[KeyRoot, KeyMode]], Tuple[KeyRoot, KeyMode]]] = None):
            nonlocal augmentation_chains
            if not default_transformer is None:
                default_chain.append(default_transformer)
                if not default_label_op is None:
                    default_label_ops.append(default_label_op)
            if isinstance(augmenting_transformer, list):
                old_chains = augmentation_chains
                augmentation_chains = []
                for old_chain, old_label_ops in old_chains:
                    for i_t, transformer in enumerate(augmenting_transformer):
                        if not label_op is None:
                            if isinstance(label_op, list):
                                new_label_ops = old_label_ops + [label_op[i_t]]
                            else:
                                new_label_ops = old_label_ops + [label_op]
                        else:
                            new_label_ops = old_label_ops
                        augmentation_chains.append((old_chain + [transformer], new_label_ops))
            else:
                for chain, label_ops in augmentation_chains:
                    chain.append(augmenting_transformer)
                    if not label_op is None:
                        label_ops.append(label_op)

        if self.hyperparameters[Hyperparameters.GAUSSIAN_NOISE_AUGMENTATION_MAX].value > 0:
            append_augmenting_transformer(AdditiveGaussianNoiseAugmentationTransformer(0, self.hyperparameters[Hyperparameters.GAUSSIAN_NOISE_AUGMENTATION_MAX].value))

        if self.hyperparameters[Hyperparameters.RANDOM_EQ_AUGMENTATION_S_MAX].value > 0:
            append_augmenting_transformer(TimeLogFrequencyDomainBellCurveEQAugmentationTransformer((self.hyperparameters[Hyperparameters.RANDOM_EQ_AUGMENTATION_I_FREQ_MIN].value, self.hyperparameters[Hyperparameters.RANDOM_EQ_AUGMENTATION_I_FREQ_MAX].value), self.hyperparameters[Hyperparameters.RANDOM_EQ_AUGMENTATION_SIGMA_MAX].value, self.hyperparameters[Hyperparameters.RANDOM_EQ_AUGMENTATION_S_MAX].value))

        if self.hyperparameters[Hyperparameters.LOUDNESS_AUGMENTATION_FACTOR_RANGE].value > 1:
            append_augmenting_transformer(LoudnessAugmentationTransformer(self.hyperparameters[Hyperparameters.LOUDNESS_AUGMENTATION_FACTOR_RANGE].value))
        
        if self.hyperparameters[Hyperparameters.TIME_WARPING_DISTANCE_RANGE].value > 0:
            append_augmenting_transformer(TimeFrequencyDomainTimeWarpAugmentationTransformer(self.hyperparameters[Hyperparameters.TIME_WARPING_DISTANCE_RANGE].value))
        
        if self.hyperparameters[Hyperparameters.TIME_MASKING_LENGTH_RANGE].value > 0:
            append_augmenting_transformer(TimeMaskingAugmentationTransformer(self.hyperparameters[Hyperparameters.TIME_MASKING_LENGTH_RANGE].value))
        
        if self.hyperparameters[Hyperparameters.FREQUENCY_MASKING_LENGTH_RANGE].value > 0:
            append_augmenting_transformer(FrequencyMaskingAugmentationTransformer(self.hyperparameters[Hyperparameters.FREQUENCY_MASKING_LENGTH_RANGE].value))
        


        def create_label_op_applicator(label_ops: List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]) -> Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]:
            if len(label_ops) > 0:
                def apply_label_ops(root: KeyRoot, mode: KeyMode) -> Tuple[KeyRoot, KeyMode]:
                    for label_op in label_ops:
                        root, mode = label_op(root, mode)
                    return root, mode
            else:
                def apply_label_ops(root: KeyRoot, mode: KeyMode) -> Tuple[KeyRoot, KeyMode]:
                    return root, mode
            return apply_label_ops

        default_chain = TransformerChain(default_chain)
        buffered_chains: List[Tuple[str, TransformerChain, List[Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]]] = [(default_chain.get_disk_name(), default_chain, create_label_op_applicator(default_label_ops))] # Index 0 is default, all others are with augmentation
        for chain, label_ops in augmentation_chains:
            chain = TransformerChain(chain)
            buffered_chains.append((chain.get_disk_name(), chain, create_label_op_applicator(label_ops)))
        
        return buffered_chains

    def __get_dataset_output_info(self) -> Tuple[Tuple[int], Tuple[tf.DType]]:
        # TODO: Remove hardcoded input data shape
        if self.hyperparameters[Hyperparameters.MODEL_INPUT_TYPE].value == ModelInputType.SPECTRUM:
            input_shape = (self.hyperparameters[Hyperparameters.TRANSFORMATION_TYPE].value.get_spectrogram_width(),)
        else:
            input_shape = (self.hyperparameters[Hyperparameters.TRANSFORMATION_TYPE].value.get_spectrogram_width(), None, 1)
        
        if self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_SEPARATE:
            output_shapes = (input_shape, { 'output_root': (1,), 'output_mode': (1,), 'output_sequential': (14,) })
            output_types = (tf.float32, { 'output_root': tf.int32, 'output_mode': tf.int32, 'output_sequential': tf.float32 })
        elif self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_SEQUENTIAL:
            output_shapes = (input_shape, 14)
            output_types = (tf.float32, tf.int32)
        else:
            output_shapes = (input_shape, 1)
            output_types = (tf.float32, tf.int32)

        if not self.apply_sample_weight:
            return output_shapes, output_types
        else:
            return output_shapes + ((1,),), output_types + (tf.float32,)



    def __get_target(self, root: KeyRoot, mode: KeyMode) -> Any:
        return self.__get_target_internal(root, mode)

    def __get_sample_weight(self, entry: MusicPiece) -> Any:
        if self.apply_sample_weight:
            return tf.constant(min(1, max(0, (entry.root.certainty if not entry.root.certainty is None else 1) / 2 + (entry.mode.certainty if not entry.mode.certainty is None else 1) / 2)), dtype=tf.float32)
        else:
            return tf.constant([1], dtype=tf.float32)
    
    def __get_preprocessing_chain(self, perform_augmentation: bool = False) -> Tuple[str, Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]:
        chain_name, _, label_op = self.__buffered_preprocessing_chains[self.__random.randrange(0, len(self.__buffered_preprocessing_chains))] if perform_augmentation else self.__buffered_preprocessing_chains[0]
        return chain_name, label_op
    
    def __get_processing_chain(self, perform_augmentation: bool = False) -> Tuple[TransformerChain, Callable[[KeyRoot, KeyMode], Tuple[KeyRoot, KeyMode]]]:
        _, chain, label_op = self.__random.choice(self.__buffered_processing_chains) if perform_augmentation else self.__buffered_processing_chains[0]
        return chain, label_op
    


    def __get_preprocessed_entries(self, entries: Iterable[MusicPiece], perform_augmentation: bool) -> List[MusicPiece]:
        n_before = len(entries)
        for chain_name, chain, _ in (self.__buffered_preprocessing_chains if perform_augmentation else self.__buffered_preprocessing_chains[0:1]):
            available_entries = []
            for entry in entries:
                if chain_name in entry.files_relative and isinstance(entry.files_relative[chain_name], list) and len(entry.files_relative[chain_name]) > 0 and all(os.path.isfile(inceptionkeynet.io.get_file_path(file)) and os.path.getsize(inceptionkeynet.io.get_file_path(file)) > 0 for file in entry.files_relative[chain_name]):
                    available_uuids = [os.path.splitext(os.path.split(inceptionkeynet.utils.make_path_compatible(entry.files_relative[miner_name]))[-1])[0] for miner_name in (self.miner_names if self.miner_whitelist_strict else self.miner_names + self.additional_miner_names) if miner_name in entry.files_relative and not entry.files_relative[miner_name] is None]
                    if len([file for file in entry.files_relative[chain_name] if any((uuid in file) for uuid in available_uuids)]) > 0:
                        available_entries.append(entry)
            entries = available_entries
        if len(entries) < n_before:
            logging.getLogger(__name__).warning(f'{len(entries)}/{n_before} entries available.')
        else:
            logging.getLogger(__name__).info(f'{len(entries)}/{n_before} entries available.')
        return entries

    def __get_random_sample(self, entry: MusicPiece, chain_name: str, ignore_whitelist: bool = False) -> np.ndarray:
        if self.miner_whitelist and not ignore_whitelist:
            accepted_uuids = [os.path.splitext(os.path.split(inceptionkeynet.utils.make_path_compatible(entry.files_relative[miner_name]))[-1])[0] for miner_name in self.miner_names if miner_name in entry.files_relative and not entry.files_relative[miner_name] is None]
            valid_files = [file for file in entry.files_relative[chain_name] if any((uuid in file) for uuid in accepted_uuids)]
            if len(valid_files) == 0 and not self.miner_whitelist_strict:
                accepted_uuids = [os.path.splitext(os.path.split(inceptionkeynet.utils.make_path_compatible(entry.files_relative[miner_name]))[-1])[0] for miner_name in self.additional_miner_names if miner_name in entry.files_relative and not entry.files_relative[miner_name] is None]
                valid_files = [file for file in entry.files_relative[chain_name] if any((uuid in file) for uuid in accepted_uuids)]
            if len(valid_files) == 0:
                raise AssertionError(f'No sample available for "{entry.title}" by "{entry.artist}" (chain name "{chain_name}") despite prior checking indicating one is available.')
            file = self.__random.choice(valid_files)
        else:
            valid_files = entry.files_relative[chain_name]
            file = self.__random.choice(valid_files)
        return inceptionkeynet.io.read_data(file)

    def __get_testing_sample(self, entry: MusicPiece, chain_name: str, ignore_whitelist: bool = False) -> np.ndarray:
        # Same as __get_ramdom_sample, but deterministically returns the same sample. For all usecases in this project, just choosing the first available sample (from the whitelist if possible, then from another source if no strict whitelist is applied) suffices, as there are no cases where this behaviour wouldn't make sense with the datasets available
        # TODO: If you have multiple datasets with a clear hierarchy beyond the whitelist, which you want to apply for testing, extend this method
        if self.miner_whitelist and not ignore_whitelist:
            accepted_uuids = [os.path.splitext(os.path.split(inceptionkeynet.utils.make_path_compatible(entry.files_relative[miner_name]))[-1])[0] for miner_name in self.miner_names if miner_name in entry.files_relative and not entry.files_relative[miner_name] is None]
            valid_files = [file for file in entry.files_relative[chain_name] if any((uuid in file) for uuid in accepted_uuids)]
            if len(valid_files) == 0 and not self.miner_whitelist_strict:
                accepted_uuids = [os.path.splitext(os.path.split(inceptionkeynet.utils.make_path_compatible(entry.files_relative[miner_name]))[-1])[0] for miner_name in self.additional_miner_names if miner_name in entry.files_relative and not entry.files_relative[miner_name] is None]
                valid_files = [file for file in entry.files_relative[chain_name] if any((uuid in file) for uuid in accepted_uuids)]
            if len(valid_files) == 0:
                raise AssertionError(f'No sample available for "{entry.title}" by "{entry.artist}" (chain name "{chain_name}") despite prior checking indicating one is available.')
            file = valid_files[0]
        else:
            valid_files = entry.files_relative[chain_name]
            file = valid_files[0]
        return inceptionkeynet.io.read_data(file)

    def create_tf_dataset(self, entries: Iterable[MusicPiece], perform_augmentation: Union[bool, Callable[[], bool]] = False, model_execution_type: Optional[ModelExecutionType] = None, processing_threads: int = 12) -> tf.data.Dataset:
        entries = list(entries) if not isinstance(entries, list) else entries
        logging.getLogger(__name__).info(f'Checking preprocessed files...')
        entries = self.__get_preprocessed_entries(entries, perform_augmentation=(perform_augmentation if isinstance(perform_augmentation, bool) else perform_augmentation()))

        min_confidence = self.hyperparameters[Hyperparameters.MINIMUM_CONFIDENCE].value
        if min_confidence > 0:
            n_prior = len(entries)
            entries = [entry for entry in entries if (entry.root.certainty is None or entry.root.certainty >= min_confidence) and (entry.mode.certainty is None or entry.mode.certainty >= min_confidence)]
            logging.getLogger(__name__).info(f'Using {len(entries)}/{n_prior} samples due to a minimum confidence restriction of {min_confidence}.')

        logging.getLogger(__name__).info(f'Done checking preprocessed files.')

        if model_execution_type != ModelExecutionType.TESTING:
            dataset = tf.data.Dataset.from_tensor_slices(np.arange(len(entries))).shuffle(len(entries))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(np.arange(len(entries)))

        is_spectrogram = self.hyperparameters[Hyperparameters.MODEL_INPUT_TYPE].value == ModelInputType.SPECTROGRAM
        max_spectrogram_length = self.hyperparameters[Hyperparameters.SPECTROGRAM_MAX_LENGTH_FRAMES].value
        if not max_spectrogram_length is None:
            if (self.hyperparameters[Hyperparameters.USE_FULL_TRACKS_FOR_VALIDATION].value and model_execution_type == ModelExecutionType.VALIDATING) or (self.hyperparameters[Hyperparameters.USE_FULL_TRACKS_FOR_TESTING].value and model_execution_type == ModelExecutionType.TESTING):
                max_spectrogram_length = None
        def load_entry(i: tf.Tensor):
            nonlocal perform_augmentation
            i = int(i.numpy())
            entry = entries[i]
            perform_augmentation_on_entry = perform_augmentation if isinstance(perform_augmentation, bool) else perform_augmentation()
            chain_name, label_op = self.__get_preprocessing_chain(perform_augmentation_on_entry)
            # print(f'Preprocessing: {chain_name}: {(entry.root.value, entry.mode.value)} -> {label_op(entry.root.value, entry.mode.value)}')
            processing_chain, processing_label_op = self.__get_processing_chain(perform_augmentation_on_entry)
            # print(f'Postprocessing: {processing_chain}: {label_op(entry.root.value, entry.mode.value)} -> {processing_label_op(*label_op(entry.root.value, entry.mode.value))}')
            
            x = processing_chain(self.__get_random_sample(entry, chain_name) if model_execution_type == ModelExecutionType.TRAINING else self.__get_testing_sample(entry, chain_name))
            if is_spectrogram and not max_spectrogram_length is None:
                rounds = 0
                while x.shape[1] < max_spectrogram_length: # Hotfix for audio files that are too small, only works when at least one audio preview is long enough though and causes an endless loop otherwise
                    x = processing_chain(self.__get_random_sample(entry, chain_name) if model_execution_type == ModelExecutionType.TRAINING else self.__get_testing_sample(entry, chain_name))
                    rounds += 1
                    if rounds == 100:
                        logging.getLogger(__name__).error(f'No spectrogram of sufficient length found for "{entry.name}" by "{entry.artist}" (uuids {[os.path.splitext(os.path.split(inceptionkeynet.utils.make_path_compatible(entry.files_relative[miner_name]))[-1])[0] for miner_name in (self.miner_names if self.miner_whitelist_strict else self.miner_names + self.additional_miner_names) if miner_name in entry.files_relative and not entry.files_relative[miner_name] is None]}).')
                        x = np.repeat(x, 10, axis=1)
                offset = self.__random.randrange(0, x.shape[1] - max_spectrogram_length + 1)
                x = x[:,offset:offset + max_spectrogram_length]

            if is_spectrogram:
                x = np.expand_dims(x, -1)
            y = self.__get_target(*processing_label_op(*label_op(entry.root.value, entry.mode.value)))
            if not self.apply_sample_weight:
                return x, y
            else:
                w = self.__get_sample_weight(entry)
                return x, y, w
        if not self.apply_sample_weight:
            @tf.autograph.experimental.do_not_convert
            def load_entry_wrapper(i: int):
                x, y = tf.py_function(load_entry, inp=(i,), Tout=self.__output_types)
                x.set_shape(self.__output_shapes[0])
                y.set_shape(self.__output_shapes[1])
                return x, y
        else:
            @tf.autograph.experimental.do_not_convert
            def load_entry_wrapper(i: int):
                x, y, w = tf.py_function(load_entry, inp=(i,), Tout=self.__output_types)
                x.set_shape(self.__output_shapes[0])
                y.set_shape(self.__output_shapes[1])
                # w.set_shape(self.__output_shapes[2])
                return x, y, w
        dataset = dataset.map(load_entry_wrapper, num_parallel_calls=processing_threads)

        return dataset