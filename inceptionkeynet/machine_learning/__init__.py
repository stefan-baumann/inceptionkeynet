from __future__ import annotations # Enable postponed evaluation of annotations to allow for annotating methods returning their enclosing type

from datetime import datetime
import time
import glob
import multiprocessing
import signal
import re
from typing import Any, List, Optional, Tuple

# Import tensorflow with correct log level
import inceptionkeynet
__log_levels = { 'DEBUG': '0', 'INFO': '1', 'WARNING': '2' }
if inceptionkeynet.TERMINAL_LOG_LEVEL in __log_levels:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = __log_levels[inceptionkeynet.TERMINAL_LOG_LEVEL]

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Sequential
import matplotlib.pyplot as plt
from tqdm import tqdm

from inceptionkeynet.machine_learning.__hyperparameters import * # Hyperparameter, HyperparameterValue, Hyperparameters
from inceptionkeynet.machine_learning.__data import DataPipeline, ModelExecutionType
from inceptionkeynet.machine_learning.__models import *
from inceptionkeynet.machine_learning.__optimizers import *
from inceptionkeynet.machine_learning.model_utils import *
from inceptionkeynet.machine_learning.train_utils import EarlyStoppingAfterThreshold, EarlyStoppingAtThreshold, ThresholdedModelCheckpoint, sparse_categorical_crossentropy_with_label_smoothing
from inceptionkeynet.data import *
from inceptionkeynet.datasets import Datasets
from inceptionkeynet.processing import *
from inceptionkeynet.processing import transformers
import inceptionkeynet.utils
import inceptionkeynet
from inceptionkeynet.machine_learning.mirex import *
from inceptionkeynet.data_mining import AudioMiners



def __process_loaded_hyperparameters(h: Hyperparameters, path: str) -> Hyperparameters:
    if h[Hyperparameters.TRAINABLE_PARAMETER_COUNT].value is None:
        summary_path = os.path.join(path, 'metadata/summary.txt')
        if os.path.isfile(summary_path):
            with open(summary_path, 'r') as file:
                summary = file.read()
                parameter_count = int(re.search(r'Trainable params:\s*(?P<n>[\d,]+)\s*\n', summary).group('n').replace(',', ''))
                h[Hyperparameters.TRAINABLE_PARAMETER_COUNT] = parameter_count
    return h


def list_models_on_disk(add_metadata_hyperparameters: bool = True) -> Iterator[Tuple[Hyperparameters, str]]:
    for model_hs_path in (f for f in glob.glob(os.path.join(inceptionkeynet.MODELS_PATH, '*/*/metadata/hyperparameters.*'), recursive=True) if os.path.splitext(f)[1] in inceptionkeynet.utils.SUPPORTED_SERIALIZATION_FORMATS and not '.temp' in f):
        try:
            model_path = os.path.dirname(os.path.dirname(model_hs_path))
            if add_metadata_hyperparameters:
                yield (__process_loaded_hyperparameters(Hyperparameters.read(model_hs_path), model_path), model_path)
            else:
                yield (Hyperparameters.read(model_hs_path), model_path)
        except Exception:
            logging.getLogger(__name__).exception('Unexpected exception while listing models on disk.')
    yield from ()

def get_model_paths_from_hyperparameters(hyperparameters: Hyperparameters) -> Iterator[str]:
    yield from (path for h, path in list_models_on_disk(add_metadata_hyperparameters=False) if h == hyperparameters)

def get_hyperparameters_from_path(path: str, add_metadata_hyperparameters: bool = True) -> Hyperparameters:
    matching_paths = glob.glob(os.path.join(path, 'metadata/hyperparameters.*'), recursive=True)
    return __process_loaded_hyperparameters(Hyperparameters.read(matching_paths[0]), path) if add_metadata_hyperparameters else Hyperparameters.read(matching_paths[0])

def is_model_fully_trained(model_path: str, hyperparameters: Optional[Hyperparameters] = None) -> bool:
    hyperparameters = hyperparameters if not hyperparameters is None else Hyperparameters.read(next(f for f in glob.glob(os.path.join(model_path, 'metadata', 'hyperparameters.*'), recursive=True) if os.path.splitext(f)[1] in inceptionkeynet.utils.SUPPORTED_SERIALIZATION_FORMATS and not '.temp' in f))
    return all(os.path.exists(os.path.join(model_path, 'folds', str(i_fold), 'checkpoints', 'final')) for i_fold in range(hyperparameters[Hyperparameters.CROSS_VALIDATION_FOLD_COUNT].value)) and os.path.exists(os.path.join(model_path)) and os.path.exists(os.path.join(model_path, 'graphs'))

def does_trained_model_exist(hyperparameters: Hyperparameters) -> bool:
    for path in get_model_paths_from_hyperparameters(hyperparameters):
        if all(os.path.exists(os.path.join(path, 'folds', str(i_fold), 'checkpoints', 'final')) for i_fold in range(hyperparameters[Hyperparameters.CROSS_VALIDATION_FOLD_COUNT].value)):
            return True
    return False

def list_trained_models_on_disk(add_metadata_hyperparameters: bool = True) -> Iterator[Tuple[Hyperparameters, str]]:
    yield from ((h, path) for h, path in list_models_on_disk(add_metadata_hyperparameters=add_metadata_hyperparameters) if is_model_fully_trained(path, h))







class ModelTrainer:
    # initialized_session = False

    def __init__(self, hyperparameters: Hyperparameters):
        # if not ModelTrainer.initialized_session:
        #     # Workaround for currently unfixed error in latest tf (https://github.com/tensorflow/tensorflow/issues/43174)
        #     # Until https://github.com/tensorflow/tensorflow/pull/44486 is merged, there are still gonna be some slowdowns though when ptxas isn't found (can be fixed by adding /usr/local/cuda-11.1/bin to $PATH if using CUDA 11.1)
        #     from tensorflow.compat.v1 import ConfigProto
        #     from tensorflow.compat.v1 import InteractiveSession
        #     config = ConfigProto()
        #     config.gpu_options.allow_growth = True
        #     session = InteractiveSession(config=config)
        #     ModelTrainer.initialized_session = True

        self.hyperparameters = hyperparameters
        self.__timestamp = datetime.now()

        if self.hyperparameters[Hyperparameters.VALIDATE_ON_TEST_DATASET].value and not self.hyperparameters[Hyperparameters.CROSS_VALIDATION_FOLD_COUNT].value == 1:
            raise ValueError('If VALIDATE_ON_TEST_DATASET is set to true, CROSS_VALIDATION_FOLD_COUNT must be set to 1.')
    


    def get_model_name(self) -> str:
        # return f'{self.hyperparameters[Hyperparameters.MODEL_NAME].value}_{self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value.value}_{"_".join(f"{h.hyperparameter.name}-{inceptionkeynet.utils.name_to_path_snake_case(str(h.value))}" for h in self.hyperparameters if h.value != h.hyperparameter.default_value and not h.hyperparameter in [Hyperparameters.MODEL_NAME, Hyperparameters.MODEL_CLASSIFICATION_TYPE])}'

        hs_string = "_".join(f"{'_'.join([sp[0] for sp in h.hyperparameter.name.split('_') if len(sp) > 0])}-{inceptionkeynet.utils.name_to_path_snake_case(str(h.value))}" for h in self.hyperparameters if h.value != h.hyperparameter.default_value and not h.hyperparameter in [Hyperparameters.MODEL_NAME, Hyperparameters.MODEL_CLASSIFICATION_TYPE])
        name_string = f'{self.hyperparameters[Hyperparameters.MODEL_NAME].value}_{self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value.value}_{hs_string}'
        result = ('_'.join([sp for sp in name_string.split('_') if len(sp) > 0])).replace('-_', '-').replace('_-', '-')
        result_truncated = result[:200] + (result[200:] and '___')
        return result_truncated

    def get_timestamp(self) -> str:
        return self.__timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    def get_path(self, subpath: str = '', filename: Optional[str] = None, fold: Optional[Union[int, str]] = None, create_if_needed: bool = True) -> str:
        if fold is None:
            path = os.path.join(inceptionkeynet.MODELS_PATH, self.get_model_name(), self.get_timestamp(), inceptionkeynet.utils.make_path_compatible(subpath))
        else:
            path = os.path.join(inceptionkeynet.MODELS_PATH, self.get_model_name(), self.get_timestamp(), 'folds', str(fold), inceptionkeynet.utils.make_path_compatible(subpath))
        
        if create_if_needed and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        if not filename is None:
            return os.path.join(path, filename)
        else:
            return path



    def get_train_dataset(self) -> Dataset:
        if self.hyperparameters[Hyperparameters.TRAIN_DATASETS].value is None:
            excluded_dataset_names = self.hyperparameters[Hyperparameters.TEST_DATASET].value
            if excluded_dataset_names is None:
                excluded_dataset_names = []
            elif isinstance(excluded_dataset_names, list):
                pass # nothing to do
            else:
                excluded_dataset_names = [excluded_dataset_names]
            datasets = [dataset for dataset in Datasets if not dataset.name in excluded_dataset_names]
        else:
            train_dataset_names = self.hyperparameters[Hyperparameters.TRAIN_DATASETS].value
            if isinstance(train_dataset_names, list):
                pass # nothing to do
            else:
                train_dataset_names = [train_dataset_names]
            datasets = [Datasets.get_from_name(name) for name in train_dataset_names]
        DictSerializable.write_dict({ 'datasets': [dataset.name for dataset in datasets] }, self.get_path('metadata', 'datasets' + inceptionkeynet.DEFAULT_FORMAT))
        merged = Dataset.merge_multi(datasets, 'Overall', include_original_metadata=False)
        min_confidence = self.hyperparameters[Hyperparameters.MINIMUM_CONFIDENCE].value
        merged.entries = [entry for entry in merged.entries if (entry.root.certainty is None or entry.root.certainty >= min_confidence) and (entry.mode.certainty is None or entry.mode.certainty >= min_confidence)]
        merged.entries = [entry.remove_nonexistent_file_links() for entry in merged.entries]
        return merged

    def get_test_dataset(self) -> Optional[Dataset]:
        if self.hyperparameters[Hyperparameters.TEST_DATASETS].value is None:
            return None
        else:
            dataset_names = self.hyperparameters[Hyperparameters.TEST_DATASETS].value
            if isinstance(dataset_names, list):
                pass # nothing to do
            else:
                dataset_names = [dataset_names]
            datasets = [Datasets.get_from_name(name) for name in dataset_names]
        DictSerializable.write_dict({ 'datasets': [dataset.name for dataset in datasets] }, self.get_path('metadata', 'datasets' + inceptionkeynet.DEFAULT_FORMAT))
        merged = Dataset.merge_multi(datasets, 'Overall', include_original_metadata=False)
        min_confidence = self.hyperparameters[Hyperparameters.MINIMUM_CONFIDENCE].value
        merged.entries = [entry for entry in merged.entries if (entry.root.certainty is None or entry.root.certainty >= min_confidence) and (entry.mode.certainty is None or entry.mode.certainty >= min_confidence)]
        merged.entries = [entry.remove_nonexistent_file_links() for entry in merged.entries]
        return merged
    
    def get_mined_indices(self, dataset: Dataset) -> List[int]:
        miner_names = [miner.name for miner in AudioMiners.list_audio_miners()]
        return [i for i in range(len(dataset)) if any(miner_name in dataset[i].files_relative and not dataset[i].files_relative[miner_name] is None for miner_name in miner_names)]

    def create_folds(self, dataset: Dataset) -> List[List[MusicPiece]]:
        processed_piece_indices = self.get_mined_indices(dataset) # list(range(len(dataset.entries))) # self.get_processed_indices(dataset, chain_name)
        logging.getLogger(__name__).info(f'{len(processed_piece_indices)} of {len(dataset)} training entries have audio available.')
        if self.hyperparameters[Hyperparameters.CROSS_VALIDATION_FOLD_COUNT].value == 1:
            return [dataset.entries]
        if len(processed_piece_indices) < self.hyperparameters[Hyperparameters.CROSS_VALIDATION_FOLD_COUNT].value:
            raise ValueError('Cannot create folds if the amount of samples is smaller than the amount of folds.')
        random.shuffle(processed_piece_indices)
        fold_indices = np.array_split(processed_piece_indices, self.hyperparameters[Hyperparameters.CROSS_VALIDATION_FOLD_COUNT].value)
        return [[dataset[i] for i in fold] for fold in fold_indices]

    def apply_batching(self, dataset: tf.data.Dataset, model_execution_type: Optional[ModelExecutionType] = None, batch_size_override : Optional[int] = None) -> tf.data.Dataset:
        batch_size = self.hyperparameters[Hyperparameters.BATCH_SIZE].value if batch_size_override is None else batch_size_override
        return dataset.batch(batch_size).prefetch(1024 // batch_size)
    
    def create_model(self) -> tf.keras.Model:
        return Models.get_by_name(self.hyperparameters[Hyperparameters.MODEL_NAME].value).create_model(self.hyperparameters)
    
    def create_loss_and_metrics(self) -> Tuple[List[Any], List[Any]]:
        losses = []
        metrics = []
        if self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_SEQUENTIAL:
            losses.append(create_sequential_output_double_categorical_crossentropy_loss())

            metrics.append(create_sequential_output_combined_accuracy_metric())
            metrics.append(get_mirex_score(length=14))
        elif self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_SEPARATE:
            losses = {
                'output_root': sparse_categorical_crossentropy_with_label_smoothing(len(KeyRoot), self.hyperparameters[Hyperparameters.LABEL_SMOOTHING].value),
                'output_mode': sparse_categorical_crossentropy_with_label_smoothing(len(KeyMode), self.hyperparameters[Hyperparameters.LABEL_SMOOTHING].value),
                'output_sequential': None
            }

            metrics = {
                'output_root': 'accuracy',
                'output_mode': 'accuracy',
                'output_sequential': get_mirex_score(length=14)
            }
        elif self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_COMBINED:
            losses.append(sparse_categorical_crossentropy_with_label_smoothing(len(KeyRoot) * len(KeyMode), self.hyperparameters[Hyperparameters.LABEL_SMOOTHING].value))

            metrics.append('accuracy')
            metrics.append(get_mirex_score(length=(len(KeyRoot) * len(KeyMode))))
        else: # Only root or mode
            losses.append(sparse_categorical_crossentropy_with_label_smoothing(len(KeyRoot) if self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT else len(KeyMode), self.hyperparameters[Hyperparameters.LABEL_SMOOTHING].value))

            metrics.append('accuracy')
        return losses, metrics

    
    
    def train(self) -> List[Dict[str, float]]:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        logging.getLogger(__name__).info(f'Training with hyperparameters:')
        for hyperparameter in self.hyperparameters:
            print(f' - {hyperparameter.hyperparameter.name}: {hyperparameter.value} (default: {hyperparameter.hyperparameter.default_value})')
        self.hyperparameters.write(self.get_path('metadata', 'hyperparameters' + inceptionkeynet.DEFAULT_FORMAT))
        
        logging.getLogger(__name__).debug(f'Loading train dataset...')
        train_dataset = self.get_train_dataset()
        logging.getLogger(__name__).debug(f'Loading test dataset...')
        test_dataset = self.get_test_dataset()
        # logging.getLogger(__name__).debug(f'Loading transformer chain...')
        # chain = self.get_transformer_chain()
        logging.getLogger(__name__).debug(f'Creating folds...')
        # folds = self.create_folds(train_dataset, chain.get_disk_name())
        folds = self.create_folds(train_dataset)
        pipeline = DataPipeline(self.hyperparameters)

        test_datset_indices = self.get_mined_indices(test_dataset) # list(range(len(test_dataset.entries))) # self.get_processed_indices(test_dataset, chain.get_disk_name())
        
        histories = []
        fold_test_metrics = []
        for i_val in range(len(folds)):

            logging.getLogger(__name__).info(f'Training fold {i_val + 1}/{len(folds)}...')
            logging.getLogger(__name__).debug(f'Assembling datasets...')

            if not self.hyperparameters[Hyperparameters.VALIDATE_ON_TEST_DATASET].value:
                train_entries = itertools.chain(*[fold for i, fold in enumerate(folds) if i != i_val])
            else:
                train_entries = folds[0]
            train = pipeline.create_tf_dataset(train_entries, perform_augmentation=True, model_execution_type=ModelExecutionType.TRAINING)
            
            if not self.hyperparameters[Hyperparameters.VALIDATE_ON_TEST_DATASET].value:
                train = self.apply_batching(train, ModelExecutionType.TRAINING)
                validation = self.apply_batching(pipeline.create_tf_dataset(folds[i_val], model_execution_type=ModelExecutionType.VALIDATING), ModelExecutionType.VALIDATING, batch_size_override=(1 if self.hyperparameters[Hyperparameters.USE_FULL_TRACKS_FOR_VALIDATION].value else None)) # self.apply_batching(fold_datasets[i_val], len(folds[i_val]))
            else:
                train = self.apply_batching(train, ModelExecutionType.TRAINING)
                validation = self.apply_batching(pipeline.create_tf_dataset([test_dataset.entries[i] for i in test_datset_indices], model_execution_type=ModelExecutionType.TESTING), model_execution_type=ModelExecutionType.TESTING, batch_size_override=(1 if self.hyperparameters[Hyperparameters.USE_FULL_TRACKS_FOR_TESTING].value else None))
            
            logging.getLogger(__name__).debug(f'Compiling model...')
            model = self.create_model()

            # Transform models with separate outputs to a single model with a sequential output for training so that mirex metrics can be computed without adding much overhead from re-predicting each sample
            if self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_SEPARATE:
                model = add_sequential_output_to_split_model(model)

            if self.hyperparameters[Hyperparameters.MODEL_INPUT_TYPE].value == ModelInputType.SPECTRUM:
                model.build(input_shape=model.input_shape)

            # Create losses & metrics according to model structure
            losses, test_metrics = self.create_loss_and_metrics()

            model.compile(loss=losses, optimizer=get_optimizer(self.hyperparameters), metrics=test_metrics)
            model.summary()
            with inceptionkeynet.utils.open_mkdirs(self.get_path('metadata', 'summary.txt'),'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\r\n'))
            
            logging.getLogger(__name__).debug(f'Training model...')
            
            callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=inceptionkeynet.utils.make_path_compatible(f'logs/{self.get_model_name()}-{self.get_timestamp()}-fold-{i_val}')),
                ThresholdedModelCheckpoint(self.get_path('checkpoints/loss', fold=i_val), monitor='val_loss', mode='min', threshold=3, save_best_only=True)
            ]
            if self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value in [ModelClassificationType.ROOT_MODE_SEQUENTIAL, ModelClassificationType.ROOT_MODE_COMBINED]:
                callbacks.append(ThresholdedModelCheckpoint(self.get_path('checkpoints/mirex', fold=i_val), monitor='val_mirex_score', mode='max', threshold=.3, save_best_only=True))
            elif self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_SEPARATE:
                callbacks.append(ThresholdedModelCheckpoint(self.get_path('checkpoints/mirex', fold=i_val), monitor='val_output_sequential_mirex_score', mode='max', threshold=.3, save_best_only=True))
            if not self.hyperparameters[Hyperparameters.EARLY_STOPPING].value is None:
                callbacks.append(EarlyStoppingAfterThreshold(patience=self.hyperparameters[Hyperparameters.EARLY_STOPPING].value, threshold=3)) # Like normal early stopping but the early stopping behaviour is only enabled after reaching a threshold
                callbacks.append(tf.keras.callbacks.EarlyStopping(patience=self.hyperparameters[Hyperparameters.EARLY_STOPPING].value * 5)) # Normal early stopping for backup incase the threshold is never reached
            if self.hyperparameters[Hyperparameters.REDUCE_LR_ON_PLATEAU].value[1] != None:
                metric, epochs, factor = self.hyperparameters[Hyperparameters.REDUCE_LR_ON_PLATEAU].value
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(metric, patience=epochs, factor=factor))



            history = model.fit(train, epochs=self.hyperparameters[Hyperparameters.EPOCHS].value, callbacks=callbacks, validation_data=validation, shuffle=False)



            try:
                model.save(self.get_path('checkpoints/final', fold=i_val))
            except Exception:
                logging.getLogger(__name__).exception('Unhandled exception while trying to save model.')
                pass
            DictSerializable.write_dict(history.history, self.get_path(filename=('history' + inceptionkeynet.DEFAULT_FORMAT), fold=i_val))
            histories.append(history.history)

            

            logging.getLogger(__name__).info(f'Evaluating model...')
            logging.getLogger(__name__).debug(f'Loading test dataset...')
            if not self.hyperparameters[Hyperparameters.VALIDATE_ON_TEST_DATASET].value:
                test_dataset_tf = self.apply_batching(pipeline.create_tf_dataset([test_dataset.entries[i] for i in test_datset_indices], model_execution_type=ModelExecutionType.TESTING), model_execution_type=ModelExecutionType.TESTING, batch_size_override=(1 if self.hyperparameters[Hyperparameters.USE_FULL_TRACKS_FOR_TESTING].value else None))
            else:
                test_dataset_tf = validation

            test_metrics = { }
            checkpoint_names = ['final', 'mirex', 'loss']
            for checkpoint_name in checkpoint_names:
                checkpoint_path = self.get_path(f'checkpoints/{checkpoint_name}', fold=i_val)
                if os.path.exists(checkpoint_path):
                    checkpoint_model = self.load_model_recompile(checkpoint_path)
                    logging.getLogger(__name__).info(f'Evaluating checkpoint for {checkpoint_name}...')
                    checkpoint_evaluation = checkpoint_model.evaluate(test_dataset_tf)
                    checkpoint_metrics = dict([(name, checkpoint_evaluation[i]) for i, name in enumerate(checkpoint_model.metrics_names)])
                    test_metrics[checkpoint_name] = checkpoint_metrics
            
            DictSerializable.write_dict(test_metrics, self.get_path('evaluation', 'test' + inceptionkeynet.DEFAULT_FORMAT, fold=i_val))
            fold_test_metrics.append(test_metrics)

            # TODO: Save model predictions on test data
        

        
        def duplicateless_legend(axs):
            handles, labels = axs.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            axs.legend(*zip(*unique))
        
        def plot_average_curve(curves, label: str, c: str = 'C1', c2: str = 'C3', axs = None):
            length = max(len(c) for c in curves)
            x = list(range(1, length + 1))

            mean, upper, lower = [], [], []
            for i in range(length):
                values = [c[i] for c in curves if len(c) > i]
                mean.append(np.mean(values))
                upper.append(np.max(values))
                lower.append(np.min(values))

            for i, len_c in enumerate(sorted([len(c) for c in curves])):
                if i < len(curves) - 1:
                    if axs == None:
                        plt.axvline(x=(len_c + 1), label='early stopping points', ls='--', color='C2')
                    else:
                        axs.axvline(x=(len_c + 1), label='early stopping points', ls='--', color='C2')
            
            if axs == None:
                plt.plot(x, mean, c=c, label=label)
                plt.fill_between(x, lower, upper, alpha=.3, facecolor=c)
            else:
                axs.plot(x, mean, c=c, label=label)
                axs.fill_between(x, lower, upper, alpha=.3, facecolor=c)
                duplicateless_legend(axs)

        if self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_SEPARATE:
            fig, axs = plt.subplots(nrows=1, ncols=3)
            fig.set_figwidth(15)
            fig.set_figheight(3)
            axs[0].set_title('Key root accuracy')
            axs[0].set_ylim([0, 1])
            plot_average_curve([history['val_output_root_accuracy'] for history in histories], 'validation', 'C1', 'C3', axs[0])
            plot_average_curve([history['output_root_accuracy'] for history in histories], 'train', 'C0', 'C2', axs[0])
            axs[1].set_title('Key mode accuracy')
            axs[1].set_ylim([0, 1])
            plot_average_curve([history['val_output_mode_accuracy'] for history in histories], 'validation', 'C1', 'C3', axs[1])
            plot_average_curve([history['output_mode_accuracy'] for history in histories], 'train', 'C0', 'C2', axs[1])
            axs[2].set_title('MIREX score')
            axs[2].set_ylim([0, 1])
            plot_average_curve([history['val_output_sequential_mirex_score'] for history in histories], 'validation', 'C1', 'C3', axs[2])
            plot_average_curve([history['output_sequential_mirex_score'] for history in histories], 'train', 'C0', 'C2', axs[2])
        elif self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value in [ModelClassificationType.ROOT_MODE_SEQUENTIAL, ModelClassificationType.ROOT_MODE_COMBINED]:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            fig.set_figwidth(15)
            fig.set_figheight(5)
            axs[0].set_title('Accuracy')
            axs[0].set_ylim([0, 1])
            plot_average_curve([history['val_accuracy'] for history in histories], 'validation', 'C1', 'C3', axs[0])
            plot_average_curve([history['accuracy'] for history in histories], 'train', 'C0', 'C2', axs[0])
            axs[1].set_title('MIREX score')
            axs[1].set_ylim([0, 1])
            plot_average_curve([history['val_mirex_score'] for history in histories], 'validation', 'C1', 'C3', axs[1])
            plot_average_curve([history['mirex_score'] for history in histories], 'train', 'C0', 'C2', axs[1])
        else:
            plt.figure(figsize=(10, 6))
            plt.ylim([0, 1])
            plot_average_curve([history['val_accuracy'] for history in histories], 'validation', 'C1', 'C3')
            plot_average_curve([history['accuracy'] for history in histories], 'train', 'C0', 'C2')
            plt.legend()
        plt.tight_layout()
        plt.savefig(self.get_path('graphs', 'accuracy.png'))
        plt.clf()

        return fold_test_metrics



    def load_model(self, path: str, compile: bool = True) -> tf.keras.Model:
        custom_objects = { }
        if self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value in [ModelClassificationType.ROOT_MODE_SEQUENTIAL, ModelClassificationType.ROOT_MODE_SEPARATE]:
            custom_objects['mirex_score'] = get_mirex_score(length=14)
        elif self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_COMBINED:
            custom_objects['mirex_score'] = get_mirex_score(length=24)
        if self.hyperparameters[Hyperparameters.MODEL_CLASSIFICATION_TYPE].value == ModelClassificationType.ROOT_MODE_SEQUENTIAL:
            custom_objects['combined_accuracy'] = create_sequential_output_combined_accuracy_metric()
        
        return tf.keras.models.load_model(path, custom_objects=custom_objects, compile=compile)

    def load_model_recompile(self, path: str) -> tf.keras.Model:
        model = self.load_model(path, compile=False)
        losses, metrics = self.create_loss_and_metrics()
        model.compile(loss=losses, optimizer=get_optimizer(self.hyperparameters), metrics=metrics)
        return model


class ModelInferencer(ModelTrainer):
    def __init__(self, hyperparameters: Hyperparameters, path: str):
        super().__init__(hyperparameters)

        self.__path = path
    
    def get_path(self, subpath: str = '', filename: Optional[str] = None, fold: Optional[Union[int, str]] = None, create_if_needed: bool = True) -> str:
        if fold is None:
            path = os.path.join(self.__path, inceptionkeynet.utils.make_path_compatible(subpath))
        else:
            path = os.path.join(self.__path, 'folds', str(fold), inceptionkeynet.utils.make_path_compatible(subpath))
        
        if create_if_needed and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        if not filename is None:
            return os.path.join(path, filename)
        else:
            return path
    
    def train(self):
        raise Exception('Invalid operation on ModelInferencer.')

    def infer(self, data):
        raise NotImplementedError('')

    def get_test_dataset_tf(self, dataset: Optional[Dataset] = None) -> tf.data.Dataset:
        test_dataset = self.get_test_dataset() if dataset is None else dataset
        pipeline = DataPipeline(self.hyperparameters)
        return self.apply_batching(pipeline.create_tf_dataset(test_dataset, model_execution_type=ModelExecutionType.TESTING), model_execution_type=ModelExecutionType.TESTING, batch_size_override=(1 if self.hyperparameters[Hyperparameters.USE_FULL_TRACKS_FOR_TESTING].value else None))

    def enumerate_available_models(self) -> Iterator[Dict[str, str]]:
        checkpoint_names = ['final', 'mirex', 'loss']
        for i_fold in range(0, self.hyperparameters[Hyperparameters.CROSS_VALIDATION_FOLD_COUNT].value):
            yield dict(((cp, self.get_path(f'checkpoints/{cp}', fold=i_fold)) for cp in checkpoint_names))

    def preprocess_sample(self, entry: MusicPiece, perform_augmentation: bool = False) -> np.ndarray:
        pipeline = DataPipeline(self.hyperparameters)
        x, *_ = next(iter(pipeline.create_tf_dataset([entry], perform_augmentation=perform_augmentation, model_execution_type=ModelExecutionType.TESTING, processing_threads=1)))
        return x.numpy()

    def infer_test_data(self, test_dataset_tf: Optional[tf.data.Dataset]) -> List[Dict[str, np.ndarray]]:
        if test_dataset_tf is None:
            test_dataset_tf = self.get_test_dataset_tf()
        
        checkpoint_names = ['final', 'mirex', 'loss']
        predictions: List[Dict[str, np.ndarray]] = []
        for i_fold in tqdm(range(0, self.hyperparameters[Hyperparameters.CROSS_VALIDATION_FOLD_COUNT].value), total=self.hyperparameters[Hyperparameters.CROSS_VALIDATION_FOLD_COUNT].value, desc='Performing inference for folds'):
            fold_predictions = { }
            for checkpoint_name in tqdm(checkpoint_names, total=len(checkpoint_names), desc='Performing inference for checkpoints'):
                checkpoint_path = self.get_path(f'checkpoints/{checkpoint_name}', fold=i_fold)
                checkpoint_model = self.load_model_recompile(checkpoint_path)
                logging.getLogger(__name__).debug(f'Evaluating checkpoint for {checkpoint_name}...')
                checkpoint_predictions = checkpoint_model.predict(test_dataset_tf)
                # checkpoint_metrics = dict([(name, checkpoint_evaluation[i]) for i, name in enumerate(checkpoint_model.metrics_names)])
                fold_predictions[checkpoint_name] = checkpoint_predictions
            
            predictions.append(fold_predictions)
        
        return predictions