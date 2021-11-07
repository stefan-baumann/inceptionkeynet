from __future__ import annotations # Enable postponed evaluation of annotations to allow for annotating methods returning their enclosing type

from typing import List, Dict, Iterable, Union, Tuple, Any, Optional, Generic, TypeVar, AnyStr, Iterator
from abc import ABC, abstractmethod
import warnings
import logging
import os
import glob
from multiprocessing.pool import ThreadPool
from threading import Thread, Lock

import librosa
import numpy as np
import soundfile as sf
import progressbar

from inceptionkeynet.data import *
from inceptionkeynet.data_mining import *
import inceptionkeynet.utils
import inceptionkeynet.io
import inceptionkeynet



__audio_file_read_lock = Lock() # Appaerently audio file reading with PySoundFile does not behave properly when using multiple thread simultaneously, so let's just prevent that with a mutex
def read_audio_file(file: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with __audio_file_read_lock:
            x, sr = librosa.load(inceptionkeynet.utils.make_path_compatible(file), sr=inceptionkeynet.AUDIO_SAMPLE_RATE, mono=True)
            return x

__audio_file_write_lock = Lock()
def write_audio_file(file: str, y: np.ndarray, sr: Optional[int] = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with __audio_file_write_lock:
            output_path = inceptionkeynet.utils.make_path_compatible(file if file.lower().endswith('.wav') else file + '.wav')
            dirname = os.path.dirname(output_path)
            if len(dirname) > 0:
                os.makedirs(dirname, exist_ok=True)
            sf.write(output_path, y, (inceptionkeynet.AUDIO_SAMPLE_RATE if sr is None else sr))



TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')
class Transformer(Generic[TInput, TOutput], DictSerializable):
    def __init__(self, name: str, save_as_audio: bool = False):
        self.name = name
        self.save_as_audio = save_as_audio

    def __call__(self, value: TInput) -> TOutput:
        return self.transform(value)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={repr(self.name)})'

    @abstractmethod
    def transform(self, input: TInput) -> TOutput:
        raise NotImplementedError('This abstract method has not been implemented.')

    def transform_multi(self, inputs: Iterable[TInput]) -> Iterator[TOutput]:
        yield from (self(value) for value in inputs)



Transformers = { }



class TransformerChain(DictSerializable):
    def __init__(self, transformers: Optional[Iterable[Transformer]] = None):
        self.transformers = list(transformers) if not transformers is None else []

    def __call__(self, value):
        return self.transform(value)
    
    def __getitem__(self, index: int) -> Transformer:
        return self.transformers[index]

    def __setitem__(self, index: int, value: Transformer):
        self[index] = value

    def __len__(self) -> int:
        return len(self.transformers)

    def __repr__(self) -> str:
        return f'TransformerChain([{", ".join(repr(transformer) for transformer in self.transformers)}])'

    def append(self, transformer: Transformer):
        self.transformers.append(transformer)
    
    def __add__(self, y: Union[TransformerChain, Transformer]) -> TransformerChain:
        if isinstance(y, Transformer):
            return TransformerChain(self.transformers + [y])
        elif isinstance(y, TransformerChain):
            return TransformerChain(self.transformers + y.transformers)
        else:
            raise ValueError('Invalid addition operation for Transformerchain with ' + repr(y))

    def to_dict(self):
        def get_transformer_type(transformer: Transformer) -> str:
            try:
                return next((name for name, type in Transformers.items() if isinstance(transformer, type)))
            except StopIteration:
                raise Exception(f'The transformer "{type(transformer).__qualname__}" has not been registered in inceptionkeynet.machine_learning.Transformers. Add it to that dictionary to register it (e.g. "(\'{type(transformer).__name__}.lower()\')").')
        return {
            'transformers': list({'transformer_type': get_transformer_type(transformer), 'transformer': transformer.to_dict() } for transformer in self.transformers)
        }

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> TransformerChain:
        transformers = []
        for value in dict['transformers']:
            try:
                transformer_type = next((type for name, type in Transformers.items() if name == value['transformer_type']))
            except Exception:
                raise ValueError(f'The transformer type "{value["transformer_type"]}" could not be resolved.')
            transformers.append(transformer_type.from_dict(value['transformer']))
        return TransformerChain(transformers)



    def transform(self, input):
        value = input
        for transformer in self.transformers:
            value = transformer(value)
        return value

    def transform_multi(self, inputs: Iterable[TInput]) -> Iterator[TOutput]:
        yield from (self(value) for value in inputs)


    
    def can_extend(self, other: TransformerChain) -> bool:
        if len(self) < len(other):
            return False
        for i, transformer in enumerate(other.transformers):
            if not TransformerChain([self[i]]).to_dict() == TransformerChain([transformer]).to_dict(): # Dumb way to compare settings, but I don't think we need a lot of performance here
                return False
        return True

    def get_extending_chain(self, other: TransformerChain) -> TransformerChain:
        if not self.can_extend(other):
            raise ValueError('Cannot calculate extending chain for a chain that cannot be extended with the current chain.')

        return TransformerChain(self.transformers[len(other):])



    def get_disk_name(self) -> str:
        return inceptionkeynet.utils.name_to_path_snake_case('-'.join(transformer.name for transformer in self.transformers))

    def save_to_disk(self):
        self.write(os.path.join(inceptionkeynet.PROCESSED_DATA_PATH, self.get_disk_name() + inceptionkeynet.DEFAULT_FORMAT))

    def get_processed_file_path(self, uuid: str) -> str:
        return os.path.join(inceptionkeynet.PROCESSED_DATA_PATH, self.get_disk_name(), uuid)

    @classmethod
    def list_transformer_chains_on_disk(cls) -> Iterator[TransformerChain]:
        yielded_something = False
        if os.path.isdir(inceptionkeynet.DATASETS_PATH):
            for chain_path in (f for f in glob.glob(os.path.join(inceptionkeynet.PROCESSED_DATA_PATH, '*.*'), recursive=False) if os.path.splitext(f)[1] in inceptionkeynet.utils.SUPPORTED_SERIALIZATION_FORMATS):
                try:
                    chain = TransformerChain.read(chain_path)
                except GeneratorExit:
                    break
                except Exception:
                    pass
                else:
                    if os.path.splitext(os.path.split(inceptionkeynet.utils.make_path_compatible(chain_path))[-1])[0] != chain.get_disk_name():
                        raise Exception(f'Encountered transformer chain with invalid filename at "{chain_path}" (expected name: "{chain.get_disk_name()}").')
                    yield chain
                    yielded_something = True
        
        if not yielded_something:
            yield from ()



    def is_entry_preprocessed(self, entry: MusicPiece, miner_names: List[str]) -> bool:
        for audio_source, audio_path in [(source, inceptionkeynet.utils.make_path_compatible(path)) for source, path in entry.files_relative.items() if source in miner_names and not path is None]:
            audio_uuid = os.path.splitext(os.path.split(inceptionkeynet.utils.make_path_compatible(audio_path))[-1])[0]
            try:
                if (not any((audio_uuid in path) for path in entry.files_relative[self.get_disk_name()])): # or (not self.is_uuid_preprocessed_on_disk(audio_uuid)):
                    return False
            except KeyError:
                return False
        return True
    
    def apply_entry(self, entry: MusicPiece, miner_names: List[str], extendable_chains: List[TransformerChain], retry_on_error: bool = True) -> Optional[MusicPiece]:
        try:
            entry_modified = False
            entry_copy = MusicPiece.from_dict(entry.to_dict())
            
            # Check for the least amount of work remaining for each audio source
            if len(entry_copy.files_relative) > 0:
                for audio_source, audio_path in [(source, inceptionkeynet.utils.make_path_compatible(path)) for source, path in entry_copy.files_relative.items() if source in miner_names and not path is None]:
                    audio_uuid = os.path.splitext(os.path.split(inceptionkeynet.utils.make_path_compatible(audio_path))[-1])[0]
                    output_path = None
                    for extendable_chain, chain_extension in extendable_chains:
                        partial_path = extendable_chain.get_processed_file_path(audio_uuid) + (inceptionkeynet.DATA_FORMAT if not extendable_chain.transformers[-1].save_as_audio else '.wav')
                        if os.path.isfile(partial_path):
                            if len(chain_extension) == 0:
                                logging.getLogger(__name__).debug(f'Audio file "{audio_path}" has already been processed by this chain.')
                                output_path = partial_path
                            else:
                                logging.getLogger(__name__).debug(f'Audio file "{audio_path}" has already been partially processed, performing remaining transformations...')
                                if extendable_chain.transformers[-1].save_as_audio:
                                    partially_transformed = read_audio_file(partial_path)
                                else:
                                    partially_transformed = inceptionkeynet.io.read_data(partial_path)
                                transformed = chain_extension(partially_transformed)
                                logging.getLogger(__name__).debug(f'Audio file "{audio_path}" fully processed, saving...')
                                output_path = self.get_processed_file_path(audio_uuid)
                                if chain_extension.transformers[-1].save_as_audio:
                                    write_audio_file(output_path, transformed)
                                else:
                                    inceptionkeynet.io.write_data(transformed, output_path)
                                logging.getLogger(__name__).debug(f'Audio file "{audio_path}" done.')

                            break
                    
                    if output_path is None: # No matching partially preprocessed data found
                        logging.getLogger(__name__).debug(f'Audio file "{audio_path}" has not been processed before, applying transformations...')
                        samples = read_audio_file(entry_copy.files_relative[audio_source])
                        transformed = self(samples)
                        logging.getLogger(__name__).debug(f'Audio file "{audio_path}" fully processed, saving...')
                        output_path = self.get_processed_file_path(audio_uuid)
                        if self.transformers[-1].save_as_audio:
                            write_audio_file(output_path, transformed)
                        else:
                            inceptionkeynet.io.write_data(transformed, output_path)
                        logging.getLogger(__name__).debug(f'Audio file "{audio_path}" done.')

                    if self.get_disk_name() in entry_copy.files_relative and isinstance(entry_copy.files_relative[self.get_disk_name()], list):
                        if not output_path in entry_copy.files_relative[self.get_disk_name()]:
                            entry_copy.files_relative[self.get_disk_name()].append(output_path)
                            entry_modified = True
                    else:
                        entry_copy.files_relative[self.get_disk_name()] = [output_path]
                        entry_modified = True
            else:
                logging.getLogger(__name__).warn(f'No available audio for for "{entry_copy.name}" by "{entry_copy.artist}".')
            
            if entry_modified:
                return entry_copy
            else:
                return None
        except Exception:
            logging.getLogger(__name__).exception(f'Unexpected exception while processing "{entry.name}" by "{entry.artist}".')
            if retry_on_error:
                logging.getLogger(__name__).warn(f'Retrying to process "{entry.name}" by "{entry.artist}"...')
                return self.apply_entry(entry, miner_names, extendable_chains, retry_on_error=False)
            else:
                return None

    def apply(self, dataset: Dataset, specific_entries: Optional[Iterable[MusicPiece]] = None) -> Optional[Dataset]:
        try:
            entries = list(specific_entries) if not specific_entries is None else list(dataset)
            miner_names = [miner.name for miner in AudioMiners.list_audio_miners()]

            extendable_chains = sorted([(chain, self.get_extending_chain(chain)) for chain in TransformerChain.list_transformer_chains_on_disk() if self.can_extend(chain)], key=lambda t: len(t[1]))

            self.save_to_disk()

            entries_modified = 0
            widgets= [
                repr(self), ', ',
                progressbar.Timer(), ' ',
                progressbar.Bar(),
                ' ', progressbar.ETA(),
            ]
            with progressbar.ProgressBar(max_value=len(entries), widgets=widgets) as bar:
                for i, entry in enumerate(entries):
                    logging.getLogger(__name__).debug(f'Current entry: "{entry.name}" by "{entry.artist}" ({i + 1}/{len(entries)}).')
                    entry_modified = self.apply_entry(entry, miner_names, extendable_chains)

                    if not entry_modified is None:
                        dataset[i] = entry_modified
                        # logging.getLogger(__name__).debug(f'Entry for "{entry.name}" by "{entry.artist}" was modified, saving dataset...')
                        # dataset.save()
                        entries_modified += 1
                    if entries_modified >= 100:
                        logging.getLogger(__name__).info(f'Saving dataset...')
                        dataset.save()
                        entries_modified = 0
                    
                    bar.update(i)
        except Exception:
            logging.getLogger(__name__).exception(f'Unexpected error while processing dataset.')
            logging.getLogger(__name__).info(f'Saving dataset...')
            dataset.save()
            return None
        except BaseException as e:
            logging.getLogger(__name__).error(f'Scraping process interrupted.')
            logging.getLogger(__name__).info(f'Saving dataset...')
            dataset.save()
            raise e from None
        else:
            logging.getLogger(__name__).info(f'Saving dataset...')
            dataset.save()
            return dataset
    
    def apply_multithreaded(self, dataset: Dataset, n_workers: int = 12) -> Optional[Dataset]:
        try:
            class __DatasetManager:
                def __init__(self, dataset: Dataset, n_trigger: int = 100):
                    self.dataset = dataset
                    self.n_trigger = n_trigger
                    self.__change_counter = 0
                    self.__change_lock = Lock()
                    self.__writing = False
                    self.__writing_lock = Lock()
                    self.__iter_count = 0
                
                def notify_done(self):
                    with self.__change_lock:
                        self.__change_counter += 1
                        c = self.__change_counter
                    if c == self.n_trigger:
                        self.__write()
                
                def __write(self):
                    logging.getLogger(__name__).info(f'Saving dataset...')
                    with self.__change_lock:
                        self.__change_counter = 0
                        with self.__writing_lock:
                            self.__writing = True
                        self.dataset.save()
                        with self.__writing_lock:
                            self.__writing = False
                
                def is_writing(self):
                    with self.__writing_lock:
                        return self.__writing
            
            manager = __DatasetManager(dataset)
            miner_names = [miner.name for miner in AudioMiners.list_audio_miners()]
            extendable_chains = sorted([(chain, self.get_extending_chain(chain)) for chain in TransformerChain.list_transformer_chains_on_disk() if self.can_extend(chain)], key=lambda t: len(t[1]))
            self.save_to_disk()
            
            def worker(i: int):
                entry = manager.dataset[i]
                logging.getLogger(__name__).debug(f'Processing "{entry.name}" by "{entry.artist}"...')
                entry_modified = self.apply_entry(entry, miner_names, extendable_chains)
                if not entry_modified is None:
                    while manager.is_writing(): # Wait for manager to not be writing
                        pass
                    manager.dataset[i] = entry_modified
                    manager.notify_done()
                return None
            
            with ThreadPool(n_workers) as pool:
                widgets= [
                    repr(self),
                    progressbar.Timer(), ' ',
                    progressbar.Bar(),
                    ' ', progressbar.ETA(),
                ]
                with progressbar.ProgressBar(max_value=len(manager.dataset), widgets=widgets) as bar:
                    for i, _ in enumerate(pool.imap_unordered(worker, list(range(len(manager.dataset))))):
                        bar.update(i)
            
        except Exception:
            logging.getLogger(__name__).exception(f'Unexpected error while processing dataset.')
            logging.getLogger(__name__).info(f'Saving dataset...')
            dataset.save()
            return None
        except BaseException as e:
            logging.getLogger(__name__).error(f'Scraping process interrupted.')
            logging.getLogger(__name__).info(f'Saving dataset...')
            dataset.save()
            raise e from None
        else:
            logging.getLogger(__name__).info(f'Done with dataset "{dataset.name}". Saving dataset...')
            dataset.save()
            return dataset
        



if __name__ == '__main__':  
    from inceptionkeynet.datasets import Datasets
    from inceptionkeynet.processing.transformers import *
    chain = TransformerChain([AbsoluteConstantQTransformer(inceptionkeynet.AUDIO_SAMPLE_RATE)])
    chain.apply(Datasets.GIANTSTEPS_MTG_KEY.get_dataset())