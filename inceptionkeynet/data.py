from __future__ import annotations # Enable postponed evaluation of annotations to allow for annotating methods returning their enclosing type 

from typing import TypeVar, Generic, Union, List, Dict, Any, IO, Iterable, Iterator, Optional
from abc import ABC, abstractmethod
import inspect
import itertools
import random
from enum import Enum
import sys
import os
import copy
import logging
import shutil

import json, pickle
from io import StringIO

import inceptionkeynet.utils
import inceptionkeynet.io
import inceptionkeynet

from inceptionkeynet.utils import USE_RUAMEL_FOR_YAML
if USE_RUAMEL_FOR_YAML:
    import ruamel.yaml as yaml
    if not getattr(yaml, '_package_data', None) is None and yaml._package_data['full_package_name'] == 'ruamel.yaml':
        import warnings
        warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
else:
    import yaml



class DictSerializable(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError('This abstract method has not been implemented.')
    
    @classmethod
    def __dump(cls, dictionary: Dict[str, Any], serializer: Union[json, yaml, pickle], stream: IO, *args, **kwargs):
        if serializer != pickle:
            serializer.dump(dictionary, stream, *args, **kwargs)
        else:
            serializer._dump(dictionary, stream, *args, **kwargs)

    def dump(self, serializer: Union[json, yaml, pickle], stream: IO, *args, **kwargs):
        return DictSerializable.__dump(self.to_dict(), serializer, stream, *args, **kwargs)

    @classmethod
    def __dumps(cls, dictionary: Dict[str, Any], serializer: Union[json, yaml, pickle], stream: IO, *args, **kwargs) -> str:
        if callable(getattr(serializer, 'dumps', None)): 
            return serializer.dumps(dictionary, *args, **kwargs)
        else:
            with StringIO() as stream:
                DictSerializable.__dump(dictionary, serializer, stream, *args, **kwargs)
                return stream.getvalue()

    def dumps(self, serializer: Union[json, yaml, pickle], *args, **kwargs) -> str:
        return DictSerializable.__dumps(self.to_dict(), serializer, *args, **kwargs)

    def write(self, file: str, serializer: Union[None, json, yaml, pickle] = None, *args, **kwargs):
        serializer = serializer if not serializer is None else inceptionkeynet.utils.get_serializer_for_file(file)
        with inceptionkeynet.utils.open_mkdirs(file, 'w' if serializer != pickle else 'wb') as f:
            self.dump(serializer, f, *args, **kwargs)



    @classmethod
    @abstractmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> Any:
        return dictionary

    @classmethod
    def load(cls, serializer: Union[json, yaml, pickle], stream: IO, *args, **kwargs) -> Any:
        return cls.from_dict(serializer.load(stream, *args, **kwargs))

    @classmethod
    def loads(cls, string: str, serializer: Union[json, yaml, pickle], *args, **kwargs) -> Any:
        if callable(getattr(serializer, 'loads', None)):
            return cls.from_dict(serializer.loads(string, *args, **kwargs))
        else:
            with StringIO(string) as stream:
                return DictSerializable.load(serializer, stream, *args, **kwargs)

    @classmethod
    def read(cls, file: str, serializer: Union[None, json, yaml, pickle] = None, *args, **kwargs) -> Any:
        if file is None:
            return None
        serializer = serializer if not serializer is None else inceptionkeynet.utils.get_serializer_for_file(file)
        with open(file, 'r' if serializer != pickle else 'rb') as f:
            result = cls.load(serializer, f, *args, **kwargs)
            if hasattr(result, 'path'):
                result.path = file
            return result
    


    @classmethod
    def write_dict(cls, dictionary: Dict[str, Any], file: str, serializer: Union[None, json, yaml, pickle] = None, *args, **kwargs):
        serializer = serializer if not serializer is None else inceptionkeynet.utils.get_serializer_for_file(file)
        with inceptionkeynet.utils.open_mkdirs(file, 'w' if serializer != pickle else 'wb') as f:
            DictSerializable.__dump(dictionary, serializer, f, *args, **kwargs)


class AnnotationSource(str, Enum):
    HUMAN = 'human'
    SOFTWARE_DERIVED = 'software_derived'
    SEMI_SUPERVISED = 'semi_supervised'
    DATA_MINED = 'data_mined'



TAnnotation = TypeVar('TAnnotation')
class AnnotatedValue(Generic[TAnnotation], DictSerializable):
    def __init__(self, value: TAnnotation, source: AnnotationSource, certainty: Optional[float] = None):
        super().__init__()
        
        self.value = value
        self.source = source
        self.certainty = certainty

    def __repr__(self) -> str:
        return f'AnnotatedValue({repr(self.value)}, {repr(self.source)}, {repr(self.certainty)})'



class KeyRoot(int, Enum):
    C = 0
    Db = 1
    D = 2
    Eb = 3
    E = 4
    F = 5
    Gb = 6
    G = 7
    Ab = 8
    A = 9
    Bb = 10
    B = 11
        
    @classmethod
    def get_names(cls, use_sharps: bool = False) -> List[str]:
        if use_sharps:
            return ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        else:
            return ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    def get_name(self, use_sharps: bool = False) -> str:
        return KeyRoot.get_names(use_sharps)[self.value]
        
    @classmethod
    def from_name(cls, name: str) -> KeyRoot:
        name_lower = name.lower()
        flat_names = list(key_name.lower() for key_name in KeyRoot.get_names(use_sharps=False))
        # print(flat_names)
        if name_lower in flat_names:
            return KeyRoot(flat_names.index(name_lower))
        else:
            sharp_names = list(key_name.lower() for key_name in KeyRoot.get_names(use_sharps=True))
            if name_lower in sharp_names:
                return KeyRoot(sharp_names.index(name_lower))
            else:
                special_mappings = { 'Cb': 'B', 'Fb': 'E', 'B#': 'C', 'E#': 'F' }
                special_mappings_lower = dict((key.lower(), special_mappings[key]) for key in special_mappings)
                if name_lower in special_mappings_lower:
                    return KeyRoot.from_name(special_mappings_lower[name_lower])
                else:
                    raise ValueError(f'The name "{name}" is not a valid key name.')
    
    @classmethod
    def try_from_name(cls, name: str) -> Optional[KeyRoot]:
        try:
            return KeyRoot.from_name(name)
        except ValueError:
            return None



class AnnotatedKeyRoot(AnnotatedValue[KeyRoot]):
    def __init__(self, value: Union[KeyRoot, int], source: AnnotationSource, certainty: Optional[float] = None):
        if not isinstance(value, KeyRoot):
            if value in set(item.value for item in KeyRoot):
                value = KeyRoot(value)
            else:
                raise ValueError(f'The value "{value}" is not a valid key root.')
        super().__init__(value, source, certainty)
        
    def to_dict(self):
        return {
            'root': self.value.value,
            'source': self.source.value,
            'certainty': self.certainty
        }

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> AnnotatedKeyRoot:
        return AnnotatedKeyRoot(dict['root'], AnnotationSource(dict['source']), dict['certainty'])



class KeyMode(str, Enum):
    MAJOR = 'major'
    MINOR = 'minor'
    
    def to_int(self) -> int:
        return list(KeyMode).index(self)
    
    @classmethod
    def from_int(cls, i: int) -> KeyMode:
        if i == 0:
            return KeyMode.MAJOR
        elif i == 1:
            return KeyMode.MINOR
        else:
            raise ValueError(f'Invalid value {repr(i)} for a KeyMode')

class AnnotatedKeyMode(AnnotatedValue[KeyMode]):
    def __init__(self, value: Union[KeyMode, str], source: AnnotationSource, certainty: Union[float, None] = None):
        if isinstance(value, KeyMode):
            if value in set(item.value for item in KeyMode):
                value = KeyMode(value)
            else:
                raise ValueError(f'The value "{value}" is not a valid key mode.')
        super().__init__(value, source, certainty)
        
    def get_name(self, use_sharps: bool = True) -> str:
        if self.value == KeyMode.MAJOR:
            return 'Major'
        elif self.value == KeyMode.MINOR:
            return 'minor'
        else:
            raise NotImplementedError('The current key mode value is not supported.')
        
    def get_short_name(self, use_sharps: bool = True) -> str:
        if self.value == KeyMode.MAJOR:
            return 'M'
        elif self.value == KeyMode.MINOR:
            return 'm'
        else:
            raise NotImplementedError('The current key mode value is not supported.')
    
    def to_dict(self):
        return {
            'mode': self.value.value,
            'source': self.source.value,
            'certainty': self.certainty
        }

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> AnnotatedKeyMode:
        return AnnotatedKeyMode(KeyMode(dict['mode']), AnnotationSource(dict['source']), dict['certainty'])



class MusicPiece(DictSerializable):
    def __init__(self, name: str, artist: str, root: AnnotatedKeyRoot, mode: AnnotatedKeyMode, files_relative: Dict[str, str] = { }, mining_sources: Dict[str, str] = { }, metadata: Dict[str, Any] = { }):
        self.name = name
        self.artist = artist
        self.root = root
        self.mode = mode
        self.files_relative = files_relative
        self.mining_sources = mining_sources
        self.metadata = metadata

    def __repr__(self) -> str:
        return f'MusicPiece({repr(self.name)}, {repr(self.artist)}, {repr(self.root)}, {repr(self.mode)}, files_relative={repr(self.files_relative)}, metadata={repr(self.metadata)})'
        
    def to_dict(self):
        return {
            'name': self.name,
            'artist': self.artist,
            'root': self.root.to_dict(),
            'mode': self.mode.to_dict(),
            'files_relative': self.files_relative,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> AnnotatedKeyRoot:
        # If there are duplicate entries, the dictionaries will be linked at runtime, so all of them are copied
        
        # Backwards compatibility for when mining_sources was still a part of metadata
        mining_sources = { }
        if 'mining_sources' in dict['metadata']:
            dict['metadata'] = copy.deepcopy(dict['metadata'])
            mining_sources = dict(mining_sources, **dict['metadata'].pop('mining_sources'))
        if 'mining_sources' in dict:
            mining_sources = dict(mining_sources, **dict['mining_sources'])
        return MusicPiece(
            name=dict['name'],
            artist=dict['artist'],
            root=AnnotatedKeyRoot.from_dict(dict['root']),
            mode=AnnotatedKeyMode.from_dict(dict['mode']),
            files_relative=copy.deepcopy(dict['files_relative']),
            mining_sources=copy.deepcopy(mining_sources),
            metadata=copy.deepcopy(dict['metadata'])
        )
    
    def remove_nonexistent_file_links(self) -> MusicPiece:
        from inceptionkeynet.data_mining import AudioMiners
        miner_names = [miner.name for miner in AudioMiners.list_audio_miners()]
        self.files_relative = dict((key, file) for key, file in self.files_relative.items() if not file is None and (key in miner_names or ((isinstance(file, str) and os.path.isfile(inceptionkeynet.io.get_file_path(file))) or (isinstance(file, list) and all((os.path.isfile(inceptionkeynet.io.get_file_path(file_)) for file_ in file))))))
        return self



class Dataset(DictSerializable):
    def __init__(self, name: str, entries: Iterable[MusicPiece], metadata: Dict[str, Any] = { }):
        self.name = name
        self.entries = list(entries) if not entries is None else []
        self.metadata = metadata
        self.shuffling_seed = random.randrange(sys.maxsize)
        self.path: str = None
    
    def __getitem__(self, index: int) -> MusicPiece:
        return self.entries[index]
    
    def __setitem__(self, index: int, value: MusicPiece):
        self.entries[index] = value

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f'Dataset({repr(self.name)}, <{len(self)} entries>, {repr(self.metadata)}, path={repr(self.path)})'

    def shuffled(self) -> Dataset:
        entries = self[:]
        random.Random(self.shuffling_seed).shuffle(entries)
        return Dataset(self.name, entries, metadata=dict(self.metadata, **{ 'shuffled': True, 'shuffling_seed': self.shuffling_seed }))

    def folds(self, n: int, drop_remainder: bool = False) -> List[Dataset]:
        if n > len(self):
            raise ValueError(f'A dataset with a length of {len(self)} cannot be split into {n} folds, as each fold has to contain at least a single entry.')
        fold_lengths = [max(1, int(round(len(self) / n))) for i in range(0, n)]

        # Readjust split counts to match total count of entries
        # print(fold_lengths)
        if drop_remainder:
            while sum(fold_lengths) > len(self):
                for i in range(0, len(fold_lengths)):
                    fold_lengths[i] -= 1
        else:
            i_split = 0
            while not sum(fold_lengths) == len(self):
                fold_lengths[i_split] += 1 if sum(fold_lengths) < len(self) else -1
                i_split = (i_split + 1) % len(fold_lengths)
        # print(fold_lengths)
        
        # Generate folds
        folds = []
        i_start = 0
        shuffled = self.shuffled()
        for i_fold, fold_length in enumerate(fold_lengths):
            folds.append(Dataset(f'{self.name}/{i_fold + 1}', shuffled[i_start:(i_start + fold_length)], metadata=self.metadata))
            i_start += fold_length
        
        return folds

    def to_dict(self):
        return {
            'name': self.name,
            'metadata': self.metadata,
            'shuffling_seed': self.shuffling_seed,
            'entries': list(entry.to_dict() for entry in self.entries)
        }

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> AnnotatedKeyRoot:
        dataset = Dataset(
            name=dict['name'],
            entries=list(MusicPiece.from_dict(entry) for entry in dict['entries']),
            metadata=dict['metadata']
        )
        dataset.shuffling_seed = dict['shuffling_seed']
        return dataset
    
    def save(self):
        if self.path is None:
            raise ValueError('Cannot save a dataset without a stored path.')
        temp_path = os.path.splitext(self.path)[0] + '.temp' + os.path.splitext(self.path)[1]
        logging.getLogger(__name__).debug(f'Writing dataset "{self.name}" to temporary file...')
        self.write(temp_path)
        logging.getLogger(__name__).debug(f'Temporary file written, renaming...')
        shutil.move(temp_path, self.path)
        logging.getLogger(__name__).debug(f'Done writing dataset "{self.name}".')

    @classmethod
    def merge(cls, first: Dataset, second: Dataset, new_name: str, new_metadata: Dict[str, Any] = { }, include_original_metadata: bool = True) -> Dataset:
        return Dataset(
            name=new_name,
            entries=itertools.chain(first, second),
            metadata=dict(new_metadata, **({
                'original_datasets': [
                    {
                        'name': first.name,
                        'metadata': first.metadata
                    },
                    {
                        'name': second.name,
                        'metadata': second.metadata
                    }
                ]
            } if include_original_metadata else { }))
        )
    
    @classmethod
    def merge_multi(cls, datasets: Iterable[Dataset], new_name: str, new_metadata: Dict[str, Any] = { }, include_original_metadata: bool = True) -> Dataset:
        datasets = list(datasets)
        return Dataset(
            name=new_name,
            entries=itertools.chain(*datasets),
            metadata=dict(new_metadata, **({
                'original_datasets': [{
                        'name': dataset.name,
                        'metadata': dataset.metadata
                    } for dataset in datasets]
            } if include_original_metadata else { }))
        )



class DatasetCreator(ABC):
    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError('This abstract method has not been implemented.')

    @abstractmethod
    def get_music_pieces(self) -> Iterator[MusicPiece]:
        raise NotImplementedError('This abstract method has not been implemented.')

    def create_dataset(self) -> Dataset:
        return Dataset(self.get_name(), self.get_music_pieces())
    
    def get_dataset(self) -> Dataset:
        from inceptionkeynet.datasets import Datasets
        dataset = Datasets.get_from_name(self.get_name())
        if dataset is None:
            logging.getLogger(__name__).info(f'Creating dataset "{self.get_name()}"...')
            dataset = self.create_dataset()
            logging.getLogger(__name__).info(f'Done creating dataset "{self.get_name()}".')
            dataset.path = os.path.join(inceptionkeynet.DATASETS_PATH, inceptionkeynet.utils.name_to_path_snake_case(self.get_name()) + inceptionkeynet.DATASET_FORMAT)
            dataset.save()
            logging.getLogger(__name__).info(f'Saved dataset "{self.get_name()}" to "{dataset.path}".')
        return dataset



if __name__ == '__main__':
    print('Splitting a 10-entry dataset into 6 folds')
    print([f'{d.name} ({len(d)} entries)' for d in Dataset('dataset', range(0, 10)).folds(6)])
    print('Splitting a 10-entry dataset into 6 folds of equal size')
    print([f'{d.name} ({len(d)} entries)' for d in Dataset('dataset', range(0, 10)).folds(6, drop_remainder=True)])
    Dataset('test-dataset', [
        MusicPiece('Title', 'Artist', AnnotatedKeyRoot(KeyRoot.F, AnnotationSource.HUMAN), AnnotatedKeyMode(KeyMode.MINOR, AnnotationSource.SOFTWARE_DERIVED, .563), { 'stft_1024': 'stft_1024/title.npy' }, { 'genre': 'edm' }),
        MusicPiece('Title2', 'Artist2', AnnotatedKeyRoot(KeyRoot.Eb, AnnotationSource.DATA_MINED, .813), AnnotatedKeyMode(KeyMode.MINOR, AnnotationSource.SEMI_SUPERVISED, .759), { 'stft_1024': 'stft_1024/title2.npy' }, { 'genre': 'blues' })
    ]).write('./data/datasets/test-dataset.json')
    print(Dataset.read('./data/datasets/test-dataset.json'))
    Dataset('test-dataset', [
        MusicPiece('Title', 'Artist', AnnotatedKeyRoot(KeyRoot.F, AnnotationSource.HUMAN), AnnotatedKeyMode(KeyMode.MINOR, AnnotationSource.SOFTWARE_DERIVED, .563), { 'stft_1024': 'stft_1024/title.npy' }, { 'genre': 'edm' }),
        MusicPiece('Title2', 'Artist2', AnnotatedKeyRoot(KeyRoot.Eb, AnnotationSource.DATA_MINED, .813), AnnotatedKeyMode(KeyMode.MINOR, AnnotationSource.SEMI_SUPERVISED, .759), { 'stft_1024': 'stft_1024/title2.npy' }, { 'genre': 'blues' })
    ]).write('./data/datasets/test-dataset.yaml')
    print(Dataset.read('./data/datasets/test-dataset.yaml'))
    Dataset('test-dataset', [
        MusicPiece('Title', 'Artist', AnnotatedKeyRoot(KeyRoot.F, AnnotationSource.HUMAN), AnnotatedKeyMode(KeyMode.MINOR, AnnotationSource.SOFTWARE_DERIVED, .563), { 'stft_1024': 'stft_1024/title.npy' }, { 'genre': 'edm' }),
        MusicPiece('Title2', 'Artist2', AnnotatedKeyRoot(KeyRoot.Eb, AnnotationSource.DATA_MINED, .813), AnnotatedKeyMode(KeyMode.MINOR, AnnotationSource.SEMI_SUPERVISED, .759), { 'stft_1024': 'stft_1024/title2.npy' }, { 'genre': 'blues' })
    ]).write('./data/datasets/test-dataset.pickle')
    print(Dataset.read('./data/datasets/test-dataset.pickle'))