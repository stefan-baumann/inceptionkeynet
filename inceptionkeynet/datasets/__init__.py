import os
import glob
from typing import Iterator
from abc import ABC, abstractmethod

from inceptionkeynet.data import Dataset, DatasetCreator
import inceptionkeynet.utils
import inceptionkeynet

from inceptionkeynet.datasets.giantsteps import GiantStepsKeyDatasetCreator, GiantStepsMTGKeyDatasetCreator
from inceptionkeynet.datasets.keyfinder_v2 import KeyFinderV2DatasetCreator
from inceptionkeynet.datasets.isophonics import IsophonicsBeatlesDatasetCreator, IsophonicsCaroleKingDatasetCreator, IsophonicsQueenDatasetCreator, IsophonicsZweieckDatasetCreator
from inceptionkeynet.datasets.rock_corpus import RockCorpusDatasetCreator
from inceptionkeynet.datasets.mcgill_billboard import McGillBillboardDatasetCreator



class __DatasetsMeta(type):
    def __iter__(self) -> Iterator[Dataset]:
        yielded_something = False
        if os.path.isdir(inceptionkeynet.DATASETS_PATH):
            for dataset_path in (f for f in glob.glob(os.path.join(inceptionkeynet.DATASETS_PATH, '*.*'), recursive=False) if os.path.splitext(f)[1] in inceptionkeynet.utils.SUPPORTED_SERIALIZATION_FORMATS and not '.temp' in f):
                try:
                    yield Dataset.read(dataset_path)
                    yielded_something = True
                except GeneratorExit:
                    break
                except Exception:
                    pass
        
        if not yielded_something:
            yield from ()

class Datasets(metaclass=__DatasetsMeta):
    GIANTSTEPS_KEY: DatasetCreator = GiantStepsKeyDatasetCreator()
    GIANTSTEPS_MTG_KEY: DatasetCreator = GiantStepsMTGKeyDatasetCreator()
    KEYFINDER_V2: DatasetCreator = KeyFinderV2DatasetCreator()
    ISOPHONICS_BEATLES: DatasetCreator = IsophonicsBeatlesDatasetCreator()
    ISOPHONICS_CAROLE_KING: DatasetCreator = IsophonicsCaroleKingDatasetCreator()
    ISOPHONICS_QUEEN: DatasetCreator = IsophonicsQueenDatasetCreator()
    ISOPHONICS_ZWEIECK: DatasetCreator = IsophonicsZweieckDatasetCreator()
    ROCKCORPUS: DatasetCreator = RockCorpusDatasetCreator()
    ROCKCORPUS: DatasetCreator = RockCorpusDatasetCreator()
    MCGILL_BILLBOARD: DatasetCreator = McGillBillboardDatasetCreator()

    @classmethod
    def get_from_name(cls, name: str) -> Dataset:
        # return next((dataset for dataset in Datasets if dataset.name == name), None)
        if os.path.isdir(inceptionkeynet.DATASETS_PATH):
            for dataset_path in (f for f in glob.glob(os.path.join(inceptionkeynet.DATASETS_PATH, f'{inceptionkeynet.utils.name_to_path_snake_case(name)}.*'), recursive=False) if os.path.splitext(f)[1] in inceptionkeynet.utils.SUPPORTED_SERIALIZATION_FORMATS and not '.temp' in f):
                try:
                    dataset = Dataset.read(dataset_path)
                    if dataset.name == name:
                        return dataset
                except GeneratorExit:
                    break
                except Exception:
                    pass
        return None



if __name__ == '__main__':
    print('Available datasets:')
    # print(list(Datasets))
    print(Datasets.GIANTSTEPS_KEY.get_dataset())
    print(Datasets.GIANTSTEPS_MTG_KEY.get_dataset())
    print(Datasets.KEYFINDER_V2.get_dataset())
    print(Datasets.ISOPHONICS_BEATLES.get_dataset())
    print(Datasets.ISOPHONICS_CAROLE_KING.get_dataset())
    print(Datasets.ISOPHONICS_QUEEN.get_dataset())
    print(Datasets.ISOPHONICS_ZWEIECK.get_dataset())
    print(Datasets.ROCKCORPUS.get_dataset())
    print(Datasets.MCGILL_BILLBOARD.get_dataset())