import inspect

from inceptionkeynet.datasets import Dataset, Datasets, DatasetCreator



for dataset_creator in [member[1] for member in inspect.getmembers(Datasets) if not member[0].startswith('_') and isinstance(member[1], DatasetCreator)]:
    dataset_creator.get_dataset()