from inceptionkeynet.data_mining import AudioMiners, AudioMiner, DatasetAudioMiner
from inceptionkeynet.datasets import Datasets
from inceptionkeynet.data import DatasetCreator

for dataset in Datasets:
    DatasetAudioMiner(dataset).run()