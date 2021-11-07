# http://www.ibrahimshaath.co.uk/keyfinder/
# .ods file download: http://www.ibrahimshaath.co.uk/keyfinder/KeyFinderV2Dataset.ods

import os
import re
from typing import Iterator

import pandas as pd

from inceptionkeynet.data import Dataset, DatasetCreator, MusicPiece, AnnotatedKeyMode, AnnotatedKeyRoot, AnnotationSource
from inceptionkeynet.data_mining.utils import key_from_string
import inceptionkeynet
import inceptionkeynet.utils



SOURCE_PATH = inceptionkeynet.utils.make_path_compatible(os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/keyfinder_v2.ods')))



class KeyFinderV2DatasetCreator(DatasetCreator):
    def get_name(self) -> str:
        return 'KeyFinder v2'

    def get_music_pieces(self) -> Iterator[MusicPiece]:
        df = pd.read_excel(SOURCE_PATH, skipfooter=3)
        for _, row in df.iterrows():
            root, mode = key_from_string(row['KEY'])
            yield MusicPiece(row['TITLE'], row['ARTIST'], AnnotatedKeyRoot(root, AnnotationSource.HUMAN), AnnotatedKeyMode(mode, AnnotationSource.HUMAN))



if __name__ == '__main__':
    print(KeyFinderV2DatasetCreator().create_dataset())