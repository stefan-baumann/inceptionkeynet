# Zip file download of Filip Korzeniowski's version with annotated keys: http://www.cp.jku.at/people/korzeniowski/bb.zip

import os
import re
import glob
import logging
from typing import Iterator, Optional, Tuple
from abc import abstractmethod

from inceptionkeynet.data import Dataset, DatasetCreator, MusicPiece, AnnotatedKeyMode, AnnotatedKeyRoot, AnnotationSource, KeyRoot, KeyMode
from inceptionkeynet.data_mining.utils import key_from_string
import inceptionkeynet
import inceptionkeynet.utils



ANNOTATIONS_SOURCE_PATH = inceptionkeynet.utils.make_path_compatible(os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/mcgill_billboard/key')))

class McGillBillboardDatasetCreator(DatasetCreator):
    def get_name(self) -> str:
        return 'McGill Billboard Key'

    def get_music_pieces(self) -> Iterator[MusicPiece]:
        for filename in glob.glob(os.path.join(ANNOTATIONS_SOURCE_PATH, os.path.normpath('*.key'))):
            match = re.match(r'.*mcgill-billboard_(?P<id>[^\-]+)-(?P<artist>[^\-]+)-(?P<title>.+)\.key', filename)
            if not match:
                logging.getLogger(__name__).error(f'Found file with unexpected path for Korzeniowski\'s version of the McGill Billboard dataset: "{filename}".')
                continue
            infos = match.groupdict()

            with open(filename, 'r') as f:
                contents = f.read()

            root, mode = key_from_string(contents)
            yield MusicPiece(infos['title'].replace('_', ' '), infos['artist'].replace('_', ' '), AnnotatedKeyRoot(root, AnnotationSource.HUMAN), AnnotatedKeyMode(mode, AnnotationSource.HUMAN))



if __name__ == '__main__':
    print(McGillBillboardDatasetCreator().create_dataset())