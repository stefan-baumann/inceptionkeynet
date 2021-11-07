# Source: http://isophonics.net/datasets
# Extract the tar balls to the corresponding directories as specified below in the section directly after the imports

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



THE_BEATLES_SOURCE_PATH = inceptionkeynet.utils.make_path_compatible(os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/isophonics/the_beatles')))
CAROLE_KING_SOURCE_PATH = inceptionkeynet.utils.make_path_compatible(os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/isophonics/carole_king')))
QUEEN_SOURCE_PATH = inceptionkeynet.utils.make_path_compatible(os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/isophonics/queen')))
ZWEIECK_SOURCE_PATH = inceptionkeynet.utils.make_path_compatible(os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/isophonics/zweieck')))



class IsophonicsDatasetCreatorBase(DatasetCreator):
    @abstractmethod
    def get_extracted_tarball_root(self) -> str:
        raise NotImplementedError('This abstract method has not been implemented.')

    def get_music_pieces(self) -> Iterator[MusicPiece]:
        for filename in glob.glob(os.path.join(self.get_extracted_tarball_root(), os.path.normpath('keylab/*/*/*.lab'))):
            match = re.match(r'.*[\\/](?P<artist>[^\\/]+)[\\/](?P<album>[^\\/]+)[\\/]((?P<cd_number>CD\d+)\s*(-\s*)?)?(?P<track_number>\d+)\s*(-\s*)?(?P<title>[^\\/]+)\.lab', filename.replace('_', ' '))
            if not match:
                logging.getLogger(__name__).error(f'Found file with unexpected path for isophonics dataset: "{filename}".')
                continue
            infos = match.groupdict()

            key = self.get_overall_key_from_lab_file(filename)
            if not key is None:
                root, mode = key
                yield MusicPiece(infos['title'], infos['artist'], AnnotatedKeyRoot(root, AnnotationSource.HUMAN), AnnotatedKeyMode(mode, AnnotationSource.HUMAN))

    def get_overall_key_from_lab_file(self, filename: str) -> Optional[Tuple[KeyRoot, KeyMode]]:
        try:
            key = None
            with open(filename, 'r') as f:
                contents = f.read()
            for keychange_match in re.finditer(r'(\d+(\.\d+)?)\t(\d+(\.\d+)?)\t(?P<type>\w+)(\t(?P<key>.+))?($|\n)', contents):
                keychange_infos = keychange_match.groupdict()
                if keychange_infos['type'] == 'Key':
                    if key is None:
                        key = keychange_infos['key']
                    elif key != keychange_infos['key']:
                        logging.getLogger(__name__).error(f'Could not extract a single key from "{filename}" as the track contains key changes.')
                        return None
            if not key is None:
                return key_from_string(key)
        except ValueError:
            logging.getLogger(__name__).error(f'Could not extract key information from "{filename}" ("{key}").')
        except Exception:
            logging.getLogger(__name__).exception(f'Caught exception while trying to interpret lab file "{filename}".')
        return None



class IsophonicsBeatlesDatasetCreator(IsophonicsDatasetCreatorBase):
    def get_name(self) -> str:
        return 'Isophonics Reference Annotations: The Beatles'

    def get_extracted_tarball_root(self) -> str:
        return THE_BEATLES_SOURCE_PATH

class IsophonicsCaroleKingDatasetCreator(IsophonicsDatasetCreatorBase):
    def get_name(self) -> str:
        return 'Isophonics Reference Annotations: Carole King'

    def get_extracted_tarball_root(self) -> str:
        return CAROLE_KING_SOURCE_PATH

class IsophonicsQueenDatasetCreator(IsophonicsDatasetCreatorBase):
    def get_name(self) -> str:
        return 'Isophonics Reference Annotations: Queen'

    def get_extracted_tarball_root(self) -> str:
        return QUEEN_SOURCE_PATH

class IsophonicsZweieckDatasetCreator(IsophonicsDatasetCreatorBase):
    def get_name(self) -> str:
        return 'Isophonics Reference Annotations: Zweieck'

    def get_extracted_tarball_root(self) -> str:
        return ZWEIECK_SOURCE_PATH



if __name__ == '__main__':
    print(IsophonicsBeatlesDatasetCreator().create_dataset())
    print(IsophonicsCaroleKingDatasetCreator().create_dataset())
    print(IsophonicsQueenDatasetCreator().create_dataset())
    print(IsophonicsZweieckDatasetCreator().create_dataset())