# https://github.com/GiantSteps/giantsteps-key-dataset
# GiantSteps Key dataset is read from the sources.xlsx file: https://github.com/GiantSteps/giantsteps-key-dataset/blob/master/sources.xlsx
# https://github.com/GiantSteps/giantsteps-mtg-key-dataset
# GiantSteps MTG Key dataset is read from the annotations.txt (https://github.com/GiantSteps/giantsteps-mtg-key-dataset/blob/master/annotations/annotations.txt) and the beatport_metadata.txt (https://github.com/GiantSteps/giantsteps-mtg-key-dataset/blob/master/annotations/beatport_metadata.txt) files

import os
import re
import logging
from typing import Iterator

import pandas as pd
import numpy as np

from inceptionkeynet.data import Dataset, DatasetCreator, MusicPiece, AnnotatedKeyMode, AnnotatedKeyRoot, AnnotationSource
from inceptionkeynet.data_mining.utils import key_from_string
import inceptionkeynet
import inceptionkeynet.utils



KEY_SOURCE_PATH = inceptionkeynet.utils.make_path_compatible(os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/giantsteps_key.xlsx')))
MTG_KEY_ANNOTATIONS_SOURCE_PATH = inceptionkeynet.utils.make_path_compatible(os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/giantsteps_mtg_annotations.txt')))
MTG_KEY_BEATPORT_SOURCE_PATH = inceptionkeynet.utils.make_path_compatible(os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/giantsteps_mtg_beatport_metadata.txt')))



class GiantStepsKeyDatasetCreator(DatasetCreator):
    def get_name(self) -> str:
        return 'GiantSteps Key'

    def get_music_pieces(self) -> Iterator[MusicPiece]:
        df = pd.read_excel(KEY_SOURCE_PATH)
        for _, row in df.iterrows():
            try:
                # TODO: Parse beatport key data for reference
                # TODO: Parse genre information for reference
                root, mode = key_from_string(row['GLOBAL KEY'])
                yield MusicPiece(row['TRACK.1'] if not pd.isnull(row['TRACK.1']) else None, row['ARTIST'] if not pd.isnull(row['ARTIST']) else None, AnnotatedKeyRoot(root, AnnotationSource.HUMAN), AnnotatedKeyMode(mode, AnnotationSource.HUMAN), metadata={ 'audio_links': { 'jku_beatport_backup': f'http://www.cp.jku.at/datasets/giantsteps/backup/{row["TRACK"]}.LOFI.mp3' }}) # The beatport links all return 403 now, but there's a backup hosted at jku we can use
            except ValueError:
                logging.getLogger(__name__).info(f'Key for "{row["TRACK.1"]}" by "{row["ARTIST"]}" is ambiguous: "{row["GLOBAL KEY"]}".')



class GiantStepsMTGKeyDatasetCreator(DatasetCreator):
    def get_name(self) -> str:
        return 'GiantSteps MTG Key'

    def get_music_pieces(self) -> Iterator[MusicPiece]:
        df_key = pd.read_csv(MTG_KEY_ANNOTATIONS_SOURCE_PATH, sep='\t')
        df_key = df_key.set_index('ID')
        df_beatport = pd.read_csv(MTG_KEY_BEATPORT_SOURCE_PATH, sep='\t')
        df_beatport = df_beatport.set_index('ID')
        for id, row in df_key.iterrows():
            try:
                metadata_row = df_beatport.loc[id]

                # TODO: Parse software-derived key data for reference
                # TODO: Parse genre information for reference
                root, mode = key_from_string(row['MANUAL KEY'])
                yield MusicPiece((metadata_row['SONG TITLE'] + (f' ({metadata_row["MIX"]})' if not pd.isnull(metadata_row['MIX']) else '') if not pd.isnull(metadata_row['SONG TITLE']) else None), metadata_row['ARTIST'] if not pd.isnull(metadata_row['ARTIST']) else None, AnnotatedKeyRoot(root, AnnotationSource.HUMAN), AnnotatedKeyMode(mode, AnnotationSource.HUMAN), metadata={ 'audio_links': { 'jku_beatport_backup': f'http://www.cp.jku.at/datasets/giantsteps/mtg_key_backup/{id}.LOFI.mp3' }})
            except ValueError:
                logging.getLogger(__name__).info(f'Key for "{metadata_row["SONG TITLE"]} ({metadata_row["MIX"]})" by "{metadata_row["ARTIST"]}" is ambiguous/unclear: "{row["MANUAL KEY"]}".')



if __name__ == '__main__':
    print(GiantStepsMTGKeyDatasetCreator().create_dataset())
    # print(GiantStepsMTGKeyDatasetCreator().create_dataset().entries[0])
    print(GiantStepsKeyDatasetCreator().create_dataset())
    print(GiantStepsKeyDatasetCreator().create_dataset().entries[0])
