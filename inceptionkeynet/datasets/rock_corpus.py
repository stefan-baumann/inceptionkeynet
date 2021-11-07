# json files: https://github.com/danigb/rock-corpus/blob/master/corpus/
# Track list 1: http://rockcorpus.midside.com/overview/rs5x20.txt
# Track list 2: http://rockcorpus.midside.com/overview/rs500.txt

import os
import re
import json
import logging
from typing import Iterator, List

import pandas as pd

from inceptionkeynet.data import Dataset, DatasetCreator, KeyMode, MusicPiece, AnnotatedKeyMode, AnnotatedKeyRoot, AnnotationSource
import inceptionkeynet.data_mining.utils
import inceptionkeynet


SOURCE_PATH_SONGS = os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/rock_corpus/songs.json'))
SOURCE_PATH_CHORDS = os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/rock_corpus/chords.json'))
SOURCE_PATH_TRACKS_1 = os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/rock_corpus/rs500.txt'))
SOURCE_PATH_TRACKS_2 = os.path.join(inceptionkeynet.DATASETS_PATH, os.path.normpath('source_files/rock_corpus/rs5x20.txt'))
CLEANUP_REGEX = '(\(.*?\))|[^\w]'

class RockCorpusDatasetCreator(DatasetCreator):
    def get_name(self) -> str:
        return 'RockCorpus'

    def get_music_pieces(self) -> Iterator[MusicPiece]:
        with open(SOURCE_PATH_SONGS, 'r') as f:
            songs = json.load(f) # Contains tonics
        with open(SOURCE_PATH_CHORDS, 'r') as f:
            chords = json.load(f) # Contains chords -> presence of i vs. I chords indicates mode
        
        tracks1 = pd.read_csv(SOURCE_PATH_TRACKS_1, delimiter='\t', header=None, names=['id', 'title', 'artist', 'year'], index_col='id') # Contains each pair of title and artist
        tracks2 = pd.read_csv(SOURCE_PATH_TRACKS_2, delimiter='\t', header=None, names=['filename', 'id', 'title', 'artist', 'year'], index_col='id') # Contains each pair of title and artist
        tracks = pd.concat([tracks1, tracks2])
        # tracks1['title_cleaned'] = tracks1['title'].apply(lambda title: re.sub(CLEANUP_REGEX, '', title).lower())
        # tracks2['title_cleaned'] = tracks2['title'].apply(lambda title: re.sub(CLEANUP_REGEX, '', title).lower())
        tracks['title_cleaned'] = tracks['title'].apply(lambda title: re.sub(CLEANUP_REGEX, '', title).lower())


        for title, metadata in songs.items():
            tonic = metadata['key']
            if isinstance(tonic, List): # Sometimes, 2 tonics are annotated. We ignore them if they are actually different
                roots = [inceptionkeynet.data_mining.utils.key_root_from_string(t) for t in tonic]
                if all(root == roots[0] for root in roots):
                    root = roots[0]
                else:
                    logging.getLogger(__name__).warning(f'Multiple tonics notated for "{title}", skipping...')
                    continue
            else:
                root = inceptionkeynet.data_mining.utils.key_root_from_string(tonic)

            # Derive mode of song from mode of tonic chords, only accept the conclusion if at least 80% are in the same mode
            harmonies = chords[title]['harmony']
            n_major_tonic = sum(len(re.findall('(?<![a-zA-Z])I(?![a-zA-Z])', harmony)) for harmony in harmonies)
            n_minor_tonic = sum(len(re.findall('(?<![a-zA-Z])i(?![a-zA-Z])', harmony)) for harmony in harmonies)
            n_total = n_major_tonic + n_minor_tonic
            if n_total > 0:
                if n_major_tonic >= .8 * n_total:
                    mode = KeyMode.MAJOR
                elif n_minor_tonic >= .8 * n_total:
                    mode = KeyMode.MINOR
                else:
                    logging.getLogger(__name__).warning(f'Ambiguous tonic mode for "{title}", skipping...')
                    continue
            else:
                logging.getLogger(__name__).warning(f'Tonic mode could not be determined for "{title}", skipping...')
                continue

            # Find artist from the two tables
            title_cleaned = re.sub(CLEANUP_REGEX, '', title).lower()
            error_dict = { 'georgiaonmymind': 'georgiaonmymiind', 'inthestillofthenight': 'inthestilloftheniteye', 'rockandrollmusic': 'rockrollmusic' } # Some tracks have typos in the lists, this fixes them
            if title_cleaned in error_dict:
                title_cleaned = error_dict[title_cleaned]

            artist = tracks.loc[tracks['title_cleaned'] == title_cleaned] # Direct match
            if artist.empty:
                artist = tracks.loc[tracks['title_cleaned'].apply(lambda x: x in title_cleaned) | tracks['title_cleaned'].str.contains(title_cleaned)] # Partial match
            if artist.empty:
                logging.getLogger(__name__).warning(f'Could not find artist for "{title}" ("{title_cleaned}"), skipping...')
                continue
            artist = artist.iloc[0]['artist']

            yield MusicPiece(title, artist, AnnotatedKeyRoot(root, AnnotationSource.HUMAN), AnnotatedKeyMode(mode, AnnotationSource.HUMAN))
            
            
        
            

        

if __name__ == '__main__':
    print(RockCorpusDatasetCreator().create_dataset())