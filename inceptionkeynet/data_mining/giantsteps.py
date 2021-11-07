from __future__ import annotations # Enable postponed evaluation of annotations to allow for annotating methods returning their enclosing type

from typing import Optional

from inceptionkeynet.data_mining.common import AudioMiner
from inceptionkeynet.data import MusicPiece



class GiantStepsAudioMiner(AudioMiner):
    def __init__(self):
        super().__init__('GiantSteps')

    def try_get_download_url(self, entry: MusicPiece) -> Optional[str]:
        if 'audio_links' in entry.metadata and 'jku_beatport_backup' in entry.metadata['audio_links']:
            return entry.metadata['audio_links']['jku_beatport_backup']