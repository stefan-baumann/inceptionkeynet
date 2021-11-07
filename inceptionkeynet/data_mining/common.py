from abc import ABC, abstractmethod
from typing import Optional
import logging
import requests
import uuid
import posixpath
import os

from inceptionkeynet.data import MusicPiece



class AudioMiner(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def try_get_download_url(self, entry: MusicPiece) -> Optional[str]:
        raise NotImplementedError('This abstract method has not been implemented.')
    
    def get_file_extension_from_url(self, url: str) -> str:
        return os.path.splitext(posixpath.basename(url))[1]
    
    def get_filename_from_url(self, url: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, f'{self.name}-{url}')) + self.get_file_extension_from_url(url)

    def try_get_audio_data(self, url: str) -> Optional[bytes]:
        if not url is None:
            try:
                return requests.get(url).content
            except Exception:
                logging.getLogger(__name__).exception(f'Unexpected exception while trying to download audio data from {url}.')
        return None