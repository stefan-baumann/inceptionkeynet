import os
from typing import Any

import numpy as np

import inceptionkeynet
import inceptionkeynet.utils



def get_file_path(path: str) -> str:
    if inceptionkeynet.DATA_FORMAT.lower() == '.npy':
        if path.lower().endswith('.npy'):
            return inceptionkeynet.utils.make_path_compatible(path)
        else:
            return inceptionkeynet.utils.make_path_compatible(path + inceptionkeynet.DATA_FORMAT)
    else:
        return path

def read_data(file: str) -> Any:
    # if not os.path.splitext(file)[1].lower() == inceptionkeynet.DATA_FORMAT.lower():
    #     raise Exception(f'Encountered invalid data type while trying to read data from "{file}".')
    
    if inceptionkeynet.DATA_FORMAT.lower() == '.npy':
        return np.load(get_file_path(file), allow_pickle=True)
    else:
        raise Exception(f'Encountered unknown configured data type: "{inceptionkeynet.DATA_FORMAT}".')

def write_data(data: Any, file: str):
    """file is expected to be without any extension."""
    dir = os.path.dirname(file)
    if dir != '':
        os.makedirs(dir, exist_ok=True)

    if inceptionkeynet.DATA_FORMAT.lower() == '.npy':
        if file.lower().endswith('.npy'):
            np.save(inceptionkeynet.utils.make_path_compatible(file), data)
        else:
            np.save(inceptionkeynet.utils.make_path_compatible(file + inceptionkeynet.DATA_FORMAT), data)
    else:
        raise Exception(f'Encountered unknown configured data type: "{inceptionkeynet.DATA_FORMAT}".')