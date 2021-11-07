from __future__ import annotations # Enable postponed evaluation of annotations to allow for annotating methods returning their enclosing type

from typing import Union, Any, IO, AnyStr, Dict, Optional, List, Tuple
import os
import re
import logging

import json, pickle

USE_RUAMEL_FOR_YAML = True

if USE_RUAMEL_FOR_YAML:
    import ruamel.yaml as yaml
    if not getattr(yaml, '_package_data', None) is None and yaml._package_data['full_package_name'] == 'ruamel.yaml':
        import warnings
        warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
else:
    import yaml



def make_path_compatible(path: str) -> str:
    """Changes forward slashes into backward slashes and the reverse depending on the current platform."""
    return os.path.join(*path.replace('/', '\\').split('\\'))

def open_mkdirs(file: AnyStr, mode: str) -> IO[Any]:
    dir = os.path.dirname(file)
    if dir != '':
        os.makedirs(dir, exist_ok=True)
    return open(file, mode=mode)



def name_to_snake_case(name: str) -> str:
    return re.sub(r'\s', '_', name.lower())

def name_to_path_snake_case(name: str) -> str:
    return re.sub(r'[^\w]', '_', name.lower())



def read_serialized(file: str, serializer: Union[None, json, yaml, pickle] = None, *args, **kwargs) -> Any:
    serializer = serializer if not serializer is None else get_serializer_for_file(file)
    with open(file, 'r' if serializer != pickle else 'rb') as f:
        return serializer.load(f, *args, **kwargs)

SUPPORTED_SERIALIZATION_FORMATS = ['.yml', '.yaml', '.json', '.pickle', '.pkl']
def get_serializer_for_file(file: str):
    extension = os.path.splitext(file)[1].lower()
    if extension == '.json':
        return json
    elif extension in ['.yml', '.yaml']:
        return yaml
    elif extension in ['.pickle', '.pkl']:
        return pickle
    else:
        raise ValueError(f'The extension "{extension}" could not be automatically assigned to a serializer (json, yaml, pickle). Please specify the serializer explicitely for this extension.')



def tex_escape(text: str):
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)




if __name__ == '__main__':
    print(name_to_path_snake_case('Test v3.1'))