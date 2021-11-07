from __future__ import annotations # Enable postponed evaluation of annotations to allow for annotating methods returning their enclosing type

from typing import Union, Any, IO, AnyStr, Dict, Optional, List, Tuple
import os
import re

import inceptionkeynet.utils

CONFIG_DIR = os.path.normpath('./config')
def get_config(name: str, expected_values: Optional[List[Union[str, Tuple[str, List[Union[str, ...]]]]]] = None, throw_on_missing_file: bool = False) -> Optional[Dict[str, Any]]:
    joined_path = inceptionkeynet.utils.make_path_compatible(os.path.join(CONFIG_DIR, os.path.normpath(name)))
    config_dir = os.path.dirname(joined_path)
    config_name = os.path.basename(joined_path)

    config_name_pattern = f'{config_name}({"|".join((re.escape(ext) for ext in inceptionkeynet.utils.SUPPORTED_SERIALIZATION_FORMATS))})'
    matching_files = [f for f in os.listdir(config_dir) if re.match(config_name_pattern, f)]
    if len(matching_files) == 0:
        if throw_on_missing_file:
            raise FileNotFoundError(f'Configuration file "{joined_path}" could not be found.')
        return None
    elif len(matching_files) > 1:
        raise Exception(f'Multiple conflicting configuration files found for "{name}".')
    
    config_file_path = os.path.join(config_dir, matching_files[0])
    config = inceptionkeynet.utils.read_serialized(config_file_path)

    if not expected_values is None:
        def recursive_config_check(dict: Dict, expected_values: List[Union[str, Tuple[str, List[Union[str, ...]]]]], current_path: str = ''):
            def join_field_path(current_path: str, field: Any) -> str:
                return (f'{current_path}/' if len(current_path) > 0 else '') + (field if isinstance(field, str) else repr(field))
            for value in expected_values:
                if isinstance(value, Tuple) and len(value) == 2 and isinstance(value[1], List):
                    field_path = join_field_path(current_path, value[0])
                    if not value[0] in dict:
                        raise ValueError(f'Missing field "{field_path}" in configuration file at "{config_file_path}".')
                    recursive_config_check(dict[value[0]], value[1], current_path=field_path)
                else:
                    if not value in dict:
                        raise ValueError(f'Missing field "{join_field_path(current_path, value)}" in configuration file at "{config_file_path}".')

        recursive_config_check(config, expected_values)

    return config