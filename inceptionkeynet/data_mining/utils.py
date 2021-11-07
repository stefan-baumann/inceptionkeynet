import re
from typing import Tuple, Optional

from inceptionkeynet.data import KeyMode, KeyRoot



def key_root_from_string(string: str) -> KeyRoot:
    return KeyRoot.from_name(string)

def key_from_string(string: str) -> Tuple[KeyRoot, KeyMode]:
    # match = re.match(r'^(?-i)\s*(?P<root>[a-gA-G])\s*(?P<root_modifier>[bB]|#|((?i)sharp|flat(?-i)))?\s*((?P<minor>(m|((?i)min(or)?(?-i))))|(?P<major>(M|((?i)maj(or)?(?-i)))))\s*$', string)
    match = re.match(r'^\s*(?P<root>[a-gA-G])\s*(?P<root_modifier>[bB]|#|([sS][hH][aA][rR][pP]|[fF][lL][aA][tT]))?\s*:?\s*((?P<minor>(m|([mM][iI][nN]([oO][rR])?)))|(?P<major>(|M|([mM][aA][jJ]([oO][rR])?))))\s*$', string)
    if not match:
        raise ValueError(f'Could not extract key from string "{string}".')
    groups = match.groupdict()

    root = key_root_from_string(groups['root'] if not 'root_modifier' in groups or groups['root_modifier'] is None else groups['root'] + groups['root_modifier'].lower().replace('sharp', '#').replace('flat', 'b'))
    if 'minor' in groups and not groups['minor'] is None:
        mode = KeyMode.MINOR
    elif 'major' in groups and not groups['minor'] is None:
        mode = KeyMode.MAJOR
    elif groups['root'].islower():
        mode = KeyMode.MINOR
    elif groups['root'].isupper():
        mode = KeyMode.MAJOR
    else:
        raise ValueError(f'Could not decide key mode for string "{string}".')

    return root, mode

def try_key_from_string(string: str) -> Optional[Tuple[KeyRoot, KeyMode]]:
    if string is None:
        return None
    try:
        return key_from_string(string)
    except ValueError:
        return None



if __name__ == '__main__':
    print(key_from_string('F#min'))
    print(key_from_string('F#maj'))
    print(key_from_string('F#min / F#maj'))