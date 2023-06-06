from typing import Any
import hashlib
import yaml


def get_md5_hash(object: Any) -> str:
    """
    MD5 hash of any object.
    The object is converted to a YAML string before being hashed.
    This allows for nested dictionaries/lists and for hashing of classes and their attributes.
    """
    dhash = hashlib.md5()
    encoded = yaml.dump(object, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
