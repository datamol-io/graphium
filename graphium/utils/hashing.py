"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


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
