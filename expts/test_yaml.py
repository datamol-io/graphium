# test_yaml.py


import yaml

CONFIG_FILE = "neurips2023_configs/config_small_mpnn.yaml"

import re
from collections import defaultdict

# def get_anchors_and_aliases(filepath):
#     anchors = defaultdict(list)
#     current_level = {}

#     with open(filepath, "r") as file:
#         for line in file:
#             indent = len(line) - len(line.lstrip(' '))
#             key_match = re.search(r'(\w+):', line)
#             anchor_match = re.search(r'&(\w+)', line)
#             alias_match = re.search(r'\*(\w+)', line)
#             if key_match:
#                 key = key_match.group(1)
#                 # Compute the full path of the current key.
#                 full_path = '.'.join([current_level[i] for i in sorted(current_level.keys()) if i < indent] + [key])
#                 current_level[indent] = key
#                 # Remove any keys that are indented more than the current line.
#                 keys_to_remove = [i for i in current_level if i > indent]
#                 for i in keys_to_remove:
#                     del current_level[i]
#             else:
#                 full_path = '.'.join([current_level[i] for i in sorted(current_level.keys())])
#             if anchor_match:
#                 anchor = anchor_match.group(1)
#                 anchors[anchor].append(full_path)
#             if alias_match:
#                 alias = alias_match.group(1)
#                 anchors[alias].append(full_path)
#     return {k: v for k, v in anchors.items() if len(v) > 1}


def get_anchors_and_aliases(filepath):
    anchors = defaultdict(list)
    current_level = {}
    anchor_to_path = {}

    with open(filepath, "r") as file:
        for line in file:
            indent = len(line) - len(line.lstrip(" "))
            key_match = re.search(r"(\w+):", line)
            anchor_match = re.search(r"&(\w+)", line)
            alias_match = re.search(r"\*(\w+)", line)
            if key_match:
                key = key_match.group(1)
                # Compute the full path of the current key.
                full_path = ".".join(
                    [current_level[i] for i in sorted(current_level.keys()) if i < indent] + [key]
                )
                current_level[indent] = key
                # Remove any keys that are indented more than the current line.
                keys_to_remove = [i for i in current_level if i > indent]
                for i in keys_to_remove:
                    del current_level[i]
            else:
                full_path = ".".join([current_level[i] for i in sorted(current_level.keys())])
            if anchor_match:
                anchor = anchor_match.group(1)
                anchor_to_path[anchor] = full_path
            if alias_match:
                alias = alias_match.group(1)
                if alias in anchor_to_path:
                    anchors[anchor_to_path[alias]].append(full_path)
    return anchors


refs = get_anchors_and_aliases(CONFIG_FILE)
import ipdb

ipdb.set_trace()
pass
