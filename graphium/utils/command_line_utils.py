import re
from collections import defaultdict
from typing import List, Dict


def get_anchors_and_aliases(filepath):
    """Utility function to extract anchors and aliases from YAML config file

    In a YAML file we can specify an anchor as  `some_name: &anchor_name value`
    This is then picked up with alises in the rest of the config as
    `other_name: *anchor_name`
    Using this format in the YAML file all anchors and aliases will be extracted

    Args:
        filepath (str): path to the config file

    Returns:
        anchors (Dict): A dictionary containing the YAML paths of the anchors
                        as keys, and a list of YAML paths to alises.
    """
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


def update_config(cfg: Dict, unknown: List, anchors: List):
    """
    Update the configuration dictionary with command line arguments.
    """
    for arg in unknown:
        if arg.startswith("--"):
            key, value = arg[2:].split("=")
            if key in anchors.keys():
                all_refs = anchors[key]
            else:
                all_refs = []
            all_refs.append(key)
            for key in all_refs:
                keys = key.split(".")
                temp_cfg = cfg
                for k in keys[:-1]:
                    temp_cfg = temp_cfg[k]
                temp_cfg[keys[-1]] = type(temp_cfg[keys[-1]])(value) if keys[-1] in temp_cfg else value

    return cfg
