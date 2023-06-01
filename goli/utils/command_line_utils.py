import re
from collections import defaultdict


class ConfigDict(dict):
    """ConfigDict object that creates a nested dictionary accesible via dot noation

    Args:
        dict (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError("No such attribute: " + key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError("No such attribute: " + key)

    def to_dict(self):
        return {key: self[key].to_dict() if isinstance(self[key], ConfigDict) else self[key] for key in self}


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


def update_config(cfg: ConfigDict, unknown: list, anchors: list):
    """
    Update the configuration dictionary with command line arguments.
    """
    for arg in unknown:
        if arg.startswith("--"):
            key, value = arg[2:].split("=")
            # TODO: If key in anchor keys - loop through all of those as well.
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
