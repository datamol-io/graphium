"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""

from typing import Union
from enum import Enum


class TaskLevel(Enum):
    """Enum for the task level of a dataset. Options are graph, node, edge, nodepair."""

    GRAPH = 1  # Default
    NODE = 2
    EDGE = 3
    NODEPAIR = 4

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(label: Union[str, "TaskLevel", type(None)]) -> "TaskLevel":
        """
        Converts a string to a TaskLevel enum. If label is None, returns TaskLevel.GRAPH. If label is already a TaskLevel enum, returns label.
        """
        if label is None:
            return TaskLevel.GRAPH
        if isinstance(label, TaskLevel):
            return label
        return TaskLevel[label.upper()]
