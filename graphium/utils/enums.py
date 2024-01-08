from typing import Union
from enum import Enum

class TaskLevel(Enum):
    """Enum for the task level of a dataset. Options are graph, node, edge, nodepair."""
    GRAPH = 1 # Default
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
