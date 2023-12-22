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


from dataclasses import dataclass


@dataclass
class MovingAverageTracker:
    num_samples: int = 0
    mean_value: float = 0.0

    def update(self, value: float):
        self.mean_value = self.mean_value * (self.num_samples / (self.num_samples + 1)) + value / (
            self.num_samples + 1
        )
        self.num_samples += 1

    def reset(self):
        # no need to update mean_value, it will be reset when update is called
        self.num_samples = 0
