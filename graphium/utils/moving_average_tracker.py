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
