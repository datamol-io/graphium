import sys
from typing import Any, Callable, Dict, List, Optional, Union, Literal, Iterable
from lightning.pytorch.callbacks import TQDMProgressBar


class ProgressBarMetrics(TQDMProgressBar):
    def __init__(
        self,
        metrics_on_progress_bar: Optional[Iterable[str]] = None,
        loss_alias: Optional[str] = "_global/loss/train",
    ) -> None:
        super().__init__()
        if metrics_on_progress_bar is None:
            metrics_on_progress_bar = {}
        self.metrics_on_progress_bar = set(metrics_on_progress_bar)
        self.loss_alias = loss_alias

    def get_metrics(self, trainer, pl_module) -> Dict[str, Union[int, str, float, Dict[str, float]]]:

        metrics = super().get_metrics(trainer, pl_module)
        filtered_metrics = {}
        for key, metric in metrics.items():
            if key in self.metrics_on_progress_bar:
                if key == self.loss_alias:
                    filtered_metrics["loss"] = metric
                else:
                    filtered_metrics[key] = metric

        return filtered_metrics
