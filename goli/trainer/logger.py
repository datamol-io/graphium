"""
I don't really want to invent my own logger, 
but for now I can't log hyperparameters with metrics without a workaround.
See issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/1228.
Code provided by https://github.com/mRcSchwering/pytorch_lightning_test
"""
from typing import Dict
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import rank_zero_only
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


from typing import Dict
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import rank_zero_only
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class HyperparamsMetricsTensorBoardLogger(TensorBoardLogger):
    """
    By default hyperparameters would be logged without any metric at the beginning of the training.
    That's why I silence the "normal" `log_hyperparams` call by pytorch lightning.
    Next, to actually log hyperparameters I am using my own adaption of the original `log_hyperparams`.
    This will make sure the tensorboard event files will end up in the same subdirectory.
    This is importtant because every call will write a new file and tensorboard will interpret these
    files as one metric as long as they have the same hyperparameter set and live in the same subdir.
    (At least thats what I found out by trial+error).
    Good: log hyperparams with metrics however you want
    Bad: have many event files (but tensorboard still interprets them correctly)
    """

    def __init__(self, *args, **kwargs):
        super(HyperparamsMetricsTensorBoardLogger, self).__init__(*args, **kwargs)
        self.open()

    def open(self):
        self.tf_summary_writer = SummaryWriter(log_dir=self.log_dir).__enter__()

    def close(self):
        self.tf_summary_writer.__exit__(None, None, None)

    def log_hyperparams(self, params: Dict[str, any], metrics: Dict[str, any] = None):
        pass

    @rank_zero_only
    def log_hyperparams_metrics(self, params: Dict[str, any], metrics: Dict[str, any] = None):
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        sanitized_params = self._sanitize_params(params)
        # self.experiment.add_hparams(sanitized_params, {} if metrics is None else metrics)
        # self._add_hparams(sanitized_params, {} if metrics is None else metrics)
        self._add_hparams2(sanitized_params, {} if metrics is None else metrics)
        # self.tags.update(sanitized_params)

    def _add_hparams(self, hparam_dict: dict, metric_dict: dict):
        exp, ssi, sei = hparams(hparam_dict, metric_dict)
        with SummaryWriter(log_dir=self.log_dir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)

    def _add_hparams2(self, hparam_dict: dict, metric_dict: dict):
        exp, ssi, sei = hparams(hparam_dict, metric_dict)
        self.tf_summary_writer.file_writer.add_summary(exp)
        self.tf_summary_writer.file_writer.add_summary(ssi)
        self.tf_summary_writer.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.tf_summary_writer.add_scalar(k, v)
