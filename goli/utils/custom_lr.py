import warnings
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLinearLR(_LRScheduler):
    """Custom linear learning rate with warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_num_epochs (int): Maximum number of epochs.
        warmup_epochs (int): The number of epochs for learning rate warmup. Default: 0.
        min_lr (float): the minimum learning rate value. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, max_num_epochs, warmup_epochs=0, min_lr=0, last_epoch=-1, verbose=False):
        self.max_num_epochs = max_num_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        super(WarmUpLinearLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )
        if self.warmup_epochs > 0 and (self.last_epoch + 1) < self.warmup_epochs:
            return [(self.last_epoch + 1) * base_lr / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # check epoch_diff in case there is a division by zero error
            epoch_diff = self.max_num_epochs - self.warmup_epochs
            if epoch_diff <= 0:
                factor = 0
            else:
                factor = ((self.last_epoch + 1) - self.warmup_epochs) / epoch_diff
            return [factor * self.min_lr + (1 - factor) * base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        lr_list = []
        for base_lr in self.base_lrs:
            if self.warmup_epochs > 0 and (self.last_epoch + 1) < self.warmup_epochs:
                lr = (self.last_epoch + 1) * base_lr / self.warmup_epochs
            else:
                factor = ((self.last_epoch + 1) - self.warmup_epochs) / (
                    self.max_num_epochs - self.warmup_epochs
                )
                lr = factor * self.min_lr + (1 - factor) * base_lr
            lr_list.append(lr)
        return lr_list
