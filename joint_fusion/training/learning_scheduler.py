import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmRestartsDecay(_LRScheduler):
    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=0,
        decay_factor=1.0,
        last_epoch=-1,
        verbose=False,
    ):
        """
        Cosine annealing with warm restarts and optional learning rate decay between cycles.

        Args:
            optimizer: Wrapped optimizer
            T_0: Number of iterations for the first restart
            T_mult: A factor increases T_i after a restart
            eta_min: Minimum learning rate
            decay_factor: Factor by which max_lr is reduced after each cycle (1.0 = no decay)
            last_epoch: The index of last epoch
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        self.T_i = T_0  # Current period length
        self.T_cur = 0  # Current step within period
        self.cycle = 0  # Current cycle number

        # Store initial learning rates for each param group
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_max_lrs = self.base_lrs.copy()

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):

        return [
            self.eta_min
            + (max_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for max_lr in self.current_max_lrs
        ]

    def step(self, epoch=None):
        """
        1. First calculates and applies the current learning rate
        2. Updates the internal state for the next step
        3. Handles cycle transitions after the current step is complete
        """

        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        self.T_cur += 1

        if self.T_cur > self.T_i:
            # Cycle completed, prep for next one
            self.cycle += 1

            self.current_max_lrs = [
                lr * self.decay_factor for lr in self.current_max_lrs
            ]

            # Restart
            self.T_cur = 0
            self.T_i = int(self.T_i * self.T_mult)
