import math

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class LinearWarmupCosineAnnealingLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int = 0,
        annealing_period: int = 20000,
        warmup_init_lr: float = 2e-5,
        max_lr: float = 0.1,
        min_lr: float = 0.0,
        lr_shrink_factor: float = 0.1,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr if min_lr < max_lr else self.max_lr

        # Linear LR warmup steps.
        self.num_warmup_steps = num_warmup_steps
        self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = self.max_lr
        if self.num_warmup_steps > 0:
            self.linear_lr_step_size = (self.warmup_end_lr - self.warmup_init_lr) / self.num_warmup_steps

        # Cosine annealing steps.
        self.lr_shrink_factor = lr_shrink_factor
        self.annealing_period = annealing_period

        super().__init__(optimizer=optimizer, last_epoch=last_epoch)
        self.init_lr(self.warmup_init_lr if num_warmup_steps > 0 else self.min_lr)

    def init_lr(self, lr):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            self.base_lrs.append(lr)

    def get_lr(self):
        step_num = self.last_epoch
        if step_num == -1:
            return self.base_lrs
        elif step_num < self.num_warmup_steps:
            # Linear warmup LR scheduling.
            return [(self.linear_lr_step_size * (step_num + 1)) + base_lr for base_lr in self.base_lrs]
        else:
            # Cosine annealing after linear warmup: https://arxiv.org/abs/1608.03983.
            num_updates = step_num - self.num_warmup_steps

            # TODO: Add support for `t_mult`: factor to grow the length of each annealing period.
            cycle_num = math.floor(num_updates / self.annealing_period)
            t_current = num_updates - (self.annealing_period * cycle_num)
            t_max = self.annealing_period

            # Shrink LR for each annealing period, following the first one.
            lr_shrink = self.lr_shrink_factor**cycle_num
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink

            return [
                min_lr + (0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t_current / t_max))) for _ in self.base_lrs
            ]


if __name__ == "__main__":
    from dataclasses import dataclass

    import matplotlib.pyplot as plt
    import torch

    @dataclass
    class TestConfig:
        batch_size = 10
        max_length = 8
        embedding_dim = 30

        num_warmup_steps = [0, 1000, 2000, 4000]
        annealing_period = [5000, 5000, 2500, 1000]
        warmup_init_lr = 2e-5
        max_lr = 1.0
        min_lr = 0.1  # 10% of peak LR
        lr_shrink_factor = 0.5

        output_dim = 5
        total_steps = 10000

    test_config = TestConfig()
    test_model = torch.nn.Linear(test_config.embedding_dim, test_config.output_dim)
    test_optimizer = torch.optim.AdamW(test_model.parameters(), betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1)

    for config_num in range(len(test_config.num_warmup_steps)):
        all_lrs = []
        test_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=test_optimizer,
            num_warmup_steps=test_config.num_warmup_steps[config_num],
            annealing_period=test_config.annealing_period[config_num],
            warmup_init_lr=test_config.warmup_init_lr,
            max_lr=test_config.max_lr,
            min_lr=test_config.min_lr,
            lr_shrink_factor=test_config.lr_shrink_factor,
        )
        for _ in range(test_config.total_steps):
            test_optimizer.zero_grad()
            outputs = test_model(torch.randn(test_config.batch_size, test_config.max_length, test_config.embedding_dim))
            labels = torch.randint(
                low=0, high=test_config.output_dim, size=(test_config.batch_size, test_config.max_length)
            )
            loss = torch.nn.functional.cross_entropy(
                input=outputs.view(-1, test_config.output_dim), target=labels.view(-1)
            )
            loss.backward()
            test_optimizer.step()
            test_scheduler.step()
            all_lrs.extend([param_group["lr"] for param_group in test_optimizer.param_groups])
        plt.plot(all_lrs)
    plt.xlabel("step num")
    plt.ylabel("lr")
    plt.show()
