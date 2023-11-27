import math
from typing import Callable, Union

import torch
import torch.nn.functional as F


def compute_loss(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], preds: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    assert len(preds.shape) >= 2 and len(labels.shape) >= 1
    return loss_fn(preds.view(-1, preds.shape[-1]), labels.view(-1))


def compute_perplexity_from_entropy(entropy: Union[float, torch.Tensor]):
    return math.exp(entropy)


def compute_perplexity(preds: torch.Tensor, labels: torch.Tensor, labels_ignore_idx: int = -100) -> torch.Tensor:
    return torch.exp(
        F.cross_entropy(input=preds.view(-1, preds.shape[-1]), target=labels.view(-1), ignore_index=labels_ignore_idx)
    ).detach()


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        batch_size = 6
        max_length = 10
        output_dim = 5

        loss_fn = torch.nn.CrossEntropyLoss()

    test_config = TestConfig()
    test_output = torch.randn(test_config.batch_size, test_config.max_length, test_config.output_dim)
    test_labels = torch.randint(
        low=0, high=test_config.output_dim, size=(test_config.batch_size, test_config.max_length)
    )
    test_loss = compute_loss(loss_fn=test_config.loss_fn, preds=test_output, labels=test_labels)
    assert torch.allclose(
        test_loss, torch.nn.functional.cross_entropy(test_output.view(-1, test_config.output_dim), test_labels.view(-1))
    )
    assert compute_perplexity_from_entropy(test_loss) == torch.exp(test_loss)
    assert compute_perplexity(preds=test_output, labels=test_labels) == torch.exp(test_loss)
