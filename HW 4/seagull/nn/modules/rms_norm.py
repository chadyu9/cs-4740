from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from seagull.nn.modules.module import Module
from seagull.nn.modules.utils.utils import set_jit_flags

set_jit_flags()


@torch.jit.script
def fused_gain_bias_dropout(
    input_normed: torch.Tensor,
    gain: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dropout_proba: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    y = gain * input_normed
    y = y + bias if bias is not None else y
    return F.dropout(y, p=dropout_proba, training=training) if dropout_proba > 0.0 else y


class RMSNorm(Module):
    __constants__ = ["dimension", "p", "eps", "bias", "dropout_proba"]

    def __init__(
        self, dimension: int, p: float = -1, eps: float = 1e-8, bias: bool = False, dropout_proba: float = 0.0
    ):
        super().__init__()

        self.dimension = dimension
        self.p = p
        self.eps = eps
        self.dropout_proba = dropout_proba

        self.gain = nn.Parameter(torch.ones(dimension), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dimension), requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] == self.dimension, f"RMSNorm.dimension: {self.dimension} must equal input.shape[-1]"
        if self.p < 0.0 or self.p > 1.0:
            norm = input.norm(p=2, dim=-1, keepdim=True)
            rms_input = norm / (self.dimension**0.5)
        else:
            partial_size = int(self.dimension * self.p)
            partial_input, _ = torch.split(input, [partial_size, self.dimension - partial_size], dim=-1)
            norm = partial_input.norm(p=2, dim=-1, keepdim=True)
            rms_input = norm / (partial_size**0.5)
        input_normed = input / (rms_input + self.eps)
        return fused_gain_bias_dropout(
            input_normed=input_normed,
            gain=self.gain,
            bias=self.bias,
            dropout_proba=self.dropout_proba,
            training=self.training,
        )


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        batch_size = 100
        max_length = 300
        embedding_dim = 200

        bias = True
        p = [-1.0, 0.5]

    test_config = TestConfig()
    test_rmsnorm = RMSNorm(dimension=test_config.embedding_dim, bias=test_config.bias, p=test_config.p[0])
    test_rmsnorm.print_params()
    test_input = torch.ones(test_config.batch_size, test_config.max_length, test_config.embedding_dim)
    assert torch.allclose(test_rmsnorm(test_input), test_input)

    test_partial_rmsnorm = RMSNorm(dimension=test_config.embedding_dim, bias=test_config.bias, p=test_config.p[1])
    assert torch.allclose(test_partial_rmsnorm(test_input), test_input)
