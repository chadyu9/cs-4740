import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from seagull.nn.modules.module import Module
from seagull.nn.modules.utils.glu_activations import get_activation_fn


class GLU(Module):
    __constants__ = ["in_features", "out_features", "bias", "activation"]

    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation: Optional[str] = "swish"):
        """Gated linear units: https://arxiv.org/pdf/2002.05202v1.pdf."""
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = get_activation_fn(activation)

        self.weight = nn.Parameter(torch.Tensor(2 * out_features, in_features), requires_grad=True)
        if bias:
            self.bias_b = nn.Parameter(torch.Tensor(out_features))
            self.bias_c = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias_b", None)
            self.register_parameter("bias_c", None)

        self._reset_parameters()

    def _reset_parameters(self):
        for weight, bias in zip([self.weight], [self.bias_b, self.bias_c]):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        xW, xV = F.linear(input=input, weight=self.weight, bias=None).split(split_size=self.out_features, dim=-1)
        return self.activation_fn(xW=xW, xV=xV, bias_b=self.bias_b, bias_c=self.bias_c)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_b is not None}"


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        batch_size = 20
        max_length = 50
        embedding_dim = 300

        activation = "gelu"
        bias = True

        output_dim = 30

    test_config = TestConfig()
    test_glu = GLU(
        in_features=test_config.embedding_dim,
        out_features=test_config.output_dim,
        activation=test_config.activation,
        bias=test_config.bias,
    )
    weight_W, weight_V = test_glu.weight.split(split_size=test_config.output_dim, dim=0)
    test_glu.print_params()

    test_input = torch.randn((test_config.batch_size, test_config.max_length, test_config.embedding_dim))
    assert test_glu(test_input).shape == (test_config.batch_size, test_config.max_length, test_config.output_dim)
    assert torch.allclose(
        test_glu(test_input),
        F.gelu(F.linear(test_input, weight_W, test_glu.bias_b)) * F.linear(test_input, weight_V, test_glu.bias_c),
    )
