from typing import Optional, Callable

import torch
import torch.nn.functional as F

from seagull.nn.modules.utils.utils import set_jit_flags

set_jit_flags()

_supported_glu_activations = [
    None,
    "linear",
    "bilinear",
    "gelu",
    "geglu",
    "swish",
    "silu",
    "swiglu",
    "relu",
    "reglu",
    "glu",
    "sigm",
    "sigmoid",
    "sigmglu",
]


@torch.jit.script
def bilinear(
    xW: torch.Tensor, xV: torch.Tensor, bias_b: Optional[torch.Tensor] = None, bias_c: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return (xW + bias_b if bias_b is not None else xW) * (xV + bias_c if bias_c is not None else xV)


@torch.jit.script
def swiglu(
    xW: torch.Tensor, xV: torch.Tensor, bias_b: Optional[torch.Tensor] = None, bias_c: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return F.silu(xW + bias_b if bias_b is not None else xW) * (xV + bias_c if bias_c is not None else xV)


@torch.jit.script
def sigmglu(
    xW: torch.Tensor, xV: torch.Tensor, bias_b: Optional[torch.Tensor] = None, bias_c: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # PyTorch GLU function applies sigmoid on the second component.
    return (xW + bias_b if bias_b is not None else xW) * F.sigmoid(xV + bias_c if bias_c is not None else xV)


@torch.jit.script
def reglu(
    xW: torch.Tensor, xV: torch.Tensor, bias_b: Optional[torch.Tensor] = None, bias_c: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return F.relu(xW + bias_b if bias_b is not None else xW) * (xV + bias_c if bias_c is not None else xV)


@torch.jit.script
def geglu(
    xW: torch.Tensor, xV: torch.Tensor, bias_b: Optional[torch.Tensor] = None, bias_c: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return F.gelu(xW + bias_b if bias_b is not None else xW) * (xV + bias_c if bias_c is not None else xV)


def get_activation_fn(
    activation: str,
) -> Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]], torch.Tensor]:
    if activation is None or activation in ["linear", "bilinear"]:
        return bilinear
    if activation in ["gelu", "geglu"]:
        return geglu
    elif activation in ["swish", "silu", "swiglu"]:
        return swiglu
    elif activation in ["relu", "reglu"]:
        return reglu
    elif activation in ["glu", "sigm", "sigmoid", "sigmglu"]:
        return sigmglu
    else:
        raise ValueError(f"{activation} not supported; must be one of {_supported_glu_activations}")


if __name__ == "__main__":
    test_activation_fn = get_activation_fn("glu")
    max_length = 50000

    test_input = torch.randn((max_length,))
    xW, xV = test_input.split(split_size=int(max_length / 2))
    assert torch.allclose(test_activation_fn(xW=xW, xV=xV), F.glu(test_input, dim=0))
