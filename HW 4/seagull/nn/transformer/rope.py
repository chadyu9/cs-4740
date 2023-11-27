from typing import Tuple, Optional

import torch
from einops import rearrange

from seagull.nn.modules.module import Module
from seagull.nn.modules.utils.utils import set_jit_flags

set_jit_flags()


class RotaryPositionalEmbedding(Module):
    def __init__(self, head_dim: int = 64, max_positions: int = 512, base: int = 10000):
        """RoFormer: https://arxiv.org/pdf/2104.09864.pdf."""
        super().__init__()

        self.head_dim = head_dim
        self.max_positions = max_positions
        self.base = base

        theta = 1 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("theta", theta)

        self._seq_length_cache = None
        self.cos_cache = None
        self.sin_cache = None
        self._cache_required()

    def _cache_required(self):
        self._seq_length_cache = self.max_positions
        # position_ids: (max_length)
        position_ids = torch.arange(self.max_positions, device=self.theta.device, dtype=self.theta.dtype)

        # freqs: (max_length, embedding_dim / 2) -> (max_length, embedding_dim)
        freqs = torch.einsum("p, t -> pt", position_ids, self.theta)
        embedding = torch.cat([freqs, freqs], dim=-1)

        # cos_cache, sin_cache: (1, 1, max_length, embedding_dim)
        self.cos_cache = rearrange(embedding.cos(), "l e -> 1 1 l e")
        self.sin_cache = rearrange(embedding.sin(), "l e -> 1 1 l e")

    def forward(
        self, seq_length: int, start_pos: int = 0, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if start_pos + seq_length > self.max_positions:
            raise AssertionError(f"{(start_pos + seq_length)=} > RotaryPositionalEmbedding.max_positions")
        device = device if device is not None else self.theta.device
        if seq_length != self.max_positions:
            return (
                self.cos_cache[:, :, start_pos : start_pos + seq_length, :].to(device),
                self.sin_cache[:, :, start_pos : start_pos + seq_length, :].to(device),
            )
        return self.cos_cache.to(device), self.sin_cache.to(device)


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    # tensor_1, tensor_2: (..., max_length, head_dim) -> (..., max_length, head_dim / 2)
    tensor_1, tensor_2 = tensor[..., : tensor.shape[-1] // 2], tensor[..., tensor.shape[-1] // 2 :]
    # Note: https://discuss.pytorch.org/t/torch-jit-trace-unexpected-error-with-torch-cat-dim-1/55152; using `dim=-1`
    # below triggers a vector size error.
    return torch.cat([-tensor_2, tensor_1], dim=(tensor_1.ndim - 1))


@torch.jit.script
def apply_rope(
    query: torch.Tensor, key: torch.Tensor, cos_vals: torch.Tensor, sin_vals: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # query, key: (batch_size, num_heads, max_length, head_dim)
    # cos_vals, sin_vals: (1, 1, max_length, head_dim)
    return (
        (query * cos_vals) + (_rotate_half(query) * sin_vals),
        (key * cos_vals) + (_rotate_half(key) * sin_vals),
    )


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        batch_size = 1
        num_heads = 1
        seq_length = 4

        head_dim = 8
        max_positions = 8
        base = 10000

    test_config = TestConfig()
    test_rope = RotaryPositionalEmbedding(
        head_dim=test_config.head_dim, max_positions=test_config.max_positions, base=test_config.base
    )
    test_rope.print_params()
    test_cos, test_sin = test_rope(seq_length=test_config.seq_length)
    assert test_cos.shape == (1, 1, test_config.seq_length, test_config.head_dim)
    assert test_sin.shape == (1, 1, test_config.seq_length, test_config.head_dim)

    test_input = (
        torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [11, 22, 33, 44, 55, 66, 77, 88],
                [111, 222, 333, 444, 555, 666, 777, 888],
                [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888],
            ]
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    test_rope_query, test_rope_key = apply_rope(query=test_input, key=test_input, cos_vals=test_cos, sin_vals=test_sin)
    assert test_input.shape == test_rope_query.shape
