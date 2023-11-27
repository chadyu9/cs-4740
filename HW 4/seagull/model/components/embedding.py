from typing import Optional, Union

import torch
from torch import nn

from seagull.nn.modules.module import Module
from seagull.nn.modules.rms_norm import RMSNorm


class Embedding(Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 768,
        max_positions: int = 512,
        padding_idx: Optional[int] = None,
        use_rope: bool = True,
        layer_norm_type: Optional[str] = None,
        dropout_proba: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self._dropout_proba = dropout_proba

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.use_rope = use_rope
        if not use_rope:
            self.position_embedding = nn.Embedding(
                num_embeddings=max_positions, embedding_dim=embedding_dim
            )
        self.apply_layer_norm = layer_norm_type is not None
        if layer_norm_type is not None:
            self.layer_norm = self._get_layer_norm(layer_norm_type=layer_norm_type)

    def _get_layer_norm(self, layer_norm_type: str) -> Union[Module, nn.Module]:
        if layer_norm_type.startswith("rms"):
            return RMSNorm(
                dimension=self.embedding_dim,
                eps=1e-8,
                dropout_proba=self._dropout_proba,
            )
        else:
            return nn.Sequential(
                nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-8),
                nn.Dropout(p=self._dropout_proba),
            )

    def forward(
        self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """See: https://pages.github.coecis.cornell.edu/cs4740/hw4-fa23/seagull.model.components.embedding.html."""
        # TODO-3
        token_embeddings = self.token_embedding(input_ids)
        if not self.use_rope:
            if not position_ids:
                position_ids = torch.arange(input_ids.shape[1])
                position_ids = position_ids.unsqueeze(dim=0)
                position_ids = position_ids.expand_as(input_ids)
            positional_embeddings = self.position_embedding(position_ids)
            sum = token_embeddings + positional_embeddings
            if self.apply_layer_norm:
                sum = self.layer_norm(sum)
            return sum
        else:
            return token_embeddings
