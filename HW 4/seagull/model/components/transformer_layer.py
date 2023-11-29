from typing import Union, Literal, Optional, Tuple

import torch
from torch import nn

from seagull.nn.modules.module import Module
from seagull.nn.modules.rms_norm import RMSNorm
from seagull.nn.transformer.ffn import FFN
from seagull.nn.transformer.mha import MultiHeadAttention


class TransformerLayer(Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        intermediate_dim: int = 2048,
        max_positions: int = 512,
        dropout_proba: float = 0.1,
        num_heads: int = 12,
        use_rope: bool = True,
        base: int = 10000,
        attn_dropout_proba: float = 0.1,
        causal: bool = True,
        ffn_bias: bool = False,
        ffn_activation: str = "swish",
        layer_norm_mode: Literal["pre", "post"] = "pre",
        layer_norm_type: str = "rms",
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self._dropout_proba = dropout_proba

        self.mha = MultiHeadAttention(
            embedding_dim=embedding_dim,
            max_positions=max_positions,
            num_heads=num_heads,
            use_rope=use_rope,
            base=base,
            attn_dropout_proba=attn_dropout_proba,
            dropout_proba=dropout_proba,
            causal=causal,
        )
        self.ffn = FFN(
            embedding_dim=embedding_dim,
            intermediate_dim=intermediate_dim,
            bias=ffn_bias,
            activation=ffn_activation,
            dropout_proba=dropout_proba,
        )

        self.layer_norm_mode = layer_norm_mode
        self.layer_norm = self._get_layer_norm(layer_norm_type=layer_norm_type)

    def reset_kv_cache(self):
        self.mha.reset_kv_cache()

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

    def maybe_apply_layer_norm(
        self,
        tensor: torch.Tensor,
        current_layer_norm_application: Literal["pre", "post"],
    ) -> torch.Tensor:
        if current_layer_norm_application == self.layer_norm_mode:
            return self.layer_norm(tensor)
        return tensor

    def forward(
        self,
        input_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        See: https://pages.github.coecis.cornell.edu/cs4740/hw4-fa23/seagull.model.components.transformer_layer.html.
        """
        # TODO-6.1
        # Implement the forward pass of a single Transformer layer.
        # Apply the MHA layer and pre/post norm as needed.
        mha_output, masked_attn_probs = self.mha(
            self.maybe_apply_layer_norm(
                tensor=input_embeddings, current_layer_norm_application="pre"
            ),
            padding_mask=padding_mask,
            use_kv_cache=use_kv_cache,
        )
        mha_output = self.maybe_apply_layer_norm(
            tensor=input_embeddings + mha_output, current_layer_norm_application="post"
        )

        # Apply the FFN layer and pre/post norm as needed.
        return (
            self.maybe_apply_layer_norm(
                tensor=mha_output
                + self.ffn(
                    self.maybe_apply_layer_norm(
                        tensor=mha_output, current_layer_norm_application="pre"
                    )
                ),
                current_layer_norm_application="post",
            ),
            masked_attn_probs,
        )
