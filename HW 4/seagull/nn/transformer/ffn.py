import torch

from seagull.nn.modules.glu import GLU
from seagull.nn.modules.linear import Linear
from seagull.nn.modules.module import Module


class FFN(Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        intermediate_dim: int = 2048,
        bias: bool = False,
        activation: str = "swish",
        dropout_proba: float = 0.1,
    ):
        """SwiGLU MLP: https://openreview.net/pdf?id=AL1fq05o7H."""
        super().__init__()

        self.glu = GLU(
            in_features=embedding_dim,
            out_features=intermediate_dim,
            bias=bias,
            activation=activation,
        )
        self.linear = Linear(
            in_features=intermediate_dim,
            out_features=embedding_dim,
            bias=bias,
            activation=None,
            dropout_proba=dropout_proba,
        )

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """See: https://pages.github.coecis.cornell.edu/cs4740/hw4-fa23/seagull.nn.transformer.ffn.html."""
        # TODO-5
        # Apply GLU and linear layers to the input embeddings.
        return self.linear(self.glu(input_embeddings))
