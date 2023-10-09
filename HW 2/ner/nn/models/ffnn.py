from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ner.nn.module import Module


class FFNN(Module):
    def __init__(
        self, embedding_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1
    ) -> None:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.ffnn.html."""
        super().__init__()

        assert num_layers > 0

        self.W = nn.Linear(embedding_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, output_dim, bias=False)
        self.f = F.relu

        self.apply(self.init_weights)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.ffnn.html."""
        # TODO-4-2
        hidden = self.W(embeddings)
        hidden_relu = self.f(hidden)
        output = self.V(hidden_relu)

        return output
