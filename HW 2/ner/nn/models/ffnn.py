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

        # Setting up the transitions between layers
        self.W = nn.Linear(embedding_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, output_dim, bias=False)
        self.hid_to_hid = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.f = F.relu

        # Initialize weights of FFNN
        self.apply(self.init_weights)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.ffnn.html."""
        # Forward pass through first hidden layer
        hidden = self.W(embeddings)
        hidden_relu = self.f(hidden)

        # Forward pass through remaining hidden layers (if any)
        for i in range(len(self.hid_to_hid)):
            hidden = self.hid_to_hid[i](hidden_relu)
            hidden_relu = self.f(hidden)

        # Output layer
        output = self.V(hidden_relu)

        return output
