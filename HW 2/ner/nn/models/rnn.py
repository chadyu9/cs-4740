import logging
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from ner.nn.module import Module


class RNN(Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ):
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.rnn.html."""
        super().__init__()

        assert num_layers > 0

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        logging.info(f"no shared weights across layers")

        nonlinearity_dict = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "prelu": nn.PReLU()}
        if nonlinearity not in nonlinearity_dict:
            raise ValueError(f"{nonlinearity} not supported, choose one of: [tanh, relu, prelu]")
        self.nonlinear = nonlinearity_dict[nonlinearity]

        # TODO-5-1
        raise NotImplementedError  # remove once the method is filled

        self.apply(self.init_weights)

    def _initial_hidden_states(
        self, batch_size: int, init_zeros: bool = False, device: torch.device = torch.device("cpu")
    ) -> List[torch.Tensor]:
        if init_zeros:
            hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        else:
            hidden_states = nn.init.xavier_normal_(
                torch.empty(self.num_layers, batch_size, self.hidden_dim, device=device)
            )
        return list(map(torch.squeeze, hidden_states))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.rnn.html."""
        # TODO-5-2
        raise NotImplementedError  # remove once the method is filled
