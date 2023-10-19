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
            raise ValueError(
                f"{nonlinearity} not supported, choose one of: [tanh, relu, prelu]"
            )
        self.nonlinear = nonlinearity_dict[nonlinearity]

        # Setting up transitions between layers (W_1 is only one from embeeding to hidden)
        self.W_list = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=bias) for _ in range(num_layers - 1)
        )
        self.W_list.insert(0, nn.Linear(embedding_dim, hidden_dim, bias=bias))

        self.U_list = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=bias) for _ in range(num_layers)
        )
        self.V = nn.Linear(hidden_dim, output_dim, bias=bias)

        # Initialize weights of RNN
        self.apply(self.init_weights)

    def _initial_hidden_states(
        self,
        batch_size: int,
        init_zeros: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> List[torch.Tensor]:
        if init_zeros:
            hidden_states = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, device=device
            )
        else:
            hidden_states = nn.init.xavier_normal_(
                torch.empty(self.num_layers, batch_size, self.hidden_dim, device=device)
            )
        return list(map(torch.squeeze, hidden_states))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.rnn.html."""
        # Initialize hidden states and list of output tensors
        hidden_states = self._initial_hidden_states(
            batch_size=embeddings.shape[0], device=embeddings.device
        )
        outputs = []

        # Loop through time steps (batch max length)
        for t in range(embeddings.shape[1]):
            # Get x_t, the input at time t
            input_k = embeddings[:, t, :]

            # Pass through layers and update hidden states
            for k in range(self.num_layers):
                hidden_states[k] = self.nonlinear(
                    self.W_list[k](input_k) + self.U_list[k](hidden_states[k])
                )
                input_k = hidden_states[k]
            output_t = self.V(input_k)
            outputs.append(output_t.reshape([output_t.shape[0], 1, output_t.shape[1]]))

        return torch.cat(outputs, dim=1)
