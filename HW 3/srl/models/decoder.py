import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Tuple


class Decoder(nn.Module):
    """
    Decoder module for sequence-to-sequence models with attention mechanism.
    """

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        target_embedding: nn.Embedding,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = target_embedding
        self.output_vocab_size = self.embedding.weight.size(0)
        self.att_projection = nn.Linear(
            in_features=self.hidden_size * 2, out_features=self.hidden_size, bias=False
        )

        ### TODO 1:
        ###   YOUR CODE HERE (~3 lines)
        ###     self.decoder (LSTM Cell with bias)
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{v} above.
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{target} above.
        ###   You may find some of these functions useful:
        ###     LSTM Cell:
        ###     https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html

    def forward(
        self,
        enc_hiddens: torch.Tensor,
        dec_init_state: torch.Tensor,
        target_padded: torch.Tensor,
    ) -> torch.Tensor:
        # Chop off the <END> token for max length sentences.
        target_padded = target_padded[:, :-1]

        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ### TODO 2:
        ###     1. Construct tensor `Y` by embedding the target sentences.
        ###     2. Construct enc_hiddens_proj by using self.att_projection to project enc_hiddens into a new shape (what is this shape?)
        ###     3. Iterate over the correct dimension of Y.
        ###         Think about what the resultant shape of this new tensor should be, in relation to b = batch size and e = embedding size
        ###             - Squeeze Y_t (a 3-dimentional tensor) into a two dimensional tensor.
        ###             - Construct Ybar_t by concatenating Y_t with o_prev on their last dimension
        ###             - Use the step function to compute the the Decoder's next (cell, state) values
        ###               as well as the new combined output o_t.
        ###             - Append o_t to combined_outputs
        ###             - Update o_prev to the new o_t.
        ###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        ###         tensors output tensors, to a single tensor shape (think of the target shape of this 3-dimensional tensor as analagous to reversing
        ###             the process we used to extract singular tensors from our initial tensor Y).
        ###
        ### Note:
        ###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###      over. Otherwise, you will remove the batch dimension accidentally.
        ###
        ### You may find some of these functions useful:
        ###     Zeros Tensor:
        ###         https://pytorch.org/docs/stable/torch.html#torch.zeros
        ###     Tensor Dimension Squeezing:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat

        ### YOUR CODE HERE

        return combined_outputs

    def step(
        self,
        Ybar_t: torch.Tensor,
        dec_state: Tuple[torch.Tensor, torch.Tensor],
        enc_hiddens: torch.Tensor,
        enc_hiddens_proj: torch.Tensor,
    ) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """Compute one forward step of the LSTM decoder, including the attention computation.
        :param Ybar_t: Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        :type Ybar_t: torch.Tensor
        :param dec_state: Tensors with shape (b, h), where b = batch size, h = hidden size.
                Tensor is decoder's prev hidden state
        :type dec_state: torch.Tensor
        :param enc_hiddens: Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        :type enc_hiddens: torch.Tensor
        :param enc_hiddens_proj: Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        :type enc_hiddens_proj: torch.Tensor
        :returns dec_state: Tensors with shape (b, h), where b = batch size, h = hidden size.
                Tensor is decoder's new hidden state
        returns combined_output: Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        """

        ### TODO 3:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state` to obtain the new dec_state.
        dec_state = None
        (dec_hidden, dec_cell) = dec_state

        ### TODO 4(Attention Step):
        ###     1. Use dot product to calculate similarity between enc_hiddens_proj and dec_hidden,
        ###        and then take softmax (this is the attention weight alpha_t)
        ###     2. Dot product attention weight with enc_hiddens to get weighted context embedding a_t
        ###     3. U_t = Concatenate dec_hidden and a_t

        ### TODO 5:
        ###     1. Apply the combined output projection layer to U_t to compute tensor V_t
        V_t = None

        combined_output = V_t
        return dec_state, combined_output
