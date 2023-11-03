import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class Encoder(nn.Module):
    """
    Encoder module for sequence-to-sequence models.
    """
    
    def __init__(self, 
                 embed_size: int, 
                 hidden_size: int, 
                 source_embeddings: nn.Embedding) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding = source_embeddings

        ### TODO 1 - Initialize the following variables:
        ###     self.encoder (Bidirectional LSTM with bias)
        ###     self.h_projection (Linear Layer with bias),called W_{h} above.
        ###     self.c_projection (Linear Layer with bias),called W_{c} above.
        ### YOUR CODE HERE (~3 Lines)

        
    def forward(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        ### GOAL:
        ###     1. Construct Tensor `X` by embedding the input. Note
        ###         that there is no initial hidden state or cell for the decoder.
        ###         Note: you should study the equations/mathematical definitions above to determine
        ###         what some of these values should be.  The same holds throughout.
        ###     2. Compute `enc_hiddens`, `last_hidden`,  `last_cell_state` by applying the LSTM encoder to `X`.
        ###     3. Compute
        ###         - `init_decoder_hidden`:
        ###             `last_hidden` is a tensor shape (2, b, h). The value 2 in the first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a new tensor (what is the shape of this tensor?).
        ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
        ###             This is h_0^{dec} in above in the writeup. Here b = batch size, h = hidden size
        ###         - `init_cell_hidden`:
        ###             `last_cell_state` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a new tensor (what is the shape of this tensor?).
        ###             Apply the c_projection layer to this in order to compute init_decoder_hidden.
        ###             This is c_0^{dec} in above in the writeup. Here b = batch size, h = hidden size

        ### YOUR CODE HERE
        enc_hiddens, dec_init_state = None, None
        
        #TODO 2 
        X = None 
        X = nn.utils.rnn.pack_padded_sequence(X, source_lengths, batch_first=True)

        #TODO 3
        enc_hiddens, (last_hidden, last_cell_state) = None 
        (enc_hiddens, _) = nn.utils.rnn.pad_packed_sequence(enc_hiddens, batch_first=True)

        #TODO 4 concatenate last hidden embed from both direction and with a linear projection 
        before_hidden_projection = None
        init_decoder_hidden = None

        #TODO 5 concatenate last cell state from both direction and with a linear projection 
        before_cell_projection = None
        init_cell_state = None
        
        dec_init_state = (init_decoder_hidden, init_cell_state)

        return enc_hiddens, dec_init_state