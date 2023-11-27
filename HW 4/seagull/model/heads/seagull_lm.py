from typing import Optional, Tuple, Union, List, Any, Set

import torch
from torch import nn

from seagull.model.seagull_transformer import Seagull
from seagull.nn.modules.glu import GLU
from seagull.nn.modules.linear import Linear
from seagull.nn.modules.module import Module
from seagull.nn.modules.rms_norm import RMSNorm
from seagull.nn.modules.utils.activations import softmax


class SeagullLM(Module):
    def __init__(self, weight_tying: bool = True, **seagull_kwargs: Any):
        super().__init__()

        self._max_positions = seagull_kwargs["max_positions"]

        self.seagull = Seagull(**seagull_kwargs)
        self.weight_tying = weight_tying
        if not weight_tying:
            self.lm_head = Linear(
                in_features=seagull_kwargs["embedding_dim"],
                out_features=seagull_kwargs["vocab_size"],
                bias=False,
                activation=None,
            )

        self.apply(self._init_weights)

    def reset_kv_cache(self):
        self.seagull.reset_kv_cache()

    def _init_weights(self, module: nn.Module) -> None:
        # Initialize module bias parameters.
        if isinstance(module, (nn.Linear, Linear, RMSNorm, nn.LayerNorm)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, GLU):
            if module.bias_b is not None:
                nn.init.zeros_(module.bias_b)
                nn.init.zeros_(module.bias_c)

        # Initialize module weight parameters.
        if isinstance(module, (nn.Embedding, nn.Linear, Linear)):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, GLU):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.gain)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        return_output_at_all_layers: bool = False,
        return_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]]:
        """See: https://pages.github.coecis.cornell.edu/cs4740/hw4-fa23/seagull.model.heads.seagull_lm.html."""
        # TODO-6.2
        raise NotImplementedError  # remove once the method is filled

    @torch.no_grad()
    def talk(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        num_samples: int = 1,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> List[List[int]]:
        def _get_updated_sequence_lengths(
            input_ids_: torch.Tensor,
            current_generation_length: int,
            all_sample_sequence_lengths_: torch.Tensor,
            modified_batch_idxs_: Set,
        ):
            # Dynamically populate the sequence length to "split" the tensor later.
            eos_token_batch_idxs = torch.where(input_ids_ == eos_token_id)[0].tolist()
            if len(eos_token_batch_idxs) > 0:
                new_eos_token_batch_idxs = list(set(eos_token_batch_idxs) - modified_batch_idxs_)
                all_sample_sequence_lengths_[new_eos_token_batch_idxs] = current_generation_length - 1
                modified_batch_idxs_.update(new_eos_token_batch_idxs)
            return all_sample_sequence_lengths_, modified_batch_idxs_

        input_ids = input_ids.expand(num_samples, -1).to(self.device)
        input_ids = input_ids[:, -self._max_positions :] if input_ids.shape[1] > self._max_positions else input_ids
        position_ids = torch.arange(input_ids.shape[1]).expand(num_samples, -1).to(self.device)

        all_input_ids = input_ids
        all_sample_sequence_lengths = torch.ones(num_samples, dtype=torch.long) * (input_ids.shape[1] + max_new_tokens)
        modified_batch_idxs = set()

        self.eval()  # https://ai.stackexchange.com/a/18392
        for _ in range(max_new_tokens):
            assert input_ids.shape == position_ids.shape

            # lm_logits: (num_samples, seq_length, vocab_size)
            lm_logits = self(input_ids=input_ids.to(self.device), position_ids=position_ids, use_kv_cache=True)[0]
            # lm_logits: (num_samples, vocab_size)
            lm_logits = lm_logits[:, -1, :] / temperature

            if top_k is not None:
                top_k_vals, _ = torch.topk(lm_logits, k=top_k, dim=-1, largest=True, sorted=True)
                lm_logits[lm_logits < top_k_vals[:, [-1]]] = -torch.finfo(lm_logits.dtype).max
            probs = softmax(lm_logits, dim=-1, training=False)
            # input_ids: (num_samples, 1)
            input_ids = torch.multinomial(probs, num_samples=1)

            all_input_ids = torch.cat([all_input_ids, input_ids], dim=-1)
            position_ids = (
                (position_ids[:, -1] + 1).unsqueeze(1)
                if position_ids[-1, -1] != self._max_positions
                else position_ids[:, -1].unsqueeze(1)
            )
            if eos_token_id is not None:
                all_sample_sequence_lengths, modified_batch_idxs = _get_updated_sequence_lengths(
                    input_ids_=input_ids,
                    current_generation_length=all_input_ids.shape[1],
                    all_sample_sequence_lengths_=all_sample_sequence_lengths,
                    modified_batch_idxs_=modified_batch_idxs,
                )

        self.reset_kv_cache()  # reset start position to zero
        all_input_ids = all_input_ids.tolist()
        if eos_token_id is not None:
            all_input_ids = [seq[: all_sample_sequence_lengths[seq_idx]] for seq_idx, seq in enumerate(all_input_ids)]
        return all_input_ids
