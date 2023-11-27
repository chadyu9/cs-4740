import random
from dataclasses import dataclass
from typing import Optional, Dict

import datasets
import torch
from torch.utils.data import Dataset


class SequenceSamplingDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        model_max_positions: int = 512,
        seq_start_pos: Optional[int] = None,
    ):
        super().__init__()

        dataset.set_format(type="torch")
        self.dataset = dataset
        self.max_length = model_max_positions + 1  # autoregressive modeling
        self.seq_start_pos = seq_start_pos  # 0: truncate right, -1: truncate left

    def _process_inputs_and_padding(self, input_ids, padding_mask):
        if len(input_ids) > self.max_length:
            if self.seq_start_pos is not None and self.seq_start_pos < 0:
                # Use the last `self.max_length` entries.
                input_ids = input_ids[-self.max_length :]
                padding_mask = padding_mask[-self.max_length :]
            else:
                seq_start_pos = (
                    random.randrange(len(input_ids) - self.max_length)
                    if self.seq_start_pos is None
                    else self.seq_start_pos
                )
                # Note: The `<|endoftext|>` token might get truncated as a result of this; that's okay because we
                # want the model to learn that `<|endoftext|>` truly means that the text has "ended."
                input_ids = input_ids[seq_start_pos : seq_start_pos + self.max_length]
                padding_mask = padding_mask[seq_start_pos : seq_start_pos + self.max_length]
        return input_ids, padding_mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        input_ids, padding_mask = self.dataset[index]["input_ids"], self.dataset[index]["attention_mask"].logical_not()
        input_ids, padding_mask = self._process_inputs_and_padding(input_ids=input_ids, padding_mask=padding_mask)

        """
        See: https://pages.github.coecis.cornell.edu/cs4740/hw4-fa23/seagull.data_processing.sequence_sampler.html.
        """
        # TODO-2.3
        labels = input_ids[1:]
        input_ids = input_ids[:-1]
        padding_mask = padding_mask[:-1]
        return {"input_ids" : input_ids, "padding_mask" : padding_mask, "labels" : labels}

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        return (
            f"SequenceSamplingDataset(model_max_positions={self.max_length - 1}, "
            f"seq_start_pos={self.seq_start_pos}, "
            f"len={len(self)})"
        )
