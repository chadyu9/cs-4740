import logging
from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Any

import numpy as np
import torch

from ner.data_processing.constants import NER_ENCODING_MAP, PAD_NER_TAG
from ner.data_processing.tokenizer import Tokenizer


class DataCollator(object):
    def __init__(
        self,
        tokenizer: Tokenizer,
        padding: Union[str, bool] = "longest",
        max_length: Optional[int] = None,
        padding_side: str = "right",
        truncation_side: str = "right",
        pad_tag: str = PAD_NER_TAG,
        text_colname: str = "text",
        label_colname: str = "NER",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.pad_tag = pad_tag
        self.text_colname = text_colname
        self.label_colname = label_colname

    def _get_max_length(self, data_instances: List[Dict[str, Any]]) -> Optional[int]:
        if not (
            (self.padding == "longest" or self.padding) and self.max_length is None
        ):
            logging.warning(
                f"both max_length={self.max_length} and padding={self.padding} provided; ignoring "
                f"padding={self.padding} and using max_length={self.max_length}"
            )
            self.padding = "max_length"

        if self.padding == "longest" or (
            isinstance(self.padding, bool) and self.padding
        ):
            return max(
                [
                    len(data_instance[self.text_colname])
                    for data_instance in data_instances
                ]
            )
        elif self.padding == "max_length":
            return self.max_length
        elif isinstance(self.padding, bool) and not self.padding:
            return None
        raise ValueError(f"padding strategy {self.padding} is invalid")

    @staticmethod
    def _process_labels(labels: List) -> torch.Tensor:
        return torch.LongTensor([NER_ENCODING_MAP[label] for label in labels])

    def __call__(self, data_instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.data_processing.data_collator.html.
        """
        # Initialize batch dimensions
        batch_size = len(data_instances)
        batch_max_length = (
            self._get_max_length(data_instances)
            if self._get_max_length(data_instances)
            else len(data_instances[0][self.text_colname])
        )

        # Initialize padded batch
        padded_batch = {}
        input_ids = torch.empty((batch_size, batch_max_length), dtype=torch.long)
        padding_mask = torch.empty((batch_size, batch_max_length), dtype=torch.long)

        # Compile tokenized sequences
        for i in range(len(data_instances)):
            tokenized_seq = self.tokenizer.tokenize(
                data_instances[i][self.text_colname],
                max_length=batch_max_length,
                padding_side=self.padding_side,
                truncation_side=self.truncation_side,
            )
            input_ids[i] = tokenized_seq["input_ids"]
            padding_mask[i] = tokenized_seq["padding_mask"]

        padded_batch["input_ids"] = input_ids
        padded_batch["padding_mask"] = padding_mask

        # If labels are provided, compile padded labels
        if self.label_colname in data_instances[0]:
            labels = torch.empty((batch_size, batch_max_length), dtype=torch.long)
            for i in range(len(data_instances)):
                non_padded_labels = data_instances[i][self.label_colname]
                # Handle truncation
                if self.truncation_side == "left":
                    non_padded_labels = non_padded_labels[
                        max(
                            len(non_padded_labels) - batch_max_length,
                            0,
                        ) :
                    ]
                else:
                    non_padded_labels = non_padded_labels[
                        : -max(
                            len(non_padded_labels) - batch_max_length,
                            0,
                        )
                        or None
                    ]

                # Handle padding
                pad_length = torch.sum(padding_mask[i]).item()
                if self.padding_side == "left":
                    non_padded_labels = [self.pad_tag] * pad_length + non_padded_labels
                else:
                    non_padded_labels += [self.pad_tag] * pad_length

                # Process labels into tensors with associated integer encoding
                labels[i] = self._process_labels(non_padded_labels)

            padded_batch["labels"] = labels

        return padded_batch
