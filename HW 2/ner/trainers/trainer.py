import logging
from dataclasses import dataclass
from itertools import chain
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
from datasets import Dataset
from rich.progress import track
from sklearn.utils import class_weight
from torch import nn
from torch.utils.data import DataLoader

from ner.data_processing.constants import (
    NER_ENCODING_MAP,
    PAD_NER_TAG,
    PAD_TOKEN,
    UNK_TOKEN,
)
from ner.data_processing.data_collator import DataCollator
from ner.data_processing.tokenizer import Tokenizer
from ner.models.ner_predictor import NERPredictor
from ner.nn.module import Module
from ner.utils.metrics import compute_loss, compute_metrics
from ner.utils.tracker import Tracker
from ner.utils.utils import get_named_entity_spans


class Trainer(object):
    def __init__(
        self,
        model: Module,
        optimizer: torch.optim.Optimizer,
        data_collator: DataCollator,
        train_data: Dataset,
        val_data: Optional[Dataset] = None,
        grad_clip_max_norm: Optional[float] = None,
        use_class_weights: bool = False,
        class_weights: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
        tracker: Optional[Tracker] = None,
        device: torch.device = torch.device("cpu"),
        label_colname="NER",
    ) -> None:
        super().__init__()

        self.device = device

        self.model = model.to(device)
        self.optimizer = optimizer

        self.labels_ignore_idx = NER_ENCODING_MAP[PAD_NER_TAG]
        self.other_ner_tag_idx = NER_ENCODING_MAP["O"]

        self.train_data = train_data
        self.val_data = val_data
        self.data_collator = data_collator
        self.grad_clip_max_norm = grad_clip_max_norm
        if use_class_weights:
            class_weights = (
                class_weights
                if class_weights
                else self._compute_class_weights(train_data, label_colname)
            )
            # PyTorch FloatTensor doesn't support device: https://github.com/pytorch/pytorch/issues/20122.
            class_weights = torch.Tensor(class_weights).to(device)
            logging.info(f"class weights: {class_weights}")
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.labels_ignore_idx, weight=class_weights
        )

        self.tracker = tracker
        self._epoch = 0

    @staticmethod
    def _compute_class_weights(train_data, label_colname="NER"):
        train_labels = list(chain(*train_data[label_colname]))
        class_weights = class_weight.compute_class_weight(
            "balanced", classes=np.unique(train_labels), y=train_labels
        )
        return class_weights

    def save_checkpoint(self, checkpoint_path: str) -> None:
        torch.save(
            {
                "epoch": self._epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epoch = checkpoint["epoch"] + 1

    def _train_epoch(self, dataloader) -> Dict[str, float]:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.trainers.trainer.html."""
        # Initialize metrics dictionary
        metrics = {
            "loss": [],
            "precision": [],
            "recall": [],
            "accuracy": [],
            "f1": [],
            "entity_f1": [],
        }

        # Set model to train mode
        self.model.train()

        # Iterate through batches
        for batch in dataloader:
            # Zero out gradients
            self.optimizer.zero_grad()

            # Run forward pass, compute loss, and backpropagate
            input_ids = batch["input_ids"].to(self.device)
            preds = self.model(input_ids)
            loss = compute_loss(self.loss_fn, preds, batch["labels"].to(self.device))
            loss.backward()
            loss = loss.item()

            # Clip gradients if indicated
            if self.grad_clip_max_norm:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            # Update parameters
            self.optimizer.step()

            # Compute and compile the metrics
            batch_metrics = compute_metrics(
                preds=preds,
                labels=batch["labels"],
                padding_mask=batch["padding_mask"],
                other_ner_tag_idx=self.other_ner_tag_idx,
                average="weighted",
            )

            for key in metrics:
                metrics[key].append(batch_metrics[key]) if key != "loss" else metrics[
                    key
                ].append(loss)

        # Return the average metrics
        average_metrics = {
            metric: np.average(score) for metric, score in metrics.items()
        }
        return average_metrics

    @torch.no_grad()
    def _eval_epoch(self, dataloader) -> Dict[str, float]:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.trainers.trainer.html."""
        # Initialize metrics dictionary
        metrics = {
            "loss": [],
            "precision": [],
            "recall": [],
            "accuracy": [],
            "f1": [],
            "entity_f1": [],
        }

        # Set model to eval mode
        self.model.eval()

        # Iterate through batches
        for batch in dataloader:
            # Run forward pass and compute loss
            input_ids = batch["input_ids"].to(self.device)
            preds = self.model(input_ids)
            loss = compute_loss(self.loss_fn, preds, batch["labels"].to(self.device))
            loss = loss.item()

            # Compute and compile the metrics
            batch_metrics = compute_metrics(
                preds=preds,
                labels=batch["labels"],
                padding_mask=batch["padding_mask"],
                other_ner_tag_idx=self.other_ner_tag_idx,
                average="weighted",
            )

            for key in metrics:
                metrics[key].append(batch_metrics[key]) if key != "loss" else metrics[
                    key
                ].append(loss)

        # Return the average metrics
        average_metrics = {
            metric: np.average(score) for metric, score in metrics.items()
        }
        return average_metrics

    def train_and_eval(
        self,
        batch_size: int = 128,
        num_epochs: int = 8,
        checkpoint_every: int = 1,
        num_workers: int = 0,
    ) -> None:
        train_dataloader = DataLoader(
            self.train_data,
            collate_fn=self.data_collator,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        val_dataloader = None
        if self.val_data is not None:
            val_dataloader = DataLoader(
                self.val_data,
                collate_fn=self.data_collator,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
            )

        for epoch in range(num_epochs):
            train_metrics = self._train_epoch(train_dataloader)
            val_metrics = (
                self._eval_epoch(val_dataloader) if val_dataloader is not None else None
            )
            if self.tracker is not None:
                self.tracker.log_metrics(
                    epoch=self._epoch, split="train", metrics=train_metrics
                )
                if val_metrics is not None:
                    self.tracker.log_metrics(
                        epoch=self._epoch, split="val", metrics=val_metrics
                    )
                if (epoch + 1) % checkpoint_every == 0:
                    self.tracker.save_checkpoint(self, epoch=self._epoch)
            self._epoch = self._epoch + 1

        if self.tracker:
            self.tracker.save_model(self.model)

    @staticmethod
    @torch.no_grad()
    def test(
        test_data: Dataset,
        data_collator: DataCollator,
        model: Module,
        batch_size: int = 128,
        num_workers: int = 0,
        index_colname: str = "index",
        device=torch.device("cpu"),
    ) -> Dict[str, List[Tuple[int]]]:
        test_dataloader = DataLoader(
            test_data,
            collate_fn=data_collator,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        token_idxs = list(chain.from_iterable(test_data[index_colname]))
        all_preds = []

        model = model.to(device)
        model.eval()
        for batch in track(test_dataloader, description=f"test"):
            # preds: (batch_size, max_length, output_dim)
            preds = model(input_ids=batch["input_ids"].to(device))

            padding_mask = batch["padding_mask"]
            if padding_mask.dtype == torch.long or padding_mask.dtype == torch.int:
                padding_mask = torch.BoolTensor(padding_mask == 1)
            mask = ~padding_mask.view(-1)

            preds = preds.view(-1, preds.shape[-1]).argmax(dim=-1)
            all_preds.append(preds[mask].cpu().numpy().squeeze())
        preds_dict = get_named_entity_spans(
            encoded_ner_ids=np.concatenate(all_preds), token_idxs=token_idxs
        )
        return preds_dict
