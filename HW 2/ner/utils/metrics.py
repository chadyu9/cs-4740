from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch import nn

from ner.utils.utils import get_named_entity_spans


def compute_loss(
    loss_fn: Callable, preds: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    # preds: (batch_size, max_length, output_dim)
    # labels: (batch_size, max_length)
    assert len(preds.shape) >= 2 and len(labels.shape) >= 1
    return loss_fn(preds.view(-1, preds.shape[-1]), labels.view(-1))


def compute_entity_f1(
    y_true: Union[np.ndarray, Dict[str, List[Tuple[int]]]],
    y_pred: Union[np.ndarray, Dict[str, List[Tuple[int]]]],
    average: str = "weighted",
    token_idxs: Optional[Union[List, np.ndarray]] = None,
) -> float:
    if average not in ["weighted", "macro"]:
        raise ValueError(f"average: {average} is not supported in compute_entity_f1")

    y_true_named_ent_spans_dict = y_true
    y_pred_named_ent_spans_dict = y_pred
    if isinstance(y_true, np.ndarray):
        y_true_named_ent_spans_dict = get_named_entity_spans(
            encoded_ner_ids=y_true.squeeze(), token_idxs=token_idxs
        )
    if isinstance(y_pred, np.ndarray):
        y_pred_named_ent_spans_dict = get_named_entity_spans(
            encoded_ner_ids=y_pred.squeeze(), token_idxs=token_idxs
        )

    ent_wise_f1, support = [], []
    for ent_label in y_true_named_ent_spans_dict.keys():
        num_true, num_correct = 0, 0
        pred_spans, true_spans = (
            y_pred_named_ent_spans_dict[ent_label],
            y_true_named_ent_spans_dict[ent_label],
        )
        for true_span in true_spans:
            num_true = num_true + 1
            if true_span in pred_spans:
                num_correct = num_correct + 1
        num_pred = len(pred_spans)

        f1 = 0
        if num_true != 0:
            if num_pred != 0 and num_correct != 0:
                precision = num_correct / num_pred
                recall = num_correct / num_true
                f1 = (2 * precision * recall) / (precision + recall)
        else:
            continue
        ent_wise_f1.append(f1)
        support.append(num_true)

    if average == "weighted":
        ent_wise_f1 = [f1 * weight for f1, weight in zip(ent_wise_f1, support)]
    return (
        sum(ent_wise_f1) / (len(ent_wise_f1) + 1e-17)
        if average == "macro"
        else sum(ent_wise_f1) / (sum(support) + 1e-17)
    )


def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    labels_ignore_idx: Optional[int] = None,
    other_ner_tag_idx: Optional[int] = None,
    average: str = "weighted",
):
    # preds: (batch_size, max_length, output_dim)
    # labels, padding_mask: (batch_size, max_length)
    assert len(preds.shape) >= 2 and len(labels.shape) >= 1
    assert (
        labels_ignore_idx is not None or padding_mask is not None
    ), "labels_ignore_idx or padding_mask must be given"

    preds = preds.view(-1, preds.shape[-1]).argmax(dim=-1)
    labels = labels.view(-1)

    if padding_mask is not None and (
        padding_mask.dtype == torch.long or padding_mask.dtype == torch.int
    ):
        padding_mask = torch.BoolTensor(padding_mask == 1)
    mask = (
        (~padding_mask.view(-1))
        if padding_mask is not None
        else (labels != labels_ignore_idx)
    )
    y_true, y_pred = labels[mask].cpu().numpy(), preds[mask].cpu().numpy()
    entity_f1 = compute_entity_f1(
        y_true=y_true, y_pred=y_pred, average=average
    )  # don't apply "other" tag mask

    if other_ner_tag_idx is not None:
        mask = mask & (labels != other_ner_tag_idx)
    y_true, y_pred = labels[mask].cpu().numpy(), preds[mask].cpu().numpy()
    metrics = {
        "entity_f1": entity_f1,
        "precision": precision_score(
            y_true=y_true, y_pred=y_pred, zero_division=0, average=average
        ),
        "recall": recall_score(
            y_true=y_true, y_pred=y_pred, zero_division=0, average=average
        ),
        "f1": f1_score(y_true=y_true, y_pred=y_pred, zero_division=0, average=average),
        "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
    }
    return metrics


if __name__ == "__main__":

    @dataclass
    class TestConfig:
        loss_fn = nn.CrossEntropyLoss()

        batch_size = 6
        max_length = 10
        output_dim = 5

        pad_ner_tag_idx = 9
        other_ner_tag_idx = 4

        average = "weighted"

    test_config = TestConfig()
    test_preds = torch.randn(
        (test_config.batch_size, test_config.max_length, test_config.output_dim)
    )
    test_labels = torch.randint(
        0, test_config.output_dim, (test_config.batch_size, test_config.max_length)
    )
    test_padding_mask = torch.where(
        torch.BoolTensor(test_labels == test_config.pad_ner_tag_idx), 1, 0
    )

    test_loss = compute_loss(test_config.loss_fn, preds=test_preds, labels=test_labels)
    assert torch.isclose(
        test_loss,
        F.cross_entropy(
            input=test_preds.view(-1, test_config.output_dim),
            target=test_labels.view(-1),
        ),
    )

    test_metrics_with_padding_mask = compute_metrics(
        preds=test_preds,
        labels=test_labels,
        padding_mask=test_padding_mask,
        other_ner_tag_idx=test_config.other_ner_tag_idx,
        average=test_config.average,
    )
    test_metrics_with_ignore_idx = compute_metrics(
        preds=test_preds,
        labels=test_labels,
        labels_ignore_idx=test_config.pad_ner_tag_idx,
        other_ner_tag_idx=test_config.other_ner_tag_idx,
        average=test_config.average,
    )
    assert np.allclose(
        list(test_metrics_with_padding_mask.values()),
        list(test_metrics_with_ignore_idx.values()),
    )
