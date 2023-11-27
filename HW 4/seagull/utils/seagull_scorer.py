from typing import List, Union, Dict, Literal

import datasets
import torch
from torch import nn
from torch.utils.data import DataLoader

from seagull.data_processing.bbpe import BBPETokenizer
from seagull.data_processing.constants import SCENE_TOKEN, UNCANNY_TOKEN, CAPTION_TOKEN, END_OF_CAPTION_TOKEN
from seagull.data_processing.sequence_sampler import SequenceSamplingDataset
from seagull.nn.modules.module import Module
from seagull.utils.metrics import compute_loss


class SeagullScorer(object):
    def __init__(
        self,
        tokenizer: BBPETokenizer,
        reference_seagull: Module,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 8,
    ):
        """
        Compute deviation in probability of reference model generating ground truth vs. generating the candidate.
        The scorer measures model alignment between the reference model and the candidate model.
        """
        self.device = device
        self._batch_size = batch_size

        self.tokenizer = tokenizer
        self.reference_model = reference_seagull.to(self.device)
        self.reference_model.eval()
        self.model_max_positions = reference_seagull._max_positions

        self._labels_ignore_idx = tokenizer._pretrained_tokenizer.pad_token_id
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self._labels_ignore_idx, reduction="sum")
        self.eps = 1e-8

    @staticmethod
    def _make_instance(scene: str, uncanny: str, captions: List[str]) -> List[str]:
        scene, uncanny = scene.strip(), uncanny.strip()
        return [
            f"{SCENE_TOKEN} {scene} {UNCANNY_TOKEN} {uncanny} {CAPTION_TOKEN} {caption.strip()} {END_OF_CAPTION_TOKEN}"
            for caption in captions
        ]

    def _get_data_batch(self, references_or_candidates: List[str]) -> Dict[str, torch.Tensor]:
        data = SequenceSamplingDataset(
            datasets.Dataset.from_dict(
                self.tokenizer.tokenize(text=references_or_candidates, max_length=(self.model_max_positions + 1))
            ),
            model_max_positions=self.model_max_positions,
            seq_start_pos=-1,
        )
        return next(iter(DataLoader(dataset=data, batch_size=self._batch_size, shuffle=False)))

    def _mask_scene_and_uncanny_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        # All labels have the same scene and uncanny description; use the caption index from the first label.
        caption_idx = torch.where(labels == self.tokenizer.token2id(CAPTION_TOKEN))[1][0]
        labels[:, torch.arange(caption_idx + 1)] = self._labels_ignore_idx
        return labels

    def _compute_nll(self, batch: Dict[str, torch.Tensor], reduction: Literal["mean", "min", "max"]) -> torch.Tensor:
        logits, _ = self.reference_model(
            input_ids=batch["input_ids"].to(self.device),
            padding_mask=batch["padding_mask"].to(self.device),
            use_kv_cache=False,
            return_output_at_all_layers=False,
            return_attentions=False,
        )
        labels = self._mask_scene_and_uncanny_from_labels(batch["labels"]).to(self.device)
        if reduction == "mean":
            nll = compute_loss(loss_fn=self.cross_entropy_loss, preds=logits, labels=labels).item()
            return nll / batch["input_ids"].shape[0]  # mean NLL of generating the caption
        elif reduction in ["min", "max"]:
            # split_logits, split_labels: batch_size x (1, max_length, vocab_size)
            split_logits, split_labels = logits.split(split_size=1, dim=0), labels.split(split_size=1, dim=0)
            batch_nlls = [
                compute_loss(loss_fn=self.cross_entropy_loss, preds=_logits, labels=_labels).item()
                for _logits, _labels in zip(split_logits, split_labels)
            ]
            return min(batch_nlls) if reduction == "min" else max(batch_nlls)
        else:
            raise ValueError(f"{reduction=} not defined")

    @torch.no_grad()
    def score(
        self,
        scene: str,
        uncanny: str,
        reference_captions: Union[str, List[str]],
        candidate_captions: Union[str, List[str]],
    ):
        reference_captions = [reference_captions] if isinstance(reference_captions, str) else reference_captions
        candidate_captions = (
            [candidate_captions] if isinstance(candidate_captions, str) else candidate_captions[: self._batch_size]
        )
        if len(reference_captions) > self._batch_size:
            raise ValueError(f"for computational purposes, total references must be below {self._batch_size}")

        references_batch = self._get_data_batch(self._make_instance(scene, uncanny, captions=reference_captions))
        candidates_batch = self._get_data_batch(self._make_instance(scene, uncanny, captions=candidate_captions))

        reference_nll = self._compute_nll(batch=references_batch, reduction="mean")
        candidate_nll = self._compute_nll(batch=candidates_batch, reduction="min")  # min since nll, not proba
        return candidate_nll / (reference_nll + self.eps)  # must be low for high model alignment


if __name__ == "__main__":
    from time import process_time

    import yaml

    from seagull.model.heads.seagull_lm import SeagullLM

    pretrained_tokenizer = BBPETokenizer()
    pretrained_tokenizer.from_file("../../pretrained_artefacts/tokenizer")

    with open("../../scripts/configs/train_model.yml", "r") as fp:
        model_config = yaml.safe_load(fp)
    model = SeagullLM(
        vocab_size=pretrained_tokenizer.vocab_size,
        padding_idx=pretrained_tokenizer._pretrained_tokenizer.pad_token_id,
        **model_config["model"],
    )
    pretrained_checkpoint_path = "../../pretrained_artefacts/checkpoints/newyorker/checkpoint_0.ckpt"
    model.from_pretrained_or_checkpoint(pretrained_checkpoint_path)

    seagull_scorer = SeagullScorer(tokenizer=pretrained_tokenizer, reference_seagull=model, device=torch.device("cpu"))
    start_time = process_time()
    print(
        seagull_scorer.score(
            scene="An alligator is coming out of the floor. Two people stare at it. A waiter points at it.",
            uncanny="There is an alligator coming out of the floor.",
            reference_captions=[
                "No, you can't have another customer. You haven't finished the last one.",
                "The waiter's clearly giving a wildlife presentation, not taking orders!",
                "When the floor says 'Surprise!' and sends an alligator instead of a welcome mat!",
            ],
            candidate_captions=[
                "Hello? I'm your wife.",
                "Yes, this is the same guy who got out of a sword",
                "It's really not a hand. It's a tail.",
                "We have to get the cash register open.",
                "He said he'd be happy to help us if they had a problem.",
                "I'm not gonna wait until I see the list of people who have been fired.",
                "But you'll have to wait for the last one.",
            ],
        )
    )
    print(f"scoring time taken: {process_time() - start_time} secs")
