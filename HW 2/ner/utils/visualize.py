from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple, Union
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import nn

from ner.data_processing.constants import NER_DECODING_MAP, NER_ENCODING_MAP, PAD_TOKEN, UNK_TOKEN
from ner.data_processing.tokenizer import Tokenizer
from ner.models.ner_predictor import NERPredictor
from ner.nn.module import Module
from ner.utils.utils import colored


def _print_convention():
    print(
        f"{colored('Convention', attrs=['underline'])}: [when labels are provided,] "
        f"{colored('dark green', color='dark green', attrs=['bold'])} text indicates that the true label and "
        f"predicted \nlabel match exactly, including the BIO tagging (e.g., pred = 'B-PER', true = 'B-PER'); "
        f"{colored('green', color='green')} text \nindicates an entity match between true and predicted labels but not "
        f"the BIO tagging (e.g., pred = \n'B-PER', true = 'I-PER'); {colored('red', color='red')} text indicates that "
        f"predicted and true labels mismatch (e.g., pred = \n'B-PER', true = 'I-LOC').\n"
    )


def _get_pred_tags_and_unk_tokens_from_text(
    tokenizer: Tokenizer,
    model: Module,
    text: List,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[str], np.ndarray]:
    input_ids = tokenizer.tokenize(input_seq=text, max_length=None)["input_ids"].unsqueeze(0)
    preds = model(input_ids.to(device)).squeeze().argmax(-1).tolist()
    pred_tags = [NER_DECODING_MAP[_] for _ in preds]
    unk_tokens = np.array(text)[np.where(input_ids[0] == tokenizer.token2id[tokenizer.unk_token], True, False)]
    return pred_tags, unk_tokens


def visualize_activations(
    tokenizer: Tokenizer,
    model: Module,
    module: Union[Module, nn.Module],
    text: List,
    prev_layer_module: Optional[Union[Module, nn.Module]] = None,
    labels: Optional[List] = None,
    nonlinearity: Optional[Callable] = None,
    device: torch.device = torch.device("cpu"),
    cbar: Optional[bool] = True,
    figsize: Tuple[int, int] = None,
    fontsize: int = 8,
):
    if nonlinearity is None:
        nonlinearity = lambda _outputs: _outputs
    _nonlinearity = nonlinearity if prev_layer_module is None else lambda _outputs: _outputs

    model.hooks["outputs"] = {"current_layer": [], "previous_layer": []}

    def get_activations_hook(layer_type, _module, _inputs, _outputs):
        model.hooks["outputs"][f"{layer_type}_layer"].append(_nonlinearity(_outputs).detach().cpu().squeeze())

    model.attach_hook(module=module, hook=partial(get_activations_hook, "current"), hook_type="forward")
    if prev_layer_module is not None:
        model.attach_hook(module=prev_layer_module, hook=partial(get_activations_hook, "previous"), hook_type="forward")

    pred_tags, unk_tokens = _get_pred_tags_and_unk_tokens_from_text(
        tokenizer=tokenizer, model=model, text=text, device=device
    )
    activations = torch.vstack(model.hooks["outputs"]["current_layer"]).squeeze()[-len(text) :]
    if prev_layer_module is not None:
        prev_layer_activations = torch.vstack(model.hooks["outputs"]["previous_layer"]).squeeze()[-len(text) :]
        activations = nonlinearity(activations + prev_layer_activations)
    activations = activations.numpy()
    model.detach_all_hooks()

    _print_convention()

    fig, ax1 = plt.subplots(1, figsize=(figsize if figsize is not None else (10, int(4.5 * (len(text) / 20)))))
    ax2 = ax1.twinx()
    plt_token_labels = [token if token not in unk_tokens else f"{token}:{UNK_TOKEN}" for token in text]
    sns.set(font_scale=0.6)
    sns.heatmap(
        activations,
        ax=ax1,
        yticklabels=plt_token_labels,
        cbar=cbar,
        cbar_kws=dict(use_gridspec=False, shrink=0.5, location="right", pad=0.1),
    )

    sns.heatmap(activations, ax=ax2, yticklabels=pred_tags, cbar=False)
    ax1.set_xlabel(f"output: {module}", fontsize=fontsize)
    ax1.set_ylabel("tokens", fontsize=fontsize)
    ax2.set_ylabel("predictions", fontsize=fontsize)
    ax1.set_yticklabels(plt_token_labels, fontsize=fontsize, rotation=0)
    ax2.set_yticklabels(pred_tags, fontsize=fontsize, rotation=0)
    if labels is not None:
        for ytick, pred_tag, label in zip(ax2.get_yticklabels(), pred_tags, labels):
            if pred_tag == label:
                if pred_tag != "O":
                    ytick.set_color("green")
                    ytick.set_fontweight("bold")
            elif pred_tag.split("-")[-1] == label.split("-")[-1]:
                ytick.set_color("green")
            elif pred_tag != label:
                ytick.set_color("red")
    ax1.tick_params(axis="x", labelsize=fontsize)
    plt.show()


def inspect_preds(
    tokenizer: Tokenizer,
    model: Module,
    text: List,
    labels: Optional[List] = None,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    model = model.to(device)

    pred_tags, unk_tokens = _get_pred_tags_and_unk_tokens_from_text(
        tokenizer=tokenizer, model=model, text=text, device=device
    )

    idx_col_max_len = max([len(str(idx)) for idx in range(len(text))] + [len("idx")])
    is_unk_col_man_len = len("is-unk?")
    token_col_max_len = max([len(token) for token in text] + [len("token")])
    tag_col_max_len = max([len(tag) for tag in pred_tags] + [len("pred")])

    _print_convention()
    print(
        f"{' '.ljust(idx_col_max_len)} "
        f"{'token'.ljust(token_col_max_len)}  "
        f"{'is-unk?'.ljust(is_unk_col_man_len)}  "
        f"{'pred'.ljust(tag_col_max_len)}" + (f"  {'true'.ljust(tag_col_max_len)}" if labels is not None else "")
    )
    print(
        f"{'-' * idx_col_max_len} "
        f"{'-' * token_col_max_len}  "
        f"{'-' * is_unk_col_man_len}  "
        f"{'-' * tag_col_max_len}" + (f"  {'-' * tag_col_max_len}" if labels is not None else "")
    )
    for idx, token in enumerate(text):
        is_unk = "✓" if token in unk_tokens else " "
        color, attrs = None, []
        if labels is not None:
            if pred_tags[idx] == labels[idx]:
                if labels[idx] != "O":
                    color = "dark green"
                    attrs = ["bold"]
            elif pred_tags[idx].split("-")[-1] == labels[idx].split("-")[-1]:
                color = "green"
            elif pred_tags[idx] != labels[idx]:
                color = "red"

        print(
            f"{str(idx).ljust(idx_col_max_len)} "
            f"{token.ljust(token_col_max_len)}  "
            f"{is_unk.ljust(is_unk_col_man_len)}  "
            f"{colored(pred_tags[idx].ljust(tag_col_max_len), color=color, attrs=attrs)}"
            + (f"  {labels[idx].ljust(tag_col_max_len)}" if labels is not None else "")
        )

    return pred_tags


if __name__ == "__main__":

    @dataclass
    class TestConfig:
        batch_size = 3

        vocab_size = 10
        padding_idx = 9

        embedding_dim = 5
        hidden_dim = 3
        output_dim = len(NER_ENCODING_MAP) - 1
        num_layers = 2

    test_config = TestConfig()

    sample_text = ["My", "name", "is", "Tushaar", "Gangavarapu"]
    sample_labels = ["O", "O", "O", "B-PER", "I-PER"]
    test_tokenizer = Tokenizer()
    test_tokenizer.from_dict(
        {PAD_TOKEN: 0, UNK_TOKEN: 1, "fine": 2, "oh": 3, "Hello": 4, "My": 5, "name": 6, "is": 7, "I": 8, "am": 9}
    )
    test_model = NERPredictor(
        model="rnn",
        vocab_size=test_config.vocab_size,
        embedding_dim=test_config.embedding_dim,
        padding_idx=test_config.padding_idx,
        hidden_dim=test_config.hidden_dim,
        output_dim=test_config.output_dim,
        num_layers=test_config.num_layers,
    )
    test_model.print_params()

    test_preds = inspect_preds(tokenizer=test_tokenizer, model=test_model, text=sample_text, labels=sample_labels)
    visualize_activations(
        tokenizer=test_tokenizer,
        model=test_model,
        module=test_model.model.hidden_hidden[1],
        prev_layer_module=test_model.model.input_hidden[1],
        text=sample_text,
        labels=sample_labels,
        nonlinearity=F.tanh,
    )
