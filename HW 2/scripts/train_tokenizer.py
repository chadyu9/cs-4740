import os
from argparse import ArgumentParser

import datasets
import yaml

from ner.data_processing.constants import PAD_TOKEN, UNK_TOKEN
from ner.data_processing.tokenizer import Tokenizer
from ner.utils.utils import set_seed


def main(
    config_path: str, basepath_to_hf_dataset: str, filepath_to_store_tokenizer: str, min_freq: int, remove_frac: float
) -> None:
    set_seed(4740)

    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    hf_dataset = datasets.load_from_disk(basepath_to_hf_dataset)

    tokenizer = Tokenizer(pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, **config["init"])
    tokenizer.train(train_data=hf_dataset["train"], min_freq=min_freq, remove_frac=remove_frac, **config["train"])
    tokenizer.save(filepath_to_store_tokenizer)


def argparser():
    parser = ArgumentParser(description="Train a tokenizer from the training dataset.")
    parser.add_argument("--config-path", type=str, help="Path to the config file.", required=True)
    parser.add_argument(
        "--basepath-to-hf-dataset",
        type=str,
        help="Path to the huggingface dataset (with train, val, test splits).",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument(
        "--filepath-to-store-tokenizer",
        type=str,
        help="Path to the store tokenizer, include the filename and extension (e.g., /tmp/config.json).",
        required=True,
    )
    parser.add_argument(
        "--min-freq", type=int, help="The minimum frequency to retain tokens in the vocabulary.", required=True
    )
    parser.add_argument(
        "--remove-frac",
        type=float,
        help="The fraction of low-frequency tokens to be removed from the vocabulary.",
        required=True,
    )
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(
        config_path=args.config_path,
        basepath_to_hf_dataset=args.basepath_to_hf_dataset,
        filepath_to_store_tokenizer=args.filepath_to_store_tokenizer,
        min_freq=args.min_freq,
        remove_frac=args.remove_frac,
    )
