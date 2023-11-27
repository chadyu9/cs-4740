import os
from argparse import ArgumentParser

import datasets
import yaml

from seagull.data_processing.bbpe import BBPETokenizer
from seagull.data_processing.constants import SCENE_TOKEN, UNCANNY_TOKEN, CAPTION_TOKEN, END_OF_CAPTION_TOKEN
from seagull.utils.torch_utils import set_seed


def main(config_path: str, basepath_to_processed_dataset: str, basepath_to_store_tokenizer: str) -> None:
    set_seed(4740)

    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    processed_train_dataset = datasets.load_from_disk(basepath_to_processed_dataset)["train"]

    special_tokens = [SCENE_TOKEN, UNCANNY_TOKEN, CAPTION_TOKEN, END_OF_CAPTION_TOKEN]
    additional_special_tokens = config["init"].pop("special_tokens")
    if additional_special_tokens is not None:
        special_tokens = list(set(special_tokens + additional_special_tokens))
    bbpe_tokenizer = BBPETokenizer(special_tokens=special_tokens, **config["init"])
    bbpe_tokenizer.train(training_data=processed_train_dataset, **config["train"])
    bbpe_tokenizer.save(tokenizer_path=basepath_to_store_tokenizer)


def argparser():
    parser = ArgumentParser(description="Train a byte-level byte-pair encoding tokenizer using the training dataset.")
    parser.add_argument("--config-path", type=str, help="Path to the config file.", required=True)
    parser.add_argument(
        "--basepath-to-processed-dataset",
        type=str,
        help="Path to the processed newyorker arrow dataset with `text` field.",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument(
        "--basepath-to-store-tokenizer",
        type=str,
        help="Basepath to store the tokenizer (stores tokenizer.json and state_dict.json files).",
        required=True,
    )
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(
        config_path=args.config_path,
        basepath_to_processed_dataset=args.basepath_to_processed_dataset,
        basepath_to_store_tokenizer=args.basepath_to_store_tokenizer,
    )
