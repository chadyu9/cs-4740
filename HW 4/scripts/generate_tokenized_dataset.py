import os
from argparse import ArgumentParser

import datasets
import yaml

from seagull.data_processing.bbpe import BBPETokenizer
from seagull.utils.torch_utils import set_seed


def main(
    config_path: str,
    basepath_to_processed_dataset: str,
    tokenizer_basepath: str,
    path_to_store_tokenized_dataset: str,
) -> None:
    set_seed(4740)

    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    processed_dataset = datasets.load_from_disk(basepath_to_processed_dataset)
    bbpe_tokenizer = BBPETokenizer()
    bbpe_tokenizer.from_file(tokenizer_basepath)

    max_length = config["tokenize"].pop("max_length", None)
    max_length = max_length + 1 if max_length is not None else None  # autoregressive modeling
    tokenizer_helper = lambda batch: bbpe_tokenizer.tokenize(
        text=batch[config["dataset"]["text_colname"]], max_length=max_length, **config["tokenize"]
    )
    tokenized_dataset = processed_dataset.map(tokenizer_helper, batch_size=4000, batched=True)
    tokenized_dataset.save_to_disk(path_to_store_tokenized_dataset)


def argparser():
    parser = ArgumentParser(description="Tokenize the processed newyorker dataset using the trained BBPE tokenizer.")
    parser.add_argument("--config-path", type=str, help="Path to the config file.", required=True)
    parser.add_argument(
        "--basepath-to-processed-dataset",
        type=str,
        help="Path to the processed newyorker arrow dataset with `text` field.",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument(
        "--tokenizer-basepath",
        type=str,
        help="Basepath to the trained tokenizer (path must include tokenizer.json and state_dict.json files).",
        required=True,
    )
    parser.add_argument(
        "--path-to-store-tokenized-dataset",
        type=str,
        help="Path to store the tokenized arrow dataset.",
        default=os.getcwd(),
        required=True,
    )
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(
        config_path=args.config_path,
        basepath_to_processed_dataset=args.basepath_to_processed_dataset,
        tokenizer_basepath=args.tokenizer_basepath,
        path_to_store_tokenized_dataset=args.path_to_store_tokenized_dataset,
    )
