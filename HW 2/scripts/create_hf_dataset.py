import os
from argparse import ArgumentParser

import datasets

from ner.utils.utils import load_json, set_seed


def main(basepath_to_dataset_json_files: str, path_to_store_hf_dataset: str) -> None:
    set_seed(4740)

    train_hf_dataset = datasets.Dataset.from_dict(load_json(os.path.join(basepath_to_dataset_json_files, "train.json")))
    val_hf_dataset = datasets.Dataset.from_dict(load_json(os.path.join(basepath_to_dataset_json_files, "val.json")))
    test_hf_dataset = datasets.Dataset.from_dict(load_json(os.path.join(basepath_to_dataset_json_files, "test.json")))

    hf_dataset = datasets.DatasetDict(
        {
            "train": train_hf_dataset,
            "val": val_hf_dataset,
            "test": test_hf_dataset,
        }
    )
    hf_dataset.save_to_disk(path_to_store_hf_dataset)


def argparser():
    parser = ArgumentParser(description="Generate an arrow dataset from .json files.")
    parser.add_argument(
        "--basepath-to-dataset-json-files",
        type=str,
        help="Path to individual json files with train.json, val.json, test.json.",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument(
        "--path-to-store-hf-dataset",
        type=str,
        help="Path to store the huggingface dataset.",
        default=os.getcwd(),
        required=True,
    )
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(
        basepath_to_dataset_json_files=args.basepath_to_dataset_json_files,
        path_to_store_hf_dataset=args.path_to_store_hf_dataset,
    )
