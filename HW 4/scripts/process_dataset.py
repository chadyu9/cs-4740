import os
from argparse import ArgumentParser

import datasets

from seagull.data_processing.constants import SCENE_TOKEN, UNCANNY_TOKEN, CAPTION_TOKEN, END_OF_CAPTION_TOKEN
from seagull.data_processing.utils import generate_newyorker_lm_text_dataset
from seagull.utils.torch_utils import set_seed


def main(basepath_to_newyorker_dataset: str, path_to_store_processed_dataset: str, batch_size: int = 4000) -> None:
    set_seed(4740)

    newyorker_dataset = datasets.load_from_disk(basepath_to_newyorker_dataset)

    scene_colname_and_special_token = ("scene", SCENE_TOKEN)
    uncanny_colname_and_special_token = ("uncanny", UNCANNY_TOKEN)
    caption_colname_and_special_token = ("caption", CAPTION_TOKEN)

    processed_dataset = generate_newyorker_lm_text_dataset(
        newyorker_dataset=newyorker_dataset,
        scene_colname_and_special_token=scene_colname_and_special_token,
        uncanny_colname_and_special_token=uncanny_colname_and_special_token,
        caption_colname_and_special_token=caption_colname_and_special_token,
        end_of_caption_special_token=END_OF_CAPTION_TOKEN,
        batch_size=batch_size,
        remove_cols=["scene", "uncanny"],
    )
    processed_dataset.save_to_disk(path_to_store_processed_dataset)


def argparser():
    parser = ArgumentParser(description="Process newyorker dataset to merge scene, uncanny description, and captions.")
    parser.add_argument(
        "--basepath-to-newyorker-dataset",
        type=str,
        help="Path to newyorker arrow dataset with train, val, and test splits.",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument(
        "--path-to-store-processed-dataset",
        type=str,
        help="Path to store the processed arrow dataset.",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, help="Batch size used in dataset mapping.", default=4000)
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(
        basepath_to_newyorker_dataset=args.basepath_to_newyorker_dataset,
        path_to_store_processed_dataset=args.path_to_store_processed_dataset,
        batch_size=args.batch_size,
    )
