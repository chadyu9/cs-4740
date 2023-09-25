import csv
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional

import datasets
import torch
import yaml

from ner.data_processing.constants import PAD_NER_TAG, PAD_TOKEN, UNK_TOKEN, NER_ENCODING_MAP
from ner.data_processing.data_collator import DataCollator
from ner.data_processing.tokenizer import Tokenizer
from ner.models import NERPredictor
from ner.trainers.trainer import Trainer
from ner.utils.utils import set_seed

csv.field_size_limit(sys.maxsize)


def _write_preds_dict_to_csv(preds_dict: Dict[str, List[Tuple[int]]], preds_filepath: str) -> None:
    with open(preds_filepath, "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=["id", "pred"])
        writer.writeheader()
        for ent in preds_dict:
            pred_str = " ".join([str(start_idx) + "-" + str(end_idx) for start_idx, end_idx in preds_dict[ent]])
            writer.writerow({"id": ent, "pred": pred_str})


def _run_model_and_write_preds(
    config_path: str,
    pretrained_checkpoint_or_model_filepath: str,
    test_data: datasets.Dataset,
    tokenizer: Tokenizer,
    data_collator: DataCollator,
    model_outputs_path: str,
    is_leaderboard_submission: bool,
) -> None:
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    device = torch.device("cpu")
    if config["general"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")

    model = NERPredictor(
        vocab_size=tokenizer.vocab_size,
        padding_idx=tokenizer.token2id[tokenizer.pad_token],
        output_dim=len(NER_ENCODING_MAP) - 1,
        **config["model"],
    )
    model.print_params()

    if pretrained_checkpoint_or_model_filepath is not None:
        if pretrained_checkpoint_or_model_filepath.endswith(".ckpt"):
            checkpoint = torch.load(pretrained_checkpoint_or_model_filepath, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        elif pretrained_checkpoint_or_model_filepath.endswith(".pt"):
            model.from_pretrained(model_filepath=pretrained_checkpoint_or_model_filepath)
        else:
            raise ValueError("model file not supported")
    else:
        raise AssertionError("neither pretrained model/checkpoint paths are provided")

    preds_dict = Trainer.test(
        test_data=test_data,
        data_collator=data_collator,
        model=model,
        batch_size=config["train_and_eval"]["batch_size"],
        num_workers=config["train_and_eval"]["num_workers"],
        device=device,
    )

    if not is_leaderboard_submission:
        with open(os.path.join(model_outputs_path, "config.json"), "w") as fp:
            json.dump(config, fp)
        model.save_pretrained(os.path.join(model_outputs_path, "model.pt"))
    _write_preds_dict_to_csv(preds_dict, preds_filepath=os.path.join(model_outputs_path, "test_preds.csv"))


def _append_net_ids_to_file(filename: str, net_ids: str) -> None:
    with open(filename, "r+") as fp:
        content = fp.read().splitlines(True)
        fp.seek(0, 0)

        start_line = "# AUTO-GENERATED (DO NOT MODIFY)\n"
        net_ids = f"# NET IDS: {net_ids.upper()}\n\n"

        if content[0] == start_line:
            content = content[3:]
        fp.writelines([start_line, net_ids] + content)


def _write_ner_files(net_ids: Optional[str], ner_outputs_path: str, is_milestone_submission: bool) -> None:
    ner_basepath = "ner"
    files_to_copy = [
        os.path.join(ner_basepath, "data_processing/tokenizer.py"),
        os.path.join(ner_basepath, "data_processing/data_collator.py"),
        os.path.join(ner_basepath, "nn/embeddings/embedding.py"),
        os.path.join(ner_basepath, "trainers/trainer.py"),
    ]
    if not is_milestone_submission:
        files_to_copy = files_to_copy + [
            os.path.join(ner_basepath, "nn/models/ffnn.py"),
            os.path.join(ner_basepath, "nn/models/rnn.py"),
        ]
    for file in files_to_copy:
        shutil.copy2(file, ner_outputs_path)
        if net_ids is not None:
            _append_net_ids_to_file(filename=os.path.join(ner_outputs_path, os.path.basename(file)), net_ids=net_ids)


def _make_output_dirs(
    basepath_to_store_submission: str, is_leaderboard_submission: bool, is_milestone_submission: bool
) -> Dict[str, str]:
    ner_outputs_path = os.path.join(basepath_to_store_submission, "ner")
    tokenizer_outputs_path = os.path.join(basepath_to_store_submission, "tokenizer")
    ffnn_outputs_path = os.path.join(basepath_to_store_submission, "ffnn")
    rnn_outputs_path = os.path.join(basepath_to_store_submission, "rnn")

    os.makedirs(basepath_to_store_submission, exist_ok=True)
    if not is_leaderboard_submission:
        os.makedirs(ner_outputs_path, exist_ok=True)
        if not is_milestone_submission:
            os.makedirs(tokenizer_outputs_path, exist_ok=True)
    if not is_milestone_submission:
        os.makedirs(ffnn_outputs_path, exist_ok=True)
        os.makedirs(rnn_outputs_path, exist_ok=True)

    return {
        "basepath_to_store_submission": basepath_to_store_submission,
        "ner_outputs_path": ner_outputs_path,
        "tokenizer_outputs_path": tokenizer_outputs_path,
        "ffnn_outputs_path": ffnn_outputs_path,
        "rnn_outputs_path": rnn_outputs_path,
    }


def _delete_if_exists(filename: str) -> None:
    try:
        os.remove(filename)
    except OSError:
        pass


def main(
    basepath_to_hf_dataset: str,
    tokenizer_filepath: str,
    basepath_to_store_submission: str,
    ffnn_config_path: str = None,
    rnn_config_path: str = None,
    pretrained_ffnn_checkpoint_or_model_filepath: str = None,
    pretrained_rnn_checkpoint_or_model_filepath: str = None,
    is_milestone_submission: bool = False,
    is_leaderboard_submission: bool = False,
    net_ids: str = None,
) -> None:
    set_seed(4740)
    if basepath_to_store_submission.endswith("/"):
        basepath_to_store_submission = basepath_to_store_submission[:-1]

    if is_leaderboard_submission:
        basepath_to_store_submission = os.path.join(basepath_to_store_submission, "leaderboard_submission")
    elif is_milestone_submission:
        basepath_to_store_submission = os.path.join(basepath_to_store_submission, "milestone_submission")
    else:
        basepath_to_store_submission = os.path.join(basepath_to_store_submission, "hw2_submission")
    _delete_if_exists(f"{basepath_to_store_submission}.zip")
    all_output_paths = _make_output_dirs(
        basepath_to_store_submission,
        is_leaderboard_submission=is_leaderboard_submission,
        is_milestone_submission=is_milestone_submission,
    )
    if not is_leaderboard_submission:
        if net_ids is None:
            raise ValueError("must include '--net-ids' as a comma-separated string (e.g., '<net-id-1>, <net-id-2>')")
        _write_ner_files(
            ner_outputs_path=all_output_paths["ner_outputs_path"],
            is_milestone_submission=is_milestone_submission,
            net_ids=net_ids,
        )

    if not is_milestone_submission:
        config_path = ffnn_config_path if ffnn_config_path is not None else rnn_config_path
        assert config_path is not None, "at least one of ffnn or rnn config paths must be provided"
        with open(config_path, "r") as fp:
            config = yaml.safe_load(fp)
        tokenizer = Tokenizer(pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, **config["tokenizer"])
        tokenizer.from_file(tokenizer_filepath)
        data_collator = DataCollator(tokenizer=tokenizer, pad_tag=PAD_NER_TAG, **config["data_collator"])
        if not is_leaderboard_submission:
            tokenizer.save(os.path.join(all_output_paths["tokenizer_outputs_path"], "tokenizer.json"))

        test_data = datasets.load_from_disk(basepath_to_hf_dataset)["test"]
        if pretrained_ffnn_checkpoint_or_model_filepath is not None:
            _run_model_and_write_preds(
                config_path=ffnn_config_path,
                pretrained_checkpoint_or_model_filepath=pretrained_ffnn_checkpoint_or_model_filepath,
                test_data=test_data,
                tokenizer=tokenizer,
                data_collator=data_collator,
                model_outputs_path=all_output_paths["ffnn_outputs_path"],
                is_leaderboard_submission=is_leaderboard_submission,
            )
        if pretrained_rnn_checkpoint_or_model_filepath is not None:
            _run_model_and_write_preds(
                config_path=rnn_config_path,
                pretrained_checkpoint_or_model_filepath=pretrained_rnn_checkpoint_or_model_filepath,
                test_data=test_data,
                tokenizer=tokenizer,
                data_collator=data_collator,
                model_outputs_path=all_output_paths["rnn_outputs_path"],
                is_leaderboard_submission=is_leaderboard_submission,
            )

    shutil.make_archive(basepath_to_store_submission, "zip", basepath_to_store_submission)
    shutil.rmtree(basepath_to_store_submission)
    print(f"submission stored at: {basepath_to_store_submission}.zip\n")


def argparser():
    parser = ArgumentParser(description="Make a submission folder for the assignment.")
    parser.add_argument(
        "--ffnn-config-path",
        type=str,
        help="Path to the ffnn-based NER predictor config file (from the artefacts).",
        default=None,
    )
    parser.add_argument(
        "--rnn-config-path",
        type=str,
        help="Path to the rnn-based NER predictor config file (from the artefacts).",
        default=None,
    )
    parser.add_argument(
        "--basepath-to-hf-dataset",
        type=str,
        help="Path to the huggingface dataset (with train, val, test splits).",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument(
        "--tokenizer-filepath",
        type=str,
        help="Path to the trained tokenizer, include the filename and extension (e.g., /tmp/config.json).",
        required=True,
    )
    parser.add_argument(
        "--basepath-to-store-submission",
        type=str,
        help="The basepath to store all the files required to make a gradescope submission.",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument(
        "--pretrained-ffnn-checkpoint-or-model-filepath",
        type=str,
        help="Path to pretrained ffnn checkpoint or model file.",
        default=None,
    )
    parser.add_argument(
        "--pretrained-rnn-checkpoint-or-model-filepath",
        type=str,
        help="Path to pretrained rnn checkpoint or model file.",
        default=None,
    )
    parser.add_argument(
        "--leaderboard-submission",
        action="store_true",
        help="Flag to indicate if the current submission is for the leaderboard.",
    )
    parser.add_argument(
        "--milestone-submission",
        action="store_true",
        help="Flag to indicate if the current submission is for the milestone.",
    )
    parser.add_argument(
        "--net-ids",
        type=str,
        help="Student net-IDs as a comma-separated string (e.g., '<net-id-1>, <net-id-2>').",
        required=False,
    )
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(
        basepath_to_hf_dataset=args.basepath_to_hf_dataset,
        tokenizer_filepath=args.tokenizer_filepath,
        basepath_to_store_submission=args.basepath_to_store_submission,
        ffnn_config_path=args.ffnn_config_path,
        rnn_config_path=args.rnn_config_path,
        pretrained_ffnn_checkpoint_or_model_filepath=args.pretrained_ffnn_checkpoint_or_model_filepath,
        pretrained_rnn_checkpoint_or_model_filepath=args.pretrained_rnn_checkpoint_or_model_filepath,
        is_leaderboard_submission=args.leaderboard_submission,
        is_milestone_submission=args.milestone_submission,
        net_ids=args.net_ids,
    )
