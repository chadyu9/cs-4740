import json
import os
import shutil
from argparse import ArgumentParser
from typing import Dict, Optional

import datasets
import torch
import yaml
from rich.progress import track

from seagull.data_processing.bbpe import BBPETokenizer
from seagull.data_processing.constants import END_OF_CAPTION_TOKEN
from seagull.model.heads.seagull_lm import SeagullLM
from seagull.utils.torch_utils import set_pytorch_backends, set_seed, get_device
from seagull.utils.utils import colored

set_pytorch_backends()


@torch.no_grad()
def _run_model_and_write_preds(
    config_path: str,
    tokenizer_basepath: str,
    pretrained_checkpoint_or_model_filepath: str,
    basepath_to_tokenized_dataset: str,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    top_k: Optional[int],
    model_outputs_path: str,
    is_leaderboard_submission: bool,
) -> None:
    assert max_new_tokens <= 100, f"we don't allow for max_new_tokens > 100; provided {max_new_tokens=}"
    assert num_samples <= 8, f"we don't allow for num_samples > 8; provided {num_samples=}"

    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    device = get_device(config["general"]["device"])
    config["generate"] = {
        "max_new_tokens": max_new_tokens,
        "num_samples": num_samples,
        "temperature": temperature,
        "top_k": top_k,
    }

    bbpe_tokenizer = BBPETokenizer()
    bbpe_tokenizer.from_file(tokenizer_basepath)

    model = SeagullLM(
        vocab_size=bbpe_tokenizer.vocab_size,
        padding_idx=bbpe_tokenizer._pretrained_tokenizer.pad_token_id,
        **config["model"],
    )
    model.to(device=device)
    if pretrained_checkpoint_or_model_filepath is not None:
        model.from_pretrained_or_checkpoint(pretrained_checkpoint_or_model_filepath)
    model.eval()

    generations_dict = {}
    test_data = datasets.load_from_disk(basepath_to_tokenized_dataset)["test"]
    for data_inst in track(test_data, description=f"test"):
        attention_mask = torch.BoolTensor(data_inst["attention_mask"])
        input_ids = torch.tensor(data_inst["input_ids"], dtype=torch.long)[attention_mask][:][:-1]  # remove eos token
        input_seq_len = len(input_ids)
        generated_samples = model.talk(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_samples=num_samples,
            top_k=top_k,
            eos_token_id=bbpe_tokenizer.token2id(END_OF_CAPTION_TOKEN),
        )
        generations_dict[data_inst["id"]] = [sample[input_seq_len:] for sample in generated_samples]
    with open(os.path.join(model_outputs_path, "test_generations.json"), "w") as fp:
        json.dump(generations_dict, fp, indent=2)

    if not is_leaderboard_submission:
        with open(os.path.join(model_outputs_path, "config.json"), "w") as fp:
            json.dump(config, fp, indent=2)
        model.save_pretrained(os.path.join(model_outputs_path, "model.pt"))


def _append_net_ids_to_file(filename: str, net_ids: str) -> None:
    with open(filename, "r+") as fp:
        content = fp.read().splitlines(True)
        fp.seek(0, 0)

        start_line = "# AUTO-GENERATED (DO NOT MODIFY)\n"
        net_ids = f"# NET IDS: {net_ids.upper()}\n\n"

        if content[0] == start_line:
            content = content[3:]
        fp.writelines([start_line, net_ids] + content)


def _write_seagull_files(net_ids: Optional[str], seagull_outputs_path: str, is_milestone_submission: bool) -> None:
    seagull_basepath = "seagull"
    files_to_copy = [
        os.path.join(seagull_basepath, "data_processing/utils.py"),
        os.path.join(seagull_basepath, "data_processing/sequence_sampler.py"),
        os.path.join(seagull_basepath, "model/components/embedding.py"),
        os.path.join(seagull_basepath, "nn/transformer/mha.py"),
    ]
    if not is_milestone_submission:
        files_to_copy = files_to_copy + [
            os.path.join(seagull_basepath, "nn/transformer/ffn.py"),
            os.path.join(seagull_basepath, "model/components/transformer_layer.py"),
            os.path.join(seagull_basepath, "model/heads/seagull_lm.py"),
        ]
    for file in files_to_copy:
        shutil.copy2(file, seagull_outputs_path)
        if net_ids is not None:
            _append_net_ids_to_file(
                filename=os.path.join(seagull_outputs_path, os.path.basename(file)), net_ids=net_ids
            )


def _make_output_dirs(
    basepath_to_store_submission: str, is_leaderboard_submission: bool, is_milestone_submission: bool
) -> Dict[str, str]:
    seagull_outputs_path = os.path.join(basepath_to_store_submission, "seagull")
    model_outputs_path = os.path.join(basepath_to_store_submission, "model")

    os.makedirs(basepath_to_store_submission, exist_ok=True)
    if is_milestone_submission:
        os.makedirs(seagull_outputs_path, exist_ok=True)
    if is_leaderboard_submission:
        os.makedirs(model_outputs_path, exist_ok=True)
    if not is_milestone_submission and not is_leaderboard_submission:
        os.makedirs(seagull_outputs_path, exist_ok=True)
        os.makedirs(model_outputs_path, exist_ok=True)

    return {
        "basepath_to_store_submission": basepath_to_store_submission,
        "seagull_outputs_path": seagull_outputs_path,
        "model_outputs_path": model_outputs_path,
    }


def _delete_if_exists(filename: str) -> None:
    try:
        os.remove(filename)
    except OSError:
        pass


def main(
    basepath_to_store_submission: str,
    basepath_to_tokenized_dataset: Optional[str] = None,
    tokenizer_basepath: Optional[str] = None,
    config_path: Optional[str] = None,
    pretrained_checkpoint_or_model_filepath: Optional[str] = None,
    max_new_tokens: int = 30,
    num_samples: int = 3,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    is_milestone_submission: bool = False,
    is_leaderboard_submission: bool = False,
    net_ids: Optional[str] = None,
) -> None:
    set_seed(4740)
    if basepath_to_store_submission.endswith("/"):
        basepath_to_store_submission = basepath_to_store_submission[:-1]

    if is_leaderboard_submission:
        basepath_to_store_submission = os.path.join(basepath_to_store_submission, "leaderboard_submission")
    elif is_milestone_submission:
        basepath_to_store_submission = os.path.join(basepath_to_store_submission, "milestone_submission")
    else:
        basepath_to_store_submission = os.path.join(basepath_to_store_submission, "hw4_submission")
    _delete_if_exists(f"{basepath_to_store_submission}.zip")
    all_output_paths = _make_output_dirs(
        basepath_to_store_submission,
        is_leaderboard_submission=is_leaderboard_submission,
        is_milestone_submission=is_milestone_submission,
    )
    if not is_leaderboard_submission:
        if net_ids is None:
            raise ValueError("must include '--net-ids' as a comma-separated string (e.g., '<net-id-1>,<net-id-2>')")
        _write_seagull_files(
            seagull_outputs_path=all_output_paths["seagull_outputs_path"],
            is_milestone_submission=is_milestone_submission,
            net_ids=net_ids,
        )

    if not is_milestone_submission:
        _run_model_and_write_preds(
            config_path=config_path,
            tokenizer_basepath=tokenizer_basepath,
            pretrained_checkpoint_or_model_filepath=pretrained_checkpoint_or_model_filepath,
            basepath_to_tokenized_dataset=basepath_to_tokenized_dataset,
            model_outputs_path=all_output_paths["model_outputs_path"],
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            is_leaderboard_submission=is_leaderboard_submission,
        )
    if is_leaderboard_submission or is_milestone_submission:
        shutil.make_archive(basepath_to_store_submission, "zip", basepath_to_store_submission)
        shutil.rmtree(basepath_to_store_submission)
        print(f"submission stored at: {basepath_to_store_submission}.zip")
    else:
        # For final submission, create two .zip files:
        code_path = os.path.join(basepath_to_store_submission, all_output_paths["seagull_outputs_path"])
        shutil.make_archive(code_path, "zip", code_path)
        shutil.rmtree(code_path)
        print(f"CODE submission {colored('(to Gradescope)', 'red')} stored at: {code_path}.zip")

        model_path = os.path.join(basepath_to_store_submission, all_output_paths["model_outputs_path"])
        shutil.make_archive(model_path, "zip", model_path)
        shutil.rmtree(model_path)
        print(f"MODEL submission {colored('(to CMS)', 'red')} stored at: {model_path}.zip")


def argparser():
    parser = ArgumentParser(description="Make a submission folder for the assignment.")
    parser.add_argument(
        "--basepath-to-tokenized-dataset",
        type=str,
        help="Path to the tokenized newyorker dataset with `input_ids` and `padding_mask`.",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--tokenizer-basepath",
        type=str,
        help="Basepath to the trained tokenizer (path must include tokenizer.json and state_dict.json files).",
    )
    parser.add_argument(
        "--basepath-to-store-submission",
        type=str,
        help="The basepath to store all the files required to make a gradescope submission.",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument("--config-path", type=str, help="Path to the finetuned model config file.", required=False)
    parser.add_argument(
        "--pretrained-checkpoint-or-model-filepath",
        type=str,
        help="Path to pretrained checkpoint or model file.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, help="Maximum number of new tokens to generate (must be <= 100).", default=30
    )
    parser.add_argument("--num-samples", type=int, help="Number of samples to generate (must be <= 8).", default=3)
    parser.add_argument("--temperature", type=float, help="The generation temperature (higher = diverse).", default=0.7)
    parser.add_argument(
        "--top-k",
        type=int,
        help="If specified, only the top-k candidates will be used in generating samples.",
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
        help="Student net-IDs as a comma-separated string (e.g., '<net-id-1>,<net-id-2>').",
        required=False,
    )
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(
        basepath_to_tokenized_dataset=args.basepath_to_tokenized_dataset,
        tokenizer_basepath=args.tokenizer_basepath,
        basepath_to_store_submission=args.basepath_to_store_submission,
        config_path=args.config_path,
        pretrained_checkpoint_or_model_filepath=args.pretrained_checkpoint_or_model_filepath,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        is_leaderboard_submission=args.leaderboard_submission,
        is_milestone_submission=args.milestone_submission,
        net_ids=args.net_ids,
    )
