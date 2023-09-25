import copy
import os
from argparse import ArgumentParser

import datasets
import torch
import yaml

from ner.data_processing.constants import PAD_NER_TAG, PAD_TOKEN, UNK_TOKEN, NER_ENCODING_MAP
from ner.data_processing.data_collator import DataCollator
from ner.data_processing.tokenizer import Tokenizer
from ner.models import NERPredictor
from ner.trainers.trainer import Trainer
from ner.utils.tracker import Tracker
from ner.utils.utils import set_seed


def main(
    config_path: str,
    tokenizer_config_path: str,
    basepath_to_hf_dataset: str,
    tokenizer_filepath: str,
    model_type: str,
    num_layers: int,
    batch_size: int,
    num_epochs: int,
    basepath_to_store_results: str,
    experiment_name: str,
    pretrained_checkpoint_or_model_filepath: str = None,
) -> None:
    set_seed(4740)

    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    with open(tokenizer_config_path, "r") as fp:
        tokenizer_config = yaml.safe_load(fp)
    config_to_save = copy.deepcopy(config)
    config_to_save["tokenizer"] = {"lowercase": tokenizer_config["init"]["lowercase"]}
    config_to_save["model"].update({"model": model_type, "num_layers": num_layers})
    config_to_save["train_and_eval"].update({"batch_size": batch_size, "num_epochs": num_epochs})
    tracker = Tracker(
        config=config_to_save, basepath_to_store_results=basepath_to_store_results, experiment_name=experiment_name
    )

    device = torch.device("cpu")
    if config["general"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")

    hf_dataset = datasets.load_from_disk(basepath_to_hf_dataset)

    tokenizer = Tokenizer(pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, **tokenizer_config["init"])
    tokenizer.from_file(tokenizer_filepath)
    data_collator = DataCollator(tokenizer=tokenizer, pad_tag=PAD_NER_TAG, **config["data_collator"])

    model = NERPredictor(
        model=model_type,
        vocab_size=tokenizer.vocab_size,
        padding_idx=tokenizer.token2id[tokenizer.pad_token],
        num_layers=num_layers,
        output_dim=len(NER_ENCODING_MAP) - 1,
        **config["model"],
    )
    model.print_params()
    optimizer = torch.optim.AdamW(params=model.parameters(), **config["optimizer"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_collator=data_collator,
        train_data=hf_dataset["train"],
        val_data=hf_dataset["val"],
        label_colname=config["data_collator"]["label_colname"],
        device=device,
        tracker=tracker,
        **config["trainer"],
    )

    if pretrained_checkpoint_or_model_filepath is not None:
        if pretrained_checkpoint_or_model_filepath.endswith(".ckpt"):
            trainer.from_checkpoint(checkpoint_path=pretrained_checkpoint_or_model_filepath)
        elif pretrained_checkpoint_or_model_filepath.endswith(".pt"):
            model.from_pretrained(model_filepath=pretrained_checkpoint_or_model_filepath)
        else:
            raise ValueError("model file not supported")
    trainer.train_and_eval(batch_size=batch_size, num_epochs=num_epochs, **config["train_and_eval"])


def argparser():
    parser = ArgumentParser(description="Train a neural network for named-entity recognition.")
    parser.add_argument("--config-path", type=str, help="Path to the config file.", required=True)
    parser.add_argument("--tokenizer-config-path", type=str, help="Path to the tokenizer config file.", required=True)
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
    parser.add_argument("--model-type", choices=["ffnn", "rnn"], help="Chooses which model type to use.", required=True)
    parser.add_argument("--num-layers", type=int, help="Number of hidden/stacked layers.", required=True)
    parser.add_argument("--batch-size", type=int, help="Training (and validation) batch size.", required=True)
    parser.add_argument("--num-epochs", type=int, help="Number of training epochs.", required=True)
    parser.add_argument(
        "--basepath-to-store-results",
        type=str,
        help="The basepath to store experimental results.",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument("--experiment-name", type=str, help="Experiment name.", default="ner_experiment", required=True)
    parser.add_argument(
        "--pretrained-checkpoint-or-model-filepath",
        type=str,
        help="Path to pretrained checkpoint or model file.",
        default=None,
    )
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(
        config_path=args.config_path,
        tokenizer_config_path=args.tokenizer_config_path,
        basepath_to_hf_dataset=args.basepath_to_hf_dataset,
        tokenizer_filepath=args.tokenizer_filepath,
        model_type=args.model_type,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        basepath_to_store_results=args.basepath_to_store_results,
        experiment_name=args.experiment_name,
        pretrained_checkpoint_or_model_filepath=args.pretrained_checkpoint_or_model_filepath,
    )
