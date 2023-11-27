import copy
import logging
import os
from argparse import ArgumentParser
from typing import Optional, Literal, List

import datasets
import yaml
from torch.optim import AdamW

from seagull.data_processing.bbpe import BBPETokenizer
from seagull.model.heads.seagull_lm import SeagullLM
from seagull.nn.optim.lr_schedulers.cosine_lr_scheduler import LinearWarmupCosineAnnealingLR
from seagull.trainers.trainer import Trainer
from seagull.utils.torch_utils import set_seed, get_device, ddp_setup, ddp_cleanup
from seagull.utils.tracker import Tracker


def main(
    config_path: str,
    basepath_to_tokenized_dataset: str,
    tokenizer_basepath: str,
    basepath_to_store_results: str,
    batch_size: int,
    num_epochs: int,
    experiment_name: str,
    pretrained_checkpoint_or_model_filepath: Optional[str] = None,
    wandb_entity_name: Optional[str] = None,
    detect_anomaly: bool = False,
    ddp_backend: Literal["nccl", "mpi", "gloo", "ucc"] = "nccl",
    pretraining: bool = False,
    freeze_layer_ids: List[str] = None,
) -> None:
    if not pretraining:
        set_seed(4740)
    use_ddp = ddp_setup(ddp_backend=ddp_backend)

    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    device = get_device(config["general"]["device"])
    training_mode = "pretraining" if pretraining else "finetuning"
    for key, value in config.items():
        if isinstance(value, dict) and training_mode in value:
            config[key] = value[training_mode]

    config_to_save = copy.deepcopy(config)
    config_to_save["general"].update({"pretraining": pretraining, "ddp_backend": ddp_backend})
    config_to_save["train_and_eval"].update(
        {"batch_size": batch_size, "num_epochs": num_epochs, "freeze_layer_ids": freeze_layer_ids}
    )

    tracker = Tracker(
        config=config_to_save,
        basepath_to_store_results=basepath_to_store_results,
        experiment_name=experiment_name,
        log_to_wandb=(wandb_entity_name is not None),
        wandb_entity_name=wandb_entity_name,
        master_process_does_setup=True,
    )

    tokenized_dataset = datasets.load_from_disk(basepath_to_tokenized_dataset)
    bbpe_tokenizer = BBPETokenizer()
    bbpe_tokenizer.from_file(tokenizer_basepath)

    model = SeagullLM(
        vocab_size=bbpe_tokenizer.vocab_size,
        padding_idx=bbpe_tokenizer._pretrained_tokenizer.pad_token_id,
        **config["model"],
    )
    # Freeze before creating the DDP model: https://tinyurl.com/34kfsz3v.
    if freeze_layer_ids is not None:
        for name, param in model.named_parameters():
            if any(f".{layer_id}." in name for layer_id in freeze_layer_ids):
                logging.info(f"freezing {name}")
                param.requires_grad = False

    # LLaMa-2: https://arxiv.org/pdf/2307.09288.pdf.
    optimizer = AdamW(model.parameters(), **config["optimizer"])
    lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, **config["lr_scheduler"])

    val_split_name = "validation" if "validation" in tokenized_dataset else "val"  # wikitext-103
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_data=tokenized_dataset["train"],
        val_data=(tokenized_dataset[val_split_name] if val_split_name in tokenized_dataset else None),
        labels_ignore_idx=bbpe_tokenizer._pretrained_tokenizer.pad_token_id,
        lr_scheduler=lr_scheduler,
        tracker=tracker,
        detect_anomaly=detect_anomaly,
        device=device,
        use_ddp=use_ddp,
        **config["trainer"],
    )
    if trainer.is_master_process:
        model.print_params()

    if pretrained_checkpoint_or_model_filepath is not None:
        if pretrained_checkpoint_or_model_filepath.endswith(".ckpt"):
            trainer.from_checkpoint(checkpoint_path=pretrained_checkpoint_or_model_filepath)
        elif pretrained_checkpoint_or_model_filepath.endswith(".pt"):
            model.from_pretrained(model_filepath=pretrained_checkpoint_or_model_filepath)
        else:
            raise ValueError("model file not supported")
    trainer.train_and_eval(batch_size=batch_size, num_epochs=num_epochs, **config["train_and_eval"])

    tracker.done()
    if use_ddp:
        ddp_cleanup()


def argparser():
    parser = ArgumentParser(description="Train a generative model using the tokenized newyorker dataset.")
    parser.add_argument("--config-path", type=str, help="Path to the config file.", required=True)
    parser.add_argument(
        "--basepath-to-tokenized-dataset",
        type=str,
        help="Path to the tokenized newyorker dataset with `input_ids` and `padding_mask`.",
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
        "--basepath-to-store-results",
        type=str,
        help="Basepath to store all the experimental results and model/checkpoint artefacts.",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training and validation; use a multiple of 8: "
        "https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html",
        required=True,
    )
    parser.add_argument("--num-epochs", type=int, help="Number of training epochs.", required=True)
    parser.add_argument(
        "--pretrained-checkpoint-or-model-filepath",
        type=str,
        help="Path to pretrained checkpoint or model file.",
        default=None,
    )
    parser.add_argument(
        "--freeze-layer-ids",
        nargs="+",
        help="Indicates which layers to be frozen (helpful in finetuning); the name matching is performed "
        "using `if .{layer_id}. in module_name`",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--experiment-name", type=str, help="Experiment name.", default="seagull_experiment", required=True
    )
    parser.add_argument("--wandb-entity-name", type=str, help="Entity name for wandb team.", default=None)
    parser.add_argument("--detect-anomaly", action="store_true", help="Model runs in autograd anomaly detection mode.")
    parser.add_argument("--ddp-backend", choices=["nccl", "mpi", "gloo", "ucc"], help="DDP backend.", default="nccl")
    parser.add_argument("--pretraining", action="store_true", help="Indicates if model is being pretrained.")
    return parser


if __name__ == "__main__":
    # To run on 4 GPUs: `torchrun --standalone --nproc_per_node=4 train_model.py (--cli-arg ...)`.
    args = argparser().parse_args()
    main(
        config_path=args.config_path,
        basepath_to_tokenized_dataset=args.basepath_to_tokenized_dataset,
        tokenizer_basepath=args.tokenizer_basepath,
        basepath_to_store_results=args.basepath_to_store_results,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        pretrained_checkpoint_or_model_filepath=args.pretrained_checkpoint_or_model_filepath,
        freeze_layer_ids=args.freeze_layer_ids,
        experiment_name=args.experiment_name,
        wandb_entity_name=args.wandb_entity_name,
        detect_anomaly=args.detect_anomaly,
        ddp_backend=args.ddp_backend,
        pretraining=args.pretraining,
    )
