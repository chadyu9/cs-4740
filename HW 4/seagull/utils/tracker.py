import json
import logging
import os
from typing import Dict, Literal, Any, Optional

import jsonlines
from rich.console import Console
from rich.logging import RichHandler

try:
    import wandb
except ImportError:
    wandb = None


class Tracker(object):
    def __init__(
        self,
        config: Dict[str, Any],
        basepath_to_store_results: str,
        experiment_name: str,
        log_level: int = logging.DEBUG,
        log_to_wandb: bool = False,
        wandb_entity_name: Optional[str] = None,
        master_process_does_setup: bool = True,
    ):
        super().__init__()

        self.basepath_to_store_results = basepath_to_store_results
        self.config = config
        self.experiment_name = experiment_name
        self.log_level = log_level
        self.log_to_wandb = log_to_wandb and wandb is not None and wandb_entity_name is not None
        self.wandb_entity_name = wandb_entity_name

        self.run_path = None
        self.checkpoints_path = None
        self._wandb_run = None
        if not master_process_does_setup:
            self.setup()

    def setup(self):
        self.run_path = os.path.join(self.basepath_to_store_results, self.experiment_name)
        self.checkpoints_path = os.path.join(self.run_path, "checkpoints")
        os.makedirs(self.run_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)

        config_path = os.path.join(self.run_path, "config.json")
        with open(config_path, "w") as fp:
            json.dump(self.config, fp, indent=2)

        log_path = os.path.join(self.run_path, "log.txt")
        console = Console(quiet=False, force_terminal=True, tab_size=4)
        logging.basicConfig(
            level=self.log_level,
            format="%(message)s",
            handlers=[logging.FileHandler(log_path), RichHandler(show_path=False, console=console)],
        )

        if self.log_to_wandb:
            self._wandb_run = wandb.init(
                entity=self.wandb_entity_name, project="seagull", name=self.experiment_name, config=self.config
            )

    def log_metrics(
        self,
        epoch_or_step: Literal["epoch", "step"],
        epoch_or_step_num: int,
        split: str,
        metrics: Dict[str, float],
        log_to_console: bool = True,
    ) -> None:
        split_metrics_file = os.path.join(self.run_path, f"{split}_metrics.jsonl")
        epoch_or_step_and_metrics = {epoch_or_step: epoch_or_step_num, "metrics": metrics}
        with jsonlines.open(split_metrics_file, "a") as fp:
            fp.write(epoch_or_step_and_metrics)

        if self.log_to_wandb:
            metrics_ = {f"{split}/{epoch_or_step}/{metric}": score for metric, score in metrics.items()}
            metrics_[epoch_or_step] = epoch_or_step_num
            wandb.log(metrics_)

        if log_to_console:
            logging.info(f"{split}/{epoch_or_step} metrics: {epoch_or_step_and_metrics}")

    def save_model(self, model):
        model_filepath = os.path.join(self.run_path, "model.pt")
        model.save_pretrained(model_filepath)

    def save_checkpoint(self, trainer, epoch: int):
        checkpoint_filepath = os.path.join(self.checkpoints_path, f"checkpoint_{epoch}.ckpt")
        trainer.save_checkpoint(checkpoint_filepath)

    def done(self):
        if self.log_to_wandb:
            wandb.finish()
