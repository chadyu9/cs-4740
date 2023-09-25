import json
import logging
import os
from typing import Dict

import jsonlines
from rich.console import Console
from rich.logging import RichHandler


class Tracker(object):
    def __init__(self, config, basepath_to_store_results, experiment_name, log_level=logging.DEBUG):
        super().__init__()

        self.basepath_to_store_results = basepath_to_store_results
        self.config = config
        self.experiment_name = experiment_name
        self.log_level = log_level

        self._setup()

    def _setup(self):
        self.run_path = os.path.join(self.basepath_to_store_results, self.experiment_name)
        self.checkpoints_path = os.path.join(self.run_path, "checkpoints")
        os.makedirs(self.run_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)

        config_path = os.path.join(self.run_path, "config.json")
        with open(config_path, "w") as fp:
            json.dump(self.config, fp)

        log_path = os.path.join(self.run_path, "log.txt")
        console = Console(quiet=False, force_terminal=True, tab_size=4)
        logging.basicConfig(
            level=self.log_level,
            format="%(message)s",
            handlers=[logging.FileHandler(log_path), RichHandler(show_path=False, console=console)],
        )

    def log_metrics(self, epoch: int, split: str, metrics: Dict[str, float]) -> None:
        split_metrics_file = os.path.join(self.run_path, f"{split}_metrics.jsonl")
        epoch_and_metrics = {"epoch": epoch, "metrics": metrics}
        with jsonlines.open(split_metrics_file, "a") as fp:
            fp.write(epoch_and_metrics)
        logging.info(f"{split} metrics: {epoch_and_metrics}")

    def save_model(self, model):
        model_filepath = os.path.join(self.run_path, "model.pt")
        model.save_pretrained(model_filepath)

    def save_checkpoint(self, trainer, epoch: int):
        checkpoint_filepath = os.path.join(self.checkpoints_path, f"checkpoint_{epoch}.ckpt")
        trainer.save_checkpoint(checkpoint_filepath)
