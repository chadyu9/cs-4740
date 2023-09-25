import abc
import logging
import time
from functools import partial
from typing import Callable, Union, Generator

import torch.nn
import torchinfo
from prettytable import PrettyTable
from torch import nn
from torchinfo import summary

from ner.utils.utils import warn_once


class Module(nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.hooks = {"hooks": {}, "outputs": {}}  # dict to store outputs from registered hooks

    def __str__(self) -> str:
        return f"module: {self._get_name()}, num_params: {sum(param.numel() for param in self.parameters())}"

    def save_pretrained(self, model_filepath) -> None:
        torch.save(self.state_dict(), model_filepath)

    def from_pretrained(self, model_filepath) -> None:
        self.load_state_dict(torch.load(model_filepath, map_location=torch.device("cpu")))

    @staticmethod
    def init_weights(module: nn.Module) -> None:
        warn_once(f"the init_weights supports nn.Embedding, nn.Linear initializations with xavier_normal")
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_normal_(module.weight)
            if module.padding_idx is not None:
                torch.nn.init.zeros_(module.weight[module.padding_idx])

    def summary(self) -> torchinfo.ModelStatistics:
        return summary(self)

    def get_trainable_params(self) -> Generator:
        return (param for param in self.parameters() if param.requires_grad)

    def print_params(self) -> None:
        params_table = PrettyTable(["module", "num_params", "requires_grad"])
        total_trainable_params = 0
        for name, param in self.named_parameters():
            params_table.add_row([name, param.numel(), param.requires_grad])
            if param.requires_grad:
                total_trainable_params = total_trainable_params + param.numel()
        print(params_table)
        if total_trainable_params >= 1e3:
            print(f"total trainable params: {(total_trainable_params / 1e6):0.2f}M")
        else:
            print(f"total trainable params: {total_trainable_params}")

    def _get_hook_name(self, hook: Union[Callable, partial[Callable]]) -> str:
        try:
            hook_name = hook.__name__
        except AttributeError:
            # Partial functions.
            hook_name = hook.func.__name__

        if hook_name in self.hooks["hooks"]:
            hook_name = hook_name + "_" + str(int(time.time()))
            logging.warning(f"another hook with the same name exists, using {hook_name} instead")
        return hook_name

    def attach_hook(self, module: nn.Module, hook: Union[Callable, partial[Callable]], hook_type: str) -> None:
        hook_name = self._get_hook_name(hook)
        if hook_type == "forward":
            self.hooks["hooks"][hook_name] = module.register_forward_hook(hook)
        elif hook_type == "backward":
            self.hooks["hooks"][hook_name] = module.register_full_backward_hook(hook)

    def detach_hook(self, hook: Callable) -> None:
        logging.warning(f"detach_hook doesn't remove the output from self.hooks['outputs']")
        hook_name = self._get_hook_name(hook)
        self.hooks["hooks"][hook_name].remove()
        self.hooks["hooks"].pop(hook_name)

    def detach_all_hooks(self) -> None:
        for handle in self.hooks["hooks"].values():
            handle.remove()
        self.hooks = {"hooks": {}, "outputs": {}}

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError
