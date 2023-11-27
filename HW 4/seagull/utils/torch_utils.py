import os
import random
from typing import Literal, Dict, Any

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group


def set_seed(seed: int = 4740):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Setting `torch.backends.cudnn.benchmark = False` slows down training.
    # Reference: https://pytorch.org/docs/stable/notes/randomness.html.
    torch.backends.cudnn.benchmark = True


def set_pytorch_backends():
    # TF32: https://docs.monai.io/en/stable/precision_accelerating.html.
    # Set TF32 for speedup: https://x.com/GuggerSylvain/status/1599190137367166977?s=20.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.opt_einsum.enabled = True


def get_device(device_type: str = "auto") -> torch.device:
    if device_type == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_type == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_type == "mps":
        raise ValueError(f"{device_type=} not supported; must be one of ['cpu', 'cuda']")
    return torch.device("cpu")


def ddp_setup(ddp_backend: Literal["nccl", "mpi", "gloo", "ucc"] = "nccl") -> bool:
    if int(os.environ.get("RANK", -1)) != -1:
        init_process_group(backend=ddp_backend)
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def ddp_cleanup():
    destroy_process_group()


def remove_compiled_model_prefix_from_model_state_dict(model_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # See: https://discuss.pytorch.org/t/how-to-save-load-a-model-with-torch-compile/179739/2.
    unwanted_prefix = "_orig_mod."
    for module_name, value in list(model_state_dict.items()):
        if module_name.startswith(unwanted_prefix):
            model_state_dict[module_name[len(unwanted_prefix) :]] = model_state_dict.pop(module_name)
    return model_state_dict
