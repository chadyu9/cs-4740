from argparse import ArgumentParser
from typing import Optional

import torch
import yaml

from seagull.data_processing.bbpe import BBPETokenizer
from seagull.model.heads.lm_utils import make_seagull_talk
from seagull.model.heads.seagull_lm import SeagullLM
from seagull.utils.torch_utils import get_device, set_pytorch_backends, set_seed

set_pytorch_backends()


@torch.no_grad()
def main(
    config_path: str,
    tokenizer_basepath: str,
    pretrained_checkpoint_or_model_filepath: Optional[str] = None,
    prompt: str = None,
    max_new_tokens: int = 20,
    num_samples: int = 5,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    set_seed(4740)
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    device = get_device(config["general"]["device"])

    bbpe_tokenizer = BBPETokenizer()
    bbpe_tokenizer.from_file(tokenizer_basepath)

    model = SeagullLM(
        vocab_size=bbpe_tokenizer.vocab_size,
        padding_idx=bbpe_tokenizer._pretrained_tokenizer.pad_token_id,
        **config["model"],
    )
    model = model.to(device)
    if pretrained_checkpoint_or_model_filepath is not None:
        model.from_pretrained_or_checkpoint(pretrained_checkpoint_or_model_filepath)
    make_seagull_talk(
        model=model,
        bbpe_tokenizer=bbpe_tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
    )


def argparser():
    parser = ArgumentParser(description="Generate text (make the seagull talk!) using the pretrained model.")
    parser.add_argument("--config-path", type=str, help="Path to the config file.", required=True)
    parser.add_argument(
        "--tokenizer-basepath",
        type=str,
        help="Basepath to the trained tokenizer (path must include tokenizer.json and state_dict.json files).",
        required=True,
    )
    parser.add_argument(
        "--pretrained-checkpoint-or-model-filepath",
        type=str,
        help="Path to pretrained checkpoint or model file; if not provided, a random initialization will be used.",
        default=None,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="The input prompt to autocomplete; if not specified, with default to `<|endoftext|>` token.",
        required=False,
        default=None,
    )
    parser.add_argument("--max-new-tokens", type=int, help="Maximum number of new tokens to generate.", default=100)
    parser.add_argument("--num-samples", type=int, help="Number of samples to generate.", default=8)
    parser.add_argument("--temperature", type=float, help="The generation temperature (higher = diverse).", default=0.7)
    parser.add_argument(
        "--top-k",
        type=int,
        help="If specified, only the top-k candidates will be used in generating samples.",
        default=None,
    )
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(
        config_path=args.config_path,
        tokenizer_basepath=args.tokenizer_basepath,
        pretrained_checkpoint_or_model_filepath=args.pretrained_checkpoint_or_model_filepath,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
    )
