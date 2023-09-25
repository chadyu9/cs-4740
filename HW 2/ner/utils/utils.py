import json
import logging
import os
import random
from functools import lru_cache
from typing import List, Union, Dict, Tuple, Optional

import numpy as np
import torch
from IPython.display import HTML

from ner.data_processing.constants import NER_DECODING_MAP
from ner.utils.styling import COLOR_MAP, ATTRS_MAP


def get_named_entity_spans(
    encoded_ner_ids: Union[List, np.ndarray], token_idxs: Optional[Union[List, np.ndarray]] = None
) -> Dict[str, List[Tuple[int]]]:
    label_dict = {"LOC": [], "MISC": [], "PER": [], "ORG": []}

    decoded_ner_tags = np.array([NER_DECODING_MAP[tag_id] for tag_id in encoded_ner_ids])
    token_idxs = token_idxs if token_idxs is not None else list(range(len(decoded_ner_tags)))
    ner_tag_idxs = np.where(decoded_ner_tags != "O")[0]

    # TODO: Add "notation" section while releasing the assignment.
    ent_start_idx, ent_end_idx, running_ner_tag = None, None, None
    for curr_ner_tag_idx in ner_tag_idxs:
        curr_ner_tag = decoded_ner_tags[curr_ner_tag_idx].split("-")[-1]
        if decoded_ner_tags[curr_ner_tag_idx].startswith("B-"):
            if ent_start_idx is not None and ent_end_idx is not None:
                # Update previous named-entity span.
                label_dict[running_ner_tag].append((token_idxs[ent_start_idx], token_idxs[ent_end_idx]))

            ent_start_idx, ent_end_idx = curr_ner_tag_idx, curr_ner_tag_idx
            running_ner_tag = curr_ner_tag
        elif (
            decoded_ner_tags[curr_ner_tag_idx].startswith("I-")
            and curr_ner_tag == running_ner_tag
            and curr_ner_tag_idx == ent_end_idx + 1
        ):
            ent_end_idx = curr_ner_tag_idx
    if ent_start_idx is not None and ent_end_idx is not None:
        # If this is the last tag, just update the dict.
        label_dict[running_ner_tag].append((token_idxs[ent_start_idx], token_idxs[ent_end_idx]))
    return label_dict


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


@lru_cache(10)
def warn_once(message: str):
    logging.warning(message)


def load_json(filepath: str):
    with open(filepath, "r") as fp:
        data = json.loads(fp.read())
    return data


def colored(text, color: str = "", attrs: Optional[List] = None) -> str:
    if attrs is None:
        attrs = []
    for attr in attrs:
        text = f"{ATTRS_MAP.get(attr, '')}{text}\033[0m"
    return f"{COLOR_MAP.get(color, '')}{text}\033[0m"


def success() -> HTML:
    success_videos = [
        """<img src="https://media.giphy.com/media/3oEdv6UTqzNk9Y5i36/giphy.gif"/>""",
        """<img src="https://media.giphy.com/media/3oz9ZE2Oo9zRC/giphy.gif"/>""",
        """<iframe frameBorder="0" height="270" width="480"
        src="https://giphy.com/embed/8VDO7Fy2PFohfdAnpJ/video"></iframe>""",
        """<iframe frameBorder="0" height="360" width="480"
        src="https://giphy.com/embed/rwqt1f492BBGpAbbSY/video">""",
        """<img src="https://media.giphy.com/media/Srf1W4nnQIb0k/giphy.gif"/>""",
        """<img src="https://media.giphy.com/media/xT8qBepJQzUjXpeWU8/giphy.gif"/>""",
        """<img src="https://media.giphy.com/media/cOvgh3VjLmeg8LLBtk/giphy.gif">""",
        """<img src="https://media.giphy.com/media/12d19apJyRsmA/giphy.gif"/>""",
        """<iframe frameBorder="0" height="320"  width="480"
        src="https://giphy.com/embed/uh26nURBaRpBzy8YRo/video"></iframe>""",
        """<img src="https://media.giphy.com/media/lxyDpcWSJ0a3UdkOfx/giphy.gif">""",
        """<iframe frameBorder="0" height="270" width="480"
        src="https://giphy.com/embed/U4dLVG7d5KsqnN8pBG/video"></iframe>""",
        """<img src="https://media.giphy.com/media/rLENR3QvrRf4A/giphy.gif"/>""",
        """<iframe frameBorder="0" height="270" width="480"
        src="https://giphy.com/embed/cNdJPpoJhOz4D3Aw8G/video"></iframe>""",
        """<img src="https://media.giphy.com/media/3o6Mbnm7WMv7O6yj5K/giphy.gif"/>""",
        """<img src="https://media.giphy.com/media/3o6Mbolqx8Ses8KXoQ/giphy.gif"/>""",
        """<img src="https://media.giphy.com/media/QW5nKIoebG8y4/giphy.gif"/>""",
    ]
    return HTML(random.sample(success_videos, 1)[0])


if __name__ == "__main__":
    test_encoded_ner_ids = [0, 1, 2, 0, 0, 0, 3, 4, 4, 3, 3, 1, 4, 4, 2, 4, 3]
    test_named_ent_spans = get_named_entity_spans(test_encoded_ner_ids)

    assert test_named_ent_spans["LOC"] == test_named_ent_spans["MISC"] and len(test_named_ent_spans["LOC"]) == 0
    assert len(test_named_ent_spans["ORG"]) == 2 and len(test_named_ent_spans["PER"]) == 4

    assert (1, 2) in test_named_ent_spans["ORG"]
    assert (11, 14) not in test_named_ent_spans["ORG"]
    assert (10, 10) in test_named_ent_spans["PER"]
    assert (6, 8) in test_named_ent_spans["PER"]
