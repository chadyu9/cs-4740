import json
import logging
import os
import random
from functools import lru_cache
from typing import List, Union, Dict, Tuple, Optional

import numpy as np
import torch
from IPython.display import HTML

from utils.styling import COLOR_MAP, ATTRS_MAP

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