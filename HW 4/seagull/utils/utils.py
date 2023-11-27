import json
import logging
import random
from functools import lru_cache
from typing import List, Optional

from IPython.display import HTML

from seagull.utils.styling import COLOR_MAP, ATTRS_MAP


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
    # Puppy videos, idea borrowed from Sasha Rush's GPU puzzles!
    puppy_video_ids = [
        "zJx2skh",
        "udLK6FS",
        "UTdqQ2w",
        "Wj2PGRl",
        "Dx2k4gf",
        "pCAIlxD",
        "rJEkEw4",
        "Vy9JShx",
        "eAhk0q7",
        "1xeUYme",
        "qqXt0XG",
        "9xCV5v1",
        "eyxH0Wc",
        "yWdsKTY",
        "bRKfspn",
        "fqHxOGI",
        "HlaTE8H",
        "lvzRF3W",
        "tNPC84t",
        "XNp6i0w",
        "Z0TII8i",
        "DKLBJh7",
        "UNIBvxI",
        "2E81PWN",
        "CCxZ6Wr",
        "ros6RLC",
        "6tVqKyM",
        "akVmh3i",
        "lqxIBsu",
        "fiJxCVA",
        "wScLiVz",
        "NQWTWXs",
        "6Kmg87X",
        "sWp0Dqd",
        "2Gdl1u7",
        "2m78jPG",
        "aydRUz8",
        "HblQhgb",
        "MVUdQYK",
        "2F6j2B4",
        "DS2IZ6K",
        "3V37Hqr",
        "Eq2uMTA",
        "djeivlK",
        "kLvno0p",
        "lMW0OPQ",
        "F1SChho",
        "qYpCMnM",
        "0n25aBB",
        "9hRi2jN",
        "qawCMl5",
        "Nu4RH7f",
        "FPxZ8WK",
        "14QJ3Mv",
        "ZNem5o3",
        "wHVpHVG",
        "pn1e9TO",
        "rFU7vEe",
        "g9I2ZmK",
        "k5jALH0",
        "dGW4BE3",
        "MQCIwzT",
        "9O1rLtw",
        "aJJAY4c",
        "bDYdPSV",
        "iwe1n1K",
    ]
    print(colored("Success!\n", "green"), end="")
    return HTML(
        f"""<video alt="success, happy puppy!" width="400" height="240" controls autoplay=1>
                <source src="https://openpuppies.com/mp4/{random.sample(puppy_video_ids, 1)[0]}.mp4" type="video/mp4"/> 
            </video>"""
    )
