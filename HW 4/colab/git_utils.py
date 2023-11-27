import os
from typing import List

import IPython


def clone(repository: str = "hw4"):
    ipython = IPython.get_ipython()
    if os.environ.get("PERSONAL_ACCESS_TOKEN") is not None or os.environ.get("PERSONAL_ACCESS_TOKEN") == "":
        ipython.run_line_magic(
            "sx",
            f"GIT_LFS_SKIP_SMUDGE=1 git clone https://{os.environ['PERSONAL_ACCESS_TOKEN']}@github.coecis.cornell.edu/"
            f"cs4740-fa23-public/{repository}.git",
        )
    else:
        raise ValueError("PERSONAL_ACCESS_TOKEN not set")


def get_lfs_files(filepaths: List[str]):
    ipython = IPython.get_ipython()
    for filepath in filepaths:
        # See: https://github.com/git-lfs/git-lfs/issues/1351.
        ipython.run_line_magic("sx", f"git lfs pull --include {filepath}")
