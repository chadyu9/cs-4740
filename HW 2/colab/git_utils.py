import os

import IPython


def git_clone(repository: str = "hw2"):
    ipython = IPython.get_ipython()
    if os.environ.get("PERSONAL_ACCESS_TOKEN") is not None or os.environ.get("PERSONAL_ACCESS_TOKEN") == "":
        ipython.run_line_magic(
            "sx",
            f"git clone https://{os.environ['PERSONAL_ACCESS_TOKEN']}@github.coecis.cornell.edu/"
            f"cs4740-fa23-public/{repository}.git",
        )
    else:
        raise ValueError("PERSONAL_ACCESS_TOKEN not set")


def git_pull():
    ipython = IPython.get_ipython()
    print("\n".join(ipython.run_line_magic("sx", "git pull")))


def git_status():
    ipython = IPython.get_ipython()
    print("\n".join(ipython.run_line_magic("sx", "git status")))
