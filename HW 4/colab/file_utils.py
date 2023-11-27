import locale
import os
import sys
import warnings
from typing import Optional, List, Union

import IPython
from google.colab import drive


def cd(load_dir: str) -> None:
    ipython = IPython.get_ipython()
    if os.path.abspath(os.getcwd()) != load_dir:
        ipython.run_line_magic("cd", load_dir)


def set_environment_variables(path: str):
    with open(path, "r") as fp:
        data = fp.readlines()
        for line in data:
            var_name, value = line.strip().split()[1].split("=")
            os.environ[var_name] = value.strip('"').strip("'")
    fp.close()


def reload_files(load_dir: Optional[str] = None) -> None:
    drive.flush_and_unmount()
    drive.mount("/content/drive")
    if load_dir is not None:
        cd(load_dir)


def install_seagull():
    ipython = IPython.get_ipython()

    warnings.filterwarnings("ignore")
    ipython.run_line_magic("run", "setup.py -qqq develop")
    sys.path.append(os.getcwd())


def pip_install(packages: Union[str, List]) -> None:
    packages = [packages] if isinstance(packages, str) else packages
    ipython = IPython.get_ipython()
    ipython.run_line_magic("sx", f"pip install -qqq --progress-bar off {' '.join(packages)}")
    print(f"installed {packages=}")


def load_required(
    load_dir: Optional[str] = None,
    environment_variables_filepath: Optional[str] = None,
    install_packages: Optional[List] = None,
    force_reinstall_pytorch: bool = True,
) -> None:
    # See: https://github.com/googlecolab/colabtools/issues/3409.
    locale.getpreferredencoding = lambda x=None: "UTF-8"

    ipython = IPython.get_ipython()
    if load_dir is not None:
        cd(load_dir)
    if environment_variables_filepath is not None:
        set_environment_variables(path=environment_variables_filepath)

    _pytorch = "torch==2.0.1"
    _needed_packages = [
        "huggingface-hub~=0.16.4",
        "datasets~=2.14.4",
        "tokenizers~=0.14.1",
        "transformers~=4.35.0",
        "einops~=0.7.0",
        "torchinfo~=1.8.0",
        "jsonlines~=3.1.0",
    ]
    if force_reinstall_pytorch:
        _needed_packages.append(_pytorch)
    if install_packages is not None:
        _needed_packages.extend(install_packages)
    ipython.run_line_magic("sx", f"pip install -qqq --progress-bar off {' '.join(_needed_packages)}")
    install_seagull()
    ipython.run_line_magic("sx", f"chmod +x scripts/*.py")

    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
