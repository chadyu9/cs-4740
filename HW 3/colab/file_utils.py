import os
import sys
import warnings
from typing import Optional, List

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


def activate(venv_path: str = "./venv") -> None:
    ipython = IPython.get_ipython()
    ipython.run_line_magic("run", f"{venv_path}/bin/activate_this.py")


def create_venv(venv_path: str = "venv", activate_venv: bool = True) -> None:
    ipython = IPython.get_ipython()
    try:
        __import__("virtualenv")
    except ImportError:
        ipython.run_line_magic("sx", "pip install -qqq virtualenv --progress-bar off")
    ipython.run_line_magic("sx", f"virtualenv {venv_path}")
    if activate_venv:
        activate(venv_path)


def install_requirements() -> None:
    ipython = IPython.get_ipython()
    ipython.run_line_magic("sx", "pip install -qqq -r requirements.txt --progress-bar off")


def install_srl() -> None:
    ipython = IPython.get_ipython()

    warnings.filterwarnings("ignore")
    ipython.run_line_magic("run", "setup.py -qqq develop")
    sys.path.append(os.getcwd())


def setup(
    load_dir: Optional[str] = None,
    venv_path: Optional[str] = None,
    environment_variables_filepath: Optional[str] = None,
) -> None:
    ipython = IPython.get_ipython()
    if load_dir is not None:
        cd(load_dir)
    if environment_variables_filepath is not None:
        set_environment_variables(path=environment_variables_filepath)

    if venv_path is not None:
        create_venv(venv_path=venv_path, activate_venv=True)
    install_requirements()
    install_ner()

    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")


def load_required(
    venv_path: Optional[str] = None,
    load_dir: Optional[str] = None,
    environment_variables_filepath: Optional[str] = None,
    install_packages: Optional[List] = None,
) -> None:
    ipython = IPython.get_ipython()
    if load_dir is not None:
        cd(load_dir)
    if environment_variables_filepath is not None:
        set_environment_variables(path=environment_variables_filepath)
    if venv_path is not None:
        activate(venv_path=venv_path)
        ipython.run_line_magic("sx", f"chmod +x {venv_path}/bin/*")

    if install_packages is not None:
        ipython.run_line_magic("sx", f"pip install -qqq --progress-bar off {' '.join(install_packages)}")
    install_srl()
    ipython.run_line_magic("sx", f"chmod +x scripts/*.py")

    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
