from glob import glob

from setuptools import setup, find_packages

with open("requirements.txt") as fp:
    requirements = fp.read().splitlines()

setup(
    name="srl",
    version="1.0",
    description="a library for semantic role labelling using neural networks",
    author=(
        "Vivian Nguyen, Sasha Boguraev, Han Xia, Travis Zhange, Vinh Nguyen, "
        "Sienna Hu, Gavin Fogel, Vivian Chen,  Benjamin Hu"
    ),
    author_email="ljl2@cornell.edu (Lillian Lee)",
    scripts=[],
    py_modules=[],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    url="https://github.coecis.cornell.edu/cs4740/hw3-fa23",
)