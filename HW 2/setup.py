from glob import glob

from setuptools import setup, find_packages

with open("requirements.txt") as fp:
    requirements = fp.read().splitlines()

setup(
    name="ner",
    version="1.0",
    description="a library for named-entity recognition using neural networks",
    author=(
        "Tushaar Gangavarapu, Pun Chaixanien, Kai Horstmann, Dave Jung, Aaishi Uppuluri, Lillian Lee, "
        "Darren Key, Logan Kraver, Lionel Tan"
    ),
    author_email="ljl2@cornell.edu (Lillian Lee)",
    scripts=glob("scripts/*.py", recursive=True),
    py_modules=[],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    url="https://github.coecis.cornell.edu/cs4740/hw2-fa23",
)
