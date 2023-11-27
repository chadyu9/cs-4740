from glob import glob

from setuptools import setup, find_packages

with open("requirements.txt") as fp:
    requirements = fp.read().splitlines()

setup(
    name="seagull",
    version="1.0",
    description="a library for generating humorous captions from scene descriptions",
    author=(
        "Tushaar Gangavarapu, Darren Key, Logan Kraver, Lionel Tan, Pun Chaixanien, Kai Horstmann, "
        "Dave Jung, Aaishi Uppuluri"
    ),
    author_email="tg352@cornell.edu (Tushaar Gangavarapu)",
    scripts=glob("scripts/*.py", recursive=True),
    py_modules=[],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    url="https://github.coecis.cornell.edu/cs4740/hw4-fa23",
)
