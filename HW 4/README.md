# CS 4740 Fa'23 HW4: Seagull

Generating humorous captions from scene descriptions using transformer-based large language models. The documentation
for the assignment is hosted at: https://pages.github.coecis.cornell.edu/cs4740/hw4-fa23/.

---

## Installation

### Google Colab

Using Google Colab is the preferred approach to run this assignment. Download
[`setup.ipynb`](https://github.coecis.cornell.edu/cs4740-fa23-public/hw4-fa23/blob/main/notebooks/setup.ipynb) file
from GitHub (see: https://stackoverflow.com/a/45645081 for download instructions). Next, upload it to your drive (the
upload location of this file on your drive is insignificant). Open it in Google Colab and follow the instructions in
the notebook file.

### Local installation

This package can also be installed locally (requires `python >= 3.9`). Before installing, you will need a personal
access token to be able to clone the repository:
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
(you can use the same personal access token used for HW2). Once you've secured your personal access token, set the
`PERSONAL_ACCESS_TOKEN` variable below and run the following in your terminal:

```shell
# Set your personal access token
cd $HOME
export PERSONAL_ACCESS_TOKEN=""
```

Once the personal access token is set, you can clone the repository:

```shell
# Clone the repository, but don't clone any of the pretrained model files! 
GIT_LFS_SKIP_SMUDGE=1 \
    git clone https://"$PERSONAL_ACCESS_TOKEN"@github.coecis.cornell.edu/cs4740-fa23-public/hw4-fa23.git \
    2> ~/error_log.txt || chmod +x $repo/.git/hooks/post-checkout
cd $HOME/hw4-fa23

# Create a venv 'venv' in hw4-fa23 dir.
python3 -m venv venv
source ./venv/bin/activate

# Install the needed requirements and setup the package.
pip3 install -r requirements.txt
pip3 install --editable .
```

---

## Running scripts

To run scripts on command line, mimic the commands in
[`hw4.ipynb`](https://github.coecis.cornell.edu/cs4740-fa23-public/hw4-fa23/blob/main/notebooks/hw4.ipynb).

---

## Citation

**Authors.** Tushaar Gangavarapu, Darren Key<sup>&#129433;</sup>, Logan Kraver<sup>&#129433;</sup>,
Lionel Tan<sup>&#129433;</sup>, Pun Chaixanien<sup>&#129436;</sup>, Kai Horstmann<sup>&#129436;</sup>,
Dave Jung<sup>&#129436;</sup>, Aaishi Uppuluri<sup>&#129436;</sup>

&nbsp;&nbsp;&nbsp;&nbsp;<sup>&#129433;</sup> software creators, equal contribution, ordered alphabetically <br/>
&nbsp;&nbsp;&nbsp;&nbsp;<sup>&#129436;</sup> software testers, equal contribution, ordered alphabetically

Cite the software as:

> Tushaar Gangavarapu, Darren Key<sup>&#129433;</sup>, Logan Kraver<sup>&#129433;</sup>,
> Lionel Tan<sup>&#129433;</sup>, Pun Chaixanien<sup>&#129436;</sup>, Kai Horstmann<sup>&#129436;</sup>,
> Dave Jung<sup>&#129436;</sup>, Aaishi Uppuluri<sup>&#129436;</sup>. 2023. [CS 4740 Fa'23 HW4] Hush, the seagulls
> are purring: On generating humorous captions from scene descriptions. GitHub.
> https://github.coecis.cornell.edu/cs4740-fa23-public/hw4-fa23/.

**Acknowledgments.** This assignment was inspired from the award-winning work: [Do Androids Laugh at Electric Sheep?
Humor "Understanding" Benchmarks from The New Yorker Caption Contest](https://aclanthology.org/2023.acl-long.41/) by
Jack Hessel, Ana Marasović, Jena D. Hwang, Lillian Lee, Jeff Da, Rowan Zellers, Robert Mankoff, and Yejin Choi.
