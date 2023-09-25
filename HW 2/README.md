# CS 4740 Fa'23 HW2

Named-entity recognition using FFNNs and RNNs. The documentation for the assignment is hosted at:
https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/.

---

## Installation

### Google Colab

Using Google Colab to run this assignment is the preferred approach. Download the
[`setup.ipynb`](https://github.coecis.cornell.edu/cs4740-fa23-public/hw2-fa23/blob/main/notebooks/setup.ipynb) file.
Next, upload it to your drive (the upload location of this file on your drive is insignificant). Open it in Google
Colab and follow the instructions in the notebook file.

### Local installation

This package can also be installed locally (requires `python >= 3.9`), to do so, run:

```shell
# Clone the repository!
cd $HOME
git clone https://github.coecis.cornell.edu/cs4740-fa23-public/hw2-fa23.git
cd $HOME/hw2-fa23

# Create a venv 'venv' in hw2-fa23 dir.
python3 -m venv venv
source ./venv/bin/activate

# Install the needed requirements and setup the package.
pip3 install -r requirements.txt
pip3 install --editable .
```

---

## Running scripts

To run scripts on command line, mimic the commands in
[`hw2.ipynb`](https://github.coecis.cornell.edu/cs4740-fa23-public/hw2-fa23/blob/main/notebooks/hw2.ipynb).

---

## Citation

**Authors.** Tushaar Gangavarapu, Pun Chaixanien<sup>&#9728;</sup>, Kai Horstmann<sup>&#9728;</sup>,
Dave Jung<sup>&#9728;</sup>, Aaishi Uppuluri<sup>&#9728;</sup>, Lillian Lee, Darren Key<sup>&#9729;</sup>,
Logan Kraver<sup>&#9729;</sup>, Lionel Tan<sup>&#9729;</sup>

&nbsp;&nbsp;&nbsp;&nbsp;<sup>&#9728;</sup> software creators, equal contribution, ordered alphabetically <br/>
&nbsp;&nbsp;&nbsp;&nbsp;<sup>&#9729;</sup> software testers, equal contribution, ordered alphabetically

Cite the software as:

> Tushaar Gangavarapu, Pun Chaixanien, Kai Horstmann, Dave Jung, Aaishi Uppuluri, Lillian Lee, Darren Key,
> Logan Kraver, Lionel Tan. 2023. CS 4740 Fa'23 HW2: Named-entity recognition using FFNNs and RNNs. GitHub.
> https://github.coecis.cornell.edu/cs4740-fa23-public/hw2-fa23.

**Acknowledgments.** This work is inspired from the assignment "CS 4740 FA'22 HW2: Neural NER" developed by
John Chung, Renee Shen, John R. Starr, Tushaar Gangavarapu, Fangcong Yin, Shaden Shaar, Marten van Schijndel, and
Lillian Lee.
