# Setup

## VSCode

This year, our recommended IDE is [Visual Studio Code](https://code.visualstudio.com).

### Installing VSCode

More information can be found on the [VSCode Download](https://code.visualstudio.com/download) web page. Just pick your operating system, and follow the instructions you are prompted with.

## Installing VSCode Extensions

We recommend installing the following extensions:

1. [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python): Intellisense and basic Python formatting
2. [Live Share](https://marketplace.visualstudio.com/items?itemName=MS-vsliveshare.vsliveshare): Work on the same codebase at the same time with your partner

## VSCode Live Share

We recommend using VSCode Live Share to facilitate pair programming and collaboration. This will allow you to work with your partner on the same codebase at the same time. You can read more about VSCode Live Share [here](https://code.visualstudio.com/learn/collaboration/live-share).

### Installation

You can download the extension [here](https://marketplace.visualstudio.com/items?itemName=MS-vsliveshare.vsliveshare). Upon pressing install, your browser will prompt you with an alert giving you the option to open in VSCode. Click 'yes' to start the installation.

![image](/guide_images/install-button.png)

### Alternative Live Share installation instructions

Didn't work for you? Try these steps instead:

1. Go to the extension tab in VSCode (the icon with four squares on the left side of the window)
2. Search `ms-vsliveshare.vsliveshare`, which is the Live Share extension's ID
3. Click install

### How to use Live Share

For more instructions on using VSCode Live Share, please refer to guide receference above, also linked [here](https://code.visualstudio.com/learn/collaboration/live-share#_get-started-with-live-share).

## Python Extension (formatting)

### Installation

For installing these extensions, follow the same steps as above. Here are the download links:

1. [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

### How to use the formatter

To format your Python using VS Code, check out [this link](https://code.visualstudio.com/docs/python/formatting). Using this formatter, your code will be formatted to the [PEP 8](https://www.python.org/dev/peps/pep-0008/) standard. This will help standardize the students code, specifically settling on
4 spaces as the standard indentation.

## Auto reload

### Installation

`%load_ext autoreload` should be included in the first cell of your notebook. This will allow you to reload any changes you make to your code automatically. It was one of the standard extensions included in the IPython notebook, so we don't need to install it separately. With this, you won't need to re-import your `*.py` files every time you make changes to them, and want to run them in your `.ipynb` notebook.

## Creating a Python virtual environment

Since we are using Python 3, we will use the `venv` module to create a virtual environment. This is a built-in module, so we don't need to install anything.

### Creating a New Virtual Environment

To create a virtual environment, we need to run the following command:

```bash
python3 -m venv cs4740
```

This creates a new virtual environment in a directory called `cs4740`.

### Activating the Virtual Environment

To activate the virtual environment, we need to run the following command:

```bash
source cs4740/bin/activate
```

This executes a binary, which will activate the virtual environment in the current shell session. You should see the name of the virtual environment in parentheses in your terminal prompt.

### Installing Dependencies

All of the Python dependencies for the project are in a `requirements.txt` file. To install them, we need to run the following command:

```bash
pip install -r requirements.txt
```

This will install all of the dependencies in the virtual environment you just created.