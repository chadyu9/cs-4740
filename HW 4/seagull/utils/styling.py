#: Colors and their ANSI escape sequences.
COLOR_MAP = {
    "red": "\033[91m",
    "green": "\033[92m",
    "dark green": "\033[32m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "pink": "\033[95m",
    "teal": "\033[96m",
    "grey": "\033[97m",
}

# Reference: https://stackoverflow.com/a/33206814.
#: Text styles and their ANSI escape sequences.
ATTRS_MAP = {
    "underline": "\033[4m",
    "bold": "\033[1m",
    "italic": "\033[3m",
    "framed": "\033[51m ",
    "crossed out": "\033[9m",
}
