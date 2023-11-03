import torch

'''
Constant - srl_map
Description: maps srl tag to integer index value
'''
SRL_MAP = {'O': 0,
           'B-ARG0': 1,
           'I-ARG0': 2,
           'B-ARG1': 3,
           'I-ARG1': 4,
           'B-ARG2': 5,
           'I-ARG2': 6,
           'B-ARGM-LOC': 7,
           'I-ARGM-LOC': 8,
           'B-ARGM-TMP': 9,
           'I-ARGM-TMP': 10}

'''
Constant - srl_frames
Description: list of all possible srl frames
'''
SRL_FRAMES = ["ARGM-TMP", "ARG0", "ARG1", "ARG2", "ARGM-LOC"]

'''
Constant - SEPT
Description: separator token
'''
SEPT = 'SEP_T'

