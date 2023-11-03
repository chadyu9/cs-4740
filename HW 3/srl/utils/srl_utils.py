from srl.utils.constants import * 
from typing import List
import numpy as np
from collections import Counter
from itertools import chain
import torch


'''
Gets current device of notebook
'''
get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"

def encode_srl_category(category_data: List[List[str]])->List[List[int]]:
  """ Encoding SRL category from a list of strings to a list of integers

  Arguments: 
    category_data (list(list(str))): SRL categories

  Returns:
    encoded category (list(list(int))): Numerical conversions of SRL categories
  """
  encoded_category = []
  for srl_list in category_data:
    encoded_srl_list = []
    for srl in srl_list:
      try:
        encoded_srl_list.append(SRL_MAP[srl])
      except:
        encoded_srl_list.append(0)
    encoded_category.append(encoded_srl_list)
  return encoded_category

def format_output_labels(token_labels, token_indices):
    """
    Returns a dictionary that has the labels (ARG0,ARG1,ARG2,TMP,LOC) as the keys, 
    with the associated value being the list of entities predicted to be of that key label. 
    Each entity is specified by its starting and ending position indicated in [token_indices].

    :param token_labels: A list of token labels 
    :type token_labels: List[String]
    :param token_indices: A list of token indices (taken from the dataset) 
                              corresponding to the labels in [token_labels].
    :type token_indices: List[int]
    """
    label_dict = {"ARG0":[], "ARG1":[], "ARG2":[], "LOC":[],"TMP":[]}
    prev_label = 'O'
    start = token_indices[0]
    for idx, label in enumerate(token_labels):
      curr_label = label.split('-')[-1]
      if label.startswith('B-') or curr_label != prev_label:
        if prev_label != 'O':
          label_dict[prev_label].append((start, token_indices[idx-1]))
        if curr_label != 'O':
          start = token_indices[idx]
        else:
          start = None
      
      prev_label = curr_label

    if start is not None and prev_label != 'O':
      label_dict[prev_label].append((start, token_indices[idx]))
    return label_dict

def mean_f1(y_pred_dict, y_true_dict):
    """
    Calculates the mean f1 score based on predictions and ground truth

    :param y_pred_dict: A list of predictions 
    :param y_true_dict: A list of the ground truths

    Returns: mean f1 score
    :type float
    """
    F1_lst = []
    for key in y_true_dict:
        TP, FN, FP = 0, 0, 0
        num_correct, num_true = 0, 0
        preds = y_pred_dict[key]
        trues = y_true_dict[key]
        for true in trues:
            num_true += 1
            if true in preds:
                num_correct += 1
            else:
                continue
        num_pred = len(preds)
        if num_true != 0:
            if num_pred != 0 and num_correct != 0:
                R = num_correct / num_true
                P = num_correct / num_pred
                F1 = 2*P*R / (P + R)
            else:
                F1 = 0      # either no predictions or no correct predictions
        else:
            continue
        F1_lst.append(F1)
    return np.mean(F1_lst)


def get_srl_frames_indices(token_labels, token_indices):
  """
    Gets the indices corresponding to their respective srl frames

    :param token_labels: A list of labels for the tokens
    :param token_indices: A list of indices of said labels' tokens

    Returns: dictionary of labels and their index spans
  """
  label_dict = {"ARGM-TMP":[], "ARG0":[], "ARG1":[], "ARG2":[], "ARGM-LOC":[]}
  prev_label = 'O'
  start = token_indices[0]

  for idx, label in enumerate(token_labels):
    curr_label = '-'.join(label.split('-')[1:]) if label != 'O' else 'O'

    if label.startswith("B-") or (curr_label != prev_label and curr_label != "O"):
      if prev_label != "O":
        label_dict[prev_label].append((start, token_indices[idx-1]))
      start = token_indices[idx]
    elif label == "O" and prev_label != "O":
      label_dict[prev_label].append((start, token_indices[idx-1]))
      start = None

    prev_label = curr_label

  if start is not None and prev_label != 'O':
    label_dict[prev_label].append((start, token_indices[idx-1]))

  return label_dict

def generate_source_corpus(source_text: List[List[str]], source_verb: List[int]):
  """
    Generates the source corpus for the SRL task 

    :param source_text: stored as as full document[sentence[word]]
    :param source_verb: the source verb for each sentence

    Returns: source corpus
  """
  assert len(source_text) == len(source_verb)
  return [[source_text[i][source_verb[i]]] + [SEPT] + [token for token in source_text[i]] + [arg] for i in range(len(source_text)) for arg in SRL_FRAMES]

def generate_target_corpus(source_text: List[List[str]], source_verb: List[int], source_srl: List[List[str]], source_indices: List[List[str]]):
  """
    Generates the target corpus for the SRL task 

    :param source_text: stored as as full document[sentence[word]]
    :param source_verb: the source verb for each sentence

    Returns: target corpus
  """
  assert len(source_text) == len(source_verb)
  assert len(source_text) == len(source_srl)

  ans = []
  for i in range(len(source_text)):
    text = source_text[i]
    verb = text[source_verb[i]]
    srl = source_srl[i]
    indices = source_indices[i]
    indice_start = indices[0]
    label_dict = get_srl_frames_indices(srl, indices)
    for key in label_dict.keys():
      arg_lst = []
      for arg_idx in label_dict[key]:
        arg_lst += text[(arg_idx[0] - indice_start):(arg_idx[1]- indice_start +1)]
      ans += [['<s>'] +[token for token in arg_lst] + ['</s>']]

  return ans

# SRL-specific utils

def convert_output(biolst, input, predict_output, tag):
  """
    Converts list of labels

    :param biolst: list of bios
    :param input: input sequence
    :param predict_output: model's predicted output
    :param tag: tag

    Returns: modified biolst
  """
  word_to_idx = {word:idx for idx, word in enumerate(input)}
  all_idx = []
  for w in predict_output:
    if w in word_to_idx:
      all_idx.append(word_to_idx[w])
  all_idx.sort()
  prev = None
  for idx in all_idx:
    if prev and prev == idx-1:
      biolst[idx] = "I-"+tag
    else:
      biolst[idx] = "B-"+tag
    prev = idx
  return biolst

def generate_predictions_using_beam_search(model, data_src):
  """
    Generates predictions on SRL model using beam search

    :param model: SRL model
    :param data_src: source of data for predictions

    Returns: predictions from model on data_src
  """
  val_pred = []

  t = len(data_src) // 5
  for i in range(t):
      input = data_src[i * 5][2:-1]
      BIOlst_output = ['O'] * len(input)
      for j in range(5):
          pos = i * 5 + j
          s = data_src[pos]
          tag = s[-1]
          result = model.beam_search(
              s,
              beam_size=16,
              max_decoding_time_step=len(s)
          )
          pred = result[0].value
          input = s[2:-1]
          BIOlst_output = convert_output(BIOlst_output, input, pred, tag)
      val_pred.append(BIOlst_output)

  return [item for sublist in val_pred for item in sublist]