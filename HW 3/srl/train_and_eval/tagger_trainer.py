import torch
from tqdm import tqdm, trange
from srl.utils.vocab import Vocab
import torch.optim as optim
import math
import numpy as np
import torch.nn as nn

from srl.utils.srl_utils import *

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of input sentence, verb indices, target output labels 
    :param data: list of tuples containing source and target sentence. ie.
        (list of (src_sent, tgt_sent))
    :type data: List[Tuple[List[str], List[str], List[str]]]
    :param batch_size: batch size
    :type batch_size: int
    :param shuffle: whether to randomly shuffle the dataset
    :type shuffle: boolean
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        srl = [e[0] for e in examples]
        verb_index = [e[1] for e in examples]
        target = [e[2] for e in examples]

        yield srl, verb_index, target


def evaluation(model, val_data, optimizer, criterion, batch_size=64):
  """ Evaluate loss on val sentences using LSTMTagger model
    :param model: LSTMTagger Model
    :type model: LSTMTagger
    :param val_data: List[Tuple[List[str], List[str]]]
    :param criterion: loss criterion
    :type batch_size: int
    :returns loss
  """
  model.eval()
  loss = 0
  correct = 0
  total = 0
  batch = 0
  for (input_batch, verb_indices, expected_out) in tqdm(batch_iter(val_data, batch_size=batch_size, shuffle=True)):
    output = model.forward(input_batch, torch.tensor(verb_indices).to(get_device()))
    total += output.size()[0] * output.size()[1]
    _, predicted = torch.max(output, 2)
    expected_out = torch.tensor(Vocab.pad_sents(expected_out))
    correct += (expected_out.to("cpu") == predicted.to("cpu")).cpu().numpy().sum()

    loss += model.compute_Loss(criterion, output.to("cpu"), expected_out.to("cpu"))
    batch += 1
  loss /= batch
  print("Validation Loss: " + str(loss.item()))
  print("Validation Accuracy: " + str(correct/total))
  print()
  return loss.item()

def train_epoch(model, train_data, optimizer, criterion, batch_size=64):
  """ trains LSTMTagger model for singular epoch
    :param model: LSTMTagger Model
    :type model: LSTMTagger
    :param train_data: List[Tuple[List[str], List[str]]]
    :param optimizer: optimizer for model
    :param criterion: loss criterion
    :type batch_size: int
    :returns average loss over the batch
  """
  model.train()
  total = 0
  batch = 0
  total_loss = 0
  correct = 0
  for (input_batch, verb_indices, expected_out) in tqdm(batch_iter(train_data, batch_size=batch_size, shuffle=True)):
    optimizer.zero_grad()
    batch += 1
    output = model.forward(input_batch, torch.tensor(verb_indices).to(get_device()))
    total += output.size()[0] * output.size()[1]
    _, predicted = torch.max(output, 2)

    expected_out = torch.tensor(Vocab.pad_sents(expected_out))
    correct += (expected_out.to("cpu") == predicted.to("cpu")).cpu().numpy().sum()
    
    loss = model.compute_Loss(criterion, output.to("cpu"), expected_out.to("cpu")) 
    total_loss += loss.item()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step() 
  print("Loss: " + str(total_loss/batch))
  print("Training Accuracy: " + str(correct/total))
  return total_loss/batch

def tagger_train_and_evaluate(number_of_epochs, model, train_data, val_data, criterion, min_loss=0, lr=.01):
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.9)
  loss_values = [[],[]]
  for epoch in trange(number_of_epochs, desc="Epochs"):
    cur_loss = train_epoch(model, train_data, optimizer, criterion)
    loss_values[0].append(cur_loss)
    cur_loss_val = evaluation(model, val_data, optimizer, criterion)
    loss_values[1].append(cur_loss_val)
    if cur_loss <= min_loss: return loss_values
  return loss_values