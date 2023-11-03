import math
import numpy as np
import torch
import sys
import time
from tqdm import tqdm

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    :param data: list of tuples containing source and target sentence. ie.
        (list of (src_sent, tgt_sent))
    :type data: List[Tuple[List[str], List[str]]]
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
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

def evaluate_ppl(model, val_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    :param model: SRL Model
    :type model: SRL
    :param dev_data: list of tuples containing source and target sentence.
        i.e. (list of (src_sent, tgt_sent))
    :param val_data: List[Tuple[List[str], List[str]]]
    :param batch_size: size of batches to extract
    :type batch_size: int
    :returns ppl: perplixty on val sentences
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(val_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl

def srl_train_and_evaluate(model, train_data, val_data, optimizer, train_batch_size=32, clip_grad=2, log_every = 100, 
                       valid_niter = 500, model_save_path="srl.ckpt", num_epoch=6):
    num_trail = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0

    print('begin Maximum Likelihood training')
    train_time = begin_time = time.time()

    val_data_tgt = [tgt for _, tgt in val_data]

    for epoch in tqdm(range(num_epoch)):
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            
            optimizer.zero_grad()
            
            batch_size = len(src_sents)
            
            example_losses = -model(src_sents, tgt_sents)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size
            loss.backward()
            
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val
            
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                        'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                            report_loss / report_examples,
                                                                                            math.exp(report_loss / report_tgt_words),
                                                                                            cum_examples,
                                                                                            report_tgt_words / (time.time() - train_time),
                                                                                            time.time() - begin_time))
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

                

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                            cum_loss / cum_examples,
                                                                                            np.exp(cum_loss / cum_tgt_words),
                                                                                            cum_examples))
                
                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl 
                dev_ppl = evaluate_ppl(model, val_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl


                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
