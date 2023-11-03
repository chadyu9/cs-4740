import gensim.downloader as api
import torch
import numpy as np

def load_gensim_embeddings(model_name="glove-wiki-gigaword-300"):
    """
    Load embeddings using gensim.

    :param model_name: Name of the pre-trained model to load.
    :return: Loaded model.
    """
    return api.load(model_name)

def create_vocab_embeddings(model, vocab):
    """
    Create embeddings for the given vocabulary using the provided model.

    :param model: Pre-trained embeddings model.
    :param vocab: Vocabulary for which embeddings are to be created.
    :return: Tensor containing embeddings for the vocabulary.
    """
    embeddings = []
    for word in vocab.word2id.keys():
        try: 
            embeddings.append(model[word])
        except:
            embeddings.append(torch.rand(300))

    embeddings = np.stack(embeddings, 0)
    return torch.from_numpy(embeddings)
