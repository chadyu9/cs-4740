# Name(s):
# Netid(s):
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
from collections import defaultdict
from nltk import classify
from nltk import download
from nltk import pos_tag
import numpy as np

download('averaged_perceptron_tagger')

class HMM: 

  def __init__(self, documents, labels, vocab, all_tags, k_t, k_e, k_s, smoothing_func): 
    """
    Initializes HMM based on the following properties.

    Input:
      documents: List[List[String]], dataset of sentences to train model
      labels: List[List[String]], NER labels corresponding the sentences to train model
      vocab: List[String], dataset vocabulary
      all_tags: List[String], all possible NER tags 
      k_t: Float, add-k parameter to smooth transition probabilities
      k_e: Float, add-k parameter to smooth emission probabilities
      k_s: Float, add-k parameter to smooth starting state probabilities
      smoothing_func: (Dict<key Tuple[String, String] : value Float>, Float) -> 
      Dict<key Tuple[String, String] : value Float>
    """
    self.documents = documents
    self.labels = labels
    self.vocab = vocab
    self.all_tags = all_tags
    self.k_t = k_t
    self.k_e = k_e
    self.k_s = k_s
    self.smoothing_func = smoothing_func
    self.emission_matrix = self.build_emission_matrix()
    self.transition_matrix = self.build_transition_matrix()
    self.start_state_probs = self.get_start_state_probs()


  def build_transition_matrix(self):
    """
    Returns the transition probabilities as a dictionary mapping all possible
    (tag_{i-1}, tag_i) tuple pairs to their corresponding smoothed 
    log probabilities: log[P(tag_i | tag_{i-1})]. 
    
    Note: Consider all possible tags. This consists of everything in 'all_tags', but also 'qf' our end token.
    Use the `smoothing_func` and `k_t` fields to perform smoothing.

    Output: 
      transition_matrix: Dict<key Tuple[String, String] : value Float>
    """
    # YOUR CODE HERE
    raise NotImplementedError()


  def build_emission_matrix(self): 
    """
    Returns the emission probabilities as a dictionary, mapping all possible 
    (tag, token) tuple pairs to their corresponding smoothed log probabilities: 
    log[P(token | tag)]. 
    
    Note: Consider all possible tokens from the list `vocab` and all tags from 
    the list `all_tags`. Use the `smoothing_func` and `k_e` fields to perform smoothing.
  
    Output:
      emission_matrix: Dict<key Tuple[String, String] : value Float>
      Its size should be len(vocab) * len(all_tags).
    """
    # YOUR CODE HERE
    raise NotImplementedError()


  def get_start_state_probs(self):
    """
    Returns the starting state probabilities as a dictionary, mapping all possible 
    tags to their corresponding smoothed log probabilities. Use `k_s` smoothing
    parameter to manually perform smoothing.
    
    Note: Do NOT use the `smoothing_func` function within this method since 
    `smoothing_func` is designed to smooth state-observation counts. Manually
    implement smoothing here.

    Output: 
      start_state_probs: Dict<key String : value Float>
    """
    # YOUR CODE HERE 
    raise NotImplementedError()


  def get_trellis_arc(self, predicted_tag, previous_tag, document, i): 
    """
    Returns the trellis arc used by the Viterbi algorithm for the label 
    `predicted_tag` conditioned on the `previous_tag` and `document` at index `i`.
    
    For HMM, this would be the sum of the smoothed log emission probabilities and 
    log transition probabilities: 
    log[P(predicted_tag | previous_tag))] + log[P(document[i] | predicted_tag)].
    
    Note: Treat unseen tokens as an <unk> token.
    Note: Make sure to handle the case where we are dealing with the first word. Is there a transition probability for this case?
    Note: Make sure to handle the case where the predicted tag is an end token. Is there an emission probability for this case?
  
    Input: 
      predicted_tag: String, predicted tag for token at index `i` in `document`
      previous_tag: String, previous tag for token at index `i` - 1
      document: List[String]
      i: Int, index of the `document` to compute probabilities 
    Output: 
      result: Float
    """
    # YOUR CODE HERE 
    raise NotImplementedError()

 

################################################################################
################################################################################



class MEMM: 

  def __init__(self, documents, labels): 
    """
    Initializes MEMM based on the following properties.

    Input:
      documents: List[List[String]], dataset of sentences to train model
      labels: List[List[String]], NER labels corresponding the sentences to train model
    """
    self.documents = documents
    self.labels = labels
    self.classifier = self.generate_classifier()


  def extract_features_token(self, document, i, previous_tag):
    """
    Returns a feature dictionary for the token at document[i].

    Input: 
      document: List[String], representing the document at hand
      i: Int, representing the index of the token of interest
      previous_tag: string, previous tag for token at index `i` - 1

    Output: 
      features_dict: Dict<key String: value Any>, Dictionaries of features 
                    (e.g: {'Is_CAP':'True', . . .})
    """
    features_dict = {}

    # YOUR CODE HERE 
    ### TODO: ADD FEATURES
    raise NotImplementedError()


  def generate_classifier(self):
    """
    Returns a trained MaxEnt classifier for the MEMM model on the featurized tokens.
    Use `extract_features_token` to extract features per token.

    Output: 
      classifier: nltk.classify.maxent.MaxentClassifier 
    """
    # YOUR CODE HERE 
    raise NotImplementedError()


  def get_trellis_arc(self, predicted_tag, previous_tag, document, i): 
    """
    Returns the trellis arc used by the Viterbi algorithm for the label 
    `predicted_tag` conditioned on the features of the token of `document` at 
    index `i`.
    
    For MEMM, this would be the log classifier output log[P(predicted_tag | features_i)].
  
    Input: 
      predicted_tag: string, predicted tag for token at index `i` in `document`
      previous_tag: string, previous tag for token at index `i` - 1
      document: string
      i: index of the `document` to compute probabilities 
    Output: 
      result: Float
    """
    # YOUR CODE HERE 
    raise NotImplementedError()
