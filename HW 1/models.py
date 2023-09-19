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

download("averaged_perceptron_tagger")


class HMM:
    def __init__(
        self, documents, labels, vocab, all_tags, k_t, k_e, k_s, smoothing_func
    ):
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
          smoothing_func: (Float, Dict<key Tuple[String, String] : value Float>, List[String]) ->
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

        Note: The final state "qf" can only be transitioned into, there should be no
        transitions from 'qf' to any other tag in your matrix

        Output:
          transition_matrix: Dict<key Tuple[String, String] : value Float>
        """
        # Initializing raw transition counts
        raw_transition_counts = {
            (tag1, tag2): 0 for tag1 in self.all_tags for tag2 in self.all_tags
        }
        for tag in self.all_tags:
            raw_transition_counts[(tag, "qf")] = 0

        # Updating raw transition counts, including transition to qf
        for sentence_label in self.labels:
            labels_with_qf = sentence_label + ["qf"]
            for i in range(len(labels_with_qf) - 1):
                raw_transition_counts[(labels_with_qf[i], labels_with_qf[i + 1])] += 1

        return self.smoothing_func(
            self.k_t, raw_transition_counts, self.all_tags + ["qf"]
        )

    def build_emission_matrix(self):
        """
        Returns the emission probabilities as a dictionary, mapping all possible
        (tag, token) tuple pairs to their corresponding smoothed log probabilities:
        log[P(token | tag)].

        Note: Consider all possible tokens from the list `vocab` and all tags from
        the list `all_tags`. Use the `smoothing_func` and `k_e` fields to perform smoothing.

        Note: The final state "qf" is final, as such, there should be no emissions from 'qf'
        to any token in your matrix (this includes a special end token!). This means the tag
        'qf' should not have any emissions, and thus not appear in your emission matrix.

        Output:
          emission_matrix: Dict<key Tuple[String, String] : value Float>
          Its size should be len(vocab) * len(all_tags).
        """
        # Initializing raw emission counts
        raw_emission_counts = {
            (tag, token): 0 for tag in self.all_tags for token in self.vocab
        }

        # Updating raw emission counts
        for sentence, sentence_labels in zip(self.documents, self.labels):
            for token, tag in zip(sentence, sentence_labels):
                raw_emission_counts[(tag, token)] += 1

        return self.smoothing_func(self.k_e, raw_emission_counts, self.vocab)

    def get_start_state_probs(self):
        """
        Returns the starting state probabilities as a dictionary, mapping all possible
        tags to their corresponding smoothed log probabilities. Use `k_s` smoothing
        parameter to manually perform smoothing.

        Note: Do NOT use the `smoothing_func` function within this method since
        `smoothing_func` is designed to smooth state-observation counts. Manually
        implement smoothing here.

        Note: The final state "qf" can only be transitioned into, as such, there should be no
        transitions from 'qf' to any token in your matrix. This means the tag 'qf' should
        not be able to start a sequence, and thus not appear in your start state probs.

        Output:
          start_state_probs: Dict<key String : value Float>
        """
        # Initialize raw start state counts
        raw_start_state_counts = {tag: 0 for tag in self.all_tags}

        # Updating raw start state counts
        for sentence_labels in self.labels:
            raw_start_state_counts[sentence_labels[0]] += 1

        # Manual smoothing
        smoothed_log_probs = {
            tag: np.log((freq + self.k_s) / (len(self.labels) * (self.k_s + 1)))
            for tag, freq in raw_start_state_counts.items()
        }
        return smoothed_log_probs

    def get_trellis_arc(self, predicted_tag, previous_tag, document, i):
        """
        Returns the trellis arc used by the Viterbi algorithm for the label
        `predicted_tag` conditioned on the `previous_tag` and `document` at index `i`.

        For HMM, this would be the sum of the smoothed log emission probabilities and
        log transition probabilities:
        log[P(predicted_tag | previous_tag))] + log[P(document[i] | predicted_tag)].

        Note: Treat unseen tokens as an <unk> token.
        Note: Make sure to handle the case where we are dealing with the first word. Is there a transition probability for this case?

        Note: Make sure to handle the case where predicted_tag is 'qf'. This corresponds to predicting the last token for a sequence.
        We can transition into this tag, but (as per our emission matrix spec), there should be no emissions leaving.
        As such, our probability when predicted_tag = 'qf' should merely be log[P(predicted_tag | previous_tag))].

        Input:
          predicted_tag: String, predicted tag for token at index `i` in `document`
          previous_tag: String, previous tag for token at index `i` - 1
          document: List[String]
          i: Int, index of the `document` to compute probabilities
        Output:
          result: Float
        """
        # Calculating log[P(predicted_tag | previous_tag))] from the transition matrix
        log_transition_prob = (
            self.transition_matrix[(previous_tag, predicted_tag)]
            if i != 0
            else self.start_state_probs[predicted_tag]
        )

        # Calculating log[P(document[i] | predicted_tag)] from the emission matrix
        log_emission_prob = (
            self.emission_matrix[
                (predicted_tag, document[i] if document[i] in self.vocab else "<unk>")
            ]
            if predicted_tag != "qf"
            else 0
        )

        # Summing them up
        return log_transition_prob + log_emission_prob


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
        # Construction set of features for the token at document[i]
        features_dict = {
            "POS_TAG": pos_tag([document[i]])[0][1],
            "Is_CAP": 1 if document[i][0].isupper() else 0,
            # "Is_FIRST": 1 if i == 0 else 0,
            # "Special_CHAR": 1 if not document[i].isalpha() else 0,
            "TOKEN_FREQ": document.count(document[i]),
            "TOKEN_LEN": len(document[i]),
            "PREV_TAG_NOT_O": previous_tag != "O" if previous_tag else False,
        }

        return features_dict

    def generate_classifier(self):
        """
        Returns a trained MaxEnt classifier for the MEMM model on the featurized tokens.
        Use `extract_features_token` to extract features per token.

        Output:
          classifier: nltk.classify.maxent.MaxentClassifier
        """
        # Featurizing tokens in documents
        featurized_tokens = [
            (
                self.extract_features_token(self.documents[i], j, None),
                self.labels[i][j],
            )
            for i in range(len(self.documents))
            for j in range(len(self.documents[i]))
        ]

        # Return trained classifier
        return classify.MaxentClassifier.train(
            train_toks=featurized_tokens, max_iter=10
        )

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
        if predicted_tag == "qf":
            return 0

        # Extracting features
        features = self.extract_features_token(document, i, previous_tag)

        # Calculating log[P(predicted_tag|features_i)] from the classifier
        return self.classifier.prob_classify(features).logprob(predicted_tag)
