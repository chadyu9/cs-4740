# Name(s):
# Netid(s):
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import numpy as np


def viterbi(model, observation, tags):
    """
    Returns the model's predicted tag sequence for a particular observation.
    Use `get_trellis_arc` method to obtain model scores at each iteration.

    Input:
      model: HMM/MEMM model
      observation: List[String]
      tags: List[String]
    Output:
      predictions: List[String]
    """
    # Initialize prediction tags, Viterbi matrix, and backpointer matrix
    predictions = []
    vb = np.zeros((len(tags), len(observation)), dtype=int)
    v = np.zeros((len(tags), len(observation)))
    v[:, 0] = [
        model.get_trellis_arc(tags[j], None, observation, 0) for j in range(len(tags))
    ]

    # Compute Viterbi matrix and associated predictions for middle transitions
    for i in range(1, len(observation)):
        for j in range(len(tags)):
            # Assign the backpointer first
            vb[j, i] = int(
                np.argmax(
                    [
                        v[k, i - 1]
                        + model.get_trellis_arc(tags[j], tags[k], observation, i)
                        for k in range(len(tags))
                    ]
                )
            )
            # Evaluate the recurrence at the vb[j, i] index of the v matrix
            v[j, i] = v[vb[j, i], i - 1] + model.get_trellis_arc(
                tags[j], tags[vb[j, i]], observation, i
            )
        # Append the tag associated with the backpointer of the greatest v value in the column
        predictions.append(tags[vb[np.argmax(v[:, i]), i]])

    # Compute final prediction
    predictions.append(
        tags[
            np.argmax(
                [
                    v[j, len(observation) - 1]
                    + model.get_trellis_arc(
                        "qf", tags[j], observation, len(observation)
                    )
                    for j in range(len(tags))
                ]
            )
        ]
    )

    return predictions
