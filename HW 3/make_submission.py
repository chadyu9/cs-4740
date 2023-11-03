import argparse
import os
import json
import csv

from srl.utils.constants import SRL_MAP
from srl.utils.srl_utils import *
from srl.utils.vocab import Vocab
import subprocess
import importlib
import sys


def load_SRL(model_path: str):
    """Load the model from a file.
    @param model_path (str): path to model
    """
    sys.path.append(os.getcwd() + "/srl")
    sys.path.append(os.getcwd())
    srl_class = importlib.import_module("SRL")
    SRL = getattr(srl_class, "SRL")
    params = torch.load(
        model_path,
        map_location=torch.device(get_device()),
    )
    args = params["args"]
    model = SRL(
        src_vocab=params["vocab"]["source"], tgt_vocab=params["vocab"]["target"], **args
    )
    model.load_state_dict(params["state_dict"])
    return model


def hash_preds(pred_dict, netIDOffset):
    new_dict = {}
    for key in pred_dict.keys():
        lst = pred_dict[key]
        for val_i in range(len(lst)):
            new_tuple = (lst[val_i][0] + netIDOffset, lst[val_i][1] + netIDOffset)
            lst[val_i] = new_tuple
        new_dict[key] = lst

    return new_dict


def string_to_int(netid):
    result = 0
    split_netID = netid.split("_")
    first_netID = None
    second_netID = None
    if len(split_netID) > 1:
        first_netID, second_netID = split_netID[0], split_netID[1]
    else:
        first_netID = split_netID[0]
    for char in first_netID:
        ascii_value = ord(char)

        # Check if the character is an integer
        if char.isdigit():
            result += int(char)
        else:
            result += ascii_value

    if len(split_netID) > 1:
        for char in second_netID:
            ascii_value = ord(char)

            # Check if the character is an integer
            if char.isdigit():
                result += int(char)
            else:
                result += ascii_value
    return result


def make_predictions(test_data, lstm):
    inv_srl_map = {SRL_MAP[key]: key for key in SRL_MAP}
    test_predict = []
    test_idx = []

    for idx in range(len(test_data)):
        out = lstm.forward(
            [test_data[idx][0]],
            torch.tensor([test_data[idx][1]], device=torch.device(get_device())),
        )
        _, predicted = torch.max(out, 2)

        len_sent = len(test_data[idx][0])
        result = predicted.cpu().numpy()[0]

        for t in range(len_sent):
            test_predict.append(inv_srl_map[result[t]])

        test_idx.extend(test["words_indices"][idx])

    y_pred_dict = format_output_labels(test_predict, test_idx)

    return y_pred_dict


def create_submission(output_filepath, label_dict, netid):
    """
    :parameter output_filepath: The full path (including file name) of the output file,
                                with extension .csv
    :type output_filepath: [String]
    :parameter token_labels: A list of token labels (eg. PER, LOC, ORG or MISC).
    :type token_labels: List[String]
    :parameter token_indices: A list of token indices (taken from the dataset)
                              corresponding to the labels in [token_labels].
    :type token_indices: List[int]
    """
    with open(output_filepath, mode="w") as csv_file:
        fieldnames = ["Id", "Predicted"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"Id": netid})
        for key in label_dict:
            p_string = " ".join(
                [str(start) + "-" + str(end) for start, end in label_dict[key]]
            )
            writer.writerow({"Id": key, "Predicted": p_string})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hashes predictions")
    parser.add_argument(
        "--lstm_weights",
        type=str,
        required=True,
        help="Path to your lstm weights. Please make sure when saving your model weights you call \{model\}.save({name of pth file})",
    )
    parser.add_argument("--srl_weights", type=str, help="Path to your srl weights")
    parser.add_argument("--netid", type=str, required=True, help="Your NetID")
    args = parser.parse_args()

    print("Running submission script...")

    sys.path.append(os.getcwd() + "/srl/models")
    lstm_class = importlib.import_module("lstm_tagger")
    LSTMTagger = getattr(lstm_class, "LSTMTagger")

    with open(os.getcwd() + "/dataset/test.json", "r") as f:
        test = json.loads(f.read())

    test_data = list(zip(test["text"], test["verb_index"]))

    ### LSTM STUFF

    model = torch.load(args.lstm_weights, map_location=torch.device(get_device()))

    y_pred_dict = make_predictions(test_data=test_data, lstm=model)

    netIDOffset = string_to_int(args.netid)

    hashed_y_pred_dict = hash_preds(y_pred_dict, netIDOffset)

    create_submission("output_lstm_hashed.csv", hashed_y_pred_dict, args.netid)

    print("output written to output_lstm_hashed.csv")

    if args.srl_weights != None:
        test_data_src = generate_source_corpus(test["text"], test["verb_index"])
        model = load_SRL(args.srl_weights)
        preds = generate_predictions_using_beam_search(model, test_data_src)
        test_idx = []
        for i in test["words_indices"]:
            test_idx.extend(i)

        pred_dict = format_output_labels(preds, test_idx)
        hashed_pred_dict = hash_preds(pred_dict, netIDOffset)
        create_submission("output_srl_hashed.csv", hashed_pred_dict, args.netid)
        print("output written to output_srl_hashed.csv")
        subprocess.run(
            [
                "zip",
                "-r",
                "submission.zip",
                "output_lstm_hashed.csv",
                "output_srl_hashed.csv",
                args.srl_weights,
                args.lstm_weights,
                "srl",
            ],
            capture_output=True,
        )
    else:
        subprocess.run(
            [
                "zip",
                "-r",
                "submission.zip",
                "output_lstm_hashed.csv",
                args.lstm_weights,
                "srl",
            ],
            capture_output=True,
        )

    print("Files are zipped to submission.zip")
