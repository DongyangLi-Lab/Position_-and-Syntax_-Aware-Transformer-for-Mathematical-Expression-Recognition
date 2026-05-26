import subprocess
from threading import Timer
import nltk
import Levenshtein
import numpy as np
import tabulate
import torch
import yaml


def run(cmd, timeout_sec):
    """Run a shell command with timeout."""
    proc = subprocess.Popen(cmd, shell=True)

    def kill_proc(p):
        p.kill()

    timer = Timer(timeout_sec, kill_proc, [proc])

    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()


def load_config(config_file, arguments=None):
    """Load yaml config and apply command-line overrides."""
    with open(config_file) as f:
        config = yaml.load(f, yaml.loader.SafeLoader)

    config["device"] = choose_device()

    if arguments is not None:
        config["arguments"] = vars(arguments)
        update_config_from_args(config)

    if "train" in config["dataset"]:
        batch_size = config["dataset"]["train"]["batch_size"]
        config["train"]["init_lr"] *= batch_size / 48

    return config


def choose_device():
    """Select cuda, mps, or cpu."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("devices: ")

    if device == "cuda":
        device_info = [[
            device,
            torch.cuda.get_device_name(device),
            torch.cuda.get_device_properties(device)
        ]]
        print(tabulate.tabulate(
            device_info,
            headers=["id", "type", "properties"],
            tablefmt="fancy_outline"
        ))
    else:
        print(device)

    return device


def update_config_from_args(config):
    """Apply optional CLI arguments to config."""
    args = config["arguments"]

    if args.get("num_workers"):
        num_workers = args["num_workers"]

        config["dataset"]["train"]["num_workers"] = num_workers
        config["dataset"]["val"]["num_workers"] = num_workers

        if "test" in config["dataset"]:
            config["dataset"]["test"]["num_workers"] = num_workers

    if args.get("test_set"):
        config["dataset"]["test"]["files"] = {
            "test": [args["test_set"]]
        }


def load_formulas(filename):
    """
    Load formula file.

    Expected line format:
        image_name.png: token token token ...
    """
    formulas = {}

    with open(filename) as f:
        for line in f:
            if ".png: " not in line:
                continue

            image_name, formula = line.split(".png: ", 1)
            formula = formula.rstrip("\n")

            if " style: " in formula:
                formula = formula.split(" style: ", 1)[0]

            if not image_name.endswith(".png"):
                image_name = image_name + ".png"

            formulas[image_name] = formula.split(" ")

    print(f"Loaded {len(formulas)} formulas from {filename}")
    return formulas


def score_files(path_ref, path_hyp, more_information=False):
    references = load_formulas(path_ref)
    hypotheses = load_formulas(path_hyp)

    assert len(references) == len(hypotheses), \
        f"Reference count {len(references)} != prediction count {len(hypotheses)}"

    return {
        "BLEU-4": compute_bleu4(references, hypotheses) * 100,
        "Edit": compute_edit_similarity(references, hypotheses) * 100
    }


def compute_edit_similarity(references, hypotheses):
    """
    Compute normalized Levenshtein edit similarity.

    Formula:
        1 - total_edit_distance / total_max_length

    Example:
        1.0 means perfect.
        0.9 means about 90% edit similarity.
    """
    total_distance = 0
    total_length = 0.0

    for image_name, pred_tokens in hypotheses.items():
        ref_tokens = references[image_name]

        distance = Levenshtein.distance(ref_tokens, pred_tokens)
        length = max(len(ref_tokens), len(pred_tokens))

        total_distance += distance
        total_length += float(length)

    if total_length == 0:
        return 0.0

    return 1.0 - total_distance / total_length

def compute_bleu4(references, hypotheses):
    """
    Compute corpus BLEU-4 score.

    references:
        dict[image_name] -> list[token]

    hypotheses:
        dict[image_name] -> list[token]
    """
    refs = [[references[image_name]] for image_name in hypotheses.keys()]
    hyps = list(hypotheses.values())

    return nltk.translate.bleu_score.corpus_bleu(
        refs,
        hyps,
        weights=(0.25, 0.25, 0.25, 0.25)
    )

def get_accuracy(labels, predictions, end_number):
    """
    Compute token accuracy before END token.

    labels:      [B, T]
    predictions: list/tensor of predicted token ids
    end_number:  id of END token
    """
    batch_acc = []

    for batch_i in range(len(labels)):
        gt_end_positions = (labels[batch_i] == end_number).nonzero(as_tuple=True)[0] - 1
        pred_end_positions = (predictions[batch_i] == end_number).nonzero(as_tuple=True)[0]

        if predictions[batch_i].shape != torch.Size([]):
            pred_len = torch.tensor([len(predictions[batch_i])]).to(labels.device)
        else:
            pred_len = torch.tensor([0]).to(labels.device)

        if len(gt_end_positions) > 0:
            gt_end = gt_end_positions[0]
        else:
            gt_end = torch.tensor([len(labels[batch_i]) - 1]).to(labels.device)

        if len(pred_end_positions) > 0:
            pred_end = pred_end_positions[0]
        else:
            pred_end = torch.tensor([len(predictions[batch_i])]).to(labels.device)

        max_end = torch.max(gt_end, pred_end)
        valid_end = torch.min(max_end, pred_len)

        acc = torch.true_divide(
            torch.sum(predictions[batch_i][:valid_end] == labels[batch_i][1:valid_end + 1]),
            max_end
        )

        if acc.shape != torch.Size([]):
            batch_acc.append(acc)
        else:
            batch_acc.append(torch.tensor([acc]).to(labels.device))

    return torch.mean(torch.stack(batch_acc)) * 100


def remove_style(predictions):
    """
    Remove simple LaTeX style wrappers from tokenized predictions.
    """
    for i, prediction in enumerate(predictions):
        cleaned_tokens = []
        skip_count = 0

        for token_i, token in enumerate(prediction):
            if skip_count > 0:
                skip_count -= 1
                continue

            if token == "\\boldsymbol":
                close_pos = find_closing_bracket(prediction[token_i + 1:])
                cleaned_tokens += prediction[token_i + 1:token_i + 1 + close_pos]
                skip_count = close_pos
                continue

            if "\\mathcal" in token:
                cleaned_tokens.append(
                    token.replace("\\mathcall{", "").replace("}", "")
                )
                continue

            if "\\mathbb" in token:
                cleaned_tokens.append(
                    token.replace("\\mathbb{", "").replace("}", "")
                )
                continue

            cleaned_tokens.append(token)

        predictions[i] = cleaned_tokens

    return predictions


def find_closing_bracket(formula):
    """
    Find the index right after the matching closing bracket.
    """
    open_brackets = 0
    closed_brackets = 0

    for i, token in enumerate(formula):
        if token == "{":
            open_brackets += 1
        elif token == "}":
            closed_brackets += 1

        if open_brackets == closed_brackets:
            return i + 1

    return len(formula)


def gen_counting_label(labels, channel, ignore):
    """
    Generate counting labels for token-count supervision.
    """
    batch_size, seq_len = labels.size()
    device = labels.device

    counting_labels = torch.zeros((batch_size, channel))

    for i in range(batch_size):
        for j in range(seq_len):
            token_id = labels[i][j]

            if token_id in ignore:
                continue

            counting_labels[i][token_id] += 1

    return counting_labels.to(device)