import argparse
import pickle
import re
import string

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from chunker import Chunker
from config import data_config, annotation_config, device
from models import EPRModel
from util import example_to_device, load_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("local", "global", "concat"))
    parser.add_argument("--dataset", default="snli")
    args = parser.parse_args()
    return args.mode, args.dataset


def num_true_pos(retrieved: list, relevant: list):
    return len(set(retrieved).difference(set(relevant)))


def phrases2word_indices(substr, sentence):
    chunker = Chunker()
    doc = chunker.get_doc(sentence)
    phrases = [
        phrase.translate(str.maketrans("", "", string.punctuation)).strip()
        for phrase in substr.split("\u2022")
    ]
    num_words = len(doc)
    indices = []
    for phrase in phrases:
        for i in range(num_words - 1):
            for j in range(1, num_words):
                if doc[i:j].text == phrase:
                    indices += list(range(i, j))

    return indices


def get_phrases_from_model(model: EPRModel, ex: dict):
    phrases = {
        "UP": [],
        "UH": [],
        "EP": [],
        "EH": [],
        "CP": [],
        "CH": [],
        "NP": [],
        "NH": [],
    }
    ex = example_to_device(ex, device)
    p_phrases_idx = ex["p_phrases_idx"]
    h_phrases_idx = ex["h_phrases_idx"]
    phrase_probs, _, _ = model.predict_phrasal_label(ex)

    for key in phrase_probs.keys():
        p, h = key
        if h is None:
            phrases["UP"] += list(range(*p_phrases_idx[p]))
        elif p is None:
            phrases["UH"] += list(range(*h_phrases_idx[h]))
        else:
            label = torch.argmax(phrase_probs[p, h], -1)
            if label == 0:
                phrases["EP"] += list(range(*p_phrases_idx[p]))
                phrases["EH"] += list(range(*h_phrases_idx[h]))
            elif label == 1:
                phrases["CP"] += list(range(*p_phrases_idx[p]))
                phrases["CH"] += list(range(*h_phrases_idx[h]))
            elif label == 2:
                phrases["NP"] += list(range(*p_phrases_idx[p]))
                phrases["NH"] += list(range(*h_phrases_idx[h]))

    return phrases


def get_phrases_from_annotation(annotation: dict, ex: dict):
    premise, hypothesis = ex["p_sent"], ex["h_sent"]

    return {
        "UP": phrases2word_indices(annotation["UP"], premise),
        "UH": phrases2word_indices(annotation["UH"], hypothesis),
        "EP": phrases2word_indices(annotation["EP"], premise),
        "EH": phrases2word_indices(annotation["EH"], hypothesis),
        "CP": phrases2word_indices(annotation["CP"], premise),
        "CH": phrases2word_indices(annotation["CH"], hypothesis),
        "NP": phrases2word_indices(annotation["NP"], premise),
        "NH": phrases2word_indices(annotation["NH"], hypothesis),
    }


def sentence_accuracy(model: EPRModel, test_dl: DataLoader):
    model.eval()
    hit_count = 0
    len_test = len(test_dl)
    pbar = tqdm(iter(test_dl))
    for i, test_ex in enumerate(pbar):
        input = example_to_device(test_ex, device)
        test_label = input["label"]

        with torch.no_grad():
            pred = model(input)
            hit_count += int(torch.argmax(pred, dim=-1) == test_label)

        pbar.set_description(f"Test accuracy: {hit_count/(i+1)}")

    return hit_count / len_test


def evaluation(model, test_ds, annotations_set):
    F_E, F_C, F_N, F_UP, F_UH = 0, 0, 0, 0, 0

    for annotations in annotations_set:
        num_true_pos_EP, num_retrieved_EP, num_relevant_EP = 0, 0, 0
        num_true_pos_CP, num_retrieved_CP, num_relevant_CP = 0, 0, 0
        num_true_pos_NP, num_retrieved_NP, num_relevant_NP = 0, 0, 0
        num_true_pos_UP, num_retrieved_UP, num_relevant_UP = 0, 0, 0
        num_true_pos_EH, num_retrieved_EH, num_relevant_EH = 0, 0, 0
        num_true_pos_CH, num_retrieved_CH, num_relevant_CH = 0, 0, 0
        num_true_pos_NH, num_retrieved_NH, num_relevant_NH = 0, 0, 0
        num_true_pos_UH, num_retrieved_UH, num_relevant_UH = 0, 0, 0

        for annotation in annotations:
            id = annotation["snli_id"]
            ex = test_ds[id]

            model_result = get_phrases_from_model(model, ex)
            annotated_result = get_phrases_from_annotation(annotation, ex)
            num_retrieved_EP += len(model_result["EP"])
            num_relevant_EP += len(annotated_result["EP"])
            num_true_pos_EP += num_true_pos(model_result["EP"], annotated_result["EP"])

            num_retrieved_CP += len(model_result["CP"])
            num_relevant_CP += len(annotated_result["CP"])
            num_true_pos_CP += num_true_pos(model_result["CP"], annotated_result["CP"])

            num_retrieved_NP += len(model_result["NP"])
            num_relevant_NP += len(annotated_result["NP"])
            num_true_pos_NP += num_true_pos(model_result["NP"], annotated_result["NP"])

            num_retrieved_UP += len(model_result["UP"])
            num_relevant_UP += len(annotated_result["UP"])
            num_true_pos_UP += num_true_pos(model_result["UP"], annotated_result["UP"])

            num_retrieved_EH += len(model_result["EH"])
            num_relevant_EH += len(annotated_result["EH"])
            num_true_pos_EH += num_true_pos(model_result["EH"], annotated_result["EH"])

            num_retrieved_CH += len(model_result["CH"])
            num_relevant_CH += len(annotated_result["CH"])
            num_true_pos_CH += num_true_pos(model_result["CH"], annotated_result["CH"])

            num_retrieved_NH += len(model_result["NH"])
            num_relevant_NH += len(annotated_result["NH"])
            num_true_pos_NH += num_true_pos(model_result["NH"], annotated_result["NH"])

            num_retrieved_UH += len(model_result["UH"])
            num_relevant_UH += len(annotated_result["UH"])
            num_true_pos_UH += num_true_pos(model_result["UH"], annotated_result["UH"])

        precision_EP = (
            num_true_pos_EP / num_retrieved_EP if num_retrieved_EP != 0 else 0
        )
        recall_EP = num_true_pos_EP / num_relevant_EP if num_relevant_EP != 0 else 0

        precision_CP = (
            num_true_pos_CP / num_retrieved_CP if num_retrieved_CP != 0 else 0
        )
        recall_CP = num_true_pos_CP / num_relevant_CP if num_relevant_CP != 0 else 0

        precision_NP = (
            num_true_pos_NP / num_retrieved_NP if num_retrieved_NP != 0 else 0
        )
        recall_NP = num_true_pos_NP / num_relevant_NP if num_relevant_NP != 0 else 0

        precision_UP = (
            num_true_pos_UP / num_retrieved_UP if num_retrieved_UP != 0 else 0
        )
        recall_UP = num_true_pos_UP / num_relevant_UP if num_relevant_UP != 0 else 0

        precision_EH = (
            num_true_pos_EH / num_retrieved_EH if num_retrieved_EH != 0 else 0
        )
        recall_EH = num_true_pos_EH / num_relevant_EH if num_relevant_EH != 0 else 0

        precision_CH = (
            num_true_pos_CH / num_retrieved_CH if num_retrieved_CH != 0 else 0
        )
        recall_CH = num_true_pos_CH / num_relevant_CH if num_relevant_CH != 0 else 0

        precision_NH = (
            num_true_pos_NH / num_retrieved_NH if num_retrieved_NH != 0 else 0
        )
        recall_NH = num_true_pos_NH / num_relevant_NH if num_relevant_NH != 0 else 0

        precision_UH = (
            num_true_pos_UH / num_retrieved_UH if num_retrieved_UH != 0 else 0
        )
        recall_UH = num_true_pos_UH / num_relevant_UH if num_relevant_UH != 0 else 0

        precision_E = (precision_EP * precision_EH) ** (1 / 2)
        recall_E = (recall_EP * recall_EH) ** (1 / 2)

        precision_C = (precision_CP * precision_CH) ** (1 / 2)
        recall_C = (recall_CP * recall_CH) ** (1 / 2)

        precision_N = (precision_NP * precision_NH) ** (1 / 2)
        recall_N = (recall_NP * recall_NH) ** (1 / 2)

        F_E += 2 * precision_E * recall_E / (precision_E + recall_E)
        F_C += 2 * precision_C * recall_C / (precision_C + recall_C)
        F_N += 2 * precision_N * recall_N / (precision_N + recall_N)
        F_UP += 2 * precision_UP * recall_UP / (precision_UP + recall_UP)
        F_UH += 2 * precision_UH * recall_UH / (precision_UH + recall_UH)

    F_E /= len(annotations_set)
    F_C /= len(annotations_set)
    F_N /= len(annotations_set)
    F_UP /= len(annotations_set)
    F_UH /= len(annotations_set)

    geometric_mean = (F_E * F_C * F_N) ** (1 / 3)
    arithmetic_mean = (F_E + F_C + F_N) / 3

    return F_E, F_C, F_N, F_UP, F_UH, geometric_mean, arithmetic_mean


if __name__ == "__main__":
    # mode, dataset_name = get_args()
    mode, dataset_name = "local", "snli"

    with open(data_config[dataset_name]["test"]["tokens"], "rb") as f:
        test_tokens: Dataset = pickle.load(f)
    with open(data_config[dataset_name]["test"]["alignments"], "rb") as f:
        test_alignments = pickle.load(f)

    annotations_set = []
    for path in annotation_config[dataset_name]:
        annotations_set.append(Dataset.from_json(path))

    test_ds: Dataset = test_tokens.add_column("alignment", test_alignments).with_format(
        "torch"
    )
    test_dl = DataLoader(test_ds)

    model, _, _, _, _ = load_checkpoint(dataset_name, mode, device)

    F_E, F_C, F_N, F_UP, F_UH, geometric_mean, arithmetic_mean = evaluation(
        model, test_ds, annotations_set
    )

    print(F_E, F_C, F_N, F_UP, F_UH, geometric_mean, arithmetic_mean)
