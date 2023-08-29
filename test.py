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
from evaluation import *
from models import EPRModel
from util import example_to_device, load_checkpoint

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

dataset_name = "snli"

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

    print(len(annotations_set))

    annotations = annotations_set[0]
    annotation = annotations[2]
    id = int(annotation["snli_id"])
    ex = test_ds[id]
    model_result = get_phrases_from_model(model, ex)
    annotated_result = get_phrases_from_annotation(annotation, ex)
    print(model_result)
    print(annotated_result)
