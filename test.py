import pickle
import torch

from config import data_config
from models import EPRModel, EmptyToken, SBert
import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
from util import example_to_device

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
    with open(data_config[dataset_name]["test"]["tokens"], "rb") as f:
        test_tokens: Dataset = pickle.load(f)
    with open(data_config[dataset_name]["test"]["alignments"], "rb") as f:
        test_alignments = pickle.load(f)

    test_ds: Dataset = test_tokens.add_column("alignment", test_alignments).with_format(
        "torch"
    )
