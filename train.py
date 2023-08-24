import pickle

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

data_config = {
    "snli": {
        "train": {
            "tokens": "data/encodings/snli/tokens/train_tokens.pkl",
            "alignments": "data/encodings/snli/alignments/train_alignments.pkl",
        },
        "val": {
            "tokens": "data/encodings/snli/tokens/val_tokens.pkl",
            "alignments": "data/encodings/snli/alignments/val_alignments.pkl",
        },
        "test": {
            "tokens": "data/encodings/snli/tokens/test_tokens.pkl",
            "alignments": "data/encodings/snli/alignments/test_alignments.pkl",
        },
    },
    "mnli": {
        "train": {
            "tokens": "data/encodings/mnli/tokens/train_tokens.pkl",
            "alignments": "data/encodings/mnli/alignments/train_alignments.pkl",
        },
        "val": {
            "tokens": "data/encodings/mnli/tokens/val_tokens.pkl",
            "alignments": "data/encodings/mnli/alignments/val_alignments.pkl",
        },
        "test": {
            "tokens": "data/encodings/mnli/tokens/test_tokens.pkl",
            "alignments": "data/encodings/mnli/alignments/test_alignments.pkl",
        },
    },
}


if __name__ == "__main__":
    with open(data_config["snli"]["train"]["tokens"], "rb") as f:
        tokens: Dataset = pickle.load(f)
    with open(data_config["snli"]["train"]["alignments"], "rb") as f:
        alignments = pickle.load(f)

    ds: Dataset = tokens.add_column("alignment", alignments)

    ds_torch = ds.with_format("torch")
