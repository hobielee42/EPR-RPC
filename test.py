import pickle

from datasets import Dataset
import torch
from torch import tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import EPR, MLP, SBert

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


def to_device(ex: dict, device: str):
    ex["idx"] = ex["idx"].to(device)
    ex["p_phrase_tokens"]["input_ids"] = ex["p_phrase_tokens"]["input_ids"].to(device)
    ex["p_phrase_tokens"]["attention_mask"] = ex["p_phrase_tokens"][
        "attention_mask"
    ].to(device)
    ex["h_phrase_tokens"]["input_ids"] = ex["h_phrase_tokens"]["input_ids"].to(device)
    ex["h_phrase_tokens"]["attention_mask"] = ex["h_phrase_tokens"][
        "attention_mask"
    ].to(device)
    ex["p_sent_tokens"]["input_ids"] = ex["p_sent_tokens"]["input_ids"].to(device)
    ex["p_sent_tokens"]["attention_mask"] = ex["p_sent_tokens"]["attention_mask"].to(
        device
    )
    ex["h_sent_tokens"]["input_ids"] = ex["h_sent_tokens"]["input_ids"].to(device)
    ex["h_sent_tokens"]["attention_mask"] = ex["h_sent_tokens"]["attention_mask"].to(
        device
    )
    ex["p_masks"] = ex["p_masks"].to(device)
    ex["h_masks"] = ex["h_masks"].to(device)
    ex["label"] = ex["label"].to(device)

    return ex


if __name__ == "__main__":
    with open(data_config["snli"]["train"]["tokens"], "rb") as f:
        tokens: Dataset = pickle.load(f)
    with open(data_config["snli"]["train"]["alignments"], "rb") as f:
        alignments = pickle.load(f)

    ds: Dataset = tokens.add_column("alignment", alignments)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    mode = "local"

    ds_torch = ds.with_format("torch")

    ex = to_device(ds_torch[0], device)

    empty_tokens = torch.nn.Embedding(2, 768).to(device)
    empty_token_indices=[tensor(0).to(device),tensor(1).to(device)]

    # sbert = SBert().to(device)
    # mlp = MLP(768).to(device)

    # local_ps = sbert(
    #     ex["p_phrase_tokens"]["input_ids"].to(device),
    #     ex["p_phrase_tokens"]["attention_mask"].to(device),
    # )

    # local_hs = sbert(
    # ex["h_phrase_tokens"]["input_ids"].to(device),
    # ex["h_phrase_tokens"]["attention_mask"].to(device),
    # )

    # p=local_ps[1]
    # h=local_hs[2]

    epr = EPR(mode).to(device)

    output=epr(ex,empty_tokens,empty_token_indices)
