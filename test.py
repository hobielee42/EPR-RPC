import pickle

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import EPRModel, EmptyToken
from train import example_to_device

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

    mode = "local"

    ds_torch = ds.with_format("torch")
    dataloader = DataLoader(ds_torch)

    none_ex=[]
    for i, ex in enumerate(tqdm(ds_torch)):
        if ex is None:
            none_ex.append(i)

    print(none_ex)
    # ex = example_to_device(ds_torch[0], device)
    # print(ex)

    # empty_tokens = torch.nn.Embedding(2, 768).to(device)
    # empty_token_indices = [tensor(0).to(device), tensor(1).to(device)]

    # empty_tokens = EmptyToken(2, 768, device=device)

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

    # epr = EPRModel(mode, device=device)
    # epr.eval()

    # print(epr.empty_tokens[0, 1])
    # with torch.no_grad():
    #     output = epr(ex)
