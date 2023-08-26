import pickle
import torch


from models import EPRModel, EmptyToken, SBert
import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
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
ds_config = {
    "snli": {
        "path": {
            "train": "data/datasets/snli_1.0/snli_1.0_train.jsonl",
            "val": "data/datasets/snli_1.0/snli_1.0_dev.jsonl",
            "test": "data/datasets/snli_1.0/snli_1.0_test.jsonl",
        },
    },
    "mnli": {
        "path": {
            "train": "data/datasets/multinli_1.0/multinli_1.0_train.jsonl",
            "val": "data/datasets/multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
            "test": "data/datasets/multinli_1.0/multinli_1.0_dev_matched.jsonl",
        },
    },
}


if __name__ == "__main__":
    # load dataset
    with open(data_config["snli"]["train"]["tokens"], "rb") as f:
        train_tokens: Dataset = pickle.load(f)
    with open(data_config["snli"]["train"]["alignments"], "rb") as f:
        train_alignments = pickle.load(f)
    with open(data_config["snli"]["test"]["tokens"], "rb") as f:
        test_tokens: Dataset = pickle.load(f)
    with open(data_config["snli"]["test"]["alignments"], "rb") as f:
        test_alignments = pickle.load(f)

    train_ds: Dataset = train_tokens.add_column("alignment", train_alignments)
    ex = train_ds[11845 + 7 * 54936]
    train_ds = train_ds.with_format("torch")
    ex_ = train_ds[11845 + 7 * 54936]

    ex = example_to_device(ex_, device)
    epr = EPRModel("local", device=device)

    # # labels = ["entailment", "contradiction", "neutral"]

    # # ds: Dataset = tokens.add_column("alignment", alignments).with_format("torch")

    # # mode = "local"

    # # ex = deepcopy(ds[0])
    # # ex["h_phrases_idx"] = []
    # # ex["h_phrase_tokens"] = {"input_ids": [[]], "attention_mask": [[]]}
    # # ex["h_masks"] = [[]]
    # # ds_ = Dataset.from_list([ex]).with_format("torch")
    # # dl = DataLoader(ds_)

    # sbert = SBert()

    # empty_phrase_tokens = {
    #     "input_ids": torch.empty((0, 1), dtype=torch.int),
    #     "attention_mask": torch.empty((0, 1), dtype=torch.int),
    # }
    # empty_local_embedding = sbert(
    #     empty_phrase_tokens["input_ids"], empty_phrase_tokens["attention_mask"]
    # )
    # num_sent_tokens = 10
    # sent_tokens = {
    #     "input_ids": torch.randint(high=1000, size=(1, num_sent_tokens)),
    #     "attention_mask": torch.randint(high=1000, size=(1, num_sent_tokens)),
    # }
    # empty_masks = torch.empty((0, num_sent_tokens), dtype=torch.int)
    # empty_global_embedding = sbert(sent_tokens["input_ids"], empty_masks)

    # ex = {"h_phrase_tokens": {"attention_mask": [[]], "input_ids": [[]]}}
    # ds = datasets.Dataset.from_list([ex])
    # dspt = ds.with_format("torch")
    # dl = DataLoader(dspt)
    # ex = next(iter(dl))
    # ex["h_phrase_tokens"]["input_ids"] = (
    #     (ex["h_phrase_tokens"]["input_ids"].squeeze(0).to(device))
    #     if ex["h_phrase_tokens"]["input_ids"] != [[]]
    #     else torch.empty((0, 1), dtype=torch.int, device=device)
    # )
    # ex["h_phrase_tokens"]["attention_mask"] = (
    #     (ex["h_phrase_tokens"]["attention_mask"].squeeze(0).to(device))
    #     if ex["h_phrase_tokens"]["attention_mask"] != [[]]
    #     else torch.empty((0, 1), dtype=torch.int, device=device)
    # )
    # sbert = sbert.to(device)
    # output = sbert(
    #     ex["h_phrase_tokens"]["input_ids"], ex["h_phrase_tokens"]["attention_mask"]
    # )
