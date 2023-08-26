import torch


from models import EPRModel, EmptyToken, SBert


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
    # with open(data_config["snli"]["train"]["tokens"], "rb") as f:
    #     tokens: Dataset = pickle.load(f)
    # with open(data_config["snli"]["train"]["alignments"], "rb") as f:
    #     alignments = pickle.load(f)

    # labels = ["entailment", "contradiction", "neutral"]

    # ds: Dataset = tokens.add_column("alignment", alignments).with_format("torch")

    # mode = "local"

    # ex = deepcopy(ds[0])
    # ex["h_phrases_idx"] = []
    # ex["h_phrase_tokens"] = {"input_ids": [[]], "attention_mask": [[]]}
    # ex["h_masks"] = [[]]
    # ds_ = Dataset.from_list([ex]).with_format("torch")
    # dl = DataLoader(ds_)

    sbert = SBert()

    empty_phrase_tokens = {
        "input_ids": torch.empty((0, 1), dtype=torch.int),
        "attention_mask": torch.empty((0, 1), dtype=torch.int),
    }
    empty_phrase_embedding = sbert(
        empty_phrase_tokens["input_ids"], empty_phrase_tokens["attention_mask"]
    )
    num_sent_tokens=10
    sent_tokens = {
        "input_ids": torch.randint(high=1000, size=(1, num_sent_tokens)),
        "attention_mask": torch.randint(high=1000, size=(1, num_sent_tokens)),
    }
    empty_masks = torch.empty((0, num_sent_tokens), dtype=torch.int)
    empty_sent_embedding = sbert(sent_tokens["input_ids"], empty_masks)
