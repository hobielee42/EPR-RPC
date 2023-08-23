import pickle

data_config = {
    "snli": {
        "train": {
            "tokens": "data/encodings/snli/tokens/train_tokens.pkl",
            "alignments": "data/encodings/snli/alignment/train_alignments.pkl",
        },
        "val": {
            "tokens": "data/encodings/snli/tokens/val_tokens.pkl",
            "alignments": "data/encodings/snli/alignment/val_alignments.pkl",
        },
        "test": {
            "tokens": "data/encodings/snli/tokens/test_tokens.pkl",
            "alignments": "data/encodings/snli/alignment/test_alignments.pkl",
        },
    },
    "mnli": {
        "train": {
            "tokens": "data/encodings/mnli/tokens/train_tokens.pkl",
            "alignments": "data/encodings/mnli/alignment/train_alignments.pkl",
        },
        "val": {
            "tokens": "data/encodings/mnli/tokens/val_tokens.pkl",
            "alignments": "data/encodings/mnli/alignment/val_alignments.pkl",
        },
        "test": {
            "tokens": "data/encodings/mnli/tokens/test_tokens.pkl",
            "alignments": "data/encodings/mnli/alignment/test_alignments.pkl",
        },
    },
}


if __name__ == "__main__":
    with open(data_config["snli"]["train"]["tokens"], "rb") as f:
        tokens = pickle.load(f)
    with open(data_config["snli"]["train"]["alignments"], "rb") as f:
        alignments = pickle.load(f)
