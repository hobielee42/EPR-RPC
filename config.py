import torch

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
        "model_path": "data/model/snli/",
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
        "model_path": "data/model/mnli/",
    },
}

dataset_config = {
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

model_config = {
    "snli": {
        "local": "data/model/snli/local.pth",
        "concat": "data/model/snli/concat.pth",
        "global": "data/model/snli/global.pth",
    },
    "mnli": {
        "local": "data/model/mnli/local.pth",
        "concat": "data/model/mnli/concat.pth",
        "global": "data/model/mnli/global.pth",
    },
}

annotation_config = {
    "snli": [
        "data/annotation/snli/1.jsonl",
        "data/annotation/snli/2.jsonl",
        "data/annotation/snli/3.jsonl",
    ],
    "mnli": [
        "data/annotation/mnli/1.jsonl",
        "data/annotation/mnli/2.jsonl",
        "data/annotation/mnli/3.jsonl",
    ],
}

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
