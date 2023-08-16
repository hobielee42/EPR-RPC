from datasets import load_dataset, Features, Value, DatasetDict
import datasets
from datasets import Dataset

snli_hg = load_dataset("snli")
snli_hg["val"] = snli_hg["validation"]
del snli_hg["validation"]

mnli_hg = load_dataset("multi_nli")
mnli_hg["val"] = mnli_hg["validation_mismatched"]
del mnli_hg["validation_mismatched"]
mnli_hg["test"] = mnli_hg["validation_matched"]
del mnli_hg["validation_matched"]

snli_local = DatasetDict.from_json(
    {
        "train": "data/datasets/snli_1.0/snli_1.0_train.jsonl",
        "val": "data/datasets/snli_1.0/snli_1.0_dev.jsonl",
        "test": "data/datasets/snli_1.0/snli_1.0_test.jsonl",
    }
)

mnli_local = DatasetDict.from_json(
    {
        "train": "data/datasets/multinli_1.0/multinli_1.0_train.jsonl",
        "val": "data/datasets/multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
        "test": "data/datasets/multinli_1.0/multinli_1.0_dev_matched.jsonl",
    }
)
