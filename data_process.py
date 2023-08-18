import argparse
import pickle
from os import makedirs

from datasets import Dataset

from preprocessor import Preprocessor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", action="store", choices=("snli", "mnli"))
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    return parser.parse_args()


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
    args = get_args()
    ds_name = args.dataset
    splits = [
        _[0]
        for _ in zip(("train", "val", "test"), (args.train, args.val, args.test))
        if _[1]
    ]

    preprocessor = Preprocessor()

    if ds_name not in ds_config:
        raise ValueError('Please enter either "snli" or "mnli" for dataset.')
    else:
        labels = ["entailment", "contradiction", "neutral"]
        for split in splits:
            ds = Dataset.from_json(ds_config[ds_name]["path"][split])
            ds = ds.filter(lambda ex: ex["gold_label"] in labels)
            ds = ds.map(preprocessor.process, remove_columns=ds.column_names)
            print(ds)
            save_dir = f"data/encodings/{ds_name}/tokens/"
            makedirs(save_dir, exist_ok=True)
            with open(save_dir + f"{split}_tokens.pkl", "wb") as f:
                print(f"Saving preprocessed dataset {ds_name}->{split}...")
                pickle.dump(ds, f)
                print(f"{ds_name}->{split} saved.")

    print(f"All splits preprocessed and saved.")
