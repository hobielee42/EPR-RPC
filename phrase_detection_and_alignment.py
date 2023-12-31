import argparse
import pickle
from os import makedirs

import torch
from datasets import Dataset
from tqdm import tqdm

from aligner import Aligner
from config import dataset_config, device
from preprocessor import Preprocessor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("snli", "mnli"))
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ds_name: str = args.dataset
    splits = [
        _[0]
        for _ in zip(("train", "val", "test"), (args.train, args.val, args.test))
        if _[1]
    ]

    print(f"Device: {device}")
    preprocessor = Preprocessor()
    aligner = Aligner(device)

    if ds_name not in dataset_config:
        raise ValueError('Please enter either "snli" or "mnli" for dataset.')
    else:
        labels = ["entailment", "contradiction", "neutral"]
        for split in splits:
            ds = Dataset.from_json(dataset_config[ds_name]["path"][split])
            ds = ds.filter(lambda ex: ex["gold_label"] in labels)
            ds = ds.map(
                preprocessor.process,
                with_indices=True,
                remove_columns=ds.column_names,
                load_from_cache_file=False,
            )
            print(ds)

            tokens_save_dir = f"data/encodings/{ds_name}/tokens/"
            makedirs(tokens_save_dir, exist_ok=True)
            with open(tokens_save_dir + f"{split}_tokens.pkl", "wb") as f:
                print(f"Saving preprocessed dataset {ds_name}->{split}...")
                pickle.dump(ds, f)
                print(f"{ds_name.capitalize()}->{split} saved.")

            alignment = []
            # for ex in ds:
            #     alignment.append(aligner.compute(ex))
            pbar = tqdm(ds)
            for i, ex in enumerate(pbar):
                pbar.set_description(f"Aligning example {i}")
                alignment.append(aligner.compute(ex))

            alignment_save_dir = f"data/encodings/{ds_name}/alignments/"
            makedirs(alignment_save_dir, exist_ok=True)
            with open(alignment_save_dir + f"{split}_alignments.pkl", "wb") as f:
                print(f"Saving alignments of {ds_name}->{split}...")
                pickle.dump(alignment, f)
                print(f"Alignments of {ds_name}->{split} saved.")

    print(f"All splits preprocessed and saved.")
