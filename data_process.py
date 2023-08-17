import argparse
import pickle

import numpy as np
from datasets import load_dataset, Dataset
from numpy import ndarray
from spacy.tokens.span import Span
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding
from chunker import Chunker
from os import makedirs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", action="store", choices=("snli", "mnli"))
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    return parser.parse_args()


def get_phrase_masks(phrases: list[Span], sent_tokens: BatchEncoding):
    masks = []
    for phrase in phrases:
        start_char, end_char = phrase.start_char, phrase.end_char
        start_token, end_token = sent_tokens.char_to_token(
            start_char
        ), sent_tokens.char_to_token(end_char - 1)

        mask = np.zeros(len(sent_tokens[0]))
        mask[start_token : end_token + 1] = 1
        masks.append(mask)

    return masks


class Preprocessor:
    def __init__(
        self,
        chunker_model_name="en_core_web_sm",
        pretrained_tokenizer_name_or_path="sentence-transformers/all-mpnet-base-v2",
    ):
        self.chunker = Chunker(chunker_model_name)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_tokenizer_name_or_path
        )

    def process(self, ex):
        premise = ex["premise"]
        hypothesis = ex["hypothesis"]

        p_phrases = self.chunker.chunk(premise)
        h_phrases = self.chunker.chunk(hypothesis)

        p_phrases_text = [_.text for _ in p_phrases]
        h_phrases_text = [_.text for _ in h_phrases]
        p_phrase_tokens = (
            self.tokenizer(
                p_phrases_text, padding=True, truncation=True, max_length=256
            )
            if p_phrases_text
            else None
        )
        h_phrase_tokens = (
            self.tokenizer(
                h_phrases_text, padding=True, truncation=True, max_length=256
            )
            if h_phrases_text
            else None
        )

        p_sent_tokens = (
            self.tokenizer(premise, padding=True, truncation=True, max_length=256)
            if premise
            else None
        )
        h_sent_tokens = (
            self.tokenizer(hypothesis, padding=True, truncation=True, max_length=256)
            if hypothesis
            else None
        )

        p_masks = get_phrase_masks(p_phrases, p_sent_tokens) if p_phrases else None
        h_masks = get_phrase_masks(h_phrases, h_sent_tokens) if h_phrases else None

        # output['p_phrase_idx'] = [(_.start, _.end) for _ in p_phrases]
        # output['h_phrase_idx'] = [(_.start, _.end) for _ in h_phrases]

        return {
            "p_sent": premise,
            "h_sent": hypothesis,
            "p_phrases": [_.text for _ in p_phrases],
            "h_phrases": [_.text for _ in h_phrases],
            "p_phrase_tokens": p_phrase_tokens,
            "h_phrase_tokens": h_phrase_tokens,
            "p_sent_tokens": p_sent_tokens,
            "h_sent_tokens": h_sent_tokens,
            "p_masks": p_masks,
            "h_masks": h_masks,
            "label": ex["label"],
        }

    def process_dataset(self, dataset: Dataset):
        dataloader = []
        print(f"Preprocessing {len(dataset)} examples...")
        pbar = tqdm(dataset)
        for i, ex in enumerate(pbar):
            dataloader.append(self.process(ex))
            pbar.set_description(f"Preprocessing sample #{i}")
        print(
            f'{len(dataloader)} out of the total {len(dataset)} {"is" if len(dataloader) in [0,1] else "are"} preprocessed.'
        )

        return dataloader


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
    ds_opt = args.dataset
    splits = [
        _[0]
        for _ in zip(("train", "val", "test"), (args.train, args.val, args.test))
        if _[1]
    ]
    preprocessor = Preprocessor()

    if ds_opt not in ds_config.keys():
        raise ValueError('Please enter either "--ds snli" or "--ds mnli".')
    else:
        for split in splits:
            dataset = Dataset.from_json(ds_config[ds_opt]["path"][split])
            dataset=dataset.rename_columns(
                {
                    "sentence1": "premise",
                    "sentence2": "hypothesis",
                    "gold_label": "label",
                }
            )
            print(type(dataset))
            print(dataset)
            split_processed = preprocessor.process_dataset(dataset)
            save_dir = f"data/encodings/{ds_opt}/tokens/"
            makedirs(save_dir, exist_ok=True)
            with open(save_dir + f"{split}_tokens.pkl", "wb") as f:
                print(f"Saving preprocessed dataset {ds_opt}->{split}...")
                pickle.dump(split_processed, f)
                print(f"{ds_opt}->{split} saved.")

    print(f"All splits preprocessed and saved.")
