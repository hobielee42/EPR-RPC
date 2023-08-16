import argparse
import pickle

import numpy as np
from datasets import load_dataset, Dataset
from numpy import ndarray
from spacy.tokens.span import Span
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding
from chunker import Chunker


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", action="store")
    parser.add_argument(
        "variation", action="store", default="hg", choices=["hg", "loc"]
    )
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
        for ex in tqdm(dataset, desc=f"Preprocessing {len(dataset)} examples..."):
            dataloader.append(self.process(ex))
        print(
            f'{len(dataloader)} out of the total {len(dataset)} {"is" if len(dataloader) <= 1 else "are"} preprocessed.'
        )

        return dataloader


if __name__ == "__main__":
    args = get_args()
    preprocessor = Preprocessor()

    if args.ds == "snli":
        dataset = load_dataset("snli")
        split_names = {"train": "train", "val": "validation", "test": "test"}
    elif args.ds == "mnli":
        dataset = load_dataset("multi_nli")
        split_names = {
            "train": "train",
            "val": "validation_mismatched",
            "test": "validation_matched",
        }
    else:
        raise ValueError('Please enter either "--ds snli" or "--ds mnli".')

    splits = [
        "train" if args.train else None,
        "val" if args.val else None,
        "test" if args.test else None,
    ]

    for split in splits:
        print(f"Preprocessing {args.ds}/{split}...")
        if split is not None:
            split_preprocessed = preprocessor.process_dataset(
                dataset[split_names[split]]
            )
            with open(
                f"./data/encodings/{args.ds}/tokens/{split}_tokens.pkl", "wb+"
            ) as f:
                print(f"Saving preprocessed dataset {args.ds}/{split}...")
                pickle.dump(split_preprocessed, f)
                print(f"{args.ds}/{split} saved.")

    print(f"All splits preprocessed and saved.")
