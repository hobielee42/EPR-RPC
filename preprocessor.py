import numpy as np
from spacy.tokens import Span
from transformers import BatchEncoding, PreTrainedTokenizer, AutoTokenizer

from chunker import Chunker


def get_phrase_masks(phrases: list[Span], sent_tokens: BatchEncoding):
    masks = []
    for phrase in phrases:
        start_char, end_char = phrase.start_char, phrase.end_char
        start_token, end_token = sent_tokens.char_to_token(
            start_char
        ), sent_tokens.char_to_token(end_char - 1)

        if start_token is None or end_token is None:
            return masks

        mask = np.zeros(len(sent_tokens[0]))
        #
        # print(f"Phrase: {phrase}")
        # print(
        #     f"phrase.start_char: {phrase.start_char}; phrase.end_char: {phrase.end_char}"
        # )
        # print(f"start_token: {start_token}; end_token: {end_token}")
        #
        mask[start_token : (end_token + 1)] = 1
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

    def process(self, ex: dict, idx=None):
        premise = ex["sentence1"]
        hypothesis = ex["sentence2"]

        p_phrases = self.chunker.chunk(premise)
        h_phrases = self.chunker.chunk(hypothesis)

        p_phrases_text = [_.text for _ in p_phrases]
        h_phrases_text = [_.text for _ in h_phrases]

        p_phrases_idx = [(_.start, _.end) for _ in p_phrases]
        h_phrases_idx = [(_.start, _.end) for _ in h_phrases]

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

        label2id = {"entailment": 0, "contradiction": 1, "neutral": 2}

        return {
            "idx": idx,
            "p_sent": premise,
            "h_sent": hypothesis,
            "p_phrases": p_phrases_text,
            "h_phrases": h_phrases_text,
            "p_phrases_idx": p_phrases_idx,
            "h_phrases_idx": h_phrases_idx,
            "p_phrase_tokens": p_phrase_tokens,
            "h_phrase_tokens": h_phrase_tokens,
            "p_sent_tokens": p_sent_tokens,
            "h_sent_tokens": h_sent_tokens,
            "p_masks": np.array(p_masks),
            "h_masks": np.array(h_masks),
            "label": label2id[ex["gold_label"]],
        }

    # def process_dataset(self, dataset: Dataset):
    #     dataloader = []
    #     print(f"Preprocessing {len(dataset)} examples...")
    #     pbar = tqdm(dataset)
    #     for i, ex in enumerate(pbar):
    #         dataloader.append(self.process(ex))
    #         pbar.set_description(f"Preprocessing sample #{i}")
    #     print(
    #         f'{len(dataloader)} out of the total {len(dataset)} {"is" if len(dataloader) in [0,1] else "are"} preprocessed.'
    #     )
    #
    #     return dataloader
