from typing import Dict, Any, List

import numpy as np
from datasets import load_dataset
from numpy import ndarray
from spacy.tokens.span import Span
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding

from chunker import Chunker


def get_phrase_masks(phrases: list[Span], sent_tokens: BatchEncoding):
    masks = []
    for phrase in phrases:
        start_char, end_char = phrase.start_char, phrase.end_char
        start_token, end_token = sent_tokens.char_to_token(start_char), sent_tokens.char_to_token(end_char - 1)

        mask = np.zeros(len(sent_tokens[0]))
        mask[start_token:end_token + 1] = 1
        masks.append(mask)

    return masks


class Preprocessor:
    def __init__(self, chunker_model_name='en_core_web_sm',
                 pretrained_tokenizer_name_or_path='sentence-transformers/all-mpnet-base-v2'):
        self.chunker = Chunker(chunker_model_name)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path)

    def encode(self, ex):
        premise = ex['premise']
        hypothesis = ex['hypothesis']

        p_phrases = self.chunker.chunk(premise)
        h_phrases = self.chunker.chunk(hypothesis)

        p_phrases_text = [_.text for _ in p_phrases]
        h_phrases_text = [_.text for _ in h_phrases]
        p_phrase_tokens = self.tokenizer(p_phrases_text, padding=True, truncation=True, max_length=256)
        h_phrase_tokens = self.tokenizer(h_phrases_text, padding=True, truncation=True, max_length=256)

        p_sent_tokens = self.tokenizer(premise, padding=True, truncation=True, max_length=256)
        h_sent_tokens = self.tokenizer(hypothesis, padding=True, truncation=True, max_length=256)

        p_masks = get_phrase_masks(p_phrases, p_sent_tokens)
        h_masks = get_phrase_masks(h_phrases, h_sent_tokens)

        # output['p_phrase_idx'] = [(_.start, _.end) for _ in p_phrases]
        # output['h_phrase_idx'] = [(_.start, _.end) for _ in h_phrases]

        return {'p_sent'         : premise, 'h_sent': hypothesis, 'p_phrases': [_.text for _ in p_phrases],
                'h_phrases'      : [_.text for _ in h_phrases], 'p_phrase_tokens': p_phrase_tokens,
                'h_phrase_tokens': h_phrase_tokens, 'p_sent_tokens': p_sent_tokens, 'h_sent_tokens': h_sent_tokens,
                'p_masks'        : p_masks, 'h_masks': h_masks, 'label': ex['label']}


if __name__ == '__main__':
    ds_snli = load_dataset("snli")
