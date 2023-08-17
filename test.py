import spacy
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from chunker import Chunker
from data_process import Preprocessor

# chunker & tokenizer test

# chunker = Chunker()
# ex = "During the intense workout in the gym, the tall muscular constructor John not only lifted weights but also showed off, though he hadn't tried it before."
# print(ex)
#
# # doc = chunker._nlp(ex)
# # for token in doc:
# #     print(token.text, token.tag_)
#
# phrases = chunker.chunk(ex)
# phrases_text = [phrase.text for phrase in phrases]
# c_offsets = [(phrase.start_char, phrase.end_char) for phrase in phrases]
#
# nlp = spacy.load("en_core_web_sm")
# doc = nlp(ex)
#
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-mpnet-base-v2"
)
#
# encoded_phrases = tokenizer(phrases_text, padding=True, truncation=True, max_length=256)
# encoded_sent = tokenizer(ex, padding=True, truncation=True, max_length=256, return_offsets_mapping=True)

# preprocessor test

# ds_snli = load_dataset("snli")
#
# ex_snli = ds_snli['test'][1024]
# print(ex_snli)
#
# preprocessor = Preprocessor()
# encoded_ex = preprocessor.encode(ex_snli)
#
# for key in encoded_ex.keys():
#     print(key + ': ' + str(encoded_ex[key]))

preprocessor = Preprocessor()
# # for split_name in ds_snli:
# #     split = ds_snli[split_name]

# ds_test = ds_snli['test']
# ds_test_processed = []
# for ex in tqdm(ds_test):
#     ds_test_processed.append(preprocessor.process(ex))

# dataset test

# ex = ds_snli["train"][122238]
# ex = {"premise": "sample sample samp", "hypothesis": "hi", "label": 0}
# _ = preprocessor.process(ex)
chunker = Chunker()
# _ = chunker.chunk("fo")

mnli = Dataset.from_json("data/datasets/multinli_1.0/multinli_1.0_train.jsonl")
mnli = mnli.rename_columns(
    {
        "sentence1": "premise",
        "sentence2": "hypothesis",
        "gold_label": "label",
    }
)

ex = mnli[21853]
output = preprocessor.process(ex)
