import torch
from datasets import Dataset
from transformers import AutoModel
from torch import tensor
import torch
import numpy as np
import torch.nn.functional as F
from aligner import Aligner
from models import SBert
from preprocessor import Preprocessor


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

labels = ["entailment", "contradiction", "neutral"]
ds = Dataset.from_json("data/datasets/multinli_1.0/multinli_1.0_train.jsonl")
ds = ds.filter(lambda ex: ex["gold_label"] in labels)
preprocessor = Preprocessor()
aligner = Aligner(device)
ex = ds[21853]
model = SBert().to(device)


# ex = {
#     "sentence1": "An elderly couple in heavy coats are looking at black and white photos displayed on the wall.",
#     "sentence2": "octogenarians admiring the old photographs that decorated the wall",
#     "gold_label": "contradiction",
# }

ex = preprocessor.process(ex)

# local_ = model(
#     tensor(ex["p_phrase_tokens"]["input_ids"]).to(device),
#     tensor(ex["p_phrase_tokens"]["attention_mask"]).to(device),
# )
# global_ = model(
#     tensor(ex["p_sent_tokens"]["input_ids"]).to(device),
#     tensor(ex["p_masks"]).to(device),
# )

aligned_phrase_pairs = aligner.compute(ex)

print(
    f"Premise: {ex['p_sent']}\n"
    f"Hypothesis: {ex['h_sent']}\n"
    f"Phrases in premise: {ex['p_phrases']}\n"
    f"Phrases in hypothesis: {ex['h_phrases']}\n"
    f"Phrase alignment:"
)
for aligned_phrases in aligned_phrase_pairs:
    print(
        f'   p: "{ex["p_phrases"][aligned_phrases[0]]}"; h: "{ex["h_phrases"][aligned_phrases[1]]}"'
    )
