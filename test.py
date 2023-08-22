import torch
from datasets import Dataset
from transformers import AutoModel
from torch import tensor
import torch
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

snli = Dataset.from_json("data/datasets/snli_1.0/snli_1.0_test.jsonl")
preprocessor = Preprocessor()
ex = snli[1024]

case_study = {
    "sentence1": "Some dogs are running on a deserted beach.",
    "sentence2": "There is only one dog at the beach.",
    "gold_label": "contradiction",
}
case_study = preprocessor.process(case_study)
ex = preprocessor.process(ex, 1024)

model = SBert().to(device)
# local_ = model(
#     tensor(ex["p_phrase_tokens"]["input_ids"]).to(device),
#     tensor(ex["p_phrase_tokens"]["attention_mask"]).to(device),
# )
# global_ = model(
#     tensor(ex["p_sent_tokens"]["input_ids"]).to(device),
#     tensor(ex["p_masks"]).to(device),
# )

aligner = Aligner(SBert(), device)
output=aligner.compute(case_study)
