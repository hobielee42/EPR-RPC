from typing import Literal
import numpy as np

import torch
from torch import Tensor, nn, tensor
import torch.nn.functional as F
from models import SBert


def cosine_similarity_matrix(A: Tensor, B: Tensor):
    # Normalize the tensors
    A_normd = F.normalize(A, dim=1)
    B_normd = F.normalize(B, dim=1)

    # Compute cosine similarity matrix
    return A_normd @ B_normd.t()


class Aligner:
    def __init__(
        self,
        device: str,
        model: nn.Module = SBert(),
        lambda_=0.6,
    ):
        self.device = device
        self.model = model.to(device)
        self.lambda_ = lambda_

    def compute(self, ex: dict):
        pairs = []
        (
            p_phrase_tokens,
            p_sent_tokens,
            p_masks,
            h_phrase_tokens,
            h_sent_tokens,
            h_masks,
        ) = (
            ex["p_phrase_tokens"],
            ex["p_sent_tokens"],
            ex["p_masks"],
            ex["h_phrase_tokens"],
            ex["h_sent_tokens"],
            ex["h_masks"],
        )
        if (
            None
            not in [
                p_phrase_tokens,
                p_sent_tokens,
                p_masks,
                h_phrase_tokens,
                h_sent_tokens,
                h_masks,
            ]
            and p_masks
            and h_masks
        ):
            self.model.eval()
            with torch.no_grad():
                local_embeddings_p = self.model(
                    tensor(ex["p_phrase_tokens"]["input_ids"]).to(self.device),
                    tensor(ex["p_phrase_tokens"]["attention_mask"]).to(self.device),
                )
                global_embeddings_p = self.model(
                    tensor(ex["p_sent_tokens"]["input_ids"]).to(self.device),
                    tensor(np.array(ex["p_masks"])).to(self.device),
                )
                local_embeddings_h = self.model(
                    tensor(ex["h_phrase_tokens"]["input_ids"]).to(self.device),
                    tensor(ex["h_phrase_tokens"]["attention_mask"]).to(self.device),
                )
                global_embeddings_h = self.model(
                    tensor(ex["h_sent_tokens"]["input_ids"]).to(self.device),
                    tensor(np.array(ex["h_masks"])).to(self.device),
                )
            local_sims = cosine_similarity_matrix(
                local_embeddings_p, local_embeddings_h
            )
            global_sims = cosine_similarity_matrix(
                global_embeddings_p, global_embeddings_h
            )

            if global_sims.size() != local_sims.size():
                return pairs

            similarities = self.lambda_ * global_sims + (1 - self.lambda_) * local_sims

            # print(similarities)

            # Find the maximum element indices along each row
            max_row_indices = torch.argmax(similarities, dim=1)
            # print(max_row_indices)

            # Find the maximum element indices along each column
            max_col_indices = torch.argmax(similarities, dim=0)
            # print(max_col_indices)

            # Initialize a list to store the results

            # Iterate over each row and check if the maximum element is also the maximum in its column
            for p, h in enumerate(max_row_indices):
                if p == max_col_indices[h]:
                    pairs.append((p, int(h)))
        return pairs
