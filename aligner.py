from typing import Literal

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
        self, model: nn.Module, device: Literal["cuda", "mps", "cpu"], lambda_=0.6
    ):
        self.device = device
        self.model = model.to(device)
        self.lambda_ = lambda_

    def compute(self, ex: dict):
        local_embeddings_p = self.model(
            tensor(ex["p_phrase_tokens"]["input_ids"]).to(self.device),
            tensor(ex["p_phrase_tokens"]["attention_mask"]).to(self.device),
        )
        global_embeddings_p = self.model(
            tensor(ex["p_sent_tokens"]["input_ids"]).to(self.device),
            tensor(ex["p_masks"]).to(self.device),
        )
        local_embeddings_h = self.model(
            tensor(ex["h_phrase_tokens"]["input_ids"]).to(self.device),
            tensor(ex["h_phrase_tokens"]["attention_mask"]).to(self.device),
        )
        global_embeddings_h = self.model(
            tensor(ex["h_sent_tokens"]["input_ids"]).to(self.device),
            tensor(ex["h_masks"]).to(self.device),
        )
        local_sims = cosine_similarity_matrix(local_embeddings_p, local_embeddings_h)
        global_sims = cosine_similarity_matrix(global_embeddings_p, global_embeddings_h)
        similarities = self.lambda_ * global_sims + (1 - self.lambda_) * local_sims

        # print(similarities)

        # Find the maximum element indices along each row
        max_row_indices = torch.argmax(similarities, dim=1)
        # print(max_row_indices)

        # Find the maximum element indices along each column
        max_col_indices = torch.argmax(similarities, dim=0)
        # print(max_col_indices)

        # Initialize a list to store the results
        results = []

        # Iterate over each row and check if the maximum element is also the maximum in its column
        for p, h in enumerate(max_row_indices):
            if p == max_col_indices[h]:
                results.append((p, int(h)))
        return results
