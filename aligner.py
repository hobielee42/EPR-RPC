from typing import Literal

from torch import Tensor, nn
import torch.nn.functional as F
from models import SBert


def cosine_similarity_matrix(A: Tensor, B: Tensor):
    # Normalize the tensors
    A_normd = F.normalize(A, dim=1)
    B_normd = F.normalize(B, dim=1)

    # Compute cosine similarity matrix
    cosine_sim_matrix1 = A_normd @ B_normd.t()


class Aligner:
    def __init__(
        self, model: nn.Module, device: Literal["cuda", "mps", "cpu"], lambda_=0.6
    ):
        self.device = device
        self.model = model.to(device)
        self.lambda_ = lambda_
