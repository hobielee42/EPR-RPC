from typing import Literal

import torch
from torch import nn, Tensor
from transformers import AutoModel, PreTrainedModel


def mean_pooling(token_embeddings: Tensor, masks: Tensor) -> Tensor:
    input_mask_expanded = masks.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class SBert(nn.Module):
    def __init__(
        self, pretrained_model_name_or_path="sentence-transformers/all-mpnet-base-v2"
    ):
        super().__init__()
        self.model: PreTrainedModel = AutoModel.from_pretrained(
            pretrained_model_name_or_path
        )

    def get_local_embeddings(self, input_ids: Tensor, attention_mask: Tensor):
        model_output = self.model(input_ids, attention_mask)
        return mean_pooling(model_output[0], attention_mask)

    def get_global_embeddings(self, input_ids: Tensor, phrase_masks: Tensor):
        model_output = self.model(input_ids)
        return mean_pooling(
            model_output[0].expand(phrase_masks.size(0), -1, -1), phrase_masks
        )

    def forward(self, input_ids: Tensor, masks: Tensor):
        if input_ids.dim() == 2:  # local mode
            return self.get_local_embeddings(input_ids, masks)
        elif input_ids.dim() == 1:  # global mode
            return self.get_global_embeddings(input_ids.unsqueeze(0), masks)


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim1=1024, hidden_dim2=256):
        super().__init__()
        self.input_dim = embed_dim
        num_labels = 3
        # self.activate = nn.ReLU()
        # self.softmax = nn.Softmax(dim=-1)
        # self.fc1 = nn.Linear(embed_dim * 4, hidden_dim1)
        # self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # self.fc3 = nn.Linear(hidden_dim2, 3)
        self.model = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_labels),
            nn.Softmax(dim=-1),
        )

    def forward(self, p: Tensor, h: Tensor) -> Tensor:
        return self.model(torch.cat([p, h, torch.abs(p - h), p * h], dim=-1))


class Inducer(nn.Module):
    def __init__(self, embed_dim, local_=False, global_=False):
        if not local_ and not global_:
            raise ValueError(
                "Must use at least one of or both local and global features."
            )
        super().__init__()

        local_sbert, global_sbert = SBert(), SBert()
        if local_ and global_:
            self.lm = [local_sbert, global_sbert]
        elif local_:
            self.lm = local_sbert
        elif global_:
            self.lm = global_sbert

        self.MLP = MLP(embed_dim)

    def forward(self, ex):
        pass
