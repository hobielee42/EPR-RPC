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
