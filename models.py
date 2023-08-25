from typing import Optional

import torch
from numpy import arange
from torch import nn, Tensor, tensor
from transformers import AutoModel, PreTrainedModel


def mean_pooling(token_embeddings: Tensor, masks: Tensor) -> Tensor:
    input_mask_expanded = masks.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_sent_score_E(phrasal_probs_E: Tensor):
    return torch.exp(torch.mean(torch.log(phrasal_probs_E)))


def get_sent_score_C(phrasal_probs_C_without_unaligned: Tensor):
    return torch.max(phrasal_probs_C_without_unaligned)


def get_sent_score_N(phrasal_probs_N: Tensor, sent_score_C: Tensor):
    return torch.max(phrasal_probs_N) * (1 - sent_score_C)


class SBert(nn.Module):
    def __init__(
        self, pretrained_model_name_or_path="sentence-transformers/all-mpnet-base-v2"
    ):
        super().__init__()
        self.model: PreTrainedModel = AutoModel.from_pretrained(
            pretrained_model_name_or_path
        )

    def forward(self, input_ids: Tensor, masks: Tensor):
        if input_ids.dim() == 2:  # local mode
            return self.get_local_embeddings(input_ids, masks)
        elif input_ids.dim() == 1:  # global mode
            return self.get_global_embeddings(input_ids.unsqueeze(0), masks)

    def get_local_embeddings(self, input_ids: Tensor, attention_mask: Tensor):
        model_output = self.model(input_ids, attention_mask)
        return mean_pooling(model_output[0], attention_mask)

    def get_global_embeddings(self, input_ids: Tensor, phrase_masks: Tensor):
        model_output = self.model(input_ids)
        return mean_pooling(
            model_output[0].expand(phrase_masks.size(0), -1, -1), phrase_masks
        )


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
        )

    def forward(self, p: Tensor, h: Tensor) -> Tensor:
        return self.model(torch.cat([p, h, torch.abs(p - h), p * h], dim=-1))


class EmptyToken(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype,
        )

    def __getitem__(self, key):
        index = tensor(key, device=self.weight.device)
        return self(index)


class EPRModel(nn.Module):
    def __init__(
        self,
        mode: str,
        language_model_name_or_path="sentence-transformers/all-mpnet-base-v2",
        embed_dim=768,
        device: torch.device = None,
    ):
        if mode not in ["local", "global", "concat"]:
            raise ValueError("Invalid mode.")
        super().__init__()

        self.mode = mode
        # if mode == "concat":
        #     self.lm = [SBert(), SBert()]
        #     self.input_dim = embed_dim * 2
        # else:
        #     self.input_dim = embed_dim
        #     self.lm = SBert()

        # self.lm = [SBert(), SBert()] if mode == "concat" else SBert()
        self.local_sbert = SBert(language_model_name_or_path)
        self.global_sbert = SBert(language_model_name_or_path)
        self.input_dim = embed_dim * ((mode == "concat") + 1)

        self.mlp = MLP(self.input_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.empty_tokens = EmptyToken(2, self.input_dim, device=device)

        if device:
            self.to(device)

    def forward(self, ex: dict):
        return self.induce_sentence_label(ex)

    def predict_phrasal_label(self, ex: dict):
        p_phrase_tokens = ex["p_phrase_tokens"]
        p_sent_tokens = ex["p_sent_tokens"]
        p_masks = ex["p_masks"]
        h_phrase_tokens = ex["h_phrase_tokens"]
        h_sent_tokens = ex["h_sent_tokens"]
        h_masks = ex["h_masks"]
        alignment: Tensor = ex["alignment"]

        # assert empty_tokens.embedding_dim == self.input_dim
        assert all(
            [
                p_masks.size(dim=0),
                h_masks.size(dim=0),
            ]
        )

        num_p_phrases = len(p_masks)
        num_h_phrases = len(h_masks)

        # get embeddings
        local_embeddings_p = self.local_sbert(
            p_phrase_tokens["input_ids"], p_phrase_tokens["attention_mask"]
        )
        local_embeddings_h = self.local_sbert(
            h_phrase_tokens["input_ids"], h_phrase_tokens["attention_mask"]
        )
        global_embeddings_p = self.global_sbert(p_sent_tokens["input_ids"], p_masks)
        global_embeddings_h = self.global_sbert(h_sent_tokens["input_ids"], h_masks)

        if self.mode == "concat":
            embeddings_p = torch.cat((local_embeddings_p, global_embeddings_p), dim=1)
            embeddings_h = torch.cat((local_embeddings_h, global_embeddings_h), dim=1)

        elif self.mode == "local":
            embeddings_p = local_embeddings_p
            embeddings_h = local_embeddings_h

        else:  # global
            embeddings_p = global_embeddings_p
            embeddings_h = global_embeddings_h

        embedding_p_empty = self.empty_tokens[0]
        embedding_h_empty = self.empty_tokens[1]

        phrasal_probs = {}
        for p in arange(num_p_phrases):
            # unaligned premise phrases
            if p not in alignment[:, 0]:
                pr_phrases = self.mlp(embeddings_p[p], embedding_h_empty)
                phrasal_probs[p, None] = self.softmax(pr_phrases)

        for h in arange(num_h_phrases):
            # unaligned hypothesis phrases
            if h not in alignment[:, 1]:
                pr_phrases = self.mlp(
                    embedding_p_empty,
                    embeddings_h[h],
                )
                phrasal_probs[None, h] = self.softmax(pr_phrases)

        for p, h in alignment:
            pr_phrases = self.mlp(embeddings_p[p], embeddings_h[h])
            phrasal_probs[p.item(), h.item()] = self.softmax(pr_phrases)

        return phrasal_probs, local_embeddings_p, local_embeddings_h

    def induce_sentence_label(self, ex: dict):
        phrasal_probs, _, _ = self.predict_phrasal_label(ex)
        phrasal_probs_without_unaligned = {
            key: value for key, value in phrasal_probs.items() if None not in key
        }
        # phrase_pairs = tuple(phrasal_probs.keys())

        phrasal_probs_values = torch.stack(list(phrasal_probs.values()))
        phrasal_probs_values_without_unaligned = torch.stack(
            list(phrasal_probs_without_unaligned.values())
        )

        sent_score_E = get_sent_score_E(phrasal_probs_values[:, 0])
        sent_score_C = get_sent_score_C(phrasal_probs_values_without_unaligned[:, 1])
        sent_score_N = get_sent_score_N(phrasal_probs_values[:, 2], sent_score_C)

        sent_scores = torch.stack((sent_score_E, sent_score_C, sent_score_N))
        sent_probs = sent_scores / torch.sum(sent_scores)

        return sent_probs


# class Explainer(nn.Module):
#     def __init__(self):
#         super().__init__()
