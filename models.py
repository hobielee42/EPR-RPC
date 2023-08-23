import torch
from numpy import arange
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
            nn.Softmax(dim=-1),
        )

    def forward(self, p: Tensor, h: Tensor) -> Tensor:
        return self.model(torch.cat([p, h, torch.abs(p - h), p * h], dim=-1))


class EPR(nn.Module):
    def __init__(self, mode, embed_dim=768):
        if mode not in ["local", "global", "concat"]:
            raise ValueError("Invalid mode.")
        super().__init__()

        self.mode = mode
        if mode == "concat":
            self.lm = [SBert(), SBert()]
            self.input_dim = embed_dim * 2
        else:
            self.input_dim = embed_dim
            self.lm = SBert()

        self.mlp = MLP(self.input_dim)

    def predict_phrasal_label(
        self,
        ex: dict,
        empty_tokens: torch.nn.Embedding,
        empty_token_indices: list[Tensor],
    ):
        p_phrase_tokens = ex["p_phrase_tokens"]
        p_sent_tokens = ex["p_sent_tokens"]
        p_masks = ex["p_masks"]
        h_phrase_tokens = ex["h_phrase_tokens"]
        h_sent_tokens = ex["h_sent_tokens"]
        h_masks = ex["h_masks"]
        alignment = ex["alignment"]

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
        if self.mode == "concat":
            local_embeddings_p = self.lm(
                p_phrase_tokens["input_ids"], p_phrase_tokens["attention_mask"]
            )
            local_embeddings_h = self.lm(
                h_phrase_tokens["input_ids"], h_phrase_tokens["attention_mask"]
            )
            global_embeddings_p = self.lm(p_sent_tokens["input_ids"], p_masks)
            global_embeddings_h = self.lm(h_sent_tokens["input_ids"], h_masks)

            embeddings_p = torch.cat((local_embeddings_p, local_embeddings_h), dim=1)
            embeddings_h = torch.cat((global_embeddings_p, global_embeddings_h), dim=1)

        elif self.mode == "local":
            embeddings_p = self.lm(
                p_phrase_tokens["input_ids"], p_phrase_tokens["attention_mask"]
            )
            embeddings_h = self.lm(
                h_phrase_tokens["input_ids"], h_phrase_tokens["attention_mask"]
            )

        else:  # global
            embeddings_p = self.lm(p_sent_tokens["input_ids"], p_masks)
            embeddings_h = self.lm(h_sent_tokens["input_ids"], h_masks)

        embedding_p_empty = empty_tokens(empty_token_indices[0])
        embedding_h_empty = empty_tokens(empty_token_indices[1])

        phrasal_labels = {}
        for p in arange(num_p_phrases):
            # unaligned premise phrases
            if p not in alignment[:, 0]:
                pr_phrases = self.mlp(embeddings_p[p], embedding_h_empty)
                phrasal_labels[p, None] = pr_phrases

        for h in arange(num_h_phrases):
            # unaligned hypothesis phrases
            if h not in alignment[:, 1]:
                pr_phrases = self.mlp(
                    embedding_p_empty,
                    embeddings_h[h],
                )
                phrasal_labels[None, h] = pr_phrases

        for p, h in alignment:
            pr_phrases = self.mlp(embeddings_p[p], embeddings_h[h])
            phrasal_labels[p, h] = pr_phrases

        return phrasal_labels

    def induce_sentence_label(
        self,
        ex: dict,
        empty_tokens: torch.nn.Embedding,
        empty_token_indices: list[Tensor],
    ):
        phrasal_labels = self.predict_phrasal_label(
            ex, empty_tokens, empty_token_indices
        )
