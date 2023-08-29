import torch
from torch import device
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

from config import model_config
from models import EPRModel


def example_to_device(ex: dict, device: torch.device):
    ex["idx"] = (
        ex["idx"].squeeze(0).to(device) if not isinstance(ex["idx"], int) else ex["idx"]
    )
    ex["p_phrases_idx"] = (
        ex["p_phrases_idx"].squeeze(0).to(device)
        if ex["p_phrases_idx"] != []
        else torch.empty((0, 2), dtype=torch.int, device=device)
    )
    ex["h_phrases_idx"] = (
        ex["h_phrases_idx"].squeeze(0).to(device)
        if ex["h_phrases_idx"] != []
        else torch.empty((0, 2), dtype=torch.int, device=device)
    )
    ex["p_phrase_tokens"]["input_ids"] = (
        (ex["p_phrase_tokens"]["input_ids"].squeeze(0).to(device))
        if ex["p_phrase_tokens"]["input_ids"] != [[]]
        and ex["p_phrase_tokens"]["input_ids"].size() != (1, 0)
        else torch.empty((0, 1), dtype=torch.int, device=device)
    )
    ex["p_phrase_tokens"]["attention_mask"] = (
        (ex["p_phrase_tokens"]["attention_mask"].squeeze(0).to(device))
        if ex["p_phrase_tokens"]["attention_mask"] != [[]]
        and ex["p_phrase_tokens"]["attention_mask"].size() != (1, 0)
        else torch.empty((0, 1), dtype=torch.int, device=device)
    )
    ex["h_phrase_tokens"]["input_ids"] = (
        (ex["h_phrase_tokens"]["input_ids"].squeeze(0).to(device))
        if ex["h_phrase_tokens"]["input_ids"] != [[]]
        and ex["h_phrase_tokens"]["input_ids"].size() != (1, 0)
        else torch.empty((0, 1), dtype=torch.int, device=device)
    )
    ex["h_phrase_tokens"]["attention_mask"] = (
        (ex["h_phrase_tokens"]["attention_mask"].squeeze(0).to(device))
        if ex["h_phrase_tokens"]["attention_mask"] != [[]]
        and ex["h_phrase_tokens"]["attention_mask"].size() != (1, 0)
        else torch.empty((0, 1), dtype=torch.int, device=device)
    )
    ex["p_sent_tokens"]["input_ids"] = (
        ex["p_sent_tokens"]["input_ids"].squeeze(0).to(device)
    )
    ex["p_sent_tokens"]["attention_mask"] = (
        ex["p_sent_tokens"]["attention_mask"].squeeze(0).to(device)
    )
    ex["h_sent_tokens"]["input_ids"] = (
        ex["h_sent_tokens"]["input_ids"].squeeze(0).to(device)
    )
    ex["h_sent_tokens"]["attention_mask"] = (
        ex["h_sent_tokens"]["attention_mask"].squeeze(0).to(device)
    )
    num_sent_tokens_p = len(ex["p_sent_tokens"]["attention_mask"])
    num_sent_tokens_h = len(ex["h_sent_tokens"]["attention_mask"])
    ex["p_masks"] = (
        ex["p_masks"].squeeze(0).to(device)
        if ex["p_masks"] != [[]] and ex["p_masks"].size() != (1, 0)
        else torch.empty((0, num_sent_tokens_p), dtype=torch.int, device=device)
    )
    ex["h_masks"] = (
        ex["h_masks"].squeeze(0).to(device)
        if ex["h_masks"] != [[]] and ex["h_masks"].size() != (1, 0)
        else torch.empty((0, num_sent_tokens_h), dtype=torch.int, device=device)
    )
    ex["label"] = ex["label"].squeeze(0).to(device)
    ex["alignment"] = (
        ex["alignment"].squeeze(0).to(device)
        if ex["alignment"] != []
        else torch.empty((0, 2), dtype=torch.int, device=device)
    )

    return ex


def save_checkpoint(
    dataset_name: str,
    mode: str,
    model: EPRModel,
    optimizer: Optimizer,
    scheduler: StepLR,
    epoch: int,
    acc: float,
):
    torch.save(
        {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch,
            "accuracy": acc,
        },
        model_config[dataset_name][mode],
    )


def load_checkpoint(dataset_name: str, mode: str, map_location: device = None):
    path = model_config[dataset_name][mode]
    checkpoint = (
        torch.load(path)
        if not map_location
        else torch.load(path, map_location=map_location)
    )
    model: EPRModel = checkpoint["model"]
    optimizer: Optimizer = checkpoint["optimizer"]
    scheduler: StepLR = checkpoint["scheduler"]
    start_epoch: int = checkpoint["epoch"]
    acc: float = checkpoint["accuracy"]

    return model, optimizer, scheduler, start_epoch, acc
