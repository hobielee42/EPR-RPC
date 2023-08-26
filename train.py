import argparse
import os
import pickle

import torch
from datasets import Dataset
from torch import Tensor, tensor
from torch.nn import NLLLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from models import EPRModel

data_config = {
    "snli": {
        "train": {
            "tokens": "data/encodings/snli/tokens/train_tokens.pkl",
            "alignments": "data/encodings/snli/alignments/train_alignments.pkl",
        },
        "val": {
            "tokens": "data/encodings/snli/tokens/val_tokens.pkl",
            "alignments": "data/encodings/snli/alignments/val_alignments.pkl",
        },
        "test": {
            "tokens": "data/encodings/snli/tokens/test_tokens.pkl",
            "alignments": "data/encodings/snli/alignments/test_alignments.pkl",
        },
        "model_path": "data/model/snli/",
    },
    "mnli": {
        "train": {
            "tokens": "data/encodings/mnli/tokens/train_tokens.pkl",
            "alignments": "data/encodings/mnli/alignments/train_alignments.pkl",
        },
        "val": {
            "tokens": "data/encodings/mnli/tokens/val_tokens.pkl",
            "alignments": "data/encodings/mnli/alignments/val_alignments.pkl",
        },
        "test": {
            "tokens": "data/encodings/mnli/tokens/test_tokens.pkl",
            "alignments": "data/encodings/mnli/alignments/test_alignments.pkl",
        },
        "model_path": "data/model/mnli/",
    },
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("local", "global", "concat"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--continue", action="store_true", default=False, dest="continue_"
    )
    args = parser.parse_args()
    return args.mode, args.epochs, args.continue_


def save_checkpoint(path, model, mode, optimizer, scheduler, epoch, acc):
    os.makedirs(path, exist_ok=True)
    torch.save(
        {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch,
            "accuracy": acc,
        },
        path + mode + ".pth",
    )


def load_checkpoint(path, mode):
    path = path + mode + ".pth"
    checkpoint = torch.load(path)
    return (
        checkpoint["model"],
        checkpoint["optimizer"],
        checkpoint["scheduler"],
        checkpoint["epoch"],
        checkpoint["accuracy"],
    )


def example_to_device(ex: dict, device: torch.device):
    ex["idx"] = ex["idx"].squeeze(0).to(device)
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
        else torch.empty((0, 1), dtype=torch.int, device=device)
    )
    ex["p_phrase_tokens"]["attention_mask"] = (
        (ex["p_phrase_tokens"]["attention_mask"].squeeze(0).to(device))
        if ex["p_phrase_tokens"]["attention_mask"] != [[]]
        else torch.empty((0, 1), dtype=torch.int, device=device)
    )
    ex["h_phrase_tokens"]["input_ids"] = (
        (ex["h_phrase_tokens"]["input_ids"].squeeze(0).to(device))
        if ex["h_phrase_tokens"]["input_ids"] != [[]]
        else torch.empty((0, 1), dtype=torch.int, device=device)
    )
    ex["h_phrase_tokens"]["attention_mask"] = (
        (ex["h_phrase_tokens"]["attention_mask"].squeeze(0).to(device))
        if ex["h_phrase_tokens"]["attention_mask"] != [[]]
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
        if ex["p_masks"] != [[]]
        else torch.empty((0, num_sent_tokens_p), dtype=torch.int, device=device)
    )
    ex["h_masks"] = (
        ex["h_masks"].squeeze(0).to(device)
        if ex["h_masks"] != [[]]
        else torch.empty((0, num_sent_tokens_h), dtype=torch.int, device=device)
    )
    ex["label"] = ex["label"].squeeze(0).to(device)
    ex["alignment"] = (
        ex["alignment"].squeeze(0).to(device)
        if ex["alignment"] != []
        else torch.empty((0, 2), dtype=torch.int, device=device)
    )

    return ex


def train_epoch(
    model: EPRModel,
    train_dl: DataLoader,
    optimizer: Optimizer,
    scheduler: StepLR,
):
    model.train()
    print(model.local_sbert.training)
    print(model.global_sbert.training)
    print(model.mlp.training)
    print(model.empty_tokens.training)
    len_epoch = len(train_dl)
    loss_fn = NLLLoss()
    batch_loss = tensor(0.0, device=device)
    epoch_loss = tensor(0.0, device=device)
    hit_count = 0
    num_batches = len_epoch // batch_size
    pbar = tqdm(iter(train_dl), total=len_epoch)
    for i, ex in enumerate(pbar):
        ex = example_to_device(ex, device)
        label: Tensor = ex["label"]

        pred: Tensor = model(ex)

        loss: Tensor = loss_fn(torch.log(pred.unsqueeze(0)), label.unsqueeze(0))
        batch_loss += loss
        epoch_loss += loss

        hit_count += int(torch.argmax(pred, dim=-1) == label)

        # end of batch
        if (i + 1) % batch_size == 0 or i >= len_epoch:
            i_batch = i // batch_size + 1
            this_batch_size = i % batch_size + 1
            batch_loss /= this_batch_size

            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_description(
                f"Batch {i_batch}/{num_batches}; Training loss: {batch_loss:.4f}; Training accuracy: {(hit_count/(i+1)):.4f}"
            )
            batch_loss = 0

    return hit_count / len_epoch


def evaluation(model: EPRModel, test_dl: DataLoader):
    model.eval()
    print(not model.local_sbert.training)
    print(not model.global_sbert.training)
    print(not model.mlp.training)
    print(not model.empty_tokens.training)
    hit_count = 0
    len_test = len(test_dl)
    pbar = tqdm(iter(test_dl))
    for i, test_ex in enumerate(pbar):
        input = example_to_device(test_ex, device)
        test_label = input["label"]

        with torch.no_grad():
            pred = model(input)
            hit_count += int(torch.argmax(pred, dim=-1) == test_label)

        pbar.set_description(f"Test accuracy: {hit_count/(i+1)}")

    return hit_count / len_test


lr = 5e-5
batch_size = 256

if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    # load dataset
    with open(data_config["snli"]["train"]["tokens"], "rb") as f:
        train_tokens: Dataset = pickle.load(f)
    with open(data_config["snli"]["train"]["alignments"], "rb") as f:
        train_alignments = pickle.load(f)
    with open(data_config["snli"]["test"]["tokens"], "rb") as f:
        test_tokens: Dataset = pickle.load(f)
    with open(data_config["snli"]["test"]["alignments"], "rb") as f:
        test_alignments = pickle.load(f)

    mode, num_epochs, continue_ = get_args()

    train_ds: Dataset = train_tokens.add_column(
        "alignment", train_alignments
    ).with_format("torch")
    test_ds: Dataset = test_tokens.add_column("alignment", test_alignments).with_format(
        "torch"
    )
    train_dl: DataLoader = DataLoader(train_ds, shuffle=True)
    test_dl: DataLoader = DataLoader(test_ds)

    if not continue_:
        epoch_start = 0
        track_acc = 0
        model = EPRModel(mode, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        num_training_steps = int(len(train_dl) / batch_size) * num_epochs
        num_warmup_steps = num_training_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    else:
        model, optimizer, scheduler, epoch_start, track_acc = load_checkpoint(
            data_config["snli"]["model_path"], mode
        )

    for epoch in range(epoch_start, num_epochs):
        train_acc = train_epoch(model, train_dl, optimizer, scheduler)

        with torch.no_grad():
            test_acc = evaluation(model, test_dl)

            if test_acc > track_acc:
                track_acc = test_acc
                print(f"Saving checkpoint with track accuracy = {track_acc}")
                save_checkpoint(
                    data_config["snli"]["model_path"],
                    model,
                    mode,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    track_acc,
                )
