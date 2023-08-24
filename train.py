import argparse
import os
import pickle

import torch
from torch import nn, Tensor, tensor
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from models import EPRModel

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

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
        path + mode + "pth",
    )


def load_checkpoint(path, mode):
    path = path + mode + "pth"
    checkpoint = torch.load(path)
    return (
        checkpoint["model"],
        checkpoint["optimizer"],
        checkpoint["scheduler"],
        checkpoint["epoch"],
        checkpoint["accuracy"],
    )


def train_epoch(
    model: EPRModel,
    train_dl: DataLoader,
    optimizer: Optimizer,
    scheduler: lr_scheduler,
):
    len_epoch = len(train_dl)
    loss_fn = CrossEntropyLoss()
    batch_loss = tensor(0, device=device)
    epoch_loss = tensor(0, device=device)
    hit_count = 0
    num_batches = len_epoch // batch_size
    pbar = tqdm(train_dl, total=len_epoch)
    for i, ex in pbar:
        input, label = ex, ex["label"]

        pred = model(input)

        loss: Tensor = loss_fn(pred, label)
        batch_loss += loss
        epoch_loss += loss

        hit_count += int(torch.argmax(pred) == label)

        # end of batch
        if (i + 1) % batch_size == 0 or i >= len_epoch:
            this_batch_size = i % batch_size + 1
            batch_loss /= this_batch_size
            batch_acc = hit_count / this_batch_size

            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_description(
                f"Training loss: {batch_loss.item()}; Training accuracy: {batch_acc}"
            )

            batch_loss = 0

    return hit_count / len_epoch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("local", "global", "concat"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--continue", action="store_true", default=False, dest="continue_"
    )
    args = parser.parse_args()
    return args.mode, args.epochs, args.continue_


lr = 5e-5
batch_size = 256

if __name__ == "__main__":
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
    train_dl: DataLoader = DataLoader(train_ds)
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
        model, optimizer, scheduler, epoch, track_acc = load_checkpoint(
            data_config["snli"]["model_path"], mode
        )

