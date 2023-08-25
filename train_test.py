import argparse
import os
import pickle

import torch
from torch.optim.lr_scheduler import StepLR
from torch import nn, Tensor, tensor
from datasets import Dataset
from torch.nn import NLLLoss
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("local", "global", "concat"), default="local")
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


def example_to_device(ex: dict, device: torch.device):
    ex["idx"] = ex["idx"][0].to(device)
    ex["p_phrases_idx"] = ex["p_phrases_idx"][0].to(device)
    ex["h_phrases_idx"] = ex["h_phrases_idx"][0].to(device)
    ex["p_phrase_tokens"]["input_ids"] = ex["p_phrase_tokens"]["input_ids"][0].to(
        device
    )
    ex["p_phrase_tokens"]["attention_mask"] = ex["p_phrase_tokens"]["attention_mask"][
        0
    ].to(device)
    ex["h_phrase_tokens"]["input_ids"] = ex["h_phrase_tokens"]["input_ids"][0].to(
        device
    )
    ex["h_phrase_tokens"]["attention_mask"] = ex["h_phrase_tokens"]["attention_mask"][
        0
    ].to(device)
    ex["p_sent_tokens"]["input_ids"] = ex["p_sent_tokens"]["input_ids"][0].to(device)
    ex["p_sent_tokens"]["attention_mask"] = ex["p_sent_tokens"]["attention_mask"][0].to(
        device
    )
    ex["h_sent_tokens"]["input_ids"] = ex["h_sent_tokens"]["input_ids"][0].to(device)
    ex["h_sent_tokens"]["attention_mask"] = ex["h_sent_tokens"]["attention_mask"][0].to(
        device
    )
    ex["p_masks"] = ex["p_masks"][0].to(device)
    ex["h_masks"] = ex["h_masks"][0].to(device)
    ex["label"] = ex["label"][0].to(device)
    ex["alignment"] = ex["alignment"][0].to(device)

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
    for test_ex in tqdm(iter(test_dl)):
        input = example_to_device(test_ex, device)
        test_label = input["label"]

        with torch.no_grad():
            pred = model(input)
            hit_count += int(torch.argmax(pred, dim=-1) == test_label)


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

    mode, num_epochs, continue_ = "local", 2, False

    train_ds_b4_torch: Dataset = train_tokens.add_column("alignment", train_alignments)
    train_ds: Dataset = train_ds_b4_torch.shuffle().with_format("torch")
    test_ds: Dataset = test_tokens.add_column("alignment", test_alignments).with_format(
        "torch"
    )

    test_run_ratio = 0.005
    len_train = len(train_ds)

    train_dl: DataLoader = DataLoader(
        train_ds.select(range(int(len_train * test_run_ratio)))
    )
    # train_dl: DataLoader = DataLoader(train_ds, shuffle=True)
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

    train_epoch(model, train_dl, optimizer, scheduler)

    # for epoch in range(epoch_start, num_epochs):
    #     train_acc = train_epoch(model, train_dl, optimizer, scheduler)
