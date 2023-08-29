import argparse
import pickle

import torch
from datasets import Dataset
from torch import Tensor
from torch.nn import NLLLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import data_config, device
from evaluation import sentence_accuracy
from models import EPRModel
from util import example_to_device, save_checkpoint, load_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("local", "global", "concat"))
    parser.add_argument("--dataset", default="snli")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--continue", action="store_true", default=False, dest="continue_"
    )
    args = parser.parse_args()
    return args.mode, args.dataset, args.epochs, args.continue_


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
    batch_loss = 0
    epoch_hit_count = 0
    batch_hit_count = 0
    num_batches = len_epoch // batch_size
    pbar = tqdm(iter(train_dl), total=len_epoch)
    for i, ex in enumerate(pbar):
        ex = example_to_device(ex, device)
        label: Tensor = ex["label"]

        try:
            pred: Tensor = model(ex)
        except:
            continue

        loss: Tensor = loss_fn(torch.log(pred.unsqueeze(0)), label.unsqueeze(0))
        batch_loss += loss

        hit = torch.argmax(pred, dim=-1) == label
        batch_hit_count += int(hit)
        epoch_hit_count += int(hit)

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
                f"Batch {i_batch}/{num_batches}; Loss: {batch_loss:.4f}; Accuracy: {(batch_hit_count/this_batch_size):.4f}"
            )
            batch_hit_count = 0
            batch_loss = 0

    return epoch_hit_count / len_epoch


lr = 5e-5
batch_size = 256

if __name__ == "__main__":
    print(f"Device: {device}")

    mode, dataset_name, num_epochs, continue_ = get_args()

    with open(data_config[dataset_name]["train"]["tokens"], "rb") as f:
        train_tokens: Dataset = pickle.load(f)
    with open(data_config[dataset_name]["train"]["alignments"], "rb") as f:
        train_alignments = pickle.load(f)
    with open(data_config[dataset_name]["test"]["tokens"], "rb") as f:
        test_tokens: Dataset = pickle.load(f)
    with open(data_config[dataset_name]["test"]["alignments"], "rb") as f:
        test_alignments = pickle.load(f)
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
            dataset_name, mode
        )

    for epoch in range(epoch_start, num_epochs):
        train_acc = train_epoch(model, train_dl, optimizer, scheduler)

        with torch.no_grad():
            test_acc = sentence_accuracy(model, test_dl)

            if test_acc > track_acc:
                track_acc = test_acc
                print(f"Saving checkpoint with track accuracy = {track_acc}")
                save_checkpoint(
                    dataset_name,
                    mode,
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    track_acc,
                )
