"""
Runs the same training setup across a bunch of different seeds.

Why I need this:
  - reproducibility - everything (torch, numpy, random, cuda) gets
    seeded the same way each run
  - each seed saves its own checkpoint separately, so if one seed's
    save gets messed up somehow it doesn't take down any of the others
    (learned this one the hard way after accidentally overwriting
    progress with a single shared checkpoint file earlier on)
  - results get logged to a CSV as we go, so if this gets interrupted
    partway through we know exactly which seeds are already done and
    don't waste time redoing them

Usage (once train_loader/val_loader/test_loader/device already exist):

    from seed_sweep import set_seed, run_seed_sweep
    seeds = list(range(10))
    run_seed_sweep(seeds, build_model_fn=build_model, num_epochs=1)
"""

import os
import csv
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW


def set_seed(seed):
    """Sets every random seed that could affect training - python's own
    random module, numpy, and torch (both cpu and cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # this makes cudnn deterministic, which costs a bit of speed but
    # means runs are actually reproducible bit-for-bit, not just "close
    # enough". could turn these off if speed matters more than exact
    # reproducibility, but keeping them on for now
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_results_log_path(base_dir):
    return os.path.join(base_dir, "seed_results.csv")


def load_completed_seeds(results_path):
    """Checks which seeds already finished and got logged, so we don't
    redo work after a crash/restart."""
    completed = set()
    if os.path.exists(results_path):
        with open(results_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(int(row["seed"]))
    return completed


def append_result(results_path, row, fieldnames):
    """Adds one seed's result as a new row in the CSV - never overwrites
    what's already there."""
    file_exists = os.path.exists(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def upsert_result(results_path, row, fieldnames):
    """Writes the latest result for a seed without leaving duplicate rows."""
    results = {}
    if os.path.exists(results_path):
        with open(results_path, "r", newline="") as f:
            results = {int(existing["seed"]): existing for existing in csv.DictReader(f)}
    results[int(row["seed"])] = row
    temp_path = results_path + ".tmp"
    with open(temp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results[seed] for seed in sorted(results))
    os.replace(temp_path, results_path)


def load_epoch_history(history_path, seed):
    """Returns the recorded epoch rows for one seed, keyed by epoch."""
    rows = {}
    if os.path.exists(history_path):
        with open(history_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                if int(row["seed"]) == seed:
                    rows[int(row["epoch"])] = row
    return rows


def upsert_epoch_history(history_path, row):
    """Stores one per-epoch observation while preserving other seeds."""
    rows = {}
    if os.path.exists(history_path):
        with open(history_path, "r", newline="") as f:
            rows = {
                (int(existing["seed"]), int(existing["epoch"])): existing
                for existing in csv.DictReader(f)
            }
    rows[(int(row["seed"]), int(row["epoch"]))] = row
    temp_path = history_path + ".tmp"
    with open(temp_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "epoch", "train_loss", "val_ood_acc"])
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    os.replace(temp_path, history_path)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y, metadata in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    return correct / total


def run_seed_sweep(
    seeds,
    build_model_fn,
    train_loader,
    val_loader,
    test_loader,
    device,
    ckpt_base_dir="./checkpoints/seed_sweep",
    num_epochs=1,
    lr=1e-5,
    new_lr=None,
    log_every=100,
    grad_accum_steps=1,
    record_epoch_history=False,
    retrain_completed_without_history=False,
    run_config=None,
):
    """
    Args:
        seeds: the list of seeds to run, e.g. list(range(10)). worth
            writing this exact list down somewhere for the report in
            case the code changes later.
        build_model_fn: function that returns a fresh model with no
            arguments. needs to be called AFTER set_seed() so the random
            init actually uses that seed.
        ckpt_base_dir: every seed gets its own subfolder under here
            (seed_0/, seed_1/, etc) so nothing gets shared between seeds.
        new_lr: optional separate learning rate for any parameters with
            "pos_proj" or "gating_param" in their name (i.e. GPSA's new
            params) - lets them move faster than the rest of a pretrained
            model, which only needs small nudges. leave as None to just
            use one LR for everything.
        grad_accum_steps: if the batch size had to be lowered for VRAM
            reasons, this recovers the original effective batch size by
            accumulating gradients over a few smaller batches before
            actually calling optimizer.step().
    """
    os.makedirs(ckpt_base_dir, exist_ok=True)
    if run_config is not None:
        config_path = os.path.join(ckpt_base_dir, "run_config.json")
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2, sort_keys=True)
    results_path = get_results_log_path(ckpt_base_dir)
    fieldnames = ["seed", "epoch", "final_train_loss", "val_ood_acc", "test_ood_acc"]
    history_path = os.path.join(ckpt_base_dir, "training_history.csv")

    completed = load_completed_seeds(results_path)
    print(f"seeds already completed: {sorted(completed)}")

    for seed in seeds:
        seed_history = load_epoch_history(history_path, seed) if record_epoch_history else {}
        has_complete_history = all(epoch in seed_history for epoch in range(num_epochs))
        if seed in completed:
            if not record_epoch_history or has_complete_history:
                print(f"seed {seed}: already done, skipping")
                continue
            if not retrain_completed_without_history:
                print(f"seed {seed}: results exist but epoch history is incomplete; skipping")
                continue
            print(f"seed {seed}: retraining from scratch to record missing epoch history")

        print(f"\n=== seed {seed} ===")
        set_seed(seed)  # has to happen before the model gets built, otherwise
                        # the random init won't actually be tied to this seed

        seed_ckpt_dir = os.path.join(ckpt_base_dir, f"seed_{seed}")
        os.makedirs(seed_ckpt_dir, exist_ok=True)
        seed_ckpt_path = os.path.join(seed_ckpt_dir, "latest.pt")

        model = build_model_fn().to(device)

        # split params into "the new GPSA stuff" and "everything else" so
        # they can get different learning rates if new_lr was given -
        # otherwise just use one LR for the whole model like normal
        new_param_names = ("pos_proj", "gating_param")
        new_params = [p for n, p in model.named_parameters() if any(k in n for k in new_param_names)]
        other_params = [p for n, p in model.named_parameters() if not any(k in n for k in new_param_names)]

        if new_params and new_lr is not None:
            optimizer = AdamW([
                {"params": other_params, "lr": lr},
                {"params": new_params, "lr": new_lr},
            ], weight_decay=0.01)
        else:
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        start_epoch, start_step = 0, 0
        force_fresh = seed in completed and record_epoch_history and not has_complete_history
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        if os.path.exists(seed_ckpt_path) and not force_fresh:
            ckpt = torch.load(seed_ckpt_path, map_location=device)
            if (record_epoch_history and ckpt.get("step", -1) >= 0
                    and "epoch_loss_sum" not in ckpt):
                print(f"  seed {seed}: checkpoint predates history tracking; restarting from scratch")
                start_epoch, start_step = 0, 0
            else:
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                start_epoch, start_step = ckpt["epoch"], ckpt["step"] + 1
                epoch_loss_sum = ckpt.get("epoch_loss_sum", 0.0)
                epoch_sample_count = ckpt.get("epoch_sample_count", 0)
                print(f"  resuming seed {seed} from epoch {start_epoch} step {start_step}")

        final_loss = None
        optimizer.zero_grad()
        for epoch in range(start_epoch, num_epochs):
            model.train()
            for step, (x, y, metadata) in enumerate(train_loader):
                if epoch == start_epoch and step < start_step:
                    continue  # skip past whatever was already done before a restart

                x, y = x.to(device), y.to(device)
                logits = model(x)
                # divide by grad_accum_steps so the accumulated gradient
                # ends up the same as if this were one big batch, not
                # grad_accum_steps times too large
                loss = F.cross_entropy(logits, y) / grad_accum_steps
                loss.backward()
                final_loss = loss.item() * grad_accum_steps  # undo the scaling just for logging
                if record_epoch_history:
                    epoch_loss_sum += final_loss * x.size(0)
                    epoch_sample_count += x.size(0)

                if (step + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if step % log_every == 0:
                    print(f"  seed {seed} epoch {epoch} step {step} loss {final_loss:.4f}")

                if step % 1500 == 0 and step > 0:
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": step,
                        "epoch_loss_sum": epoch_loss_sum,
                        "epoch_sample_count": epoch_sample_count,
                    }, seed_ckpt_path)

            start_step = 0
            if record_epoch_history:
                train_loss = epoch_loss_sum / epoch_sample_count if epoch_sample_count else float("nan")
                val_acc = evaluate(model, val_loader, device)
                history_row = {
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_ood_acc": val_acc,
                }
                upsert_epoch_history(history_path, history_row)
                seed_history[epoch] = history_row
                epoch_loss_sum = 0.0
                epoch_sample_count = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "step": -1,
                    "epoch_loss_sum": epoch_loss_sum,
                    "epoch_sample_count": epoch_sample_count,
                }, seed_ckpt_path)

        if record_epoch_history and num_epochs - 1 in seed_history:
            final_loss = float(seed_history[num_epochs - 1]["train_loss"])
            val_acc = float(seed_history[num_epochs - 1]["val_ood_acc"])
        else:
            val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"  seed {seed}: val_ood_acc={val_acc:.4f} test_ood_acc={test_acc:.4f}")

        result_row = {
            "seed": seed,
            "epoch": num_epochs - 1,
            "final_train_loss": final_loss,
            "val_ood_acc": val_acc,
            "test_ood_acc": test_acc,
        }
        if record_epoch_history:
            upsert_result(results_path, result_row, fieldnames)
        else:
            append_result(results_path, result_row, fieldnames)

    print(f"\nall done. results log: {results_path}")


def summarize_results(ckpt_base_dir="./checkpoints/seed_sweep"):
    """Prints mean/std across every completed seed - this is the number
    that actually goes in the report, not any single seed on its own."""
    results_path = get_results_log_path(ckpt_base_dir)
    val_accs, test_accs = [], []
    with open(results_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_accs.append(float(row["val_ood_acc"]))
            test_accs.append(float(row["test_ood_acc"]))

    val_accs, test_accs = np.array(val_accs), np.array(test_accs)
    print(f"n = {len(val_accs)} seeds")
    print(f"val_ood_acc:  mean={val_accs.mean():.4f}  std={val_accs.std():.4f}")
    print(f"test_ood_acc: mean={test_accs.mean():.4f}  std={test_accs.std():.4f}")
    return val_accs, test_accs
