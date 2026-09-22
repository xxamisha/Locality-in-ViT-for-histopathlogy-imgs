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
    results_path = get_results_log_path(ckpt_base_dir)
    fieldnames = ["seed", "epoch", "final_train_loss", "val_ood_acc", "test_ood_acc"]

    completed = load_completed_seeds(results_path)
    print(f"seeds already completed: {sorted(completed)}")

    for seed in seeds:
        if seed in completed:
            print(f"seed {seed}: already done, skipping")
            continue

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
        if os.path.exists(seed_ckpt_path):
            ckpt = torch.load(seed_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch, start_step = ckpt["epoch"], ckpt["step"] + 1
            print(f"  resuming seed {seed} from epoch {start_epoch} step {start_step}")

        model.train()
        final_loss = None
        optimizer.zero_grad()
        for epoch in range(start_epoch, num_epochs):
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
                    }, seed_ckpt_path)

            start_step = 0

        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"  seed {seed}: val_ood_acc={val_acc:.4f} test_ood_acc={test_acc:.4f}")

        append_result(results_path, {
            "seed": seed,
            "epoch": num_epochs - 1,
            "final_train_loss": final_loss,
            "val_ood_acc": val_acc,
            "test_ood_acc": test_acc,
        }, fieldnames)

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
