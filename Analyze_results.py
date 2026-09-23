"""
Puts together all the tables and plots for the results section.

Reads straight from the seed_results.csv files each run already wrote,
so nothing here needs to retrain anything - just loading numbers and
turning them into something presentable (tables that actually get saved
to disk instead of just printed in a terminal, boxplots, that kind of
thing).

One thing this does NOT do: convergence curves (accuracy over epochs).
That would need per-epoch evaluation logged WHILE training happens, and
the runs so far only save the final accuracy at the end, not anything
in between. If the epoch ablation gets rerun with per-epoch eval added,
that data could go here too, but as it stands there's nothing to plot
for that.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import ViTModel
from phikon_gpsa import inject_gpsa


def load_seed_results(base_dir):
    """Loads seed_results.csv into a dict keyed by seed number."""
    path = os.path.join(base_dir, "seed_results.csv")
    results = {}
    if not os.path.exists(path):
        print(f"warning: couldn't find results at {path}")
        return results
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            results[int(row["seed"])] = row
    return results


def build_summary_table(model_dirs, metric="test_ood_acc", higher_is_better=True):
    """
    model_dirs is just {model_name: folder_path}.
    Prints a table and also saves it to results_table.csv so it's not
    just sitting in the terminal output (screenshots of console output
    aren't great evidence for a report).
    """
    rows = []
    for name, base_dir in model_dirs.items():
        results = load_seed_results(base_dir)
        if not results:
            continue
        values = np.array([float(r[metric]) for r in results.values()])
        rows.append({
            "model": name,
            "metric": metric,
            "n_runs": len(values),
            "mean": round(values.mean(), 4),
            "std": round(values.std(), 4),
            "min": round(values.min(), 4),
            "max": round(values.max(), 4),
            "higher_is_better": higher_is_better,
        })

    out_path = f"./results_table_{metric}.csv"
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved table to {out_path}")

    print(f"\n{'model':>20} | {'metric':>13} | {'n':>3} | {'mean':>7} | {'std':>7} | {'min':>7} | {'max':>7}")
    print("-" * 80)
    for r in rows:
        print(f"{r['model']:>20} | {r['metric']:>13} | {r['n_runs']:>3} | "
              f"{r['mean']:>7.4f} | {r['std']:>7.4f} | {r['min']:>7.4f} | {r['max']:>7.4f}")
    return rows


def plot_boxplot(model_dirs, metric="test_ood_acc", save_path="./boxplot_run_distributions.png"):
    """Boxplot showing the spread across seeds for each model, not just
    the average - basically the same kind of figure the lecture slides
    used as an example."""
    data, labels = [], []
    for name, base_dir in model_dirs.items():
        results = load_seed_results(base_dir)
        if not results:
            continue
        values = [float(r[metric]) for r in results.values()]
        data.append(values)
        labels.append(name)

    if not data:
        print("nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel(metric)
    ax.set_title(f"Spread across seeds ({metric})")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved boxplot to {save_path}")
    plt.close()


def plot_paired_slope(dir_a, dir_b, label_a, label_b, metric="test_ood_acc", save_path=None):
    """
    Draws a line for each seed from its dir_a result to its dir_b
    result - makes it easy to see at a glance whether one model beats
    the other pretty much every time, or if it's more mixed. Goes
    nicely with the paired statistical test since it's the same idea
    just visualized.
    """
    a_results = load_seed_results(dir_a)
    b_results = load_seed_results(dir_b)
    shared_seeds = sorted(set(a_results) & set(b_results))

    if not shared_seeds:
        print(f"no shared seeds between {label_a} and {label_b}")
        return

    if save_path is None:
        save_path = f"./paired_slope_{label_a}_vs_{label_b}.png".replace(" ", "_")

    fig, ax = plt.subplots(figsize=(5, 6))
    for seed in shared_seeds:
        a_val = float(a_results[seed][metric])
        b_val = float(b_results[seed][metric])
        color = "tab:green" if b_val > a_val else "tab:red"
        ax.plot([0, 1], [a_val, b_val], marker="o", color=color, alpha=0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel(metric)
    ax.set_title(f"{label_a} vs {label_b} ({len(shared_seeds)} shared seeds)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved {save_path}")
    plt.close()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def report_model_sizes():
    """Just loads each model type long enough to count its parameters -
    doesn't actually train anything."""
    print("\n--- parameter counts ---")

    class PhikonClassifier(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
            self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)
        def forward(self, x):
            return self.head(self.backbone(x).last_hidden_state[:, 0])

    class PhikonGPSAClassifier(nn.Module):
        def __init__(self, local_layers=10, gating_init=1.0, num_classes=2):
            super().__init__()
            self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
            inject_gpsa(self.backbone, local_layers=local_layers, gating_init=gating_init)
            self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)
        def forward(self, x):
            return self.head(self.backbone(x).last_hidden_state[:, 0])

    # gating_init doesn't actually change the parameter count, just the
    # starting value, but passing it anyway to match what's actually used
    models_to_check = [
        ("vanilla-phikon", PhikonClassifier()),
        ("GPSA-phikon (tuned config)", PhikonGPSAClassifier(local_layers=10, gating_init=0.0)),
    ]

    for name, model in models_to_check:
        total, trainable = count_params(model)
        print(f"{name}: {total/1e6:.2f}M total params, {trainable/1e6:.2f}M trainable")


if __name__ == "__main__":
    # all three models we care about now - vanilla baseline, GPSA with
    # the paper's original default settings, and GPSA with our tuned
    # settings (gating_init=0.0, new_lr=1e-3) which is the actual
    # headline result
    model_dirs = {
        "vanilla": "./checkpoints/seed_sweep_vanilla",
        "GPSA (default)": "./checkpoints/seed_sweep_gpsa",
        "GPSA (tuned)": "./checkpoints/seed_sweep_gpsa_tuned",
    }

    build_summary_table(model_dirs, metric="test_ood_acc")
    build_summary_table(model_dirs, metric="val_ood_acc")
    plot_boxplot(model_dirs, metric="test_ood_acc")

    # paired comparisons - the tuned one is the one that actually matters
    # for the headline claim, keeping the default one too since it shows
    # why the tuning was worth doing in the first place
    plot_paired_slope("./checkpoints/seed_sweep_vanilla", "./checkpoints/seed_sweep_gpsa_tuned",
                       "vanilla", "GPSA (tuned)")
    plot_paired_slope("./checkpoints/seed_sweep_vanilla", "./checkpoints/seed_sweep_gpsa",
                       "vanilla", "GPSA (default)")

    report_model_sizes()
