"""
Puts together all the tables and plots for the results section.

Reads straight from the seed_results.csv files each run already wrote,
so nothing here needs to retrain anything - just loading numbers and
turning them into something presentable (tables that actually get saved
to disk instead of just printed in a terminal, boxplots, that kind of
thing). It also records paired tests and ablation summaries. It does not run
checkpoint prediction diagnostics; convergence is plotted from the epoch-
ablation histories when available.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import torch.nn as nn
from transformers import ViTModel
from phikon_gpsa import inject_gpsa
from paired_seed_comparison import compare_results
from gpsa_hparam_search import summarize_search
from ablation_test import summarize_ablation, summarize_epoch_ablation


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
        sample_std = values.std(ddof=1) if len(values) > 1 else float("nan")
        rows.append({
            "model": name,
            "metric": metric,
            "n_runs": len(values),
            "mean": round(values.mean(), 4),
            "std": round(sample_std, 4),
            "std_method": "sample SD (ddof=1)",
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


def holm_adjust(p_values):
    """Return Holm-adjusted p-values in the input order, preserving NaNs."""
    adjusted = [float("nan")] * len(p_values)
    valid = [index for index, value in enumerate(p_values) if np.isfinite(value)]
    ordered = sorted(valid, key=lambda index: p_values[index])
    running_max = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        running_max = max(running_max, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def build_comprehensive_results_csv(
    model_dirs,
    paired_results,
    ablation_results,
    save_path="./statistical_results.csv",
):
    """Write raw seed records, summaries, paired tests, and ablations."""
    fields = [
        "record_type", "experiment", "comparison", "model", "baseline_model",
        "metric", "seed", "seeds", "n", "n_pairs", "mean",
        "sample_sd", "value", "baseline_mean", "method_mean", "mean_difference",
        "ci95_lower", "ci95_upper", "test_name", "statistic", "p_value",
        "holm_adjusted_p", "alpha", "significant", "effect_size_name", "effect_size",
        "training_seconds",
        "validation_seconds", "selected_epoch", "selected_checkpoint", "total_parameters",
        "trainable_parameters", "configuration", "condition", "std_method", "ci_method",
        "higher_is_better", "timing_scope", "notes",
    ]
    rows = []

    def add_seed_records(experiment, model, base_dir, condition=""):
        results = load_seed_results(base_dir)
        if not results:
            return
        metadata = {}
        metadata_path = os.path.join(base_dir, "run_metadata.csv")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", newline="") as f:
                metadata = {int(row["seed"]): row for row in csv.DictReader(f)}

        for seed, result in results.items():
            run = metadata.get(seed, {})
            notes = "" if run else "legacy run; runtime/config metadata not recorded"
            for metric in ("val_ood_acc", "test_ood_acc"):
                rows.append({
                    "record_type": "seed_level_result",
                    "experiment": experiment,
                    "model": model,
                    "condition": condition,
                    "metric": metric,
                    "seed": seed,
                    "value": float(result[metric]),
                    "training_seconds": run.get("training_seconds", ""),
                    "validation_seconds": run.get("validation_seconds", ""),
                    "timing_scope": run.get("timing_scope", ""),
                    "selected_epoch": run.get("selected_epoch", result.get("epoch", "")),
                    "selected_checkpoint": run.get("selected_checkpoint", ""),
                    "total_parameters": run.get("total_parameters", ""),
                    "trainable_parameters": run.get("trainable_parameters", ""),
                    "configuration": run.get("run_config_json", ""),
                    "notes": notes,
                })

        for metric in ("val_ood_acc", "test_ood_acc"):
            values = np.array([float(result[metric]) for result in results.values()])
            rows.append({
                "record_type": "model_summary",
                "experiment": experiment,
                "model": model,
                "condition": condition,
                "metric": metric,
                "n": len(values),
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "std_method": "sample SD (ddof=1)",
                "higher_is_better": True,
            })

        for seed, run in metadata.items():
            rows.append({
                "record_type": "run_metadata",
                "experiment": experiment,
                "model": model,
                "condition": condition,
                "seed": seed,
                "training_seconds": run.get("training_seconds", ""),
                "validation_seconds": run.get("validation_seconds", ""),
                "timing_scope": run.get("timing_scope", ""),
                "selected_epoch": run.get("selected_epoch", ""),
                "selected_checkpoint": run.get("selected_checkpoint", ""),
                "total_parameters": run.get("total_parameters", ""),
                "trainable_parameters": run.get("trainable_parameters", ""),
                "configuration": run.get("run_config_json", ""),
            })

    for model, base_dir in model_dirs.items():
        add_seed_records("main", model, base_dir)

    for comparison in paired_results:
        comparison_name = comparison["comparison"]
        tests = ("wilcoxon", "ttest")
        adjusted_by_test = {
            test: holm_adjust([item[f"{test}_p"] for item in paired_results])
            for test in tests
        }
        current_index = paired_results.index(comparison)
        for test in tests:
            p_value = comparison[f"{test}_p"]
            adjusted_p = adjusted_by_test[test][current_index]
            rows.append({
                "record_type": "paired_test",
                "experiment": "main",
                "comparison": comparison_name,
                "model": comparison["method"],
                "baseline_model": comparison["baseline"],
                "metric": "test_ood_acc",
                "n_pairs": comparison["n_pairs"],
                "seeds": ";".join(map(str, comparison["shared_seeds"])),
                "baseline_mean": comparison["vanilla_mean"],
                "method_mean": comparison["gpsa_mean"],
                "mean_difference": comparison["mean_difference"],
                "ci95_lower": comparison["ci95_lower"],
                "ci95_upper": comparison["ci95_upper"],
                "ci_method": "unadjusted two-sided paired t interval",
                "test_name": test,
                "statistic": comparison[f"{test}_stat"],
                "p_value": p_value,
                "holm_adjusted_p": adjusted_p,
                "alpha": comparison["alpha"],
                "significant": adjusted_p < comparison["alpha"],
                "effect_size_name": "Cohen's dz",
                "effect_size": comparison["cohens_d"],
                "higher_is_better": True,
                "notes": "Holm correction across the three planned paired comparisons",
            })

    for summary in ablation_results:
        experiment = summary["ablation"]
        condition = summary["condition"]
        add_seed_records(experiment, "GPSA-phikon", summary["results_dir"], condition)
        for metric_prefix, metric_name in (("val", "val_ood_acc"), ("test", "test_ood_acc")):
            rows.append({
                "record_type": "ablation_summary",
                "experiment": experiment,
                "model": "GPSA-phikon",
                "condition": condition,
                "metric": metric_name,
                "n": summary["n_runs"],
                "mean": summary[f"{metric_prefix}_mean"],
                "sample_sd": summary[f"{metric_prefix}_sample_sd"],
                "std_method": "sample SD (ddof=1)",
                "higher_is_better": True,
            })

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved comprehensive statistical results to {save_path}")
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
    ax.boxplot(data, tick_labels=labels, showmeans=True)
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


def plot_mean_accuracy_ci(model_dirs, metric="test_ood_acc",
                          save_path="./mean_accuracy_confidence_intervals.png"):
    """Plot mean accuracy with an approximate 95% confidence interval."""
    labels, means, errors = [], [], []
    for name, base_dir in model_dirs.items():
        results = load_seed_results(base_dir)
        if not results:
            continue
        values = np.array([float(row[metric]) for row in results.values()])
        if len(values) < 2:
            continue
        labels.append(name)
        means.append(values.mean())
        errors.append(t.ppf(0.975, df=len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))

    if not means:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=errors, capsize=5, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylabel(metric)
    ax.set_title(f"Mean {metric} with 95% confidence intervals")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    print(f"saved {save_path}")
    plt.close()


def plot_seed_heatmap(model_dirs, metric="test_ood_acc",
                      save_path="./seed_accuracy_heatmap.png"):
    """Plot one accuracy cell per model and seed."""
    seed_values = {name: load_seed_results(path) for name, path in model_dirs.items()}
    shared_seeds = sorted(set.intersection(*(set(values) for values in seed_values.values())))
    if not shared_seeds:
        print("no shared seeds available for heatmap")
        return

    matrix = np.array([
        [float(seed_values[name][seed][metric]) for seed in shared_seeds]
        for name in model_dirs
    ])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=matrix.min(), vmax=matrix.max())
    ax.set_yticks(range(len(model_dirs)), list(model_dirs))
    ax.set_xticks(range(len(shared_seeds)), shared_seeds)
    ax.set_xlabel("Seed")
    ax.set_title(f"{metric} across shared seeds")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label=metric)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    print(f"saved {save_path}")
    plt.close()


def plot_epoch_ablation_convergence(
    results_dir="./checkpoints/ablation_epochs",
    save_path="./epoch_ablation_convergence.png",
):
    """Plot genuine epoch histories for each available epoch-ablation run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = False
    palette = plt.get_cmap("tab10")

    condition_dirs = []
    if os.path.isdir(results_dir):
        condition_dirs = sorted(
            (entry for entry in os.listdir(results_dir) if entry.startswith("epochs_")),
            key=lambda entry: int(entry.split("_", 1)[1]),
        )

    for condition in condition_dirs:
        history_path = os.path.join(results_dir, condition, "training_history.csv")
        if not os.path.exists(history_path):
            print(f"no training history for {condition}")
            continue
        by_seed = {}
        with open(history_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                by_seed.setdefault(int(row["seed"]), []).append({
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_ood_acc": float(row["val_ood_acc"]),
                })
        if not by_seed:
            print(f"no per-epoch rows in {history_path}")
            continue

        plotted = True
        epoch_count = int(condition.split("_", 1)[1])
        condition_color = palette((epoch_count - 1) % 10)
        for seed_rows in by_seed.values():
            seed_rows.sort(key=lambda row: row["epoch"])
            epochs = [row["epoch"] + 1 for row in seed_rows]
            axes[0].plot(epochs, [row["train_loss"] for row in seed_rows],
                         color=condition_color, alpha=0.2, linewidth=0.8)
            axes[1].plot(epochs, [row["val_ood_acc"] for row in seed_rows],
                         color=condition_color, alpha=0.2, linewidth=0.8)

        epochs = sorted({row["epoch"] for rows in by_seed.values() for row in rows})
        for axis, metric in zip(axes, ("train_loss", "val_ood_acc")):
            means, deviations = [], []
            for epoch in epochs:
                values = [row[metric] for rows in by_seed.values() for row in rows
                          if row["epoch"] == epoch]
                means.append(float(np.mean(values)))
                deviations.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            means = np.asarray(means)
            deviations = np.asarray(deviations)
            display_epochs = [epoch + 1 for epoch in epochs]
            seed_count = len(by_seed)
            axis.plot(display_epochs, means, color=condition_color, linewidth=2,
                      marker="o", label=f"{epoch_count} epochs ({seed_count} seeds)")
            axis.fill_between(display_epochs, means - deviations, means + deviations,
                              color=condition_color, alpha=0.15)

    if not plotted:
        plt.close(fig)
        print("no epoch-ablation histories found; rerun training to generate convergence curves")
        return

    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Mean training loss")
    axes[1].set(title="Validation performance", xlabel="Epoch", ylabel="OOD accuracy")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("Epoch-count convergence by training duration (faint: individual seeds; bands: mean +/- sample SD)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    print(f"saved {save_path}")
    plt.close(fig)


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
    plot_mean_accuracy_ci(
        model_dirs,
        metric="test_ood_acc",
        save_path="./mean_test_accuracy_confidence_intervals.png",
    )
    plot_mean_accuracy_ci(
        model_dirs,
        metric="val_ood_acc",
        save_path="./mean_val_accuracy_confidence_intervals.png",
    )
    plot_seed_heatmap(model_dirs, metric="test_ood_acc")
    plot_epoch_ablation_convergence()

    # Paired comparisons use the same seed identifiers. The unified report
    # applies Holm correction across these two planned GPSA-vs-vanilla tests.
    paired_results = []
    for comparison_name, gpsa_label, gpsa_dir in (
        ("vanilla vs GPSA (default)", "GPSA (default)", "./checkpoints/seed_sweep_gpsa"),
        ("vanilla vs GPSA (tuned)", "GPSA (tuned)", "./checkpoints/seed_sweep_gpsa_tuned"),
    ):
        comparison = compare_results(
            vanilla_dir="./checkpoints/seed_sweep_vanilla",
            gpsa_dir=gpsa_dir,
        )
        comparison.update({
            "comparison": comparison_name,
            "baseline": "vanilla",
            "method": gpsa_label,
        })
        paired_results.append(comparison)

    tuned_comparison = compare_results(
        vanilla_dir="./checkpoints/seed_sweep_gpsa",
        gpsa_dir="./checkpoints/seed_sweep_gpsa_tuned",
        baseline_name="GPSA (default)",
        method_name="GPSA (tuned)",
    )
    tuned_comparison.update({
        "comparison": "GPSA (default) vs GPSA (tuned)",
        "baseline": "GPSA (default)",
        "method": "GPSA (tuned)",
    })
    paired_results.append(tuned_comparison)
    print("\nDirect tuned-vs-untuned GPSA comparison (test OOD accuracy)")
    print(
        f"{'Comparison':<34} {'n':>3} {'default':>10} {'tuned':>10} "
        f"{'delta (pp)':>11} {'95% CI (pp)':>22} {'Wilcoxon p':>12}"
    )
    print(
        f"{'GPSA (default) vs GPSA (tuned)':<34} "
        f"{tuned_comparison['n_pairs']:>3} "
        f"{tuned_comparison['vanilla_mean'] * 100:>9.2f}% "
        f"{tuned_comparison['gpsa_mean'] * 100:>9.2f}% "
        f"{tuned_comparison['mean_difference'] * 100:>+10.2f} "
        f"[{tuned_comparison['ci95_lower'] * 100:+.2f}, "
        f"{tuned_comparison['ci95_upper'] * 100:+.2f}] "
        f"{tuned_comparison['wilcoxon_p']:>12.4g}"
    )

    ablation_results = (
        summarize_search()
        + summarize_ablation()
        + summarize_epoch_ablation()
    )
    plot_paired_slope("./checkpoints/seed_sweep_vanilla", "./checkpoints/seed_sweep_gpsa_tuned",
                       "vanilla", "GPSA (tuned)")
    plot_paired_slope("./checkpoints/seed_sweep_vanilla", "./checkpoints/seed_sweep_gpsa",
                       "vanilla", "GPSA (default)")

    build_comprehensive_results_csv(
        model_dirs,
        paired_results,
        ablation_results,
    )

    report_model_sizes()
