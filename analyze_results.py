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
    result - makes it easy to see whether one model beats
    the other pretty much every time, or if it's more mixed. pairs with statistical test since it's the same idea
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
        labels.append(name)
        means.append(values.mean())
        errors.append(1.96 * values.std(ddof=1) / np.sqrt(len(values)))

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


class Camelyon17HFDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        image = example["image"].convert("RGB")
        return self.transform(image), example["label"]


def build_ood_test_loader(batch_size=16, num_workers=0):
    """Build the same center-2 OOD test loader used by training."""
    dataset = load_dataset("wltjr1007/Camelyon17-WILDS")
    all_data = concatenate_datasets([
        dataset["train"], dataset["validation"], dataset["test"]
    ])
    test_hf = all_data.filter(lambda example: example["center"] == 2)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return DataLoader(
        Camelyon17HFDataset(test_hf, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def collect_predictions(model_dirs, prediction_seeds, device, batch_size=16):
    """Evaluate selected checkpoint seeds and average their probabilities."""
    loader = build_ood_test_loader(batch_size=batch_size)
    factories = {
        "vanilla": lambda: PhikonClassifier(),
        "GPSA (default)": lambda: PhikonGPSAClassifier(local_layers=10, locality_strength=1.0,
                                                         gating_init=1.0),
        "GPSA (tuned)": lambda: PhikonGPSAClassifier(local_layers=10, locality_strength=1.0,
                                                       gating_init=0.0),
    }
    predictions = {}
    labels = None

    for name, base_dir in model_dirs.items():
        probability_sum = None
        used_seeds = []
        for seed in prediction_seeds:
            checkpoint_path = os.path.join(base_dir, f"seed_{seed}", "latest.pt")
            if not os.path.exists(checkpoint_path):
                print(f"warning: missing checkpoint {checkpoint_path}")
                continue
            model = factories[name]().to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            batch_probabilities, batch_labels = [], []
            with torch.no_grad():
                for images, batch_y in loader:
                    probabilities = torch.softmax(model(images.to(device)), dim=-1)[:, 1]
                    batch_probabilities.append(probabilities.cpu().numpy())
                    batch_labels.append(batch_y.numpy())
            current_probabilities = np.concatenate(batch_probabilities)
            current_labels = np.concatenate(batch_labels)
            if labels is None:
                labels = current_labels
            if probability_sum is None:
                probability_sum = np.zeros_like(current_probabilities)
            probability_sum += current_probabilities
            used_seeds.append(seed)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        if used_seeds:
            predictions[name] = probability_sum / len(used_seeds)
            print(f"{name}: averaged checkpoint seeds {used_seeds}")

    return labels, predictions


def binary_curves(labels, probabilities):
    """Return ROC and precision-recall points without requiring scikit-learn."""
    thresholds = np.r_[np.inf, np.sort(np.unique(probabilities))[::-1]]
    positives = max(1, np.sum(labels == 1))
    negatives = max(1, np.sum(labels == 0))
    fpr, tpr, precision, recall = [], [], [], []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = np.sum(predicted & (labels == 1))
        fp = np.sum(predicted & (labels == 0))
        fn = np.sum(~predicted & (labels == 1))
        fpr.append(fp / negatives)
        tpr.append(tp / positives)
        precision.append(tp / max(1, tp + fp))
        recall.append(tp / max(1, tp + fn))
    return np.array(fpr), np.array(tpr), np.array(precision), np.array(recall)


def area_under_curve(x, y):
    order = np.argsort(x)
    return np.trapezoid(y[order], x[order]) if hasattr(np, "trapezoid") else np.trapz(y[order], x[order])


def plot_prediction_diagnostics(model_dirs, device, prediction_seeds=(0,), batch_size=16):
    """Create normalized confusion matrices, ROC, and precision-recall plots."""
    labels, predictions = collect_predictions(model_dirs, prediction_seeds, device, batch_size)
    if not predictions:
        print("no checkpoint predictions available")
        return

    names = list(predictions)
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.8), squeeze=False)
    for axis, name in zip(axes[0], names):
        predicted = predictions[name] >= 0.5
        matrix = np.zeros((2, 2), dtype=float)
        for actual in (0, 1):
            for estimate in (0, 1):
                matrix[actual, estimate] = np.sum((labels == actual) & (predicted == estimate))
        matrix /= matrix.sum(axis=1, keepdims=True)
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        axis.set_title(name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        for row in range(2):
            for col in range(2):
                axis.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[0].tolist(), label="Row proportion")
    fig.suptitle("Normalized OOD test confusion matrices")
    plt.tight_layout()
    plt.savefig("./normalized_confusion_matrices.png", dpi=180)
    print("saved ./normalized_confusion_matrices.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probabilities in predictions.items():
        fpr, tpr, _, _ = binary_curves(labels, probabilities)
        auc = area_under_curve(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("OOD test ROC curves")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./roc_curves.png", dpi=180)
    print("saved ./roc_curves.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probabilities in predictions.items():
        _, _, precision, recall = binary_curves(labels, probabilities)
        auc = area_under_curve(recall, precision)
        ax.plot(recall, precision, label=f"{name} (AP={auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("OOD test precision-recall curves")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./precision_recall_curves.png", dpi=180)
    print("saved ./precision_recall_curves.png")
    plt.close()

#Ablation section:
def plot_hparam_search(results_dir="./checkpoints/gpsa_search", metric="test_ood_acc"):
    """Bar charts for the gating_init and new_lr sweeps."""
    gating_values = [0.0, 0.5, 1.0]
    lr_values = [1e-4, 5e-4, 1e-3]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, sweep_name, values in [(axes[0], "gating_init", gating_values),
                                     (axes[1], "new_lr", lr_values)]:
        means, stds, labels = [], [], []
        for v in values:
            path = f"{results_dir}/{sweep_name}_{v}/seed_results.csv"
            if not os.path.exists(path):
                continue
            accs = []
            with open(path, "r", newline="") as f:
                for row in csv.DictReader(f):
                    accs.append(float(row[metric]))
            accs = np.array(accs)
            means.append(accs.mean())
            stds.append(accs.std())
            labels.append(str(v))

        ax.bar(labels, means, yerr=stds, capsize=5, color="tab:blue", alpha=0.7)
        ax.set_xlabel(sweep_name)
        ax.set_ylabel(metric)
        ax.set_title(f"{sweep_name} sweep")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("./ablation_hparam_search.png", dpi=150)
    print("saved ablation_hparam_search.png")
    plt.close()

    rows = []
    for sweep_name, values in [("gating_init", gating_values), ("new_lr", lr_values)]:
        for v in values:
            path = f"{results_dir}/{sweep_name}_{v}/seed_results.csv"
            if not os.path.exists(path):
                continue
            accs = []
            with open(path, "r", newline="") as f:
                for row in csv.DictReader(f):
                    accs.append(float(row[metric]))
            accs = np.array(accs)
            rows.append({"sweep": sweep_name, "value": v, "n": len(accs),
                         "mean": round(accs.mean(), 4), "std": round(accs.std(), 4)})
    if rows:
        with open("./results_table_hparam_search.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print("saved results_table_hparam_search.csv")


def plot_local_layers_ablation(results_dir="./checkpoints/ablation_local_layers", metric="test_ood_acc"):
    """Bar chart for the local_layers sweep."""
    values = [4, 6, 8, 10, 12]
    means, stds, labels, rows = [], [], [], []

    for v in values:
        path = f"{results_dir}/local_layers_{v}/seed_results.csv"
        if not os.path.exists(path):
            continue
        accs = []
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                accs.append(float(row[metric]))
        accs = np.array(accs)
        means.append(accs.mean())
        stds.append(accs.std())
        labels.append(str(v))
        rows.append({"local_layers": v, "n": len(accs),
                     "mean": round(accs.mean(), 4), "std": round(accs.std(), 4)})

    if not means:
        print("no local_layers ablation results found")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, means, yerr=stds, capsize=5, color="tab:green", alpha=0.7)
    ax.set_xlabel("local_layers (number of GPSA layers)")
    ax.set_ylabel(metric)
    ax.set_title("local_layers ablation")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("./ablation_local_layers.png", dpi=150)
    print("saved ablation_local_layers.png")
    plt.close()

    with open("./results_table_local_layers.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("saved results_table_local_layers.csv")


def plot_epoch_ablation(results_dir="./checkpoints/ablation_epochs", metric="test_ood_acc"):
    """
    Line chart of accuracy vs epoch count - the closest thing to a
    convergence curve given what i actually logged (final accuracy at
    each of 1/3/5 epochs, not per-step). evidence for
    "performance over epochs".
    """
    values = [1, 3, 5]
    means, stds, rows = [], [], []

    for v in values:
        path = f"{results_dir}/epochs_{v}/seed_results.csv"
        if not os.path.exists(path):
            continue
        accs = []
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                accs.append(float(row[metric]))
        accs = np.array(accs)
        means.append(accs.mean())
        stds.append(accs.std())
        rows.append({"epochs": v, "n": len(accs),
                     "mean": round(accs.mean(), 4), "std": round(accs.std(), 4)})

    if not means:
        print("no epoch ablation results found")
        return

    plotted_epochs = values[:len(means)]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(plotted_epochs, means, yerr=stds, marker="o", capsize=5, color="tab:orange")
    ax.set_xlabel("epochs")
    ax.set_ylabel(metric)
    ax.set_title("Accuracy vs training length (tuned config)")
    ax.set_xticks(plotted_epochs)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./ablation_epochs.png", dpi=150)
    print("saved ablation_epochs.png")
    plt.close()

    with open("./results_table_epochs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("saved results_table_epochs.csv")


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
