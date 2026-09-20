"""
Compare vanilla Phikon with GPSA-Phikon using the same random seeds.

The same seeds are used for both models so that the results are paired.
"""

from transformers import ViTModel
import torch.nn as nn
from scipy.stats import wilcoxon, ttest_rel
import numpy as np
import csv

from phikon_gpsa import inject_gpsa
from seed_sweep import run_seed_sweep


# Normal pretrained Phikon model
class PhikonClassifier(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = ViTModel.from_pretrained(
            "owkin/phikon",
            add_pooling_layer=False
        )

        self.head = nn.Linear(
            self.backbone.config.hidden_size,
            num_classes
        )

    def forward(self, x):

        output = self.backbone(x)

        # Use the CLS token for classification
        cls_token = output.last_hidden_state[:, 0]

        return self.head(cls_token)


# Phikon with GPSA added to the first few layers
class PhikonGPSAClassifier(nn.Module):

    def __init__(
        self,
        local_layers=10,
        locality_strength=1.0,
        num_classes=2
    ):
        super().__init__()

        self.backbone = ViTModel.from_pretrained(
            "owkin/phikon",
            add_pooling_layer=False
        )

        # Replace the first few attention layers with GPSA
        inject_gpsa(
            self.backbone,
            local_layers=local_layers,
            locality_strength=locality_strength
        )

        self.head = nn.Linear(
            self.backbone.config.hidden_size,
            num_classes
        )

    def forward(self, x):

        output = self.backbone(x)

        # Use the CLS token for classification
        cls_token = output.last_hidden_state[:, 0]

        return self.head(cls_token)


def run_paired_comparison(
    train_loader,
    val_loader,
    test_loader,
    device,
    seeds=None,
    num_epochs=1
):

    # Use the same seeds for both models
    if seeds is None:
        seeds = list(range(10))

    print("=== Vanilla Phikon ===")

    run_seed_sweep(
        seeds=seeds,
        build_model_fn=lambda: PhikonClassifier(),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        ckpt_base_dir="/content/drive/MyDrive/seed_sweep_vanilla",
        num_epochs=num_epochs
    )

    print("\n=== GPSA Phikon ===")

    run_seed_sweep(
        seeds=seeds,
        build_model_fn=lambda: PhikonGPSAClassifier(
            local_layers=10,
            locality_strength=1.0
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        ckpt_base_dir="/content/drive/MyDrive/seed_sweep_gpsa",
        num_epochs=num_epochs
    )


def compare_results(
    vanilla_dir="/content/drive/MyDrive/seed_sweep_vanilla",
    gpsa_dir="/content/drive/MyDrive/seed_sweep_gpsa"
):

    # Load the test accuracy for each seed
    def load_results(folder):

        path = f"{folder}/seed_results.csv"
        results = {}

        with open(path, "r", newline="") as file:

            reader = csv.DictReader(file)

            for row in reader:
                seed = int(row["seed"])
                accuracy = float(row["test_ood_acc"])

                results[seed] = accuracy

        return results

    vanilla = load_results(vanilla_dir)
    gpsa = load_results(gpsa_dir)

    # Only use seeds that finished for both models
    shared_seeds = sorted(
        set(vanilla.keys()) & set(gpsa.keys())
    )

    print(
        f"Seeds available for both models: "
        f"{len(shared_seeds)}"
    )

    if len(shared_seeds) == 0:
        print("No matching seeds were found.")
        return

    vanilla_acc = [
        vanilla[seed]
        for seed in shared_seeds
    ]

    gpsa_acc = [
        gpsa[seed]
        for seed in shared_seeds
    ]

    # Basic summary
    print(
        f"\nVanilla Phikon: "
        f"mean={np.mean(vanilla_acc):.4f}, "
        f"std={np.std(vanilla_acc):.4f}"
    )

    print(
        f"GPSA Phikon:    "
        f"mean={np.mean(gpsa_acc):.4f}, "
        f"std={np.std(gpsa_acc):.4f}"
    )

    # The same seeds were used, so use paired tests
    wilcoxon_stat, wilcoxon_p = wilcoxon(
        gpsa_acc,
        vanilla_acc
    )

    t_stat, ttest_p = ttest_rel(
        gpsa_acc,
        vanilla_acc
    )

    # Difference between GPSA and vanilla for each seed
    differences = (
        np.array(gpsa_acc) -
        np.array(vanilla_acc)
    )

    # Cohen's d for paired results
    cohens_d = (
        differences.mean() /
        differences.std(ddof=1)
    )

    print(
        f"\nWilcoxon test: "
        f"statistic={wilcoxon_stat:.4f}, "
        f"p={wilcoxon_p:.4e}"
    )

    print(
        f"Paired t-test: "
        f"t={t_stat:.4f}, "
        f"p={ttest_p:.4e}"
    )

    print(
        f"Cohen's d: {cohens_d:.4f}"
    )

    return {
        "vanilla": vanilla_acc,
        "gpsa": gpsa_acc,
        "seeds": shared_seeds,
        "wilcoxon_p": wilcoxon_p,
        "ttest_p": ttest_p,
        "cohens_d": cohens_d
    }