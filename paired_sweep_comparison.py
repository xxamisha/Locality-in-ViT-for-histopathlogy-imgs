"""
Comparing vanilla phikon vs GPSA-phikon across multiple seeds.

Both models get trained on the same set of seeds so the comparison is
paired - each seed gives us one (vanilla_acc, gpsa_acc) pair, which is
what lets us use a paired statistical test at the end

Meant to be run after the dataset/dataloaders are already set up.
"""

from transformers import ViTModel
import torch.nn as nn
from phikon_gpsa import inject_gpsa
from seed_sweep import set_seed, run_seed_sweep, summarize_results
from scipy.stats import wilcoxon, ttest_rel
import csv


class PhikonClassifier(nn.Module):
    """Just phikon fine-tuned normally, no GPSA. This is our baseline -
    same backbone as the GPSA version just without the attention swap
    so we can actually tell if GPSA is doing anything."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        out = self.backbone(x).last_hidden_state[:, 0]
        return self.head(out)


class PhikonGPSAClassifier(nn.Module):
    """same as above but with GPSA swapped in for the first local_layers
    layers."""
    def __init__(self, local_layers=10, locality_strength=1.0, gating_init=1.0, num_classes=2):
        super().__init__()
        self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        inject_gpsa(self.backbone, local_layers=local_layers, locality_strength=locality_strength,
                    gating_init=gating_init)
        self.local_layers = local_layers
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        out = self.backbone(x).last_hidden_state[:, 0]
        return self.head(out)


def run_paired_comparison(train_loader, val_loader, test_loader, device,
                           seeds=None, num_epochs=1, log_every=100,
                           grad_accum_steps=1, gpsa_new_lr=None, gating_init=1.0,
                           vanilla_ckpt_dir="./checkpoints/seed_sweep_vanilla",
                           gpsa_ckpt_dir="./checkpoints/seed_sweep_gpsa"):
    """
    Trains both models across the same list of seeds. Each seed is
    checkpointed on its own, so if this gets interrupted partway through
    (crash, ran out of time) you can just run it again and it
    picks up from wherever it left off instead of starting over.

    grad_accum_steps: only needed if the batch size had to be lowered for
        VRAM reasons (e.g. 16 instead of 32 on an 8GB card) - this lets you
        recover the same effective batch size by accumulating gradients
        over a couple of smaller batches before actually stepping.
    gpsa_new_lr: optional separate learning rate just for GPSA's own new
        parameters (pos_proj and the gate). These start from scratch
        (unlike the rest of the backbone, which is pretrained), so giving
        them a bit more room to move can help them catch up faster.
    gating_init: what the gate starts at before training. Default 1.0
        matches the paper (~73% positional at init). Lower values (like
        0.0, which is 50/50) mean less disruption to the pretrained
        weights at the start, which might matter more when only
        fine-tuning for a short time.
    """
    if seeds is None:
        # using 10 instead of the usual 30 here because of time/compute
        # constraints - each full run takes a while, and 30x that wasn't
        # realistic given the deadline. documented properly in the report.
        seeds = list(range(10))

    print("=== vanilla phikon ===")
    run_seed_sweep(
        seeds=seeds,
        build_model_fn=lambda: PhikonClassifier(),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device,
        ckpt_base_dir=vanilla_ckpt_dir,
        num_epochs=num_epochs,
        log_every=log_every,
        grad_accum_steps=grad_accum_steps,
    )

    print("\n=== GPSA phikon (same seeds as above) ===")
    run_seed_sweep(
        seeds=seeds,
        build_model_fn=lambda: PhikonGPSAClassifier(local_layers=10, locality_strength=1.0,
                                                      gating_init=gating_init),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device,
        ckpt_base_dir=gpsa_ckpt_dir,
        num_epochs=num_epochs,
        log_every=log_every,
        grad_accum_steps=grad_accum_steps,
        new_lr=gpsa_new_lr,
    )


def compare_results(
    vanilla_dir="./checkpoints/seed_sweep_vanilla",
    gpsa_dir="./checkpoints/seed_sweep_gpsa",
):
    """Runs the actual stats comparison once both sweeps are done."""
    def load_test_accs(base_dir):
        path = f"{base_dir}/seed_results.csv"
        accs = {}
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                accs[int(row["seed"])] = float(row["test_ood_acc"])
        return accs

    vanilla_accs = load_test_accs(vanilla_dir)
    gpsa_accs = load_test_accs(gpsa_dir)

    # only compare seeds that actually finished for BOTH models - if one
    # run got interrupted and the other didn't, we don't want to compare
    # against missing data
    shared_seeds = sorted(set(vanilla_accs) & set(gpsa_accs))
    print(f"seeds with results in both: {len(shared_seeds)}")
    if len(shared_seeds) < len(vanilla_accs) or len(shared_seeds) < len(gpsa_accs):
        print("warning: some seeds are missing from one model or the other, "
              "only using the ones that are in both")

    v = [vanilla_accs[s] for s in shared_seeds]
    g = [gpsa_accs[s] for s in shared_seeds]

    import numpy as np
    print(f"\nvanilla-phikon: mean={np.mean(v):.4f} std={np.std(v):.4f}")
    print(f"GPSA-phikon:    mean={np.mean(g):.4f} std={np.std(g):.4f}")

    # using a paired test here since both models were run on the same
    # seeds - that means each pair of numbers came from a matched
    # starting point, so a paired test is more appropriate (and more
    # powerful) than treating them as two independent groups
    stat, p_wilcoxon = wilcoxon(g, v)
    t_stat, p_ttest = ttest_rel(g, v)

    print(f"\nWilcoxon signed-rank: statistic={stat:.4f}, p={p_wilcoxon:.4e}")
    print(f"Paired t-test:        t={t_stat:.4f}, p={p_ttest:.4e}")

    # effect size - p-value alone doesn't tell you how BIG the difference
    # actually is, just whether it's likely to be real
    diffs = np.array(g) - np.array(v)
    cohens_d = diffs.mean() / diffs.std(ddof=1)
    print(f"Cohen's d (paired):   {cohens_d:.4f}")

    return {"vanilla": v, "gpsa": g, "wilcoxon_p": p_wilcoxon, "ttest_p": p_ttest, "cohens_d": cohens_d}
