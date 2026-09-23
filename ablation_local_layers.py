"""
Ablation: local_layers (number of GPSA layers vs standard MHSA).

this varies ONLY local_layers, holding everything else fixed (locality_strength=1.0,
lr=1e-5, num_epochs=1, same seeds as the main sweep would use just
fewer of them, since this is exploratory/supporting evidence, not the
headline statistical claim).

Uses fewer seeds per value (3-5) since this is meant to show a trend
with a rough error bar, not a formal significance test. running the
full seed count for every ablation value would multiply compute for a
secondary analysis and will run out of time for the analysis.
"""

import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import csv
import os

from seed_sweep import set_seed, run_seed_sweep
from phikon_gpsa import inject_gpsa
from transformers import ViTModel
import torch.nn as nn

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
NUM_SEEDS_PER_VALUE = 3   # reduced — this is exploratory, not the main statistical result
NUM_EPOCHS = 1
NUM_WORKERS = 4
LOCAL_LAYERS_VALUES = [4, 6, 8, 10, 12]  # 12 = all layers use GPSA, 0 would be equivalent to vanilla

# winning config from the gating_init/new_lr hyperparameter search — using
# these here so this ablation reflects the ACTUAL tuned method, not the
# original untuned defaults
GATING_INIT = 0.0
NEW_LR = 1e-3


class PhikonGPSAClassifier(nn.Module):
    def __init__(self, local_layers, locality_strength=1.0, gating_init=1.0, num_classes=2):
        super().__init__()
        self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        inject_gpsa(self.backbone, local_layers=local_layers, locality_strength=locality_strength,
                    gating_init=gating_init)
        self.local_layers = local_layers
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        out = self.backbone(x).last_hidden_state[:, 0]
        return self.head(out)


class Camelyon17HFDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        image = ex["image"].convert("RGB")
        label = ex["label"]
        metadata = {"center": ex["center"], "patient": ex["patient"], "node": ex["node"]}
        return self.transform(image), label, metadata


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return images, labels, metadata


def summarize_ablation(results_dir="./checkpoints/ablation_local_layers"):
    """Reads every local_layers value's results and prints mean +/- std, for
    a quick bar/line chart in your report."""
    import numpy as np
    print(f"\n{'local_layers':>12} | {'n':>3} | {'val_acc mean':>12} | {'val_acc std':>11} | {'test_acc mean':>13} | {'test_acc std':>12}")
    print("-" * 75)
    for ll in LOCAL_LAYERS_VALUES:
        path = f"{results_dir}/local_layers_{ll}/seed_results.csv"
        if not os.path.exists(path):
            print(f"{ll:>12} | (no results yet)")
            continue
        val_accs, test_accs = [], []
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                val_accs.append(float(row["val_ood_acc"]))
                test_accs.append(float(row["test_ood_acc"]))
        val_accs, test_accs = np.array(val_accs), np.array(test_accs)
        print(f"{ll:>12} | {len(val_accs):>3} | {val_accs.mean():>12.4f} | {val_accs.std():>11.4f} | "
              f"{test_accs.mean():>13.4f} | {test_accs.std():>12.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    print("loading dataset (should hit local cache from the main sweep)...")
    hf_dataset = load_dataset("wltjr1007/Camelyon17-WILDS")
    all_data = concatenate_datasets([hf_dataset["train"], hf_dataset["validation"], hf_dataset["test"]])

    TRAIN_CENTERS = {0, 3, 4}
    VAL_OOD_CENTER = 1
    TEST_OOD_CENTER = 2

    train_hf = all_data.filter(lambda ex: ex["center"] in TRAIN_CENTERS)
    val_ood_hf = all_data.filter(lambda ex: ex["center"] == VAL_OOD_CENTER)
    test_ood_hf = all_data.filter(lambda ex: ex["center"] == TEST_OOD_CENTER)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_data = Camelyon17HFDataset(train_hf, transform)
    val_ood_data = Camelyon17HFDataset(val_ood_hf, transform)
    test_ood_data = Camelyon17HFDataset(test_ood_hf, transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ood_data, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ood_data, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)

    seeds = list(range(NUM_SEEDS_PER_VALUE))  # SAME small seed set reused across all local_layers values —
                                               # this keeps the ablation internally consistent (same
                                               # per-seed randomness factored out across values)

    for local_layers in LOCAL_LAYERS_VALUES:
        print(f"\n{'='*20} local_layers = {local_layers} {'='*20}")
        run_seed_sweep(
            seeds=seeds,
            build_model_fn=lambda ll=local_layers: PhikonGPSAClassifier(local_layers=ll, gating_init=GATING_INIT),  # ll=local_layers captures the loop value correctly
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            device=device,
            ckpt_base_dir=f"./checkpoints/ablation_local_layers/local_layers_{local_layers}",
            num_epochs=NUM_EPOCHS,
            log_every=100,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            new_lr=NEW_LR,
        )

    summarize_ablation()
