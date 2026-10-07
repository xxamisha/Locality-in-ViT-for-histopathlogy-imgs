"""
quick hyperparameter search for GPSA-phikon: gating_init and new_lr, This is becuase the gating-init controls the blend between two types of attention 
(content and positional). Higher accuracy on a higher gating init means more bias towards locality vs a lower gating init value means that it trusts more the pretrained content
attention. usually better for gating init to be around 1 since there is only 1 epoch because theres less to undo and it's able to have more room to learn if it wants locality or content. 
Learning rate is also a newly added parameter to the GPSA model so changing this would aim to change the accuracy of the model. It determines how quickly they learn so lower new_lr 
value means the new parameters barely move and start at random and the larger lr values are very fast so new components try to catch up quickly. 

Since its only training on one epoch, there is a constrained space so finetuning these values make all the difference. 

Sweep 1: gating_init in {0.0, 0.5, 1.0}, LR left at default (no
          differential LR) — finds the best starting gate value.
Sweep 2: new_lr in {1e-4, 5e-4, 1e-3}, gating_init held at the paper's
          default (1.0) — finds the best differential LR for GPSA's new
          params (pos_proj, gating_param).

5 seeds per value (30 runs total). 
"""

import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import csv
import os
import numpy as np

from seed_sweep import set_seed, run_seed_sweep
from phikon_gpsa import inject_gpsa
from transformers import ViTModel
import torch.nn as nn

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
NUM_SEEDS = 5
NUM_EPOCHS = 1
NUM_WORKERS = 4

GATING_INIT_VALUES = [0.0, 0.5, 1.0]
NEW_LR_VALUES = [1e-4, 5e-4, 1e-3]
DEFAULT_GATING_INIT = 1.0   # paper's default, held fixed during the new_lr sweep
DEFAULT_NEW_LR = None       # no differential LR, held fixed during the gating_init sweep


class PhikonGPSAClassifier(nn.Module):
    def __init__(self, local_layers=10, locality_strength=1.0, gating_init=1.0, num_classes=2):
        super().__init__()
        self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        inject_gpsa(self.backbone, local_layers=local_layers, locality_strength=locality_strength,
                    gating_init=gating_init)
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


def summarize_search(results_dir="./checkpoints/gpsa_search"):
    print(f"\n{'sweep':>10} | {'value':>8} | {'n':>3} | {'val_acc mean':>12} | {'val sample SD':>13} | {'test_acc mean':>13} | {'test sample SD':>14}")
    print("-" * 80)
    summaries = []
    for sweep_name, values in [("gating_init", GATING_INIT_VALUES), ("new_lr", NEW_LR_VALUES)]:
        for v in values:
            path = f"{results_dir}/{sweep_name}_{v}/seed_results.csv"
            if not os.path.exists(path):
                print(f"{sweep_name:>10} | {v:>8} | (no results yet)")
                continue
            val_accs, test_accs = [], []
            with open(path, "r", newline="") as f:
                for row in csv.DictReader(f):
                    val_accs.append(float(row["val_ood_acc"]))
                    test_accs.append(float(row["test_ood_acc"]))
            val_accs, test_accs = np.array(val_accs), np.array(test_accs)
            print(f"{sweep_name:>10} | {v:>8} | {len(val_accs):>3} | {val_accs.mean():>12.4f} | "
                f"{val_accs.std(ddof=1):>13.4f} | {test_accs.mean():>13.4f} | {test_accs.std(ddof=1):>14.4f}")
            summaries.append({
                "ablation": "hyperparameter_search",
                "condition": f"{sweep_name}={v}",
                "results_dir": os.path.dirname(path),
                "n_runs": len(val_accs),
                "val_mean": float(val_accs.mean()),
                "val_sample_sd": float(val_accs.std(ddof=1)) if len(val_accs) > 1 else float("nan"),
                "test_mean": float(test_accs.mean()),
                "test_sample_sd": float(test_accs.std(ddof=1)) if len(test_accs) > 1 else float("nan"),
            })
    return summaries


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    print("loading dataset (should hit local cache)...")
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

    seeds = list(range(NUM_SEEDS))  # SAME small seed set reused across every value in both sweeps

    # ── Sweep 1: gating_init (LR fixed at default, no differential LR) ────
    for gi in GATING_INIT_VALUES:
        print(f"\n{'='*20} gating_init = {gi} {'='*20}")
        run_seed_sweep(
            seeds=seeds,
            build_model_fn=lambda gi=gi: PhikonGPSAClassifier(gating_init=gi),
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            device=device,
            ckpt_base_dir=f"./checkpoints/gpsa_search/gating_init_{gi}",
            num_epochs=NUM_EPOCHS,
            log_every=100,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            new_lr=DEFAULT_NEW_LR,
            run_config={
                "dataset": "Camelyon17-WILDS",
                "model": "GPSA-phikon",
                "sweep": "gating_init",
                "gating_init": gi,
                "local_layers": 10,
                "locality_strength": 1.0,
                "training_centers": sorted(TRAIN_CENTERS),
                "validation_center": VAL_OOD_CENTER,
                "test_center": TEST_OOD_CENTER,
                "image_size": [224, 224],
                "normalize_mean": [0.5, 0.5, 0.5],
                "normalize_std": [0.5, 0.5, 0.5],
            },
        )

    # ── Sweep 2: new_lr (gating_init fixed at paper's default 1.0) ────────
    for nlr in NEW_LR_VALUES:
        print(f"\n{'='*20} new_lr = {nlr} {'='*20}")
        run_seed_sweep(
            seeds=seeds,
            build_model_fn=lambda: PhikonGPSAClassifier(gating_init=DEFAULT_GATING_INIT),
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            device=device,
            ckpt_base_dir=f"./checkpoints/gpsa_search/new_lr_{nlr}",
            num_epochs=NUM_EPOCHS,
            log_every=100,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            new_lr=nlr,
            run_config={
                "dataset": "Camelyon17-WILDS",
                "model": "GPSA-phikon",
                "sweep": "new_lr",
                "gating_init": DEFAULT_GATING_INIT,
                "local_layers": 10,
                "locality_strength": 1.0,
                "training_centers": sorted(TRAIN_CENTERS),
                "validation_center": VAL_OOD_CENTER,
                "test_center": TEST_OOD_CENTER,
                "image_size": [224, 224],
                "normalize_mean": [0.5, 0.5, 0.5],
                "normalize_std": [0.5, 0.5, 0.5],
            },
        )

    summarize_search()
