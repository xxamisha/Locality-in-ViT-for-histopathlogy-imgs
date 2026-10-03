# Combined epoch-count and local-layer ablation runner.

#Epoch-count search first: 
"""
Epoch-count ablation: at the winning config (gating_init=0.0, new_lr=1e-3),
does training longer actually help, or was initialization/LR the real fix?

3 seeds per epoch value, epochs in {1, 3, 5}. Small n since this is a
supplementary check, not the main statistical claim but enough points
to plot an accuracy-vs-epochs curve.

Run: python ablation_test.py
"""

import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import csv
import os
import numpy as np

from seed_sweep import run_seed_sweep
from phikon_gpsa import inject_gpsa
from transformers import ViTModel
import torch.nn as nn

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
NUM_SEEDS = 3
NUM_WORKERS = 4
NUM_EPOCHS = 1

GATING_INIT = 0.0
NEW_LR = 1e-3
EPOCH_VALUES = [1, 3, 5]
LOCAL_LAYERS_VALUES = [4, 6, 8, 10, 12]

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


def summarize_epoch_ablation(results_dir="./checkpoints/ablation_epochs"):
    print(f"\n{'epochs':>8} | {'n':>3} | {'val_acc mean':>12} | {'val sample SD':>13} | {'test_acc mean':>13} | {'test sample SD':>14}")
    print("-" * 70)
    summaries = []
    for e in EPOCH_VALUES:
        path = f"{results_dir}/epochs_{e}/seed_results.csv"
        if not os.path.exists(path):
            print(f"{e:>8} | (no results yet)")
            continue
        val_accs, test_accs = [], []
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                val_accs.append(float(row["val_ood_acc"]))
                test_accs.append(float(row["test_ood_acc"]))
        val_accs, test_accs = np.array(val_accs), np.array(test_accs)
        print(f"{e:>8} | {len(val_accs):>3} | {val_accs.mean():>12.4f} | {val_accs.std(ddof=1):>13.4f} | "
              f"{test_accs.mean():>13.4f} | {test_accs.std(ddof=1):>14.4f}")
        summaries.append({
            "ablation": "epoch_count",
            "condition": f"epochs={e}",
            "results_dir": os.path.dirname(path),
            "n_runs": len(val_accs),
            "val_mean": float(val_accs.mean()),
            "val_sample_sd": float(val_accs.std(ddof=1)) if len(val_accs) > 1 else float("nan"),
            "test_mean": float(test_accs.mean()),
            "test_sample_sd": float(test_accs.std(ddof=1)) if len(test_accs) > 1 else float("nan"),
        })
    return summaries
def summarize_ablation(results_dir="./checkpoints/ablation_local_layers"):
    """Reads every local_layers value's results and prints mean +/- std, for
    a quick bar/line chart in your report."""
    import numpy as np
    print(f"\n{'local_layers':>12} | {'n':>3} | {'val_acc mean':>12} | {'val sample SD':>13} | {'test_acc mean':>13} | {'test sample SD':>14}")
    print("-" * 75)
    summaries = []
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
        print(f"{ll:>12} | {len(val_accs):>3} | {val_accs.mean():>12.4f} | {val_accs.std(ddof=1):>13.4f} | "
              f"{test_accs.mean():>13.4f} | {test_accs.std(ddof=1):>14.4f}")
        summaries.append({
            "ablation": "local_layers",
            "condition": f"local_layers={ll}",
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

    seeds = list(range(NUM_SEEDS))  # same small seed set reused across every epoch value

    for epochs in EPOCH_VALUES:
        print(f"\n{'='*20} epochs = {epochs} {'='*20}")
        run_seed_sweep(
            seeds=seeds,
            build_model_fn=lambda: PhikonGPSAClassifier(local_layers=10, locality_strength=1.0,
                                                          gating_init=GATING_INIT),
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            device=device,
            ckpt_base_dir=f"./checkpoints/ablation_epochs/epochs_{epochs}",
            num_epochs=epochs,
            log_every=100,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            new_lr=NEW_LR,
            run_config={
                "dataset": "Camelyon17-WILDS",
                "model": "GPSA-phikon",
                "ablation": "epoch_count",
                "local_layers": 10,
                "locality_strength": 1.0,
                "gating_init": GATING_INIT,
                "epoch_condition": epochs,
                "training_centers": sorted(TRAIN_CENTERS),
                "validation_center": VAL_OOD_CENTER,
                "test_center": TEST_OOD_CENTER,
                "image_size": [224, 224],
                "normalize_mean": [0.5, 0.5, 0.5],
                "normalize_std": [0.5, 0.5, 0.5],
            },
        )
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
                run_config={
                    "dataset": "Camelyon17-WILDS",
                    "model": "GPSA-phikon",
                    "ablation": "local_layers",
                    "local_layers": local_layers,
                    "locality_strength": 1.0,
                    "gating_init": GATING_INIT,
                    "training_centers": sorted(TRAIN_CENTERS),
                    "validation_center": VAL_OOD_CENTER,
                    "test_center": TEST_OOD_CENTER,
                    "image_size": [224, 224],
                    "normalize_mean": [0.5, 0.5, 0.5],
                    "normalize_std": [0.5, 0.5, 0.5],
                },
            )
    

    summarize_epoch_ablation()
    summarize_ablation()
