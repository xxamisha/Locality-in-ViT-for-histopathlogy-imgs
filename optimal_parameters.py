"""
Combined the most optimal config: gating_init=0.0 + new_lr=1e-3, full 10 seeds.
THIS IS AFTER RUNNING HPARAM.PY THAT I FOUND THIS NEW OPTIMAL RATES. mentioned in report.

compares against the existing vanilla baseline.

Run: python run_combined_config.py
"""

import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from seed_sweep import run_seed_sweep
from paired_seed_comparison import PhikonGPSAClassifier, compare_results

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
NUM_SEEDS = 10
NUM_EPOCHS = 1
NUM_WORKERS = 4

# the winning combo from the hyperparameter search
GATING_INIT = 0.0
NEW_LR = 1e-3


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

    print(f"\n=== GPSA-phikon, tuned config (gating_init={GATING_INIT}, new_lr={NEW_LR}), 10 seeds ===")
    run_seed_sweep(
        seeds=list(range(NUM_SEEDS)),
        build_model_fn=lambda: PhikonGPSAClassifier(local_layers=10, locality_strength=1.0,
                                                      gating_init=GATING_INIT),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device,
        ckpt_base_dir="./checkpoints/seed_sweep_gpsa_tuned",
        num_epochs=NUM_EPOCHS,
        log_every=50,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        new_lr=NEW_LR,
    )

    print("\n=== comparison: vanilla vs tuned GPSA ===")
    compare_results(
        vanilla_dir="./checkpoints/seed_sweep_vanilla",
        gpsa_dir="./checkpoints/seed_sweep_gpsa_tuned",
    )
