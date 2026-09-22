"""
Runs the vanilla-phikon vs GPSA-phikon comparison locally.
Just run: python run_sweep_local.py

Need these installed first (in a venv ideally):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install transformers datasets scipy numpy

And these files need to be in the same folder:
    gpsa.py, phikon_gpsa.py, seed_sweep.py, paired_seed_comparison.py

NOTE ON THE if __name__ == "__main__" THING BELOW:
this actually matters, not just style. On Windows, DataLoader with
num_workers > 0 spawns extra processes, and each one re-imports this
whole file to set itself up. If the dataset-loading code wasn't inside
this guard, every worker process would end up re-downloading and
re-filtering the dataset from scratch, all fighting over the same cache
file at once - which is exactly what was happening before this got
fixed (kept seeing FileExistsError / WinError 1224 from multiple
processes racing each other).
"""

import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from paired_seed_comparison import run_paired_comparison, compare_results

# ── settings ────────────────────────────────────────────────────────────
BATCH_SIZE = 16          # had to drop this from 32 (what Colab used) since the
                         # 3070 only has 8GB VRAM vs the T4's 16GB
GRAD_ACCUM_STEPS = 2     # makes up for the smaller batch - 16 * 2 = effective
                         # batch of 32 again, see seed_sweep.py for why this
                         # works out exactly (not just approximately)
NUM_SEEDS = 10
NUM_EPOCHS = 1
NUM_WORKERS = 4          # drop this to 0 or 2 if DataLoader workers keep crashing


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
    if device == "cpu":
        print("no GPU found - this is going to be really slow. check torch actually has CUDA:")
        print("  py -c \"import torch; print(torch.cuda.is_available())\"")

    # ── load the dataset ───────────────────────────────────────────────
    print("loading dataset (should be cached locally after the first run)...")
    hf_dataset = load_dataset("wltjr1007/Camelyon17-WILDS")
    all_data = concatenate_datasets([hf_dataset["train"], hf_dataset["validation"], hf_dataset["test"]])

    # splitting by hospital rather than randomly - this matches the actual
    # WILDS benchmark split (train on 3 hospitals, test OOD on 2 held-out
    # ones), which is the whole point of the project
    TRAIN_CENTERS = {0, 3, 4}
    VAL_OOD_CENTER = 1
    TEST_OOD_CENTER = 2

    train_hf = all_data.filter(lambda ex: ex["center"] in TRAIN_CENTERS)
    val_ood_hf = all_data.filter(lambda ex: ex["center"] == VAL_OOD_CENTER)
    test_ood_hf = all_data.filter(lambda ex: ex["center"] == TEST_OOD_CENTER)

    print(f"train: {len(train_hf)}, val (OOD): {len(val_ood_hf)}, test (OOD): {len(test_ood_hf)}")
    # should be roughly 302,436 / 34,904 / 85,054

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

    # actually running it
    run_paired_comparison(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        seeds=list(range(NUM_SEEDS)),
        num_epochs=NUM_EPOCHS,
        log_every=50,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        vanilla_ckpt_dir="./checkpoints/seed_sweep_vanilla",
        gpsa_ckpt_dir="./checkpoints/seed_sweep_gpsa_v2",  # separate folder so this doesn't
                                                            # overwrite the original (valid) run
        gpsa_new_lr=5e-4,   # letting GPSA's new params (pos_proj, gate) move faster than the
                            # rest of the pretrained backbone, since they're starting from scratch
        gating_init=0.0,    # gentler start (50/50) instead of the paper's default (~73%
                            # positional), so there's less to "undo" during just 1 epoch
    )

    print("\n=== final comparison ===")
    compare_results(
        vanilla_dir="./checkpoints/seed_sweep_vanilla",
        gpsa_dir="./checkpoints/seed_sweep_gpsa_v2",
    )
