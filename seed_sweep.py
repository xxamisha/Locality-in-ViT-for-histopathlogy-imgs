```python
"""
Run the model using multiple random seeds.

This is useful because one training run can give a slightly different
result depending on the random seed. Running several seeds lets us
calculate the average and standard deviation.

Results are saved to a CSV file and each seed has its own checkpoint,
so the experiment can be continued if Colab disconnects.
"""

import os
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW


def set_seed(seed):
    """
    Set the random seed so that the runs are more reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CUDA operations more reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_results_path(folder):
    return os.path.join(
        folder,
        "seed_results.csv"
    )


def get_completed_seeds(results_path):
    """
    Find which seeds have already finished.
    """

    completed = set()

    if not os.path.exists(results_path):
        return completed

    with open(
        results_path,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            completed.add(
                int(row["seed"])
            )

    return completed


def save_result(results_path, result, columns):
    """
    Add the result from one seed to the CSV file.
    """

    file_exists = os.path.exists(
        results_path
    )

    with open(
        results_path,
        "a",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Calculate accuracy on a dataset.
    """

    model.eval()

    correct = 0
    total = 0

    for x, y, metadata in loader:

        x = x.to(device)
        y = y.to(device)

        predictions = model(x)

        predicted_labels = predictions.argmax(
            dim=-1
        )

        correct += (
            predicted_labels == y
        ).sum().item()

        total += y.size(0)

    return correct / total


def run_seed_sweep(
    seeds,
    build_model_fn,
    train_loader,
    val_loader,
    test_loader,
    device,
    checkpoint_folder="/content/drive/MyDrive/seed_sweep_checkpoints",
    epochs=1,
    learning_rate=1e-5,
    print_every=100
):
    """
    Train the model using each seed in the list.

    Each seed gets its own folder and checkpoint.
    """

    os.makedirs(
        checkpoint_folder,
        exist_ok=True
    )

    results_path = get_results_path(
        checkpoint_folder
    )

    columns = [
        "seed",
        "epoch",
        "train_loss",
        "val_accuracy",
        "test_accuracy"
    ]

    completed = get_completed_seeds(
        results_path
    )

    print(
        "Completed seeds:",
        sorted(completed)
    )

    for seed in seeds:

        # Don't run a seed again if it is already finished
        if seed in completed:
            print(
                f"Seed {seed} already finished"
            )
            continue

        print(
            f"\n----- Seed {seed} -----"
        )

        # Set the seed before creating the model
        set_seed(seed)

        # Give each seed its own checkpoint
        seed_folder = os.path.join(
            checkpoint_folder,
            f"seed_{seed}"
        )

        os.makedirs(
            seed_folder,
            exist_ok=True
        )

        checkpoint_path = os.path.join(
            seed_folder,
            "latest.pt"
        )

        # Create a new model
        model = build_model_fn().to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        start_epoch = 0
        start_step = 0

        # Continue from a checkpoint if one exists
        if os.path.exists(checkpoint_path):

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device
            )

            model.load_state_dict(
                checkpoint["model_state"]
            )

            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )

            start_epoch = checkpoint["epoch"]
            start_step = checkpoint["step"] + 1

            print(
                f"Resuming from epoch {start_epoch}, "
                f"step {start_step}"
            )

        model.train()

        final_loss = None

        for epoch in range(
            start_epoch,
            epochs
        ):

            for step, batch in enumerate(
                train_loader
            ):

                # Skip batches that were already completed
                if (
                    epoch == start_epoch
                    and step < start_step
                ):
                    continue

                x, y, metadata = batch

                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                logits = model(x)

                loss = F.cross_entropy(
                    logits,
                    y
                )

                loss.backward()

                optimizer.step()

                final_loss = loss.item()

                if step % print_every == 0:
                    print(
                        f"seed={seed} "
                        f"epoch={epoch} "
                        f"step={step} "
                        f"loss={loss.item():.4f}"
                    )

                # Save every 1500 steps
                if step > 0 and step % 1500 == 0:

                    torch.save(
                        {
                            "model_state":
                                model.state_dict(),

                            "optimizer_state":
                                optimizer.state_dict(),

                            "epoch":
                                epoch,

                            "step":
                                step
                        },
                        checkpoint_path
                    )

            # Start from the first batch in the next epoch
            start_step = 0

        # Test the finished model
        val_accuracy = evaluate(
            model,
            val_loader,
            device
        )

        test_accuracy = evaluate(
            model,
            test_loader,
            device
        )

        print(
            f"Seed {seed}: "
            f"val={val_accuracy:.4f}, "
            f"test={test_accuracy:.4f}"
        )

        # Save this seed's result
        save_result(
            results_path,
            {
                "seed": seed,
                "epoch": epochs - 1,
                "train_loss": final_loss,
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy
            },
            columns
        )

    print(
        "\nFinished!"
    )

    print(
        "Results saved to:",
        results_path
    )


def summarize_results(
    checkpoint_folder="/content/drive/MyDrive/seed_sweep_checkpoints"
):
    """
    Calculate the mean and standard deviation across the seeds.
    """

    results_path = get_results_path(
        checkpoint_folder
    )

    val_results = []
    test_results = []

    with open(
        results_path,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            val_results.append(
                float(row["val_accuracy"])
            )

            test_results.append(
                float(row["test_accuracy"])
            )

    val_results = np.array(
        val_results
    )

    test_results = np.array(
        test_results
    )

    print(
        f"Number of seeds: {len(val_results)}"
    )

    print(
        f"Validation accuracy: "
        f"{val_results.mean():.4f} "
        f"+/- {val_results.std():.4f}"
    )

    print(
        f"Test accuracy: "
        f"{test_results.mean():.4f} "
        f"+/- {test_results.std():.4f}"
    )

    return val_results, test_results


if __name__ == "__main__":

    # Example:
    #
    # seeds = list(range(30))
    #
    # run_seed_sweep(
    #     seeds,
    #     build_model_fn=build_model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     device=device
    # )

    pass
```
