"""
Train a baseline or GAN-augmented MNIST CNN.

Baseline:
python -m src.train_cnn --output-dir outputs/cnn_baseline

Augmented:
python -m src.train_cnn --synthetic-count 10000 \
  --gan-checkpoint outputs/gan_models/generator_epoch_50.pth \
  --teacher-checkpoint outputs/cnn_baseline/baseline_cnn.pth \
  --output-dir outputs/cnn_augmented
"""

import argparse
import json
import os
import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from src.data import get_cnn_dataloaders
from src.metrics import pseudo_labeled_synthetic_dataset
from src.models import Generator, MNISTCNN
from src.utils import save_checkpoint, set_seed

class SyntheticDigitDataset(Dataset):
    """
    Tensor-backed synthetic images with Python-int targets.

    MNIST returns targets as Python ints. Returning int labels here keeps every
    item in the ConcatDataset compatible with PyTorch's default collate logic.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], int(self.labels[index].item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline or augmented CNN")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./outputs/cnn")
    parser.add_argument("--gan-checkpoint")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--synthetic-count", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item() * len(labels)
            correct += (output.argmax(dim=1) == labels).sum().item()
            total += len(labels)

    return total_loss / total, 100 * correct / total


def main(args: argparse.Namespace) -> None:
    if args.synthetic_count and not (
        args.gan_checkpoint and args.teacher_checkpoint
    ):
        raise ValueError(
            "Synthetic augmentation requires --gan-checkpoint and "
            "--teacher-checkpoint."
        )

    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader, test_loader = get_cnn_dataloaders(
        args.data_root,
        args.batch_size,
    )

    mode = "baseline"

    if args.synthetic_count:
        teacher = MNISTCNN().to(device)
        teacher.load_state_dict(
            torch.load(args.teacher_checkpoint, map_location=device)
        )

        generator = Generator(args.latent_dim).to(device)
        generator.load_state_dict(
            torch.load(args.gan_checkpoint, map_location=device)
        )

        synthetic_images, synthetic_labels = pseudo_labeled_synthetic_dataset(
            generator,
            teacher,
            args.synthetic_count,
            args.latent_dim,
            device,
            args.confidence,
            args.max_candidates,
        )

        synthetic_dataset = SyntheticDigitDataset(
            synthetic_images,
            synthetic_labels,
        )

        augmented_dataset = ConcatDataset(
            [
                train_loader.dataset,
                synthetic_dataset,
            ]
        )

        train_loader = DataLoader(
            augmented_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        mode = "augmented"

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        print(f"Epoch {epoch}: test accuracy {test_accuracy:.2f}%")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"{mode}_cnn.pth")
    save_checkpoint(model.state_dict(), args.output_dir, f"{mode}_cnn.pth")

    results = {
        "mode": mode,
        "synthetic_count": args.synthetic_count,
        "confidence": args.confidence,
        "model_checkpoint": checkpoint_path,
        "final_test_accuracy": history[-1]["test_accuracy"],
        "history": history,
    }
    with open(
        os.path.join(args.output_dir, f"{mode}_metrics.json"),
        "w",
    ) as result_file:
        json.dump(results, result_file, indent=2)


if __name__ == "__main__":
    main(parse_args())
