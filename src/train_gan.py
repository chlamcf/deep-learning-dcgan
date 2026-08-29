"""
Train a DCGAN on MNIST.
Example: python -m src.train_gan --epochs 50 --sample-epochs 1,10,25,50
"""

import argparse
import csv
import json
import os
import torch
from torch import nn, optim
from torchvision.utils import save_image
from src.data import get_gan_dataloader
from src.metrics import fid_from_loader
from src.models import Discriminator, Generator
from src.utils import save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DCGAN on MNIST")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./outputs/gan")
    parser.add_argument("--checkpoint-dir", default="./outputs/gan_models")
    parser.add_argument("--sample-epochs", default="1,10,25,50")
    parser.add_argument("--fid-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader = get_gan_dataloader(args.data_root, args.batch_size)
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    sample_epochs = {int(epoch) for epoch in args.sample_epochs.split(",")}
    loss_path = os.path.join(args.output_dir, "loss_history.csv")

    with open(loss_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "batch", "d_loss", "g_loss"])

        for epoch in range(1, args.epochs + 1):
            generator.train()
            discriminator.train()

            for batch_index, (real_images, _) in enumerate(train_loader):
                real_images = real_images.to(device)
                real_targets = torch.ones(len(real_images), device=device)
                fake_targets = torch.zeros(len(real_images), device=device)

                optimizer_d.zero_grad()
                noise = torch.randn(len(real_images), args.latent_dim, device=device)
                fake_images = generator(noise)

                d_loss = (
                    criterion(discriminator(real_images), real_targets)
                    + criterion(discriminator(fake_images.detach()), fake_targets)
                ) / 2
                d_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()
                noise = torch.randn(len(real_images), args.latent_dim, device=device)
                generated_images = generator(noise)
                g_loss = criterion(discriminator(generated_images), real_targets)
                g_loss.backward()
                optimizer_g.step()

                writer.writerow([epoch, batch_index, d_loss.item(), g_loss.item()])

            if epoch in sample_epochs or epoch == args.epochs:
                generator.eval()
                with torch.no_grad():
                    samples = generator(fixed_noise)
                save_image(
                    samples,
                    os.path.join(args.output_dir, f"samples_epoch_{epoch:03d}.png"),
                    nrow=8,
                    normalize=True,
                    value_range=(-1, 1),
                )

            save_checkpoint(
                generator.state_dict(),
                args.checkpoint_dir,
                f"generator_epoch_{epoch}.pth",
            )
            save_checkpoint(
                discriminator.state_dict(),
                args.checkpoint_dir,
                f"discriminator_epoch_{epoch}.pth",
            )

    fid = fid_from_loader(
        generator,
        train_loader,
        args.latent_dim,
        device,
        args.fid_samples,
    )

    metrics = {
        "epoch": args.epochs,
        "fid": fid,
        "fid_samples": args.fid_samples,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Training complete. FID: {fid:.3f}")


if __name__ == "__main__":
    main(parse_args())
