"""
Plot GAN loss curves from the CSV history saved during training.
Usage:
    python -m src.plot_losses
    python -m src.plot_losses --csv-path outputs/gan_images/loss_history.csv --out outputs/gan_images/loss_curve.png
"""

import argparse
import csv
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GAN loss curves from saved CSV history")
    parser.add_argument("--csv-path", type=str, default="./outputs/gan_images/loss_history.csv")
    parser.add_argument("--out", type=str, default="./outputs/gan_images/loss_curve.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d_losses, g_losses = [], []
    with open(args.csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d_losses.append(float(row["d_loss"]))
            g_losses.append(float(row["g_loss"]))

    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.out)
    plt.close()
    print(f"Saved loss curve to {args.out}")


if __name__ == "__main__":
    main()
