"""GAN evaluation and synthetic-image augmentation helpers."""

import torch


def fid_from_loader(
    generator,
    real_loader,
    latent_dim: int,
    device: torch.device,
    n_samples: int = 10_000,
) -> float:
    """Compute Inception-feature FID for real and generated MNIST images."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise RuntimeError(
            "FID requires `torchmetrics[image]` and `torch-fidelity`. "
            "Install them before running GAN evaluation."
        ) from exc

    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    generator.eval()
    seen = 0

    with torch.no_grad():
        for real_images, _ in real_loader:
            remaining = n_samples - seen
            if remaining <= 0:
                break

            batch_size = min(len(real_images), remaining)
            real_images = real_images[:batch_size].to(device)
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)

            real_rgb = ((real_images + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1)
            fake_rgb = ((fake_images + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1)

            metric.update(real_rgb, real=True)
            metric.update(fake_rgb, real=False)
            seen += batch_size

    return float(metric.compute().cpu())


def pseudo_labeled_synthetic_dataset(
    generator,
    teacher,
    count: int,
    latent_dim: int,
    device: torch.device,
    confidence: float = 0.95,
    max_candidates: int | None = None,
):
    """
    Generate synthetic images and keep only high-confidence teacher predictions.
    The generated images are transformed from GAN normalization [-1, 1] to the
    CNN normalization used in src.data. A bounded candidate count prevents an
    endless loop when GAN quality is too poor for the confidence threshold.
    """
    max_candidates = max_candidates or count * 20
    generated = 0
    images, labels = [], []

    generator.eval()
    teacher.eval()

    with torch.no_grad():
        while sum(x.size(0) for x in images) < count and generated < max_candidates:
            batch_size = min(512, max_candidates - generated)
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake = generator(noise)

            cnn_input = (fake - (0.1307 / 0.5)) / (0.3081 / 0.5)
            probabilities = teacher(cnn_input).softmax(dim=1)
            scores, predictions = probabilities.max(dim=1)
            accepted = scores >= confidence

            if accepted.any():
                images.append(cnn_input[accepted].cpu())
                labels.append(predictions[accepted].cpu())

            generated += batch_size

    accepted_count = sum(x.size(0) for x in images)
    if accepted_count < count:
        raise RuntimeError(
            f"Only accepted {accepted_count}/{count} synthetic images after "
            f"{generated} candidates. Lower --confidence, train the GAN longer, "
            "or increase --max-candidates."
        )

    return torch.cat(images)[:count], torch.cat(labels)[:count]
