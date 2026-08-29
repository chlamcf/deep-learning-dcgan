"""
Data loading utilities shared by the GAN and CNN scripts.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_gan_dataloader(data_root: str = "./data", batch_size: int = 128, download: bool = True) -> DataLoader:
    """MNIST loader normalized to [-1, 1], matching the Tanh output of the GAN generator."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.MNIST(root=data_root, train=True, download=download, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def get_cnn_dataloaders(data_root: str = "./data", batch_size: int = 64, download: bool = True):
    """MNIST train/test loaders normalized with the dataset mean/std."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root=data_root, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=download, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
    