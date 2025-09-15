import os
from torchvision.datasets import CIFAR10

def get_data(data_dir: str = "data"):
    """Download CIFAR10 dataset."""
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download CIFAR10
    CIFAR10(
        root=data_dir,
        train=True,
        download=True,
    )
    CIFAR10(
        root=data_dir,
        train=False,
        download=True,
    )

if __name__ == "__main__":
    get_data()
