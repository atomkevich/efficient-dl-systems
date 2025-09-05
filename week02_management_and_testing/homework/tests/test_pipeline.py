import time
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, generate_samples, train_epoch
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


def test_training():
    """Integration test for the complete training procedure including model creation,
    training loop and sample generation. Tests different hyperparameters and devices."""
    # Test configurations to check different scenarios
    configs = [
        {"device": "cpu", "batch_size": 8, "hidden_size": 32, "lr": 1e-3, "num_timesteps": 100},
        {"device": "cpu", "batch_size": 8, "hidden_size": 32, "lr": 1e-6, "num_timesteps": 100},  # Very small LR
        {"device": "cuda", "batch_size": 16, "hidden_size": 64, "lr": 5e-4, "num_timesteps": 200}
    ]
    
    for config in configs:
        # Skip CUDA tests if not available
        if config["device"] == "cuda" and not torch.cuda.is_available():
            continue
            
        # Create model with different architectures
        ddpm = DiffusionModel(
            eps_model=UnetModel(3, 3, hidden_size=config["hidden_size"]),
            betas=(1e-4, 0.02),
            num_timesteps=config["num_timesteps"]
        )
        ddpm.to(config["device"])
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=config["lr"])
        
        # Create small dataset
        transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = CIFAR10(
            "./data",
            train=True,
            download=True,
            transform=transforms,
        )
        # Use only first 100 samples for faster testing
        dataset = torch.utils.data.Subset(dataset, list(range(100)))
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        
        # Time the training loop
        start_time = time.time()
        
        # Train for 2 epochs and track losses
        losses = []
        max_time = 300  # 5 minutes max
        
        for _ in range(2):
            epoch_loss = train_epoch(ddpm, dataloader, optimizer, config["device"])
            losses.append(epoch_loss)
            
            if time.time() - start_time > max_time:
                pytest.fail(f"Training took too long (>5 minutes) with config {config}")
        
        # Check training behavior based on learning rate
        if config["lr"] >= 1e-4:
            # Normal learning rate should show improvement
            assert losses[-1] < losses[0] * 0.95, (
                f"Training did not converge with lr={config['lr']}. "
                f"Initial loss: {losses[0]}, Final loss: {losses[-1]}"
            )
        else:
            # Very small learning rate should show minimal change
            assert abs(losses[-1] - losses[0]) < losses[0] * 0.1, (
                f"Loss changed too much with tiny lr={config['lr']}. "
                f"Initial loss: {losses[0]}, Final loss: {losses[-1]}"
            )
        
        # Test sample generation
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            generate_samples(ddpm, config["device"], tmp.name)
            # Verify that the file was created and is not empty
            assert tmp.tell() > 0, "Generated sample file is empty"
            
            # Check that generated images have reasonable values
            from PIL import Image
            img = Image.open(tmp.name)
            assert img.size == (64, 64), f"Unexpected image size: {img.size}"
