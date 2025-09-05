import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import wandb
import argparse
from dataclasses import asdict

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel
from config import get_config


def main(device: str, config_override: dict = None):
    config = get_config()
    
    # Override config if provided
    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Initialize W&B
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=asdict(config)
    )
    
    # Create model
    ddpm = DiffusionModel(
        eps_model=UnetModel(config.image_channels, config.image_channels, hidden_size=config.unet_hidden_size),
        betas=(config.beta_1, config.beta_2),
        num_timesteps=config.num_timesteps,
    )
    ddpm.to(device)
    
    # Log model architecture
    wandb.watch(ddpm, log_freq=100, log_graph=True)
    
    # Data preparation
    train_transforms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CIFAR10(
        config.dataset_path,
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        num_workers=config.num_workers, 
        shuffle=True
    )
    
    # Optimizer with learning rate scheduling
    optim = torch.optim.Adam(ddpm.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=config.num_epochs)
    
    # Training loop
    for epoch in range(config.num_epochs):
        # Train one epoch
        avg_loss = train_epoch(
            ddpm, 
            dataloader, 
            optim, 
            device, 
            epoch, 
            log_input_batch=(epoch == 0)  # Log input batch only in first epoch
        )
        
        # Log metrics
        current_lr = optim.param_groups[0]['lr']
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "learning_rate": current_lr
        }, step=epoch)
        
        # Generate and log samples
        generate_samples(ddpm, config, device, epoch)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {current_lr:.2e}")
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--run-name", type=str, default="ddpm-baseline", help="W&B run name")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Override config with command line args
    config_override = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "run_name": args.run_name
    }
    
    main(device=device, config_override=config_override)
