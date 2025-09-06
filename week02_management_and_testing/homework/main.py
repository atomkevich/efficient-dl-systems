import os
from typing import Dict, Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import wandb

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize W&B
    wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Create model
    ddpm = DiffusionModel(
        eps_model=UnetModel(
            cfg.model.image_channels,
            cfg.model.image_channels,
            hidden_size=cfg.model.hidden_size
        ),
        betas=(cfg.model.beta_1, cfg.model.beta_2),
        num_timesteps=cfg.model.num_timesteps,
    )
    ddpm.to(device)
    
    # Log model architecture
    wandb.watch(ddpm, log_freq=100)
    
    # Data preparation
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    if cfg.training.use_random_flip:
        transforms_list.insert(1, transforms.RandomHorizontalFlip())
    
    train_transforms = transforms.Compose(transforms_list)

    dataset = CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True
    )
    
    # Create optimizer
    if cfg.optimizer.name.lower() == "adam":
        optimizer = torch.optim.Adam(
            ddpm.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif cfg.optimizer.name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            ddpm.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")
    
    # Log config as artifact
    config_path = os.path.join(hydra.utils.get_original_cwd(), "conf/experiment", f"{cfg.experiment}.yaml")
    artifact = wandb.Artifact("experiment_config", type="config")
    artifact.add_file(config_path)
    wandb.log_artifact(artifact)
    
    # Training loop
    for epoch in range(cfg.training.epochs):
        # Train one epoch
        avg_loss = train_epoch(ddpm, dataloader, optimizer, device)
        
        # Log metrics
        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_size": cfg.training.batch_size,
            "optimizer": cfg.optimizer.name,
        }
        wandb.log(metrics, step=epoch)
        
        # Generate and log samples every 10 epochs
        if epoch % 10 == 0:
            samples_path = f"samples_epoch_{epoch}.png"
            generate_samples(ddpm, device, samples_path)
            wandb.log({"samples": wandb.Image(samples_path)}, step=epoch)
        
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {cfg.optimizer.lr:.2e}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
