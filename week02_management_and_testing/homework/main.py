import os
from typing import Dict, Any

import hydra

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
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
    print(cfg)
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
        root="data",  # это создаст data/cifar-10-batches-py
        train=True,
        download=False,  # данные уже должны быть загружены
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
    exp_name = HydraConfig.get().runtime.choices.experiment

    # путь к исходному yaml-файлу
    config_path = os.path.join(
        hydra.utils.get_original_cwd(),
        "conf/experiment",
        f"{exp_name}.yaml"
)
    artifact = wandb.Artifact("experiment_config", type="config")
    artifact.add_file(config_path)
    wandb.log_artifact(artifact)

    # Training loop
    all_metrics = []
    for epoch in range(cfg.training.epochs):
        # Train one epoch
        avg_loss = train_epoch(ddpm, dataloader, optimizer, device, epoch, cfg)
        
        # Collect metrics
        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_size": cfg.training.batch_size,
            "optimizer": cfg.optimizer.name,
        }
        all_metrics.append(metrics)
        wandb.log(metrics, step=epoch)
        
        # Generate and log samples according to config
        if epoch % cfg.sampling.save_every_n_epochs == 0:
            from config import TrainingConfig
            
            # Create config with values from Hydra
            config = TrainingConfig()
            config.num_samples = cfg.sampling.num_samples
            config.image_channels = cfg.model.image_channels
            config.image_size = cfg.sampling.image_size
            config.sample_grid_nrow = cfg.sampling.sample_grid_nrow
            config.samples_dir = cfg.sampling.samples_dir
            
            generate_samples(ddpm, config, device, epoch)
        
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {cfg.optimizer.lr:.2e}")
    
    # Save final metrics for DVC
    final_metrics = {
        "final_loss": all_metrics[-1]["train_loss"],
        "best_loss": min(m["train_loss"] for m in all_metrics),
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "optimizer": cfg.optimizer.name,
        "learning_rate": cfg.optimizer.lr,
    }
    
    # Получаем имя эксперимента из конфига Hydra
    exp_name = HydraConfig.get().runtime.choices.experiment
    
    # Сохраняем метрики в трех местах:
    import json
    
    # 1. В директории Hydra для полных логов
    with open("metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # 2. В директории metrics/<experiment_name>.json для DVC
    metrics_dir = os.path.join(hydra.utils.get_original_cwd(), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"{exp_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
        
    # 3. Копия последнего эксперимента в metrics.json для обратной совместимости
    latest_metrics_path = os.path.join(hydra.utils.get_original_cwd(), "metrics.json")
    with open(latest_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    wandb.finish()


if __name__ == "__main__":
    main()
