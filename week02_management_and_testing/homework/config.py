from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Training hyperparameters
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-5
    num_workers: int = 4
    
    # Model hyperparameters
    image_channels: int = 3
    unet_hidden_size: int = 128
    
    # Diffusion hyperparameters
    beta_1: float = 1e-4
    beta_2: float = 0.02
    num_timesteps: int = 1000
    
    # Data hyperparameters
    image_size: int = 32
    
    # Sampling hyperparameters
    num_samples: int = 8
    sample_grid_nrow: int = 4
    
    # Paths
    dataset_path: str = "cifar10"
    samples_dir: str = "samples"
    
    # W&B
    project_name: str = "ddpm-cifar10"
    run_name: str = "ddpm-baseline"


def get_config() -> TrainingConfig:
    return TrainingConfig()