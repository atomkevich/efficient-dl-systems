import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import wandb
import os

from modeling.diffusion import DiffusionModel
from config import TrainingConfig


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str, epoch: int, log_input_batch: bool = False):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (x, _) in enumerate(pbar):
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        total_loss += train_loss.item()
        num_batches += 1
        
        pbar.set_description(f"[Epoch {epoch}] loss: {loss_ema:.4f}")
        
        # Log input batch only once per epoch and only for first batch
        if log_input_batch and batch_idx == 0:
            # Denormalize images for logging (from [-1, 1] to [0, 1])
            images_denorm = (x[:8] + 1) / 2  # Take first 8 images
            images_denorm = torch.clamp(images_denorm, 0, 1)
            grid = make_grid(images_denorm, nrow=4)
            wandb.log({"input_batch": wandb.Image(grid)}, step=epoch)
    
    avg_loss = total_loss / num_batches
    return avg_loss


def generate_samples(model: DiffusionModel, config: TrainingConfig, device: str, epoch: int):
    model.eval()
    with torch.no_grad():
        samples = model.sample(config.num_samples, (config.image_channels, config.image_size, config.image_size), device=device)
        # Denormalize samples for display (from [-1, 1] to [0, 1])
        samples_denorm = (samples + 1) / 2
        samples_denorm = torch.clamp(samples_denorm, 0, 1)
        grid = make_grid(samples_denorm, nrow=config.sample_grid_nrow)
        
        # Save to file
        os.makedirs(config.samples_dir, exist_ok=True)
        save_path = os.path.join(config.samples_dir, f"epoch_{epoch:03d}.png")
        save_image(grid, save_path)
        
        # Log to W&B
        wandb.log({"generated_samples": wandb.Image(grid)}, step=epoch)
        
        return samples_denorm
