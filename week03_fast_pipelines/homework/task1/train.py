from typing import Optional

import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet
from loss_scaler import StaticLossScaler, DynamicLossScaler

from dataset import get_train_data


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional["LossScaler"] = None,
) -> None:
    model.train()
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        if scaler is not None:
            # Scale loss and backward
            scaled_loss = scaler.scale_loss(loss)
            scaled_loss.backward()
            
            # Step with scaler
            success = scaler.step(optimizer)
            if not success:
                continue
        else:
            # Regular backward and step without scaling
            loss.backward()
            optimizer.step()
        
        accuracy = ((outputs > 0.5) == labels).float().mean()
        
        pbar.set_description(
            f"Loss: {round(loss.item(), 4)} "
            f"Accuracy: {round(accuracy.item() * 100, 4)}"
            + (f" Scale: {scaler.cur_scale:.0f}" if hasattr(scaler, "cur_scale") else "")
        )


def train(use_amp: bool = True, scaler_type: str = "dynamic"):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    # Create loss scaler if using AMP
    scaler = None
    if use_amp:
        if scaler_type == "static":
            scaler = StaticLossScaler()
        elif scaler_type == "dynamic":
            scaler = DynamicLossScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device=device,
            scaler=scaler
        )
