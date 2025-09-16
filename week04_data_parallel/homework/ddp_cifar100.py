import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100

torch.set_num_threads(1)


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        self.bn1 = nn.BatchNorm1d(128, affine=False)  # to be replaced with SyncBatchNorm

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return self.fc2(x)


def sync_gradients(model):
    """Synchronize gradients across all processes."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def validate(model, val_loader, device, rank, world_size):
    """Распределенная валидация с агрегацией метрик."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    # Собираем метрики со всех процессов
    metrics = torch.tensor([total_loss, correct, total], dtype=torch.float64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    val_loss = metrics[0] / metrics[2]
    val_acc = metrics[1] / metrics[2]
    
    return val_loss, val_acc

def run_training(rank, size, use_pytorch_ddp=False):
    """
    Run distributed training with either custom or PyTorch DDP implementation
    
    Args:
        rank: Process rank
        size: World size (total number of processes)
        use_pytorch_ddp: If True, use PyTorch's DistributedDataParallel, otherwise use custom implementation
    """
    torch.manual_seed(0)
    
    # Создаем train и val датасеты
    train_dataset = CIFAR100(
        "./cifar",
        train=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
        download=True,
    )
    
    val_dataset = CIFAR100(
        "./cifar",
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
        download=True,
    )
    
    # Создаем распределенные загрузчики данных
    train_sampler = DistributedSampler(train_dataset, size, rank)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=64)
    
    val_sampler = DistributedSampler(val_dataset, size, rank, shuffle=False)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=64)

    model = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Выбираем реализацию DDP
    if use_pytorch_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # На CPU не используем SyncBatchNorm от PyTorch
        if torch.cuda.is_available():
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model)
    else:
        # Используем нашу реализацию SyncBatchNorm, которая работает и на CPU
        from syncbn import SyncBatchNorm
        model = SyncBatchNorm.convert_sync_batchnorm(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Gradient accumulation settings
    accumulation_steps = 4  # Number of batches to accumulate gradients
    optimizer.zero_grad()

    for epoch in range(10):
        model.train()
        train_sampler.set_epoch(epoch)  # Важно для правильной перемешивания данных
        epoch_loss = torch.zeros(1, device=device)
        running_acc = 0.0
        num_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            # Forward pass and loss computation
            output = model(data)
            loss = F.cross_entropy(output, target)
            epoch_loss += loss.detach()  # Сохраняем оригинальное значение loss для статистики
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()

            # Update weights only after accumulating enough gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                # Synchronize gradients only when we're about to update
                sync_gradients(model)
                optimizer.step()
                optimizer.zero_grad()

            pred = output.argmax(dim=1)
            acc = pred.eq(target).sum().item()
            running_acc += acc
            num_samples += target.size(0)

            # Логгируем только с процесса ранга 0
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                      f"Loss: {loss.item():.6f}\t"
                      f"Acc: {running_acc/num_samples:.4f}")

        # Валидация в конце каждой эпохи
        val_loss, val_acc = validate(model, val_loader, device, rank, size)
        
        # Логгируем результаты эпохи только с процесса ранга 0
        if rank == 0:
            print(f"Epoch: {epoch}, Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        
        # Handle remaining gradients at the end of epoch
        if len(train_loader) % accumulation_steps != 0:
            sync_gradients(model)
            optimizer.step()
            optimizer.zero_grad()
            
        scheduler.step()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, backend="gloo")  # use "nccl" for GPU
