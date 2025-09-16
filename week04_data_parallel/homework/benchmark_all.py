import os
import time
from collections import defaultdict

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from ddp_cifar100 import Net, validate, init_process, sync_gradients, run_training


def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def run_benchmark(rank, size):
    """Run both implementations for comparison"""
    print(f"Running benchmarks on rank {rank}")
    
    # Запускаем наш кастомный DDP
    #print("Running Custom DDP implementation...")
    custom_metrics = run_training(rank, size, use_pytorch_ddp=False)
    
    # Очищаем кэш CUDA для честного сравнения
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Запускаем встроенный DDP PyTorch
    print("\nRunning PyTorch DDP implementation...")
    pytorch_metrics = run_training(rank, size, use_pytorch_ddp=True)
    
    if rank == 0:
        print("\nPerformance Comparison:")
        print("-" * 50)
        print("Metric\t\t\tCustom DDP\tPyTorch DDP")
        print("-" * 50)
        
        custom_batch_time = np.mean(custom_metrics['batch_time'])
        pytorch_batch_time = np.mean(pytorch_metrics['batch_time'])
        print(f"Avg Batch Time\t\t{custom_batch_time:.4f}s\t{pytorch_batch_time:.4f}s")
        
        print(f"Total Time\t\t{custom_metrics['total_time']:.2f}s\t{pytorch_metrics['total_time']:.2f}s")
        print(f"Memory Usage\t\t{custom_metrics['memory_usage']:.2f}MB\t{pytorch_metrics['memory_usage']:.2f}MB")
        print(f"Final Accuracy\t\t{custom_metrics['val_acc'][-1]:.4f}\t{pytorch_metrics['val_acc'][-1]:.4f}")

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_benchmark, backend="gloo")  # use "nccl" for GPU