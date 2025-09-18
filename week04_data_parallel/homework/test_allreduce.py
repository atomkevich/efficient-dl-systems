import os
import time
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import numpy as np
import psutil
from allreduce import butterfly_allreduce, ring_allreduce, init_process

def measure_memory():
    """Измеряем использование памяти текущим процессом"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def run_test(rank, size, vector_size, num_iterations=10):
    """
    Тестируем различные реализации all_reduce
    Args:
        rank: ранг текущего процесса
        size: общее количество процессов
        vector_size: размер вектора для тестирования
        num_iterations: количество итераций для усреднения времени
    """
    results = {}
    torch.manual_seed(rank)
    
    # Создаем тензор для тестирования
    tensor = torch.randn(vector_size, dtype=torch.float32)
    tensor_copy = tensor.clone()
    
    # Тестируем Butterfly All-Reduce
    start_mem = measure_memory()
    start_time = time.time()
    for _ in range(num_iterations):
        butterfly_allreduce(tensor.clone(), rank, size)
    butterfly_time = (time.time() - start_time) / num_iterations
    butterfly_mem = measure_memory() - start_mem
    butterfly_result = tensor.clone()
    
    # Сбрасываем тензор
    tensor.copy_(tensor_copy)
    
    # Тестируем Ring All-Reduce
    start_mem = measure_memory()
    start_time = time.time()
    for _ in range(num_iterations):
        ring_allreduce(tensor.clone(), rank, size)
    ring_time = (time.time() - start_time) / num_iterations
    ring_mem = measure_memory() - start_mem
    ring_result = tensor.clone()
    
    # Сбрасываем тензор
    tensor.copy_(tensor_copy)
    
    # Тестируем встроенный all_reduce
    start_mem = measure_memory()
    start_time = time.time()
    for _ in range(num_iterations):
        dist.all_reduce(tensor.clone())
    torch_time = (time.time() - start_time) / num_iterations
    torch_mem = measure_memory() - start_mem
    torch_result = tensor.clone()
    
    # Собираем результаты
    if rank == 0:
        results = {
            'butterfly': {'time': butterfly_time, 'memory': butterfly_mem, 'result': butterfly_result},
            'ring': {'time': ring_time, 'memory': ring_mem, 'result': ring_result},
            'torch': {'time': torch_time, 'memory': torch_mem, 'result': torch_result}
        }
        
        # Проверяем точность
        torch_mean = torch_result.mean().item()
        butterfly_error = abs(butterfly_result.mean().item() - torch_mean)
        ring_error = abs(ring_result.mean().item() - torch_mean)
        
        print(f"\nResults for {size} workers, vector size {vector_size}:")
        print(f"Butterfly: {butterfly_time:.4f}s, {butterfly_mem:.2f}MB, error: {butterfly_error:.6f}")
        print(f"Ring: {ring_time:.4f}s, {ring_mem:.2f}MB, error: {ring_error:.6f}")
        print(f"Torch: {torch_time:.4f}s, {torch_mem:.2f}MB")
    
    return results

def test_allreduce():
    """
    Запускает тесты для разных конфигураций:
    - Количество процессов: 1-32
    - Размер вектора: 1000-100000
    """
    vector_sizes = [1000, 10000, 100000]
    worker_counts = [2, 4, 8, 16, 32]
    
    for vector_size in vector_sizes:
        for num_workers in worker_counts:
            if num_workers > 16 and vector_size == 100000:
                continue  # Пропускаем тяжелые конфигурации
                
            processes = []
            port = 29500 + (vector_size % 1000)  # Уникальный порт для каждого размера вектора
            
            for rank in range(num_workers):
                p = Process(
                    target=init_process,
                    args=(rank, num_workers, 
                          lambda r, s: run_test(r, s, vector_size),
                          port)
                )
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()

if __name__ == "__main__":
    test_allreduce()