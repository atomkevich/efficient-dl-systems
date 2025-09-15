import pytest
import torch
from syncbn import SyncBatchNorm


def run_worker(rank, world_size, batch_size, hid_dim, num_features, queue):
    """Worker process function."""
    import os
    import torch.distributed as dist
    
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Create layers
    syncbn = SyncBatchNorm(num_features)
    bn = torch.nn.BatchNorm1d(num_features, affine=False)
    
    # Create input data (same across all processes)
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(batch_size, num_features, hid_dim)
    
    # Create mask for first B/2 samples
    mask = torch.zeros_like(input_tensor)
    mask[:batch_size//2] = 1.0
    
    # Forward and backward pass with SyncBN
    input_tensor.requires_grad = True
    sync_out = syncbn(input_tensor)
    loss_sync = (sync_out * mask).sum()
    loss_sync.backward()
    grad_sync = input_tensor.grad.clone()
    
    # Forward and backward pass with regular BN
    input_tensor.grad = None
    bn_out = bn(input_tensor)
    loss_bn = (bn_out * mask).sum()
    loss_bn.backward()
    grad_bn = input_tensor.grad.clone()
    
    # Clean up
    dist.destroy_process_group()
    
    # Send results back to main process
    queue.put({
        'rank': rank,
        'sync_out': sync_out.detach(),
        'bn_out': bn_out.detach(),
        'grad_sync': grad_sync,
        'grad_bn': grad_bn
    })

@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hid_dim", [128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hid_dim, batch_size):
    """Test SyncBatchNorm implementation with different configurations."""
    import torch.multiprocessing as mp
    ctx = torch.multiprocessing.get_context("spawn")
    
    # Create a queue to get results from workers
    queue = ctx.Queue()
    num_features = 64
    
    # Start worker processes
    processes = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=run_worker,
            args=(rank, num_workers, batch_size, hid_dim, num_features, queue)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for _ in range(num_workers):
        results.append(queue.get())
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Compare results
    for r1, r2 in zip(results[:-1], results[1:]):
        torch.testing.assert_close(r1['sync_out'], r2['sync_out'], rtol=0, atol=1e-3)
        torch.testing.assert_close(r1['grad_sync'], r2['grad_sync'], rtol=0, atol=1e-3)
    
    # Compare with regular BatchNorm (using results from first worker)
    r0 = results[0]
    torch.testing.assert_close(r0['sync_out'], r0['bn_out'], rtol=0, atol=1e-3)
    torch.testing.assert_close(r0['grad_sync'], r0['grad_bn'], rtol=0, atol=1e-3)
