import os

import torch.distributed as dist


def run_sequential(rank, size, num_iter=10):
    """
    Prints the process rank sequentially according to its number over `num_iter` iterations,
    separating the output for each iteration by `---`
    Example (3 processes, num_iter=2):
    ```
    Process 0
    Process 1
    Process 2
    ---
    Process 0
    Process 1
    Process 2
    ```
    """

    for _ in range(num_iter):
        for r in range(size):
            if rank == r:
                print(f"Process {rank}", flush=True)
            dist.barrier()
    
            


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])          # 0..nproc-1
    dist.init_process_group(backend="gloo")             # init_method="env://" по умолчанию

    # Можно брать rank/size из dist (надёжнее), а не из env:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    run_sequential(rank, world_size, num_iter=2)

    dist.destroy_process_group()
