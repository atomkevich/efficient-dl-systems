import os
import random

import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def init_process(rank, size, fn, master_port, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def butterfly_allreduce(send, rank, size):
    """
    Performs Butterfly All-Reduce over the process group. Modifies the input tensor in place.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """

    buffer_for_chunk = torch.empty((size,), dtype=torch.float)

    send_futures = []

    send_futures.extend(
        dist.isend(elem, i) for i, elem in enumerate(send) if i != rank
    )

    recv_futures = []

    for i, elem in enumerate(buffer_for_chunk):
        if i != rank:
            recv_futures.append(dist.irecv(elem, i))
        else:
            elem.copy_(send[i])

    for future in recv_futures:
        future.wait()

    # compute the average
    torch.mean(buffer_for_chunk, dim=0, out=send[rank])

    send_futures.extend(
        dist.isend(send[rank], i) for i in range(size) if i != rank
    )

    recv_futures = []

    recv_futures.extend(
        dist.irecv(elem, i) for i, elem in enumerate(send) if i != rank
    )

    for future in recv_futures:
        future.wait()
    for future in send_futures:
        future.wait()


def ring_allreduce(send: torch.Tensor, rank: int, size: int):
    """
    Ring All-Reduce (SUM) + деление на size (AVERAGE).
    Работает для 1-D тензора любой длины: делим на 'size' чанков через torch.chunk.
    Модифицирует 'send' на месте.
    """
    assert send.dim() == 1, "Ожидается 1-D тензор"
    p = size
    if p == 1:
        return

    # Разбиваем вектор на p чанков (последние могут отличаться по длине)
    chunks = list(torch.chunk(send, p, dim=0))
    # Локальные “редукции” по чанкам (копии, чтобы аккумулировать суммы)
    red = [c.clone() for c in chunks]

    prev = (rank - 1) % p
    nxt  = (rank + 1) % p

    # -------- reduce-scatter --------
    # За p-1 шаг: на шаге s процесс r шлёт (r - s) mod p, принимает (r - s - 1) mod p, суммирует.
    for s in range(p - 1):
        idx_send = (rank - s) % p
        idx_recv = (rank - s - 1) % p

        send_buf = red[idx_send]
        recv_buf = torch.empty_like(red[idx_recv])

        req_r = dist.irecv(tensor=recv_buf, src=prev)
        req_s = dist.isend(tensor=send_buf, dst=nxt)
        req_r.wait(); req_s.wait()

        red[idx_recv].add_(recv_buf)

    # Теперь у ранга r полностью просуммирован чанк с индексом (r + 1) % p
    # -------- all-gather --------
    # Раздаём суммированные чанки по кольцу, чтобы собрать полный вектор.
    for s in range(p - 1):
        idx_send = (rank - s - 1) % p
        idx_recv = (rank - s - 2) % p

        send_buf = red[idx_send]
        recv_buf = torch.empty_like(red[idx_recv])

        req_r = dist.irecv(tensor=recv_buf, src=prev)
        req_s = dist.isend(tensor=send_buf, dst=nxt)
        req_r.wait(); req_s.wait()

        red[idx_recv].copy_(recv_buf)

    # Склеиваем и усредняем
    send.copy_(torch.cat(red, dim=0))
    send.div_(p)



def run_butterfly_allreduce(rank, size):
    """Simple point-to-point communication."""
    torch.manual_seed(rank)
    tensor = torch.randn((size,), dtype=torch.float)
    print("Rank ", rank, " has data ", tensor)
    butterfly_allreduce(tensor, rank, size)
    print("Rank ", rank, " has data ", tensor)
    

def run_ring_allreduce(rank, size):
    """Simple point-to-point communication."""
    torch.manual_seed(rank)
    tensor = torch.randn((size,), dtype=torch.float)
    print("Rank ", rank, " has data ", tensor)
    ring_allreduce(tensor, rank, size)
    print("Rank ", rank, " has data ", tensor)


if __name__ == "__main__":
    size = 5
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_butterfly_allreduce, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    # --------------------------------------------------    
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_ring_allreduce, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
