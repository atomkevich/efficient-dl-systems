import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

"""
Task 1: Tensor Parallel Multi-Head Attention
Полное решение: шардирование по головам.
 - Q,K,V: column-wise shard (каждый rank хранит subset голов => subset output dim)
 - O: row-wise shard (каждый rank хранит slice входных колонок, умножает локально, затем all_reduce суммы)
Проверяем эквивалентность с эталонной (непараллельной) реализацией по выходам и градиенту входа.
"""

# -------- Reference Attention (упрощённая Llama-подобная) --------
class MyLlamaAttention(nn.Module):
    def __init__(self, hidden_size: int = 4096, num_heads: int = 32):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)
    def _shape(self, x):
        B,S,_ = x.shape
        return x.view(B,S,self.num_heads,self.head_dim).transpose(1,2)  # (B,H,S,D)
    def forward(self, x):  # x: (B,S,H)
        q = self._shape(self.wq(x))
        k = self._shape(self.wk(x))
        v = self._shape(self.wv(x))
        attn = F.scaled_dot_product_attention(q,k,v)  # (B,H,S,D)
        B,H,S,D = attn.shape
        attn = attn.transpose(1,2).reshape(B,S,H*D)
        return self.wo(attn)

# -------- Local shard module --------
class TPAttentionShard(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, rank: int, world_size: int):
        super().__init__()
        assert num_heads % world_size == 0
        self.hidden_size = hidden_size
        self.num_heads_total = num_heads
        self.world_size = world_size
        self.rank = rank
        self.local_heads = num_heads // world_size
        self.head_dim = hidden_size // num_heads
        self.local_hidden = self.local_heads * self.head_dim
        self.wq = nn.Linear(hidden_size, self.local_hidden, bias=False)
        self.wk = nn.Linear(hidden_size, self.local_hidden, bias=False)
        self.wv = nn.Linear(hidden_size, self.local_hidden, bias=False)
        # Row-parallel output: (local_hidden -> hidden_size)
        self.wo = nn.Linear(self.local_hidden, hidden_size, bias=False)
    def _shape_local(self, x):  # (B,S,local_hidden)->(B,h_loc,S,D)
        B,S,_ = x.shape
        return x.view(B,S,self.local_heads,self.head_dim).permute(0,2,1,3)
    def forward(self, x):
        q = self._shape_local(self.wq(x))
        k = self._shape_local(self.wk(x))
        v = self._shape_local(self.wv(x))
        attn_local = F.scaled_dot_product_attention(q,k,v)
        B,h,S,D = attn_local.shape
        attn_local = attn_local.permute(0,2,1,3).reshape(B,S,h*D)
        return self.wo(attn_local)  # (B,S,H) partial

class ComputeTPAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard: TPAttentionShard, x: torch.Tensor):
        x = x.detach().requires_grad_(x.requires_grad)
        ctx.save_for_backward(x)
        ctx.shard = shard
        out_partial = shard(x)
        dist.all_reduce(out_partial, op=dist.ReduceOp.SUM)
        return out_partial
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        shard: TPAttentionShard = ctx.shard
        with torch.enable_grad():
            x_local = x.detach().requires_grad_(True)
            out_partial = shard(x_local)
            dist.all_reduce(out_partial, op=dist.ReduceOp.SUM)
            out_partial.backward(grad_output)
            grad_in = x_local.grad
        # Суммируем вклады от разных шардированных QKV
        dist.all_reduce(grad_in, op=dist.ReduceOp.SUM)
        return None, grad_in

class TPAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, rank, world_size):
        super().__init__()
        self.shard = TPAttentionShard(hidden_size, num_heads, rank, world_size)
    def forward(self, x):
        return ComputeTPAttention.apply(self.shard, x)

# -------- Weight copy helper --------
@torch.no_grad()
def copy_tp_from_reference(ref: MyLlamaAttention, tp_shard: TPAttentionShard, rank: int, world_size: int):
    heads_total = ref.num_heads
    head_dim = ref.head_dim
    hpr = heads_total // world_size
    start_h = rank * hpr
    end_h = (rank+1)*hpr
    col_start = start_h * head_dim
    col_end = end_h * head_dim
    tp_shard.wq.weight.copy_(ref.wq.weight[col_start:col_end])
    tp_shard.wk.weight.copy_(ref.wk.weight[col_start:col_end])
    tp_shard.wv.weight.copy_(ref.wv.weight[col_start:col_end])
    # row-parallel wo: take columns col_start:col_end
    tp_shard.wo.weight.copy_(ref.wo.weight[:, col_start:col_end].T)

if __name__ == "__main__":
    dist.init_process_group("gloo")
    torch.manual_seed(1337)
    rank = dist.get_rank(); world_size = dist.get_world_size()
    hidden_size = 1024  # меньше для скорости в ноутбуке
    num_heads = 16

    # Поочерёдная инициализация для экономии RAM
    for active in range(world_size):
        dist.barrier()
        if rank != active:
            continue
        ref_attn = MyLlamaAttention(hidden_size=hidden_size, num_heads=num_heads)
        x = torch.randn(2, 64, hidden_size, requires_grad=True)
        ref_out = ref_attn(x)
        ref_out.sum().backward()
        ref_grad_in = x.grad.clone()
        tp_mod = TPAttention(hidden_size, num_heads, rank, world_size)
        copy_tp_from_reference(ref_attn, tp_mod.shard, rank, world_size)
        print(f"Initialized rank={rank}", flush=True)
        del ref_attn

    dist.barrier()
    # Test forward
    tp_input = x.detach().requires_grad_(True)
    tp_out = tp_mod(tp_input)
    if rank == 0:
        print("Reference out (rank0):", ref_out[0,0,:8].data)
    for r in range(world_size):
        dist.barrier()
        if r != rank: continue
        print(f"TP out (rank={rank}):", tp_out[0,0,:8].data)
        assert torch.allclose(tp_out, ref_out, atol=1e-5), f"forward mismatch rank={rank}"

    dist.barrier()
    # Test backward
    tp_out.sum().backward()
    if rank == 0:
        print("Ref grad_in (rank0):", ref_grad_in[0,0,:8])
    for r in range(world_size):
        dist.barrier()
        if r != rank: continue
        print(f"TP grad_in (rank={rank}):", tp_input.grad[0,0,:8])
        assert torch.allclose(tp_input.grad, ref_grad_in, atol=1e-4), f"grad mismatch rank={rank}"

    if rank == 0:
        print("All tensor-parallel attention tests passed.")
o back and, well... do it)*

```

```


```

```


```

```


```

```


```

```


### Full model conversion

Now let's apply this technique to parallelize the actual Llama model. As in, with weights.

__Task 2 (1 point):__ Combine the two previous techniques in one file that parallelizes an actual Llama model and .generates meaningful output. For simplicity, you do not need to partition key-value cache here - only the forward pass itself. We will default to generating tokens with recomputation.

For the sake of formality, your task is to parallelize the following inference code:
