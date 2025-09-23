"""
Sequence Parallel Training Script (with optional Prompt Tuning + LoRA)
---------------------------------------------------------------------
This script extends the earlier naive sequence-parallel training demo by adding:
  * LoRA adapters for selected Linear layers (attention / MLP)
  * Prompt tuning (virtual tokens) optionally combined with LoRA
  * Trainable parameter reporting

Naive sequence parallel strategy (current version):
  - Shard sequence dimension across ranks (tokens) -> each rank holds (B, S_local, H)
  - Reconstruct full sequence (all_gather) for attention projections (inefficient but simple)
  - Each rank computes only its subset of heads (head shard) then all_reduce the merged head outputs
  - MLP runs locally (replicated weights)
  - Gather local logits for loss (teacher-style next-token prediction)

Future improvements (not implemented here):
  - all_to_all based attention avoiding full sequence replication
  - Overlap comm/compute, gradient checkpointing per head group
  - Combining sequence + tensor parallel parameter shards

Usage (single GPU dummy):
  python sequence_parallel_train_lora.py --dummy --layers 2 --seq-len 512 --steps 20 \
      --lora --lora-r 8 --lora-targets attn.q_proj,attn.k_proj,attn.v_proj,mlp.gate_proj,mlp.up_proj

Multi-GPU (2 GPUs example):
  torchrun --nproc_per_node 2 sequence_parallel_train_lora.py --dummy --layers 4 --seq-len 2048 \
      --steps 50 --backend nccl --dtype bf16 --lora --lora-r 16 \
      --lora-targets attn.q_proj,attn.v_proj,mlp.gate_proj --virtual-tokens 16

Flags:
  --lora              Enable LoRA (low rank adapters)
  --lora-r            Rank of LoRA decomposition
  --lora-alpha        Scaling factor (alpha / r)
  --lora-dropout      Dropout on LoRA input (optional)
  --lora-targets      Comma separated substrings to match module path names for instrumentation
  --virtual-tokens    Prompt tuning virtual tokens count (0 to disable)
  --no-prompt         Disable prompt tuning even if virtual-tokens > 0

Outputs: prints step loss, learning rate, and at end summary of trainable parameter share.
"""
import os, math, argparse, random, time
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

try:
    from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding, apply_rotary_pos_emb
    )
except Exception:
    LlamaForCausalLM = AutoTokenizer = LlamaConfig = None
    LlamaRotaryEmbedding = apply_rotary_pos_emb = None

# ================= Distributed init =================
def init_distributed(backend="nccl"):
    if dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        dist.init_process_group("gloo", rank=0, world_size=1)
    else:
        dist.init_process_group(backend=backend)

# ================= LoRA Modules =====================
class LoRALinear(nn.Module):
    """Wraps a Linear layer with low-rank (A @ B) additive adaptation.
    weight update:  y = W x + (scaling * B(A(x)))  where A: in->r, B: r->out
    Base weight is frozen by caller.
    """
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float=0.0):
        super().__init__()
        assert base.bias is None, "Bias not supported in this minimal LoRA wrapper"
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Init: as in typical LoRA: A normal, B zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # Freeze base weight
        for p in self.base.parameters():
            p.requires_grad_(False)
    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out

def apply_lora(model: nn.Module, targets: List[str], r: int, alpha: float, dropout: float):
    """Replace nn.Linear modules whose qualified name contains any pattern in targets with LoRALinear."""
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            qname = name
            if any(t in qname for t in targets):
                parent_path = qname.split('.')[:-1]
                attr_name = qname.split('.')[-1]
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)
                new_mod = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, new_mod)
                replaced.append(qname)
    return replaced

# ================= Sequence helpers =================
def split_sequence(input_ids, rank: int, world: int):
    B, S = input_ids.shape
    assert S % world == 0, "Sequence length must be divisible by world size (pad beforehand)"
    chunk = S // world
    start = rank * chunk
    end = (rank + 1) * chunk
    return input_ids[:, start:end]

def gather_sequence(x_local, world: int):
    if world == 1:
        return x_local
    gather_list = [torch.empty_like(x_local) for _ in range(world)]
    dist.all_gather(gather_list, x_local)
    return torch.cat(gather_list, dim=1)

# =============== Rotary Embeddings ===============
class RotaryContext:
    def __init__(self, dim: int, max_position: int, device, dtype):
        self.rope = LlamaRotaryEmbedding(dim=dim, max_position_embeddings=max_position, base=10000, device=device)
        # precompute cache lazily inside forward of HuggingFace variant; we mimic call
    def __call__(self, q, k, position_ids):
        cos, sin = self.rope(q, position_ids)  # returns pair (cos,sin)
        return apply_rotary_pos_emb(q, k, cos, sin)

# =============== Attention (naive gather) ===============
class SeqParAttention(nn.Module):
    def __init__(self, ref_attn, config, rank: int, world: int):
        super().__init__()
        self.hidden_size = ref_attn.q_proj.in_features
        self.num_heads = ref_attn.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert self.hidden_size % self.num_heads == 0
        assert self.num_heads % world == 0
        self.world = world
        self.rank = rank
        self.local_heads = self.num_heads // world
        # replicate projection weights for simplicity (could TP-shard)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.scale = self.head_dim ** -0.5
        self.use_rope = True
        self.rotary = RotaryContext(self.head_dim, max_position=65536, device="cpu", dtype=torch.float32) if LlamaRotaryEmbedding else None
    def forward(self, x_local, position_ids_full):
        # x_local: (B, S_local, H); reconstruct full sequence for q,k,v
        world = self.world
        B, S_local, H = x_local.shape
        x_full = gather_sequence(x_local, world)  # (B,S,H)
        q = self.q_proj(x_full).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x_full).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x_full).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)
        if self.use_rope and self.rotary is not None:
            q, k = self.rotary(q, k, position_ids_full)
        # slice local heads
        h_start = self.rank * self.local_heads
        h_end = (self.rank+1) * self.local_heads
        q_l = q[:, h_start:h_end]
        k_l = k[:, h_start:h_end]
        v_l = v[:, h_start:h_end]
        attn = torch.matmul(q_l, k_l.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        ctx = torch.matmul(attn, v_l)  # (B, local_heads, S, D)
        ctx = ctx.transpose(1,2).contiguous().view(B, -1, self.local_heads*self.head_dim)
        # all_reduce over head-split output to assemble full hidden
        if self.world > 1:
            dist.all_reduce(ctx, op=dist.ReduceOp.SUM)
        y = self.o_proj(ctx)
        # return only local sequence slice for residual path alignment
        chunk = y.shape[1] // self.world
        return y[:, self.rank*chunk:(self.rank+1)*chunk]

# =============== MLP =================
class SeqParMLP(nn.Module):
    def __init__(self, ref_mlp):
        super().__init__()
        self.gate_proj = nn.Linear(ref_mlp.gate_proj.in_features, ref_mlp.gate_proj.out_features, bias=False)
        self.up_proj = nn.Linear(ref_mlp.up_proj.in_features, ref_mlp.up_proj.out_features, bias=False)
        self.down_proj = nn.Linear(ref_mlp.down_proj.in_features, ref_mlp.down_proj.out_features, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# =============== Block =================
class SeqParBlock(nn.Module):
    def __init__(self, ref_block, config, rank, world):
        super().__init__()
        hs = ref_block.input_layernorm.weight.shape[0]
        self.rms1 = nn.RMSNorm(hs)
        self.rms2 = nn.RMSNorm(hs)
        self.attn = SeqParAttention(ref_block.self_attn, config, rank, world)
        self.mlp = SeqParMLP(ref_block.mlp)
    def forward(self, x_local, position_ids_full):
        h = self.rms1(x_local)
        h_attn = self.attn(h, position_ids_full)
        h = x_local + h_attn
        h2 = self.rms2(h)
        h = h + self.mlp(h2)
        return h

# =============== Wrapper Model =================
class SequenceParallelTrainModel(nn.Module):
    def __init__(self, ref_model, rank, world, num_layers=None):
        super().__init__()
        self.config = ref_model.config
        self.embed = nn.Embedding(ref_model.model.embed_tokens.num_embeddings,
                                  ref_model.model.embed_tokens.embedding_dim)
        layers = ref_model.model.layers if (num_layers is None) else ref_model.model.layers[:num_layers]
        self.layers = nn.ModuleList([SeqParBlock(lb, ref_model.config, rank, world) for lb in layers])
        self.final_norm = nn.RMSNorm(ref_model.model.norm.weight.shape[0])
        self.lm_head = nn.Linear(ref_model.lm_head.in_features, ref_model.lm_head.out_features, bias=False)
        self.rank = rank
        self.world = world
    def forward(self, input_ids_full):
        # input_ids_full is the FULL sequence (B,S) replicated across ranks
        B,S = input_ids_full.shape
        world = self.world
        # Convert to embeddings and slice local tokens
        emb = self.embed(input_ids_full)  # (B,S,H)
        chunk = S // world
        local = emb[:, self.rank*chunk:(self.rank+1)*chunk]
        # position ids (full) for RoPE
        device = emb.device
        position_ids = torch.arange(S, device=device).unsqueeze(0)
        for layer in self.layers:
            local = layer(local, position_ids)
        # gather sequence before final norm / lm_head so weights see full context
        full_hidden = gather_sequence(local, world)
        full_hidden = self.final_norm(full_hidden)
        logits = self.lm_head(full_hidden)
        return logits

@torch.no_grad()
def copy_weights(ref_model, sp_model):
    sp_model.embed.weight.copy_(ref_model.model.embed_tokens.weight)
    for sp_blk, ref_blk in zip(sp_model.layers, ref_model.model.layers):
        # attn proj
        sp_blk.attn.q_proj.weight.copy_(ref_blk.self_attn.q_proj.weight)
        sp_blk.attn.k_proj.weight.copy_(ref_blk.self_attn.k_proj.weight)
        sp_blk.attn.v_proj.weight.copy_(ref_blk.self_attn.v_proj.weight)
        sp_blk.attn.o_proj.weight.copy_(ref_blk.self_attn.o_proj.weight)
        # mlp
        sp_blk.mlp.gate_proj.weight.copy_(ref_blk.mlp.gate_proj.weight)
        sp_blk.mlp.up_proj.weight.copy_(ref_blk.mlp.up_proj.weight)
        sp_blk.mlp.down_proj.weight.copy_(ref_blk.mlp.down_proj.weight)
        # norms
        sp_blk.rms1.weight.copy_(ref_blk.input_layernorm.weight)
        sp_blk.rms2.weight.copy_(ref_blk.post_attention_layernorm.weight)
    sp_model.final_norm.weight.copy_(ref_model.model.norm.weight)
    sp_model.lm_head.weight.copy_(ref_model.lm_head.weight)

# =============== Prompt Tuning Module ===============
class PromptTuner(nn.Module):
    def __init__(self, virtual_tokens: int, hidden_size: int):
        super().__init__()
        self.virtual_tokens = virtual_tokens
        if virtual_tokens > 0:
            self.prompt_embed = nn.Parameter(torch.zeros(virtual_tokens, hidden_size))
            nn.init.normal_(self.prompt_embed, std=0.02)
        else:
            self.register_parameter('prompt_embed', None)
    def forward(self, input_ids_full, embedding_fn):
        # embedding_fn: function(ids)->(B,S,H)
        base_emb = embedding_fn(input_ids_full)
        if self.prompt_embed is None:
            return base_emb, input_ids_full
        B = base_emb.size(0)
        prompt = self.prompt_embed.unsqueeze(0).expand(B, -1, -1)
        emb = torch.cat([prompt, base_emb], dim=1)
        pad_ids = torch.zeros(B, self.prompt_embed.size(0), dtype=input_ids_full.dtype, device=input_ids_full.device) - 1
        new_ids = torch.cat([pad_ids, input_ids_full], dim=1)
        return emb, new_ids

# =============== Synthetic Data Loader ===============
def build_synthetic_dataset(tokenizer, vocab_size: int, seq_len: int, dataset_size: int):
    data = []
    for _ in range(dataset_size):
        length = seq_len
        ids = torch.randint(low=5, high=vocab_size-1, size=(length,), dtype=torch.long)
        data.append(ids)
    return data

def data_loader(data: List[torch.Tensor], batch_size: int, device):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch = pad_sequence(batch, batch_first=True, padding_value=0)
        yield batch.to(device)

# =============== Training =================
def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    ap.add_argument('--dummy', action='store_true')
    ap.add_argument('--layers', type=int, default=4, help='Number of layers to use (-1 = full)')
    ap.add_argument('--seq-len', type=int, default=1024)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--backend', default='gloo')
    ap.add_argument('--dtype', choices=['fp32','bf16','fp16'], default='fp32')
    ap.add_argument('--virtual-tokens', type=int, default=32)
    ap.add_argument('--no-prompt', action='store_true')
    # LoRA
    ap.add_argument('--lora', action='store_true')
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--lora-alpha', type=float, default=16.0)
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--lora-targets', type=str, default='attn.q_proj,attn.v_proj')
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    init_distributed(args.backend)
    rank = dist.get_rank(); world = dist.get_world_size()
    device = torch.device('cuda', rank) if torch.cuda.is_available() else torch.device('cpu')
    if rank == 0:
        print(f"[Init] world={world} seq_len={args.seq_len} layers={args.layers} lora={args.lora}")
    # Dtype
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Build reference model (weights source)
    if args.dummy:
        assert LlamaConfig is not None, 'transformers required for dummy'
        cfg = LlamaConfig(
            hidden_size=512,
            intermediate_size=1536,
            num_attention_heads=8,
            num_hidden_layers=max(args.layers, 2),
            vocab_size=32000,
        )
        ref_model = LlamaForCausalLM(cfg)
        tokenizer = None
    else:
        ref_model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    ref_model.to(dtype).to(device)
    dist.barrier()

    # Sequence Parallel model
    sp = SequenceParallelTrainModel(ref_model, rank, world, num_layers=None if args.layers < 0 else args.layers).to(dtype).to(device)
    copy_weights(ref_model, sp)
    # Freeze everything
    for p in sp.parameters():
        p.requires_grad_(False)

    # Prompt tuner
    prompt = None
    if not args.no_prompt and args.virtual_tokens > 0:
        prompt = PromptTuner(args.virtual_tokens, sp.config.hidden_size).to(dtype).to(device)

    # Apply LoRA if requested
    lora_modules = []
    if args.lora:
        targets = [t.strip() for t in args.lora_targets.split(',') if t.strip()]
        lora_modules = apply_lora(sp, targets, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        if rank == 0:
            print(f"[LoRA] Applied to {len(lora_modules)} modules: {lora_modules}")

    # Optimizer collects trainable params (prompt + LoRA)
    trainable_params = []
    if prompt is not None:
        trainable_params += list(prompt.parameters())
    trainable_params += [p for p in sp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    trainable_count, total_count = count_trainable(sp)  # only SP model (prompt separate)
    if prompt is not None:
        trainable_count += sum(p.numel() for p in prompt.parameters())
    if rank == 0:
        pct = 100.0 * trainable_count / (total_count + (sum(p.numel() for p in prompt.parameters()) if prompt else 0))
        print(f"[Params] Trainable={trainable_count/1e6:.2f}M total~={total_count/1e6:.2f}M ({pct:.3f}%)")

    # Synthetic dataset (shared generation for reproducibility)
    vocab = sp.embed.num_embeddings
    # Ensure divisibility by world (pad later if needed)
    base_seq = args.seq_len
    if base_seq % world != 0:
        base_seq = (base_seq + world - 1) // world * world
        if rank == 0:
            print(f"[Adjust] seq_len -> {base_seq} for divisibility")
    data = build_synthetic_dataset(tokenizer, vocab, base_seq, dataset_size=args.steps * args.batch_size * 2)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype=='fp16'))
    autocast_ctx = torch.cuda.amp.autocast if args.dtype in ('fp16','bf16') else nullcontext

    global_step = 0
    start_time = time.time()
    for batch in data_loader(data, args.batch_size, device):
        if global_step >= args.steps:
            break
        # Add prompt embeddings if present
        if prompt is not None:
            with torch.no_grad():
                emb, extended_ids = prompt(batch, sp.embed)
            # Build full ids with -1 for virtual tokens; logits only need real token loss
            input_ids_full = extended_ids  # shape (B, V+S)
        else:
            emb = sp.embed(batch)
            input_ids_full = batch
        # Forward (model reconstructs embeddings internally – for prompt we bypass by manual embed injection simplifies?)
        # Simpler: temporarily monkeypatch embed if prompt used
        if prompt is not None:
            orig_embed = sp.embed
            class TempEmbed(nn.Module):
                def __init__(self, tensor):
                    super().__init__(); self.tensor = tensor
                def forward(self, ids):
                    # ignore ids, return precomputed embeddings (B,S,H)
                    return self.tensor
            sp.embed = TempEmbed(emb)
        with autocast_ctx(dtype=emb.dtype if isinstance(emb, torch.Tensor) else dtype):
            logits = sp(input_ids_full)
            # Shift for next-token prediction (ignore virtual tokens in loss)
            if prompt is not None and prompt.prompt_embed is not None:
                vt = prompt.virtual_tokens
            else:
                vt = 0
            # logits: (B, V+S, vocab)
            target = input_ids_full.clone()
            # mask virtual positions
            if vt > 0:
                target[:, :vt] = -100
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                   target[:, 1:].reshape(-1), ignore_index=-100)
        if prompt is not None:
            sp.embed = orig_embed
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        # Log
        if dist.is_initialized() and world > 1:
            loss_tensor = loss.detach().clone()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_avg = (loss_tensor / world).item()
        else:
            loss_avg = loss.item()
        if rank == 0 and (global_step % 1 == 0):
            lr = optimizer.param_groups[0]['lr']
            print(f"step={global_step} loss={loss_avg:.4f} lr={lr:.2e}")
        global_step += 1

    if rank == 0:
        elapsed = time.time() - start_time
        print(f"[Done] steps={global_step} time={elapsed:.2f}s steps/s={global_step/elapsed:.2f}")

if __name__ == '__main__':
    from contextlib import nullcontext
    main()
