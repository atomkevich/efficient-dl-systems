"""
Task 2 (extended bonus): Tensor-Parallel Llama with KV Cache

Adds past_key_values (KV caching) support to previously implemented manual tensor-parallel
Llama layers. Each rank shards attention heads (column-wise q/k/v, row-wise o) and maintains
its own slice of key/value cache for the heads it owns. During incremental generation, we only
process the new token and concatenate it to the per-rank cache – no cross-rank communication is
required for K/V because heads are disjoint. We still all_reduce the row-parallel output (wo).

This script can run either with a real HF Llama model (default) or a tiny dummy config via --dummy
for quick correctness & speed testing on CPU or a single GPU.

Usage examples:

Single GPU dummy quick test:
  python tensor_parallel_llama.py --dummy --prompt "Hello world" --max-new-tokens 8 --use-cache

Multi-GPU (tensor parallel) real model (example for 2 GPUs):
  torchrun --nproc_per_node 2 tensor_parallel_llama.py \
      --model-name meta-llama/Llama-3.2-1B \
      --prompt "The quick brown fox" \
      --max-new-tokens 32 --dtype bf16 --use-cache --backend nccl

Compare generation w/ and w/o cache timing:
  torchrun --nproc_per_node 2 tensor_parallel_llama.py --dummy --prompt "Once upon a time" \
      --max-new-tokens 32 --time-compare

Outputs (rank 0) will print two generations: (1) without cache (recompute) and (2) with cache.
They should be identical (deterministic greedy) while the cached version is faster per-token.
"""
import os, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

try:
    from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
except Exception:
    AutoTokenizer = None
    LlamaForCausalLM = None
    LlamaConfig = None

# ----------------- Utility linear shards -----------------
class ColumnLinearShard(nn.Module):
    """Shards output dimension across ranks (column parallel). Each rank stores weight[out_slice, in]."""
    def __init__(self, in_features: int, out_features_total: int, rank: int, world_size: int, bias: bool=False):
        super().__init__()
        assert out_features_total % world_size == 0
        self.out_per_rank = out_features_total // world_size
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_per_rank)) if bias else None
        self.rank = rank
        self.world_size = world_size
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    def forward(self, x):  # x: (..., in)
        out = F.linear(x, self.weight, self.bias)  # (..., out_per_rank)
        return out

class RowLinearShard(nn.Module):
    """Shards input rows across ranks (row parallel). Each rank stores weight[out, in_slice].
    We linearly project and then all_reduce SUM to assemble full output."""
    def __init__(self, in_features_total: int, out_features: int, rank: int, world_size: int, bias: bool=False):
        super().__init__()
        assert in_features_total % world_size == 0
        self.in_per_rank = in_features_total // world_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.rank = rank
        self.world_size = world_size
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    def forward(self, x):  # x: (..., in_per_rank)
        partial = F.linear(x, self.weight, None)  # (..., out)
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        if self.bias is not None:
            partial = partial + self.bias
        return partial

# ----------------- Attention with per-rank KV cache -----------------
class TPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, rank: int, world_size: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads_total = num_heads
        self.world_size = world_size
        self.rank = rank
        self.local_heads = num_heads // world_size
        self.head_dim = hidden_size // num_heads
        local_hidden = self.local_heads * self.head_dim
        # Column shards: each rank owns subset of heads => subset of q/k/v output dim
        self.wq = ColumnLinearShard(hidden_size, hidden_size, rank, world_size, bias=False)
        self.wk = ColumnLinearShard(hidden_size, hidden_size, rank, world_size, bias=False)
        self.wv = ColumnLinearShard(hidden_size, hidden_size, rank, world_size, bias=False)
        # Row shard for output: each rank holds slice of input dimension => sums
        self.wo = RowLinearShard(hidden_size, hidden_size, rank, world_size, bias=False)
        self.scale = self.head_dim ** -0.5

    def _shape_heads(self, x):  # x: (B,S,local_hidden)
        B,S,_ = x.shape
        return x.view(B, S, self.local_heads, self.head_dim).transpose(1,2)  # (B,local_heads,S,D)

    def forward(self, x, past_kv=None, use_cache=False):
        """x: (B,S,H). past_kv: (k_cache, v_cache) each (B, local_heads, T_past, D)
        Returns: attn_out (B,S,H), present_kv (if use_cache)
        """
        B,S,H = x.shape
        q = self._shape_heads(self.wq(x))  # (B, lh, S, D)
        k = self._shape_heads(self.wk(x))
        v = self._shape_heads(self.wv(x))
        if past_kv is not None:
            past_k, past_v = past_kv
            # Concatenate along sequence dimension
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        attn_scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale  # (B, lh, S, T_total)
        attn_weights = attn_scores.softmax(dim=-1)
        attn_ctx = torch.matmul(attn_weights, v)  # (B, lh, S, D)
        # Merge heads
        attn_ctx = attn_ctx.transpose(1,2).contiguous().view(B,S,self.local_heads*self.head_dim)
        # Row-sharded output projection (collective all_reduce inside)
        out = self.wo(attn_ctx)  # (B,S,H)
        present = (k, v) if use_cache else None
        return out, present

# ----------------- MLP (gated) -----------------
class TPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, rank: int, world_size: int):
        super().__init__()
        self.gate = ColumnLinearShard(hidden_size, intermediate_size, rank, world_size, bias=False)
        self.up = ColumnLinearShard(hidden_size, intermediate_size, rank, world_size, bias=False)
        self.down = RowLinearShard(intermediate_size, hidden_size, rank, world_size, bias=False)
    def forward(self, x):
        g = F.silu(self.gate(x))
        u = self.up(x)
        h = g * u
        return self.down(h)

# ----------------- RMSNorm -----------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight).to(x.dtype)

# ----------------- TP Block -----------------
class TPBlock(nn.Module):
    def __init__(self, ref_block, rank: int, world_size: int):
        super().__init__()
        hs = ref_block.self_attn.q_proj.in_features
        self.attn = TPAttention(hs, ref_block.self_attn.num_heads, rank, world_size)
        self.mlp = TPMLP(hs, ref_block.mlp.gate_proj.out_features, rank, world_size)
        self.rms1 = RMSNorm(hs)
        self.rms2 = RMSNorm(hs)
    @torch.no_grad()
    def load_from_ref(self, ref_block, rank: int, world_size: int):
        # Attention q/k/v column shard
        for proj_ref, proj_tp in [(ref_block.self_attn.q_proj, self.attn.wq),
                                  (ref_block.self_attn.k_proj, self.attn.wk),
                                  (ref_block.self_attn.v_proj, self.attn.wv)]:
            shard_size = proj_tp.weight.shape[0]
            start = rank * shard_size
            end = (rank+1)*shard_size
            proj_tp.weight.copy_(proj_ref.weight[start:end])
        # Out proj row shard
        shard_in = self.attn.local_heads * self.attn.head_dim
        start_in = rank * shard_in
        end_in = (rank+1) * shard_in
        self.attn.wo.weight.copy_(ref_block.self_attn.o_proj.weight[:, start_in:end_in])
        # MLP shards
        for ref_lin, tp_lin in [(ref_block.mlp.gate_proj, self.mlp.gate),
                                (ref_block.mlp.up_proj, self.mlp.up)]:
            shard_size = tp_lin.weight.shape[0]
            tp_lin.weight.copy_(ref_lin.weight[rank*shard_size:(rank+1)*shard_size])
        start_in = rank * self.mlp.down.in_per_rank
        end_in = (rank+1) * self.mlp.down.in_per_rank
        self.mlp.down.weight.copy_(ref_block.mlp.down_proj.weight[:, start_in:end_in])
        # Norms
        self.rms1.weight.copy_(ref_block.input_layernorm.weight)
        self.rms2.weight.copy_(ref_block.post_attention_layernorm.weight)
    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present = self.attn(self.rms1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.rms2(x))
        return x, present

# ----------------- Full TP Llama -----------------
class TensorParallelLlama(nn.Module):
    def __init__(self, ref_model: LlamaForCausalLM, rank: int, world_size: int):
        super().__init__()
        self.config = ref_model.config
        self.embed = nn.Embedding(ref_model.model.embed_tokens.num_embeddings, ref_model.model.embed_tokens.embedding_dim)
        self.layers = nn.ModuleList([TPBlock(ref_layer, rank, world_size) for ref_layer in ref_model.model.layers])
        self.final_norm = RMSNorm(ref_model.model.norm.normalized_shape[0])
        self.lm_head = nn.Linear(ref_model.lm_head.in_features, ref_model.lm_head.out_features, bias=False)
        self.rank = rank
        self.world_size = world_size
    @torch.no_grad()
    def load_from_ref(self, ref_model: LlamaForCausalLM, rank: int, world_size: int):
        self.embed.weight.copy_(ref_model.model.embed_tokens.weight)
        for tp_block, ref_block in zip(self.layers, ref_model.model.layers):
            tp_block.load_from_ref(ref_block, rank, world_size)
        self.final_norm.weight.copy_(ref_model.model.norm.weight)
        self.lm_head.weight.copy_(ref_model.lm_head.weight)
    def forward(self, input_ids, past_key_values=None, use_cache=False):
        # past_key_values: list[ (k,v) per layer ] each k,v: (B, local_heads, T_past, D)
        x = self.embed(input_ids)  # (B,S,H)
        presents = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if (past_key_values is not None) else None
            x, present = layer(x, past_kv=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, presents

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=32, use_cache=True, temperature=0.0):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        B, S = input_ids.shape
        past = None
        # First forward pass on prompt
        logits, past = self(input_ids, past_key_values=None, use_cache=use_cache)
        generated = [input_ids]
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]  # last token
            if temperature > 0:
                probs = (next_logits / temperature).softmax(-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(-1, keepdim=True)
            generated.append(next_token)
            if use_cache:
                logits, past = self(next_token, past_key_values=past, use_cache=True)
            else:
                # Recompute full context (inefficient) for parity check
                full_ctx = torch.cat(generated, dim=1)
                logits, _ = self(full_ctx, past_key_values=None, use_cache=False)
        return torch.cat(generated, dim=1)

# ----------------- Helper functions -----------------

def init_distributed(backend="nccl"):
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        # single-process fallback
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    else:
        dist.init_process_group(backend=backend)

@torch.no_grad()
def build_reference_model(args, device):
    if args.dummy:
        assert LlamaConfig is not None, "transformers not installed"
        config = LlamaConfig(
            hidden_size=512,
            intermediate_size=1536,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=32000,
        )
        ref = LlamaForCausalLM(config)
        tok = None
    else:
        assert LlamaForCausalLM is not None, "transformers not installed"
        ref = LlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 if args.dtype=="bf16" else torch.float32, device_map=None)
        tok = AutoTokenizer.from_pretrained(args.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    ref.to(device)
    return ref, tok

@torch.no_grad()
def tensor_parallelize(ref_model, rank, world_size, device):
    tp = TensorParallelLlama(ref_model, rank, world_size).to(device)
    tp.load_from_ref(ref_model, rank, world_size)
    return tp

# ----------------- Timing utility -----------------
@torch.no_grad()
def time_generation(model: TensorParallelLlama, input_ids, max_new_tokens, use_cache):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, use_cache=use_cache)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return out, time.time() - t0

# ----------------- Main -----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', default='meta-llama/Llama-3.2-1B')
    ap.add_argument('--prompt', default='Hello world')
    ap.add_argument('--max-new-tokens', type=int, default=16)
    ap.add_argument('--backend', default='nccl')
    ap.add_argument('--dtype', choices=['fp32','bf16'], default='fp32')
    ap.add_argument('--use-cache', action='store_true')
    ap.add_argument('--time-compare', action='store_true', help='Run both cached and recompute generation & compare time')
    ap.add_argument('--dummy', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    init_distributed(args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device('cuda', rank) if torch.cuda.is_available() else torch.device('cpu')

    ref_model, tokenizer = build_reference_model(args, device)
    if args.dtype == 'bf16' and torch.cuda.is_available():
        ref_model.to(torch.bfloat16)

    # Build TP model & free reference grads
    tp_model = tensor_parallelize(ref_model, rank, world_size, device)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    tp_model.eval()

    if tokenizer is None:
        # dummy tokenizer emulation: map chars to ord % vocab
        prompt_ids = torch.tensor([[ (ord(c) % tp_model.embed.num_embeddings) for c in args.prompt  ]], device=device)
    else:
        prompt_ids = tokenizer(args.prompt, return_tensors='pt').input_ids.to(device)

    # Broadcast prompt length for consistency (optional)
    if world_size > 1:
        dist.broadcast(prompt_ids, src=0)

    if args.time_compare:
        if rank == 0:
            print(f"[Timing] Generating {args.max_new_tokens} tokens (world_size={world_size})")
        out_cache, t_cache = time_generation(tp_model, prompt_ids, args.max_new_tokens, use_cache=True)
        out_recomp, t_recomp = time_generation(tp_model, prompt_ids, args.max_new_tokens, use_cache=False)
        if rank == 0:
            if tokenizer is not None:
                text_cache = tokenizer.decode(out_cache[0], skip_special_tokens=True)
                text_recomp = tokenizer.decode(out_recomp[0], skip_special_tokens=True)
            else:
                text_cache = ''.join(chr(int(i)%128) for i in out_cache[0])
            print("With cache time:  %.3f s" % t_cache)
            print("Recompute time:   %.3f s" % t_recomp)
            print("Speedup: x%.2f" % (t_recomp / max(t_cache,1e-6)))
            print("Texts identical:", bool(torch.equal(out_cache, out_recomp)))
            if tokenizer is not None:
                print("Sample output:\n", text_cache)
        return

    # Single path generation
    out = tp_model.generate(prompt_ids, max_new_tokens=args.max_new_tokens, use_cache=args.use_cache)
    if rank == 0:
        if tokenizer is not None:
            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            decoded = ''.join(chr(int(i)%128) for i in out[0])
        print("=== Generation (use_cache=%s) ===" % args.use_cache)
        print(decoded)

if __name__ == '__main__':
    main()
