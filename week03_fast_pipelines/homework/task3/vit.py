"""
The source is ViT from lucidrains implementation
https://github.com/lucidrains/vit-pytorch
"""

import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Быстрая аппроксимация GELU с предварительно вычисленными константами
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x)  # Более быстрая аппроксимация GELU
        x = self.dropout(x)
        x = self.linear2(x)
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = heads != 1 or dim_head != dim

        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head

        # Объединяем Q,K,V в одну проекцию для эффективности
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # LayerNorm после attention для лучшей производительности
        self.norm = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        
        # Единая QKV проекция с оптимизированным reshape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Оптимизированное вычисление attention scores с fused операцией
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Оптимизированный softmax с inplace операцией
        attn = F.softmax(dots, dim=-1, inplace=True)
        attn = self.dropout(attn)
        
        # Оптимизированное вычисление выхода с предварительным reshape
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        
        out = self.to_out(out)
        return self.norm(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, dropout=dropout),
                    ]
                )
            )
        # Флаг для включения gradient checkpointing
        self.use_checkpointing = True

    def _layer_forward(self, layer, x):
        attn, ff = layer
        x = x + attn(x)
        x = x + ff(x)
        return x

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                # Используем gradient checkpointing для экономии памяти
                x = checkpoint(self._layer_forward, layer, x, preserve_rng_state=True)
            else:
                x = self._layer_forward(layer, x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        depth,
        heads,
        dim=256,  # изменили на 256
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        # Эффективный патчинг через Conv2d
        self.to_patch_embedding = nn.Conv2d(
            in_channels=channels,
            out_channels=dim,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
            padding=0
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dropout=dropout)

        self.pool = pool
        # Убираем избыточный Identity слой
        
        # Заменяем BatchNorm на LayerNorm
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, img):
        # Conv2d patching: [B, C, H, W] -> [B, D, H', W']
        x = self.to_patch_embedding(img)
        # Reshape to sequence: [B, D, H', W'] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)
        b, n, _ = x.shape

        # Add CLS token
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding (inplace)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Transformer blocks
        x = self.transformer(x)

        # Pool and classify
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        return self.mlp_head(x)
