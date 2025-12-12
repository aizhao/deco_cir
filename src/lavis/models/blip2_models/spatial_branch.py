"""
Spatial Branch with 2D Rotary Position Embedding (RoPE)
Inspired by Qwen-VL/Image architecture for spatial-aware visual feature processing.

Components:
    - VisionRotaryEmbedding2D: 2D RoPE implementation (Axial approach)
    - QwenStyleAttention: Attention with QK-Norm (Qwen characteristic)
    - QwenBlock: Transformer block with RoPE support
    - QwenSpatialAdapter: Complete adapter for integrating with SPRC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VisionRotaryEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding for Vision Transformers.
    
    Extends standard RoPE to 2D grid by splitting the embedding dimension
    into two halves: one for height (H) and one for width (W).
    
    Args:
        dim: The dimension of the rotary embedding (typically head_dim)
        theta: Base frequency for the sinusoidal functions (default: 10000.0)
    """
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Cache inverse frequencies to avoid redundant computation
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, h, w, device):
        """
        Generate 2D RoPE frequencies for a given spatial grid.
        
        Args:
            h: Height of the spatial grid
            w: Width of the spatial grid
            device: Device to create tensors on
            
        Returns:
            freqs: Tensor of shape [h*w, dim] containing the rotation frequencies
        """
        # 1. Generate position sequences for H and W axes
        seq_h = torch.arange(h, device=device, dtype=self.inv_freq.dtype)
        seq_w = torch.arange(w, device=device, dtype=self.inv_freq.dtype)
        
        # 2. Compute frequency outer products
        # Half of the dimension is allocated to H, half to W
        freqs_h = torch.outer(seq_h, self.inv_freq)  # [h, dim/2]
        freqs_w = torch.outer(seq_w, self.inv_freq)  # [w, dim/2]
        
        # 3. Broadcast and concatenate into 2D grid
        # [h, w, dim/2]
        emb_h = freqs_h.unsqueeze(1).repeat(1, w, 1)
        emb_w = freqs_w.unsqueeze(0).repeat(h, 1, 1)
        
        # Concatenate: [h, w, dim]
        # First dim/2 encodes height info, last dim/2 encodes width info
        freqs = torch.cat([emb_h, emb_w], dim=-1)
        
        # Flatten: [seq_len, dim]
        freqs = freqs.reshape(-1, self.dim)
        return freqs


def apply_rotary_pos_emb(x, freqs):
    """
    Apply rotary position embedding to the input tensor.
    
    Args:
        x: Input tensor of shape [B, Seq_Len, Num_Heads, Head_Dim]
        freqs: Frequency tensor of shape [Seq_Len, Head_Dim]
        
    Returns:
        Rotated tensor with the same shape as input
    """
    # Adjust freqs for broadcasting: [1, Seq_Len, 1, Head_Dim]
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    
    # Complex rotation: x_new = x * cos + rotate(x) * sin
    cos = freqs.cos().to(x.dtype)
    sin = freqs.sin().to(x.dtype)
    
    # rotate_half: [-x2, x1, -x4, x3, ...]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    
    return (x * cos) + (x_rotated * sin)


class QwenStyleAttention(nn.Module):
    """
    Qwen-style Multi-Head Attention with QK-Norm.
    
    Key features:
        - QK-Norm: LayerNorm applied to Q and K before attention computation
        - 2D-RoPE: Rotary position embedding applied to Q and K
    
    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        # Qwen characteristic: QK-Norm (RMSNorm or LayerNorm)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
    
    def forward(self, x, rope_freqs):
        """
        Forward pass with 2D-RoPE.
        
        Args:
            x: Input tensor of shape [B, N, C]
            rope_freqs: RoPE frequencies of shape [N, head_dim]
            
        Returns:
            Output tensor of shape [B, N, C]
        """
        B, N, C = x.shape
        
        # 1. Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, N, num_heads, head_dim]
        
        # 2. Qwen characteristic: Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # 3. Core: Apply 2D-RoPE (only to Q and K, not V)
        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)
        
        # 4. Attention calculation
        q = q.transpose(1, 2)  # [B, Heads, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class QwenBlock(nn.Module):
    """
    Transformer block with Qwen-style attention and 2D-RoPE support.
    
    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio for FFN hidden dimension expansion
        drop: Dropout rate
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = QwenStyleAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN (Feed Forward Network)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # Qwen typically uses SwiGLU, GELU is simpler
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )
    
    def forward(self, x, rope_freqs):
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor of shape [B, N, C]
            rope_freqs: RoPE frequencies of shape [N, head_dim]
            
        Returns:
            Output tensor of shape [B, N, C]
        """
        x = x + self.attn(self.norm1(x), rope_freqs)
        x = x + self.mlp(self.norm2(x))
        return x


class QwenSpatialAdapter(nn.Module):
    """
    Qwen-style Spatial Adapter for SPRC (Parallel Branch Version).
    
    This adapter outputs a spatial-aware DELTA (increment) that should be
    added to the original visual features. It does NOT include internal
    residual connection - the fusion happens in the main model.
    
    Architecture:
        1. Down-projecting from ViT dimension to adapter dimension
        2. Processing through transformer blocks with 2D-RoPE
        3. Up-projecting back to ViT dimension
        4. Zero-init gating for stable training
    
    The zero-init gate ensures that at training start, this branch outputs 0,
    preserving the original SPRC performance.
    
    Args:
        input_dim: Input feature dimension (e.g., 1408 for ViT-G)
        hidden_dim: Hidden dimension for the adapter
        num_heads: Number of attention heads
        depth: Number of transformer blocks
        mlp_ratio: Ratio for FFN hidden dimension expansion
        drop: Dropout rate
    """
    def __init__(
        self, 
        input_dim=1408, 
        hidden_dim=768, 
        num_heads=12, 
        depth=2, 
        mlp_ratio=4.0,
        drop=0.0
    ):
        super().__init__()
        
        # 1. Down projection: ViT-G (1408) -> Adapter (768)
        self.down_project = nn.Linear(input_dim, hidden_dim)
        
        # 2. RoPE initialization
        # head_dim = 768 / 12 = 64
        # RoPE dim should be head_dim (32 for H, 32 for W in 2D case)
        self.rope = VisionRotaryEmbedding2D(dim=hidden_dim // num_heads)
        
        # 3. Transformer layers
        self.blocks = nn.ModuleList([
            QwenBlock(hidden_dim, num_heads, mlp_ratio, drop) for _ in range(depth)
        ])
        
        # 4. Up projection back to original feature space
        self.up_project = nn.Linear(hidden_dim, input_dim)
        
        # 5. Zero-init gate
        # Ensures that at training start, this branch outputs 0,
        # not disturbing the original SPRC performance
        self.gate = nn.Parameter(torch.zeros(1))
        
        # Store config for reference
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.depth = depth
        
        # Initialize up_project to zero for stable training
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, patch_tokens, grid_size=None):
        """
        Forward pass through the spatial adapter.
        
        NOTE: This method expects PATCH TOKENS ONLY (no CLS token).
        The CLS token should be handled separately in the main model.
        
        Args:
            patch_tokens: Input tensor of shape [B, N, C] (Patch embeddings from ViT)
                N is typically 256 (16x16) for 224x224 images
            grid_size: Optional tuple (H, W) for the spatial grid.
                       If None, assumes square grid based on sequence length.
                       
        Returns:
            spatial_delta: Tensor of shape [B, N, C] - the spatial enhancement delta
                          to be ADDED to the original patch tokens
        """
        B, N, C = patch_tokens.shape
        
        # Infer grid size if not provided
        if grid_size is None:
            sqrt_n = int(math.sqrt(N))
            if sqrt_n * sqrt_n == N:
                grid_h, grid_w = sqrt_n, sqrt_n
            else:
                # Try to find factors for non-square grids
                for h in range(int(math.sqrt(N)), 0, -1):
                    if N % h == 0:
                        grid_h, grid_w = h, N // h
                        break
                else:
                    grid_h, grid_w = sqrt_n, sqrt_n
        else:
            grid_h, grid_w = grid_size
        
        # 1. Down projection
        x_feat = self.down_project(patch_tokens)
        
        # 2. Generate RoPE frequencies
        # Generated in forward to support dynamic resolution
        freqs = self.rope(grid_h, grid_w, patch_tokens.device)
        
        # 3. Pass through transformer blocks
        for block in self.blocks:
            x_feat = block(x_feat, freqs)
        
        # 4. Up projection to get delta
        spatial_delta = self.up_project(x_feat)
        
        # 5. Apply gate (zero-init ensures delta starts at 0)
        spatial_delta = spatial_delta * self.gate
        
        return spatial_delta
    
    def get_gate_value(self):
        """Return current gate value for monitoring during training."""
        return self.gate.item()


class SpatialBranch(nn.Module):
    """
    Alternative implementation: Parallel spatial branch that can be used
    alongside the original visual features.
    
    This creates a separate spatial-aware feature stream that is combined
    with the original features via learnable weighted sum.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for processing
        num_heads: Number of attention heads
        depth: Number of transformer blocks
        fusion_mode: How to combine with original features ('add', 'concat', 'gate')
    """
    def __init__(
        self,
        input_dim=1408,
        hidden_dim=768,
        num_heads=12,
        depth=2,
        fusion_mode='gate'
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        
        self.adapter = QwenSpatialAdapter(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=depth
        )
        
        if fusion_mode == 'concat':
            # Output dimension doubles when concatenating
            self.fusion_proj = nn.Linear(input_dim * 2, input_dim)
        elif fusion_mode == 'gate':
            # Learnable per-channel gate
            self.fusion_gate = nn.Parameter(torch.zeros(input_dim))
    
    def forward(self, x, grid_size=None):
        """
        Process input and combine with spatial features.
        
        Args:
            x: Input tensor of shape [B, N, C]
            grid_size: Optional tuple (H, W) for spatial grid
            
        Returns:
            Fused tensor of shape [B, N, C]
        """
        spatial_out = self.adapter(x, grid_size)
        
        if self.fusion_mode == 'add':
            return x + spatial_out
        elif self.fusion_mode == 'concat':
            combined = torch.cat([x, spatial_out], dim=-1)
            return self.fusion_proj(combined)
        elif self.fusion_mode == 'gate':
            gate = torch.sigmoid(self.fusion_gate)
            return x * (1 - gate) + spatial_out * gate
        else:
            return spatial_out

