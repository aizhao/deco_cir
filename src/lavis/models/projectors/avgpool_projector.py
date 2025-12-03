"""
AvgPoolProjector - Adapted from DeCo (Decoupling token compression from semantic abstraction)
Reference: https://arxiv.org/abs/2405.20985

This projector uses 2D Adaptive Average Pooling for parameter-free patch-level compression,
avoiding the "double abstraction" problem of Q-former while preserving fine-grained visual semantics.
"""

import torch
import torch.nn as nn
from einops import rearrange


class AvgPoolProjector(nn.Module):
    """
    Average Pooling Projector for vision-language models.
    
    Compresses visual tokens using 2D adaptive average pooling followed by MLP projection.
    This approach decouples compression from semantic abstraction, allowing the LLM to handle
    semantic processing while the projector only performs spatial downsampling.
    
    Args:
        layer_num (int): Number of MLP layers. Default: 2
        query_num (int): Number of output query tokens (must be a perfect square). Default: 144
        mm_hidden_size (int): Input hidden size from vision encoder. Default: 1024
        llm_hidden_size (int): Output hidden size for LLM. Default: 4096
    """
    
    def __init__(
        self,
        layer_num: int = 2,
        query_num: int = 144,
        mm_hidden_size: int = 1024,
        llm_hidden_size: int = 4096,
    ):
        super().__init__()
        self.layer_num = layer_num
        self.query_num = query_num
        self.mm_hidden_size = mm_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.build_net()
        
    def build_net(self):
        """Build the pooling and MLP projection layers."""
        # Calculate spatial dimensions for pooling
        hw = int(self.query_num ** 0.5)
        assert hw * hw == self.query_num, f"query_num must be a perfect square, got {self.query_num}"
        
        # 2D Adaptive Average Pooling (parameter-free compression)
        self.sampler = nn.AdaptiveAvgPool2d((hw, hw))
        
        # Enhanced MLP projection with LayerNorm for stable training
        # LayerNorm is critical for aligning visual and text feature distributions
        modules = [
            nn.Linear(self.mm_hidden_size, self.llm_hidden_size),
            nn.LayerNorm(self.llm_hidden_size),
            nn.GELU()
        ]
        for _ in range(1, self.layer_num):
            modules.append(nn.Linear(self.llm_hidden_size, self.llm_hidden_size))
            modules.append(nn.LayerNorm(self.llm_hidden_size))
            modules.append(nn.GELU())
        self.mlp_projector = nn.Sequential(*modules)
        
    def forward(self, visual_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AvgPoolProjector.
        
        Args:
            visual_feat (torch.Tensor): Input visual features of shape [B, seq_len, hidden_dim]
                                       where seq_len may include CLS token
        
        Returns:
            torch.Tensor: Projected features of shape [B, query_num, llm_hidden_size]
        """
        batch_size, seq_len, h_dim = visual_feat.shape
        
        # Remove CLS token if present (seq_len = 257 = 256 + 1 CLS)
        if seq_len == 257:
            visual_feat = visual_feat[:, 1:, :]  # Remove first token (CLS)
            seq_len = 256
        
        # Calculate spatial dimensions (assuming square feature maps)
        hw = int(seq_len ** 0.5)
        assert hw * hw == seq_len, f"seq_len must be a perfect square after removing CLS, got {seq_len}"
        
        # Reshape to 2D: [B, seq_len, dim] -> [B, dim, H, W]
        shaped_visual_feat = rearrange(visual_feat, "b (h w) d -> b d h w", h=hw, w=hw)
        
        # Apply adaptive average pooling: [B, dim, H, W] -> [B, dim, h', w']
        pooled_visual_feat = self.sampler(shaped_visual_feat)
        
        # Reshape back to sequence: [B, dim, h', w'] -> [B, query_num, dim]
        reshaped_visual_feat = rearrange(pooled_visual_feat, "b d h w -> b (h w) d")
        
        # Apply MLP projection: [B, query_num, mm_hidden_size] -> [B, query_num, llm_hidden_size]
        output_feat = self.mlp_projector(reshaped_visual_feat)
        
        return output_feat
