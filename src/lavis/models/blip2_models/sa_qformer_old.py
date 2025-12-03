"""
SA-QFormer: Spatially-Aware Q-Former with DeCo-inspired grid queries and dynamic instruction injection
"""
import logging
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lavis.models.blip2_models.Qformer import BertConfig, BertModel, BertLMHeadModel
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip_models.blip_outputs import BlipOutput


class AdaptiveGridPooling(nn.Module):
    """Normalize variable-sized ViT patch embeddings to fixed spatial grid"""
    
    def __init__(self, output_size: Tuple[int, int] = (8, 8)):
        super().__init__()
        self.output_size = output_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, vit_features: Tensor, original_grid_size: Tuple[int, int]) -> Tensor:
        """
        Args:
            vit_features: (B, N, D) where N = H_vit * W_vit (may include CLS token)
            original_grid_size: Tuple (H_vit, W_vit) - can be None, will auto-infer
        Returns:
            grid_features: (B, H*W, D) where H, W = output_size
        """
        B, N, D = vit_features.shape
        
        # Check if CLS token is present (N = H*W + 1)
        # Try to infer if this is a square grid + 1
        N_without_cls = N - 1
        H_vit = W_vit = int(math.sqrt(N_without_cls))
        
        if H_vit * W_vit == N_without_cls:
            # CLS token present, remove it (first token)
            vit_features = vit_features[:, 1:, :]  # Remove CLS token
            N = N_without_cls
        else:
            # No CLS token, try direct square root
            H_vit = W_vit = int(math.sqrt(N))
            if H_vit * W_vit != N:
                raise ValueError(f"Cannot infer square grid from {N} patches")
        
        # Reshape to 2D spatial grid
        features_2d = vit_features.view(B, H_vit, W_vit, D)
        # Permute to (B, D, H, W) for pooling
        features_2d = features_2d.permute(0, 3, 1, 2)
        
        # Apply adaptive pooling
        pooled = self.adaptive_pool(features_2d)  # (B, D, H_out, W_out)
        
        # Permute back and flatten
        pooled = pooled.permute(0, 2, 3, 1)  # (B, H_out, W_out, D)
        H_out, W_out = self.output_size
        grid_features = pooled.reshape(B, H_out * W_out, D)
        
        return grid_features


class Learned2DPositionalEncoding(nn.Module):
    """Learned positional encoding for 2D grid"""
    
    def __init__(self, grid_size: Tuple[int, int], hidden_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        
        # Separate embeddings for row and column
        self.row_embed = nn.Embedding(grid_size[0], hidden_size // 2)
        self.col_embed = nn.Embedding(grid_size[1], hidden_size // 2)
        
    def forward(self, batch_size: int, device: torch.device) -> Tensor:
        """
        Returns:
            pos_encoding: (B, H*W, hidden_size)
        """
        H, W = self.grid_size
        
        # Create position indices
        row_indices = torch.arange(H, device=device).repeat_interleave(W)
        col_indices = torch.arange(W, device=device).repeat(H)
        
        # Get embeddings
        row_embeds = self.row_embed(row_indices)  # (H*W, hidden_size//2)
        col_embeds = self.col_embed(col_indices)  # (H*W, hidden_size//2)
        
        # Concatenate and expand for batch
        pos_encoding = torch.cat([row_embeds, col_embeds], dim=-1)  # (H*W, hidden_size)
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pos_encoding


class Sinusoidal2DPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for 2D grid"""
    
    def __init__(self, grid_size: Tuple[int, int], hidden_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        
    def forward(self, batch_size: int, device: torch.device) -> Tensor:
        """
        Returns:
            pos_encoding: (B, H*W, hidden_size)
        """
        H, W = self.grid_size
        d_model = self.hidden_size
        
        # Create position indices
        pos_h = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).flatten()
        pos_w = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).flatten()
        
        # Compute sinusoidal encodings
        pe = torch.zeros(H * W, d_model, device=device)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Encode height (first half of dimensions)
        pe[:, 0:d_model//2:2] = torch.sin(pos_h.unsqueeze(1) * div_term[:d_model//4])
        pe[:, 1:d_model//2:2] = torch.cos(pos_h.unsqueeze(1) * div_term[:d_model//4])
        
        # Encode width (second half of dimensions)
        pe[:, d_model//2::2] = torch.sin(pos_w.unsqueeze(1) * div_term[:d_model//4])
        pe[:, d_model//2+1::2] = torch.cos(pos_w.unsqueeze(1) * div_term[:d_model//4])
        
        # Expand for batch
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pe


class GridQueryInitializer(nn.Module):
    """Create structured queries with 2D positional encodings"""
    
    def __init__(self, grid_size: Tuple[int, int] = (8, 8), 
                 hidden_size: int = 768, 
                 pos_encoding_type: str = 'learned'):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.num_queries = grid_size[0] * grid_size[1]
        
        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.zeros(1, self.num_queries, hidden_size))
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)
        
        # Positional encoding
        if pos_encoding_type == 'learned':
            self.pos_encoding = Learned2DPositionalEncoding(grid_size, hidden_size)
        elif pos_encoding_type == 'sinusoidal':
            self.pos_encoding = Sinusoidal2DPositionalEncoding(grid_size, hidden_size)
        else:
            raise ValueError(f"Invalid pos_encoding_type: {pos_encoding_type}. Must be 'learned' or 'sinusoidal'")
    
    def forward(self, batch_size: int, device: torch.device) -> Tensor:
        """
        Returns:
            grid_queries: (B, H*W, hidden_size) with positional encodings added
        """
        # Expand queries for batch
        queries = self.query_embed.expand(batch_size, -1, -1).to(device)
        
        # Add positional encodings
        pos_enc = self.pos_encoding(batch_size, device)
        grid_queries = queries + pos_enc
        
        return grid_queries


class LocalConstrainedCrossAttention(nn.Module):
    """Cross-attention with local spatial constraints"""
    
    def __init__(self, config: BertConfig, grid_size: Tuple[int, int] = (8, 8), 
                 neighborhood_radius: int = 1):
        super().__init__()
        self.grid_size = grid_size
        self.neighborhood_radius = neighborhood_radius
        
        # Standard attention components
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.encoder_width, self.all_head_size)
        self.value = nn.Linear(config.encoder_width, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Precompute and register local attention mask
        self.register_buffer('local_mask', self._create_local_mask())
    
    def _create_local_mask(self) -> Tensor:
        """Create local attention mask for grid queries"""
        H, W = self.grid_size
        r = self.neighborhood_radius
        num_queries = H * W
        
        # Create mask: True indicates valid attention
        mask = torch.zeros(num_queries, num_queries, dtype=torch.bool)
        
        for q_idx in range(num_queries):
            q_row, q_col = q_idx // W, q_idx % W
            
            for kv_idx in range(num_queries):
                kv_row, kv_col = kv_idx // W, kv_idx % W
                
                # Check if within neighborhood
                if abs(q_row - kv_row) <= r and abs(q_col - kv_col) <= r:
                    mask[q_idx, kv_idx] = True
        
        return mask
    
    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query_embeds: Tensor, key_value_embeds: Tensor, 
                attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            query_embeds: (B, H*W, D) grid queries
            key_value_embeds: (B, H*W, D) grid features from pooling
            attention_mask: Optional additional mask
        Returns:
            attended_features: (B, H*W, D)
        """
        batch_size = query_embeds.size(0)
        
        # Compute Q, K, V
        query_layer = self.transpose_for_scores(self.query(query_embeds))
        key_layer = self.transpose_for_scores(self.key(key_value_embeds))
        value_layer = self.transpose_for_scores(self.value(key_value_embeds))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply local constraint mask
        local_mask = self.local_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H*W, H*W)
        attention_scores = attention_scores.masked_fill(~local_mask, float('-inf'))
        
        # Apply additional mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        
        return context_layer


class InstructionInjectionModule(nn.Module):
    """Enable text instructions to modulate query representations via self-attention"""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        # Use a single BERT layer for self-attention
        from lavis.models.blip2_models.Qformer import BertLayer
        self.self_attn_layer = BertLayer(config, layer_num=0)
    
    def forward(self, grid_queries: Tensor, text_embeds: Tensor, 
                text_attention_mask: Tensor) -> Tensor:
        """
        Args:
            grid_queries: (B, H*W, D) initial grid queries
            text_embeds: (B, L, D) text instruction embeddings
            text_attention_mask: (B, L) mask for text
        Returns:
            modulated_queries: (B, H*W, D) instruction-aware queries
        """
        batch_size = grid_queries.size(0)
        num_queries = grid_queries.size(1)
        
        # Concatenate queries and text
        combined = torch.cat([grid_queries, text_embeds], dim=1)  # (B, H*W+L, D)
        
        # Create attention mask for combined sequence
        query_mask = torch.ones(batch_size, num_queries, dtype=torch.long, 
                               device=grid_queries.device)
        combined_mask = torch.cat([query_mask, text_attention_mask], dim=1)
        
        # Extend mask for attention
        extended_mask = combined_mask.unsqueeze(1).unsqueeze(2)
        extended_mask = (1.0 - extended_mask) * -10000.0
        
        # Apply self-attention
        layer_output = self.self_attn_layer(
            combined,
            attention_mask=extended_mask,
            query_length=num_queries
        )
        
        # Extract modulated queries (first H*W tokens)
        modulated_queries = layer_output[0][:, :num_queries, :]
        
        return modulated_queries


@registry.register_model("sa_qformer")
class SAQFormer(Blip2Base):
    """
    SA-QFormer: Spatially-Aware Q-Former with grid-based queries and instruction injection
    """
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/sa_qformer_pretrain.yaml",
    }
    
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        grid_size=(8, 8),
        neighborhood_radius=1,
        pos_encoding_type='learned',
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()
        
        # Validate configuration
        if grid_size[0] <= 0 or grid_size[1] <= 0:
            raise ValueError(f"Grid dimensions must be positive. Got: {grid_size}")
        if neighborhood_radius < 0:
            raise ValueError(f"Neighborhood radius must be non-negative. Got: {neighborhood_radius}")
        if pos_encoding_type not in ['learned', 'sinusoidal']:
            raise ValueError(f"pos_encoding_type must be 'learned' or 'sinusoidal'. Got: {pos_encoding_type}")
        
        self.grid_size = grid_size
        self.num_queries = grid_size[0] * grid_size[1]
        
        # Initialize tokenizer
        self.tokenizer = self.init_tokenizer()
        
        # Initialize vision encoder
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        
        # SA-QFormer components
        self.adaptive_pooling = AdaptiveGridPooling(output_size=grid_size)
        self.grid_query_init = GridQueryInitializer(
            grid_size=grid_size,
            hidden_size=768,  # BERT hidden size
            pos_encoding_type=pos_encoding_type
        )
        
        # Initialize Q-Former with modified config
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = self.visual_encoder.num_features
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = self.num_queries
        
        self.Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        
        # Instruction injection module
        self.instruction_injection = InstructionInjectionModule(encoder_config)
        
        # Local constrained cross-attention (replaces standard cross-attention in some layers)
        self.local_cross_attn = LocalConstrainedCrossAttention(
            encoder_config, 
            grid_size=grid_size,
            neighborhood_radius=neighborhood_radius
        )
        
        # Projection layers
        self.vision_proj = nn.Linear(encoder_config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(encoder_config.hidden_size, embed_dim)
        self.itm_head = nn.Linear(encoder_config.hidden_size, 2)
        
        # Temperature parameter
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        
        self.max_txt_len = max_txt_len
        
        logging.info(f"Initialized SA-QFormer with grid_size={grid_size}, "
                    f"num_queries={self.num_queries}, neighborhood_radius={neighborhood_radius}")
    
    def forward_image_with_spatial_awareness(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract spatially-aware image features using grid queries
        
        Args:
            image: (B, 3, H, W)
        Returns:
            query_output: (B, num_queries, D) spatially-aware features
            image_embeds: (B, N, D) original ViT features
        """
        # Get ViT features
        image_embeds = self.ln_vision(self.visual_encoder(image))
        
        # Apply adaptive pooling to normalize to fixed grid
        grid_features = self.adaptive_pooling(image_embeds, None)
        
        # Initialize grid queries with positional encoding
        batch_size = image.size(0)
        grid_queries = self.grid_query_init(batch_size, image.device)
        
        # Apply local constrained cross-attention
        # Note: For full integration, this should be integrated into Q-Former layers
        # Here we apply it as a preprocessing step
        spatially_aware_queries = self.local_cross_attn(grid_queries, grid_features)
        
        return spatially_aware_queries, image_embeds
    
    def forward(self, samples):
        """
        Forward pass with instruction injection and spatial awareness
        Compatible with CIRR/FashionIQ training that expects 'image', 'target', 'text_input'
        """
        # Handle both standard BLIP2 format and CIR format
        if "target" in samples:
            # CIR training format: reference image, target image, text
            reference_image = samples["image"]
            target_image = samples["target"]
            text = samples["text_input"]
            
            # Process reference image with spatial awareness
            ref_embeds = self.ln_vision(self.visual_encoder(reference_image))
            ref_grid_features = self.adaptive_pooling(ref_embeds, None)
            
            # Process target image
            target_embeds = self.ln_vision(self.visual_encoder(target_image))
            target_grid_features = self.adaptive_pooling(target_embeds, None)
            
            # Initialize grid queries
            batch_size = reference_image.size(0)
            grid_queries = self.grid_query_init(batch_size, reference_image.device)
            
            # Encode text
            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(reference_image.device)
            
            # Get text embeddings for instruction injection
            text_embeds = self.Qformer.bert.embeddings(text_tokens.input_ids)
            
            # Process reference image with grid queries (without instruction injection first)
            ref_atts = torch.ones(ref_grid_features.size()[:-1], dtype=torch.long).to(reference_image.device)
            query_atts = torch.ones(grid_queries.size()[:-1], dtype=torch.long).to(reference_image.device)
            
            # Fusion: reference image + text
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=grid_queries,
                attention_mask=attention_mask,
                encoder_hidden_states=ref_grid_features,
                encoder_attention_mask=ref_atts,
                return_dict=True,
            )
            
            # Second pass through Q-Former with fusion queries
            text_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=fusion_output.last_hidden_state[:, :self.num_queries, :],
                attention_mask=attention_mask,
                return_dict=True,
            )
            
            # Extract fusion features (use token at position num_queries)
            fusion_feats = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, self.num_queries, :]), dim=-1
            )
            
            # Process target image
            target_atts = torch.ones(target_grid_features.size()[:-1], dtype=torch.long).to(reference_image.device)
            target_query_output = self.Qformer.bert(
                query_embeds=grid_queries,
                encoder_hidden_states=target_grid_features,
                encoder_attention_mask=target_atts,
                use_cache=True,
                return_dict=True,
            )
            
            # Extract target features
            target_feats = F.normalize(
                self.vision_proj(target_query_output.last_hidden_state), dim=-1
            )
            
            # Compute Fusion-Target Contrastive Loss (loss_itc)
            sim_t2q = torch.matmul(
                fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
            ).squeeze()
            
            bs = reference_image.size(0)
            targets = torch.linspace(0, bs - 1, bs, dtype=int).to(reference_image.device)
            sim_i2t, _ = sim_t2q.max(-1)
            sim_i2t = sim_i2t / self.temp
            loss_itc = F.cross_entropy(sim_i2t, targets)
            
            # Compute Relative Contrastive Loss (loss_rtc)
            # Text-only features (without image fusion) - use fresh queries
            text_only_queries = self.grid_query_init(batch_size, reference_image.device)
            query_atts = torch.ones(text_only_queries.size()[:-1], dtype=torch.long).to(reference_image.device)
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            
            text_only_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=text_only_queries,
                attention_mask=attention_mask,
                return_dict=True,
            )
            text_only_feat = F.normalize(
                self.text_proj(text_only_output.last_hidden_state[:, 0, :]), dim=-1
            )
            
            sim_r2t = torch.matmul(
                text_only_feat.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
            ).squeeze()
            sim_r2t, _ = sim_r2t.max(-1)
            sim_r2t = sim_r2t / self.temp
            loss_rtc = F.cross_entropy(sim_r2t, targets)
            
            # Compute Alignment Loss (loss_align)
            # Align fusion queries with text-only queries
            loss_align = F.mse_loss(
                fusion_output.last_hidden_state[:, :self.num_queries, :].mean(1),
                text_only_queries.clone().detach().mean(1)
            )
            
            # Return dict format expected by training script
            return {
                'loss_itc': loss_itc,
                'loss_rtc': loss_rtc,
                'loss_align': loss_align
            }
        
        else:
            # Standard BLIP2 format
            image = samples["image"]
            text = samples["text_input"]
            
            # Get ViT features and apply grid pooling
            image_embeds = self.ln_vision(self.visual_encoder(image))
            grid_features = self.adaptive_pooling(image_embeds, None)
            
            # Initialize grid queries
            batch_size = image.size(0)
            grid_queries = self.grid_query_init(batch_size, image.device)
            
            # Encode text
            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            
            # Get text embeddings for instruction injection
            text_output = self.Qformer.bert.embeddings(text_tokens.input_ids)
            
            # Apply instruction injection
            instruction_aware_queries = self.instruction_injection(
                grid_queries, 
                text_output, 
                text_tokens.attention_mask
            )
            
            # Use instruction-aware queries in Q-Former
            image_atts = torch.ones(grid_features.size()[:-1], dtype=torch.long).to(image.device)
            
            query_output = self.Qformer.bert(
                query_embeds=instruction_aware_queries,
                encoder_hidden_states=grid_features,
                encoder_attention_mask=image_atts,
                use_cache=True,
                return_dict=True,
            )
            
            # Extract features
            image_feats = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )
            
            # Text features
            text_output_final = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            text_feat = F.normalize(
                self.text_proj(text_output_final.last_hidden_state[:, 0, :]), dim=-1
            )
            
            # Simple contrastive loss
            loss_itc = F.mse_loss(image_feats.mean(dim=1), text_feat)
            
            # Return dict format expected by training script
            return {'loss_itc': loss_itc}
    
    def forward_image(self, image):
        """Extract image features for retrieval"""
        image_embeds = self.ln_vision(self.visual_encoder(image))
        grid_features = self.adaptive_pooling(image_embeds, None)
        
        batch_size = image.size(0)
        grid_queries = self.grid_query_init(batch_size, image.device)
        
        image_atts = torch.ones(grid_features.size()[:-1], dtype=torch.long).to(image.device)
        
        query_output = self.Qformer.bert(
            query_embeds=grid_queries,
            encoder_hidden_states=grid_features,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds
    
    def forward_text(self, text_tokens):
        """Extract text features"""
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]
    
    def extract_features(self, samples, mode="multimodal"):
        """Extract features for evaluation"""
        from lavis.models.blip_models.blip_outputs import BlipOutputFeatures
        
        image = samples.get("image")
        caption = samples.get("text_input")
        
        assert mode in ["image", "text", "multimodal"], "mode must be one of 'image', 'text', 'multimodal'"
        
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None
        
        if mode == "image":
            assert image is not None, "Image is not provided for mode 'image'"
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            
            grid_features = self.adaptive_pooling(image_embeds_frozen, None)
            
            batch_size = image.size(0)
            grid_queries = self.grid_query_init(batch_size, self.device)
            
            image_atts = torch.ones(grid_features.size()[:-1], dtype=torch.long).to(self.device)
            
            query_output = self.Qformer.bert(
                query_embeds=grid_queries,
                encoder_hidden_states=grid_features,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        
        elif mode == "text":
            assert caption is not None, "text input is None for mode 'text'"
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)
            
            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)
        
        elif mode == "multimodal":
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            
            grid_features = self.adaptive_pooling(image_embeds_frozen, None)
            
            batch_size = image.size(0)
            grid_queries = self.grid_query_init(batch_size, self.device)
            query_atts = torch.ones(grid_queries.size()[:-1], dtype=torch.long).to(self.device)
            
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            
            image_atts = torch.ones(grid_features.size()[:-1], dtype=torch.long).to(self.device)
            
            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=grid_queries,
                attention_mask=attention_mask,
                encoder_hidden_states=grid_features,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            multimodal_embeds = output.last_hidden_state[:, : grid_queries.size(1), :]
        
        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )
    
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        grid_size = tuple(cfg.get("grid_size", [8, 8]))
        neighborhood_radius = cfg.get("neighborhood_radius", 1)
        pos_encoding_type = cfg.get("pos_encoding_type", "learned")
        
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        embed_dim = cfg.get("embed_dim", 256)
        max_txt_len = cfg.get("max_txt_len", 32)
        
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            grid_size=grid_size,
            neighborhood_radius=neighborhood_radius,
            pos_encoding_type=pos_encoding_type,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )
        
        model.load_checkpoint_from_config(cfg)
        
        return model
