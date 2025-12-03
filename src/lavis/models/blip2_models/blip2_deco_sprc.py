"""
SPRC_DeCo: Spatial Grid Prompts for Composed Image Retrieval
Replaces Q-Former with DeCo module to preserve spatial details
Uses BERT Self-Attention for Late Fusion instead of Cross-Attention

Key Differences from Original SPRC:
1. DeCo Projector: Preserves spatial structure (8x8 grid) vs Q-Former's learnable queries
2. BERT Fusion: Self-attention on [Visual_Grid + Text] instead of Cross-attention
3. Position Embeddings: Explicit 2D spatial encoding for the grid
"""
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train


class SpatialGridProjector(nn.Module):
    """
    Spatial Grid Projector for DeCo-SPRC
    
    Compresses ViT patches into a fixed 8x8 grid using adaptive pooling + MLP.
    Critical: Uses MLP (Linear-LayerNorm-GELU-Linear) instead of simple Linear
    to handle the domain gap between vision and language features.
    
    Args:
        input_dim (int): Input dimension from ViT (e.g., 1408 for EVA-CLIP-G)
        output_dim (int): Output dimension for BERT (768)
        grid_size (int): Spatial grid size (8 for 8x8 = 64 tokens)
    """
    
    def __init__(self, input_dim=1408, output_dim=768, grid_size=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size
        
        # Step 1: Adaptive Average Pooling (parameter-free spatial compression)
        self.spatial_pooling = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        # Step 2: MLP Head (CRITICAL for performance!)
        # Follow DeCo original: Linear -> GELU -> Linear (no LayerNorm)
        modules = [nn.Linear(input_dim, output_dim)]
        for _ in range(1):  # One additional layer
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_dim, output_dim))
        self.mlp_head = nn.Sequential(*modules)
    
    def forward(self, visual_feat):
        """
        Forward pass: Compress ViT patches to spatial grid
        
        Args:
            visual_feat: [B, N_patches, Dim] where N_patches includes CLS token
        
        Returns:
            grid_tokens: [B, grid_size^2, output_dim]
        """
        B, N, D = visual_feat.shape
        
        # Step A: Remove CLS token (CRITICAL!)
        # EVA-CLIP-G outputs [B, 257, 1408] where first token is CLS
        if N == 257:
            visual_feat = visual_feat[:, 1:, :]  # Remove CLS -> [B, 256, 1408]
            N = 256
        
        # Step B: Calculate spatial dimensions (assuming square patches)
        H = W = int(N ** 0.5)
        assert H * W == N, f"N_patches must be perfect square, got {N}"
        
        # Step C: Reshape to 2D image [B, Dim, H, W]
        visual_2d = visual_feat.transpose(1, 2).reshape(B, D, H, W)
        
        # Step D: Adaptive pooling to grid_size x grid_size
        pooled = self.spatial_pooling(visual_2d)  # [B, Dim, grid_size, grid_size]
        
        # Step E: Flatten spatial dimensions
        grid_flat = pooled.flatten(2).transpose(1, 2)  # [B, grid_size^2, Dim]
        
        # Step F: MLP projection
        grid_tokens = self.mlp_head(grid_flat)  # [B, grid_size^2, output_dim]
        
        return grid_tokens


@registry.register_model("sprc_deco")
class SPRC_DeCo(Blip2Base):
    """
    SPRC with DeCo: Spatial Grid Prompts for Composed Image Retrieval
    
    Key Innovations:
    1. DeCo Projector: Preserves spatial structure (8x8 grid) instead of learnable queries
    2. Positional Embeddings: Helps BERT understand 2D spatial structure
    3. Late Fusion: BERT Self-Attention on [Spatial_Prompts + Text] instead of Cross-Attention
    
    Architecture:
    - ViT (EVA-CLIP-G) -> SpatialGridProjector (8x8) -> +PosEmbed -> BERT Fusion
    """
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_deco_sprc.yaml",
    }
    
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        grid_size=8,  # 8x8 = 64 spatial tokens
        embed_dim=256,  # Contrastive embedding dimension
        max_txt_len=32,
        cross_attention_freq=2,
    ):
        super().__init__()
        
        self.tokenizer = self.init_tokenizer()
        self.grid_size = grid_size
        self.num_spatial_tokens = grid_size * grid_size  # 64
        self.max_txt_len = max_txt_len
        
        # 1. Vision Encoder (EVA-CLIP-G)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("Frozen vision encoder")
        
        # 2. Spatial Grid Projector (DeCo)
        self.spatial_projector = SpatialGridProjector(
            input_dim=self.visual_encoder.num_features,  # 1408 for EVA-CLIP-G
            output_dim=768,  # BERT hidden size
            grid_size=grid_size
        )
        
        # 3. Fusion Encoder (BERT from Q-former)
        # Initialize Q-former to get BERT, then remove CLS head
        self.Qformer, _ = self.init_Qformer(
            num_query_token=self.num_spatial_tokens,
            vision_width=self.visual_encoder.num_features,
            cross_attention_freq=cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        
        # Remove CLS head (we don't need it for contrastive learning)
        self.Qformer.cls = None
        
        # 4. Positional Embeddings for Spatial Tokens (CRITICAL!)
        # Helps BERT understand the 2D structure of the 8x8 grid
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spatial_tokens, 768)
        )
        self.pos_embed.data.normal_(mean=0.0, std=0.02)
        
        # 5. Projection Heads for Contrastive Learning
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        
        # 6. Temperature Parameter
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        
        # 7. Prompt tokens for TIC loss (visual-only branch)
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, self.num_spatial_tokens, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
    
    def forward(self, samples):
        """
        Forward pass with BERT fusion and three losses: L_ITC, L_RTC, L_Align
        
        Args:
            samples: Dict with keys:
                - image: Reference images [B, 3, H, W]
                - target: Target images [B, 3, H, W]
                - text_input: Text descript
                ions [B]
        
        Returns:
            Dict of losses
        """
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]
        B = image.size(0)
        
        ###============== 1. Visual Extraction + DeCo Compression ===================###
        with self.maybe_autocast():
            ref_vit_feats = self.ln_vision(self.visual_encoder(image))  # [B, 257, 1408]
            tgt_vit_feats = self.ln_vision(self.visual_encoder(target))  # [B, 257, 1408]
        
        ref_vit_feats = ref_vit_feats.float()
        tgt_vit_feats = tgt_vit_feats.float()
        
        # DeCo compression: preserve spatial structure
        ref_spatial_prompts = self.spatial_projector(ref_vit_feats)  # [B, 64, 768]
        tgt_spatial_prompts = self.spatial_projector(tgt_vit_feats)  # [B, 64, 768]
        
        # Add position embeddings (CRITICAL for spatial understanding!)
        ref_spatial_prompts = ref_spatial_prompts + self.pos_embed  # [B, 64, 768]
        tgt_spatial_prompts = tgt_spatial_prompts + self.pos_embed  # [B, 64, 768]
        
        ###============== 2. Text Tokenization ===================###
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        
        ###============== 3. Fusion Branch: Ref + Text (BERT Self-Attention) ===================###
        # Attention masks
        spatial_atts = torch.ones(ref_spatial_prompts.size()[:-1], dtype=torch.long).to(image.device)
        attention_mask = torch.cat([spatial_atts, text_tokens.attention_mask], dim=1)
        
        # BERT Self-Attention Fusion: [Spatial_Prompts(64) + Text_Tokens(32)]
        # This is the KEY difference from original SPRC!
        fusion_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds=ref_spatial_prompts,  # Use DeCo spatial prompts as query
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Second pass: refine with self-attention (like original SPRC)
        text_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, :self.num_spatial_tokens, :],
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Extract fusion features
        # Global feature: use text CLS position (index = num_spatial_tokens)
        fusion_query = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, self.num_spatial_tokens, :]),
            dim=-1
        )  # [B, embed_dim]
        
        # Dense features: use spatial tokens from first fusion output
        fusion_patches = F.normalize(
            self.vision_proj(fusion_output.last_hidden_state[:, :self.num_spatial_tokens, :]),
            dim=-1
        )  # [B, 64, embed_dim]
        
        ###============== 4. Target Branch: Target Image Only ===================###
        # Process target through BERT (self-attention only, no text)
        target_atts = torch.ones(tgt_spatial_prompts.size()[:-1], dtype=torch.long).to(image.device)
        
        target_output = self.Qformer.bert(
            query_embeds=tgt_spatial_prompts,
            attention_mask=target_atts,
            return_dict=True,
        )
        
        # Extract target features
        target_patches = F.normalize(
            self.vision_proj(target_output.last_hidden_state),
            dim=-1
        )  # [B, 64, embed_dim]
        
        ###============== 5. TIC Branch: Ref + Empty Text (Relative Contrastive) ===================###
        # Use learnable prompt tokens instead of spatial prompts
        prompt_tokens = self.prompt_tokens.expand(B, -1, -1)
        prompt_atts = torch.ones(prompt_tokens.size()[:-1], dtype=torch.long).to(image.device)
        
        # Create attention mask for prompt + text
        text_only_attention_mask = torch.cat([prompt_atts, text_tokens.attention_mask], dim=1)
        
        # BERT with text only (no visual information)
        text_only_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds=prompt_tokens,
            attention_mask=text_only_attention_mask,
            return_dict=True,
        )
        
        # Extract text-only feature (use first prompt token like original SPRC)
        text_only_query = F.normalize(
            self.text_proj(text_only_output.last_hidden_state[:, 0, :]),
            dim=-1
        )  # [B, embed_dim]
        
        ###============== 6. Three Losses (Following Original SPRC) ===================###
        targets_idx = torch.arange(B, dtype=torch.long).to(image.device)
        

        
        # Loss 1: L_ITC (Fusion-Target Contrastive)
        # Compute [B, B] similarity matrix for contrastive learning
        # fusion_query: [B, D], target_patches: [B, 64, D]
        
        # Correct computation: need [B, B, 64] similarity matrix
        # fusion_query: [B, D], target_patches: [B, 64, D]
        # Use einsum for clarity: batch_i query @ batch_j patches
        sim_all = torch.einsum('bd,bjd->bj', fusion_query, target_patches.reshape(B, -1, 256))
        # Actually we need max over patches, so:
        # For each query i, compute similarity with all target patches j
        sim_all = torch.einsum('id,jkd->ijk', fusion_query, target_patches)  # [B, B, 64]
        
        # Max over patches dimension
        sim_i2t, _ = sim_all.max(-1)  # [B, B]
        sim_i2t = sim_i2t / self.temp
        
        loss_itc = F.cross_entropy(sim_i2t, targets_idx)
        
        # Loss 2: L_RTC (Relative/Text-only Contrastive)
        sim_all_rtc = torch.einsum('id,jkd->ijk', text_only_query, target_patches)  # [B, B, 64]
        
        sim_r2t, _ = sim_all_rtc.max(-1)  # [B, B]
        sim_r2t = sim_r2t / self.temp
        loss_rtc = F.cross_entropy(sim_r2t, targets_idx)
        
        # Loss 3: L_Align (Alignment between fusion and text-only)
        # Force fusion spatial features to align with prompt tokens
        # This ensures the model learns meaningful spatial representations
        loss_align = F.mse_loss(
            fusion_output.last_hidden_state[:, :self.num_spatial_tokens, :].mean(1),
            prompt_tokens.clone().detach().mean(1)
        )
        

        
        return {
            'loss_itc': loss_itc,
            'loss_rtc': loss_rtc,
            'loss_align': loss_align
        }
    
    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        """
        Extract target features for retrieval (MUST match training!)
        
        Args:
            image: Target images [B, 3, H, W]
            mode: Kept for compatibility
        
        Returns:
            Tuple of (patch_features, vit_features)
        """
        with self.maybe_autocast():
            vit_feats = self.ln_vision(self.visual_encoder(image))  # [B, 257, 1408]
        vit_feats = vit_feats.float()
        
        # DeCo compression + position embeddings
        spatial_prompts = self.spatial_projector(vit_feats)  # [B, 64, 768]
        spatial_prompts = spatial_prompts + self.pos_embed
        
        # CRITICAL: Process through BERT to match training feature space!
        target_atts = torch.ones(spatial_prompts.size()[:-1], dtype=torch.long).to(image.device)
        
        target_output = self.Qformer.bert(
            query_embeds=spatial_prompts,
            attention_mask=target_atts,
            return_dict=True,
        )
        
        # Project and normalize
        target_patches = F.normalize(
            self.vision_proj(target_output.last_hidden_state),
            dim=-1
        )  # [B, 64, embed_dim]
        
        return target_patches, vit_feats
    
    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text):
        """
        Inference for retrieval using BERT fusion
        
        Args:
            reference_embeds: Reference ViT features [B, 257, 1408]
            target_feats: Pre-computed target patch features [N, 64, embed_dim]
            text: Text descriptions [B]
        
        Returns:
            Similarity scores [B, N]
        """
        B = reference_embeds.size(0)
        
        # DeCo compression + position embeddings
        ref_spatial_prompts = self.spatial_projector(reference_embeds)  # [B, 64, 768]
        ref_spatial_prompts = ref_spatial_prompts + self.pos_embed
        
        # Tokenize text
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)
        
        # BERT Fusion (same as training)
        spatial_atts = torch.ones(ref_spatial_prompts.size()[:-1], dtype=torch.long).to(reference_embeds.device)
        attention_mask = torch.cat([spatial_atts, text_tokens.attention_mask], dim=1)
        
        fusion_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds=ref_spatial_prompts,
             return_dict=True,
        )
        
        text_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, :self.num_spatial_tokens, :],
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Extract fusion query (global feature)
        fusion_query = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, self.num_spatial_tokens, :]),
            dim=-1
        )  # [B, embed_dim]
        
        # Compute similarity with all targets
        # fusion_query: [B, embed_dim], target_feats: [N, 64, embed_dim]
        sim_matrix = torch.matmul(
            fusion_query.unsqueeze(1),  # [B, 1, D]
            target_feats.permute(0, 2, 1)  # [N, D, 64]
        )  # [B, N, 64]
        
        # Max similarity across target patches
        sim_scores, _ = sim_matrix.max(-1)  # [B, N]
        
        return sim_scores
    
    @classmethod
    def from_config(cls, cfg):
        """Create model from config"""
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        grid_size = cfg.get("grid_size", 8)
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        
        max_txt_len = cfg.get("max_txt_len", 32)
        embed_dim = cfg.get("embed_dim", 256)
        
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            grid_size=grid_size,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
            cross_attention_freq=cross_attention_freq,
        )
        model.load_checkpoint_from_config(cfg)
        
        return model
