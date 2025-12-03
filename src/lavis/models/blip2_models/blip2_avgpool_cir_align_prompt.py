"""
BLIP2 with DeCo Projector for Composed Image Retrieval
Uses DeCo for visual compression + Q-former BERT for fusion
"""
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.projectors.avgpool_projector import AvgPoolProjector


@registry.register_model("blip2_avgpool_cir_align_prompt")
class Blip2AvgPoolCirAlignPrompt(Blip2Base):
    """
    BLIP2 with DeCo for CIR.
    
    Architecture:
    1. DeCo (AvgPoolProjector): Visual compression (576 -> 64 tokens)
    2. Q-former BERT: Multimodal fusion via self-attention on concatenated visual+text tokens
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_avgpool_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_avgpool_pretrain.yaml",
        "coco": "configs/models/blip2/blip2_avgpool_pretrain.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=64,  # Must be perfect square (8x8)
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        projector_layers=2,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        # 1. Vision Encoder (unchanged)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # 2. DeCo Projector: Visual compression only
        self.deco_projector = AvgPoolProjector(
            layer_num=projector_layers,
            query_num=num_query_token,
            mm_hidden_size=self.visual_encoder.num_features,
            llm_hidden_size=768,  # BERT hidden size
        )
        
        self.num_query_token = num_query_token
        
        # 3. Fusion Encoder: Reuse Q-former's BERT for multimodal fusion
        # Initialize Q-former with correct num_query_token for proper initialization
        self.Qformer, _ = self.init_Qformer(
            num_query_token=num_query_token,  # Use actual number for proper init
            vision_width=self.visual_encoder.num_features,
            cross_attention_freq=cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        
        # Remove only the CLS head (we don't need it for contrastive learning)
        self.Qformer.cls = None
        
        # 4. Projection heads for contrastive learning
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        
        # 5. Temperature parameter
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        
        self.max_txt_len = max_txt_len
        
        # 6. Visual position embeddings for DeCo tokens (spatial structure)
        self.visual_pos_embed = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.visual_pos_embed.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
        
        # 7. Empty text tokens for TIC loss (visual-only branch)
        self.register_buffer("empty_text_ids", torch.zeros(1, max_txt_len, dtype=torch.long))

    def forward(self, samples):
        """
        Forward pass using DeCo + Q-former BERT fusion
        Args:
            samples: Dictionary containing:
                - image: Reference images [B, 3, H, W]
                - target: Target images [B, 3, H, W]
                - text_input: Text descriptions [B]
        Returns:
            Dictionary of losses
        """
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]
        
        ###============== A. Visual Encoding + DeCo Compression ===================###
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            target_embeds = self.ln_vision(self.visual_encoder(target))
        image_embeds = image_embeds.float()
        target_embeds = target_embeds.float()
        
        # DeCo compression: [B, 257, 1408] -> [B, 64, 768]
        ref_grid = self.deco_projector(image_embeds)  # [B, 64, 768]
        tgt_grid = self.deco_projector(target_embeds)  # [B, 64, 768]
        
        # Add spatial position embeddings (critical for preserving spatial structure!)
        ref_grid = ref_grid + self.visual_pos_embed
        tgt_grid = tgt_grid + self.visual_pos_embed
        
        ###============== B. Text Tokenization ===================###
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        
        ###============== C. Fusion Branch: Ref + Text (Cross-Attention like Q-former) ===================###
        # Use learnable query tokens (like original Q-former)
        query_tokens = torch.zeros(image.size(0), self.num_query_token, 768).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        
        # Attention mask for query + text
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        
        # Cross-attention: query tokens attend to compressed visual features
        ref_atts = torch.ones(ref_grid.size()[:-1], dtype=torch.long).to(image.device)
        fusion_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=ref_grid,  # Cross-attend to DeCo compressed features
            encoder_attention_mask=ref_atts,
            return_dict=True,
        )
        
        # Second pass: refine with text-only self-attention
        text_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, :self.num_query_token, :],
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Extract fusion feature from text CLS position (like original)
        fusion_query = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, self.num_query_token, :]),
            dim=-1
        )  # [B, embed_dim]
        
        # Keep dense features from first fusion output
        fusion_patches = F.normalize(
            self.vision_proj(fusion_output.last_hidden_state[:, :self.num_query_token, :]),
            dim=-1
        )  # [B, 64, embed_dim]
        
        ###============== D. Target Branch: Target Image Only (Cross-Attention) ===================###
        # Use same query tokens as fusion branch
        target_query_tokens = torch.zeros(target.size(0), self.num_query_token, 768).to(image.device)
        target_atts = torch.ones(tgt_grid.size()[:-1], dtype=torch.long).to(image.device)
        
        # Cross-attention: query tokens attend to target visual features
        target_output = self.Qformer.bert(
            query_embeds=target_query_tokens,
            encoder_hidden_states=tgt_grid,  # Cross-attend to DeCo compressed target
            encoder_attention_mask=target_atts,
            return_dict=True,
        )
        
        # Extract target features (use vision_proj like original)
        target_global = F.normalize(
            self.vision_proj(target_output.last_hidden_state.mean(dim=1)),
            dim=-1
        )  # [B, embed_dim]
        
        target_patches = F.normalize(
            self.vision_proj(target_output.last_hidden_state),
            dim=-1
        )  # [B, 64, embed_dim]
        
        ###============== E. TIC Branch: Ref + Empty Text (Visual-Only) ===================###
        # Create empty text tokens
        empty_text_ids = self.empty_text_ids.expand(image.size(0), -1).to(image.device)
        empty_text_mask = torch.zeros_like(empty_text_ids)
        empty_attention_mask = torch.cat([visual_atts, empty_text_mask], dim=1)
        
        # Visual-only fusion
        visual_only_output = self.Qformer.bert(
            input_ids=empty_text_ids,
            query_embeds=ref_grid,
            attention_mask=empty_attention_mask,
            return_dict=True,
        )
        
        visual_only_query = F.normalize(
            self.vision_proj(visual_only_output.last_hidden_state[:, :self.num_query_token, :].mean(dim=1)),
            dim=-1
        )  # [B, embed_dim]
        
        ###============== F. Three Losses ===================###
        bs = image.size(0)
        targets_idx = torch.arange(bs, dtype=torch.long).to(image.device)
        
        # Loss 1: Global ITC (Image-Text-Composition Retrieval)
        # fusion_query vs target_global
        sim_global = torch.matmul(fusion_query, target_global.T) / self.temp  # [B, B]
        loss_itc = F.cross_entropy(sim_global, targets_idx)
        
        # Loss 2: Dense Spatial Alignment (利用DeCo的空间结构优势)
        # Compute dense similarity between all fusion and target pairs in the batch
        # fusion_patches: [B, 64, D], target_patches: [B, 64, D]
        
        # Expand for batch-wise comparison
        fusion_expanded = fusion_patches.unsqueeze(1)  # [B, 1, 64, D]
        target_expanded = target_patches.unsqueeze(0)  # [1, B, 64, D]
        
        # Compute similarity matrix: [B, B, 64, 64]
        sim_dense_all = torch.matmul(
            fusion_expanded,  # [B, 1, 64, D]
            target_expanded.transpose(2, 3)  # [1, B, D, 64]
        )  # [B, B, 64, 64]
        
        # Max similarity across target patches, then mean across fusion patches
        sim_dense_max, _ = sim_dense_all.max(dim=3)  # [B, B, 64]
        sim_dense_matrix = sim_dense_max.mean(dim=2) / self.temp  # [B, B]
        
        loss_dense = F.cross_entropy(sim_dense_matrix, targets_idx)
        
        # Loss 3: TIC (Textual Intra-modal Contrastive)
        # Push fusion_query (Ref+Text) away from visual_only_query (Ref+Empty)
        # This forces the model to pay attention to text modifications
        
        # Compute cosine similarity for each pair
        sim_tic_diag = F.cosine_similarity(fusion_query, visual_only_query, dim=1)  # [B]
        
        # We want similarity to be LOW (ideally negative or close to 0)
        # Use a margin loss: penalize if similarity > margin
        margin = 0.2  # Allow some similarity but not too much
        loss_tic = torch.clamp(sim_tic_diag - margin, min=0).mean()
        
        return {
            'loss_itc': loss_itc,
            'loss_dense': loss_dense,
            'loss_tic': loss_tic
        }

    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        """
        Extract target features using DeCo + BERT (must match training!)
        Args:
            image: Target images [B, 3, H, W]
            mode: Feature aggregation mode (kept for compatibility)
        Returns:
            Tuple of (patch_features, global_feature)
        """
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        
        # DeCo compression with position embeddings
        tgt_grid = self.deco_projector(image_embeds)  # [B, 64, 768]
        tgt_grid = tgt_grid + self.visual_pos_embed
        
        # CRITICAL: Process through BERT to match training feature space!
        target_visual_atts = torch.ones(tgt_grid.size()[:-1], dtype=torch.long).to(image.device)
        target_output = self.Qformer.bert(
            query_embeds=tgt_grid,
            attention_mask=target_visual_atts,
            return_dict=True,
        )
        
        # Project and normalize patches
        target_patches = F.normalize(
            self.vision_proj(target_output.last_hidden_state),  # [B, 64, embed_dim]
            dim=-1
        )
        
        # Global feature (for backward compatibility)
        target_global = F.normalize(
            self.vision_proj(target_output.last_hidden_state.mean(dim=1)),
            dim=-1
        )
        
        return target_patches, image_embeds

    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text):
        """
        Inference using DeCo + Self-Attention fusion
        Args:
            reference_embeds: Reference image embeddings [B, 257, dim]
            target_feats: Pre-computed target patch features [N, 64, embed_dim]
            text: Text descriptions [B]
        Returns:
            Similarity scores [B, N]
        """
        # DeCo compression for reference image
        ref_grid = self.deco_projector(reference_embeds)  # [B, 64, 768]
        ref_grid = ref_grid + self.visual_pos_embed
        
        # Tokenize text
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)
        
        # Self-attention fusion (NO cross-attention)
        visual_atts = torch.ones(ref_grid.size()[:-1], dtype=torch.long).to(reference_embeds.device)
        attention_mask = torch.cat([visual_atts, text_tokens.attention_mask], dim=1)
        
        fusion_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            query_embeds=ref_grid,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Extract fusion patches for dense matching
        # Use vision_proj for consistency
        fusion_patches = F.normalize(
            self.vision_proj(fusion_output.last_hidden_state[:, :self.num_query_token, :]),
            dim=-1
        )  # [B, 64, embed_dim]
        
        # Compute dense similarity with all target patches
        # fusion_patches: [B, 64, embed_dim]
        # target_feats: [N, 64, embed_dim]
        
        # Expand for batch-wise comparison
        fusion_expanded = fusion_patches.unsqueeze(1)  # [B, 1, 64, D]
        target_expanded = target_feats.unsqueeze(0)  # [1, N, 64, D]
        
        # Compute similarity: [B, N, 64, 64]
        sim_matrix = torch.matmul(
            fusion_expanded,  # [B, 1, 64, D]
            target_expanded.transpose(2, 3)  # [1, N, D, 64]
        )  # [B, N, 64, 64]
        
        # Max similarity across target patches for each fusion patch
        sim_max, _ = sim_matrix.max(dim=3)  # [B, N, 64]
        
        # Average across fusion patches
        sim_scores = sim_max.mean(dim=2)  # [B, N]
        
        return sim_scores

    @classmethod
    def from_config(cls, cfg):
        """Create model from config"""
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token", 64)  # Must be perfect square (8x8)
        projector_layers = cfg.get("projector_layers", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            max_txt_len=max_txt_len,
            projector_layers=projector_layers,
        )
        model.load_checkpoint_from_config(cfg)

        return model
