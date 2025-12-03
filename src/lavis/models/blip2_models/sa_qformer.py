"""
SA-QFormer: Based on blip2_qformer_cir_align_prompt with grid pooling
"""
import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures


class AdaptiveGridPooling(nn.Module):
    """Normalize variable-sized ViT patch embeddings to fixed spatial grid"""
    
    def __init__(self, output_size=(8, 8)):
        super().__init__()
        self.output_size = output_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, vit_features, original_grid_size=None):
        B, N, D = vit_features.shape
        
        # Handle CLS token if present (N = H*W + 1)
        # Try N-1 first (assuming CLS token at position 0)
        N_patches = N - 1
        H_vit = W_vit = int(math.sqrt(N_patches))
        
        if H_vit * W_vit == N_patches:
            # Has CLS token, remove it
            vit_features = vit_features[:, 1:, :]  # Remove CLS token
            N = N_patches
        else:
            # No CLS token, try original N
            H_vit = W_vit = int(math.sqrt(N))
            if H_vit * W_vit != N:
                raise ValueError(f"Cannot infer square grid from {N} patches")
        
        # Reshape to 2D spatial grid
        features_2d = vit_features.view(B, H_vit, W_vit, D)
        features_2d = features_2d.permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # Apply adaptive pooling
        pooled = self.adaptive_pool(features_2d)  # (B, D, H_out, W_out)
        
        # Permute back and flatten
        pooled = pooled.permute(0, 2, 3, 1)  # (B, H_out, W_out, D)
        H_out, W_out = self.output_size
        grid_features = pooled.reshape(B, H_out * W_out, D)
        
        return grid_features


@registry.register_model("sa_qformer")
class SAQFormer(Blip2Base):
    """
    SA-QFormer: blip2_cir_align_prompt + grid pooling for spatial awareness
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
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        grid_size=(8, 8),
        **kwargs  # Ignore extra params
    ):
        super().__init__()
        
        self.tokenizer = self.init_tokenizer()
        
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len
        
        # Prompt tokens for relative contrastive
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
        
        # SA-QFormer: Add grid pooling (disabled for testing)
        self.adaptive_pooling = AdaptiveGridPooling(output_size=grid_size)
        self.use_grid_pooling = False  # Disable for now
        
        logging.info(f"Initialized SA-QFormer with grid_size={grid_size}")
    
    def forward(self, samples):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]
        
        ###============== Reference-Text Fusion (NO grid pooling for now) ===================###
        image_embeds = self.ln_vision(self.visual_encoder(image))
        # image_embeds = self.adaptive_pooling(image_embeds)  # Disabled: causing loss increase
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
        
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        fusion_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )
        
        ###============== Target Features (NO grid pooling for now) ===================###
        target_embeds = self.ln_vision(self.visual_encoder(target))
        # target_embeds = self.adaptive_pooling(target_embeds)  # Disabled: causing loss increase
        target_atts = torch.ones(target_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=target_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )
        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state), dim=-1
        )
        
        ###============== Fusion-Target Contrastive ===================###
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()
        
        sim_i2t, _ = sim_t2q.max(-1)
        sim_i2t = sim_i2t / self.temp
        bs = image.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(image.device)
        loss_itc = F.cross_entropy(sim_i2t, targets)
        
        ###============== Relative Contrastive ===================###
        prompt_tokens = self.prompt_tokens.expand(image_embeds.shape[0], -1, -1)
        
        text_only_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=prompt_tokens,
            attention_mask=attention_mask,
            return_dict=True,
            no_img=True
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
        
        ###============== Alignment Loss ===================###
        loss_align = F.mse_loss(
            fusion_output.last_hidden_state[:, : query_tokens.size(1), :].mean(1),
            prompt_tokens.clone().detach().mean(1)
        )
        
        return {
            'loss_itc': loss_itc,
            'loss_rtc': loss_rtc,
            'loss_align': loss_align
        }
    
    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        # image_embeds = self.adaptive_pooling(image_embeds)  # Disabled
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds
    
    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text):
        # reference_embeds = self.adaptive_pooling(reference_embeds)  # Disabled
        
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(reference_embeds.device)
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
        
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)
        
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        fusion_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )
        
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()
        
        sim_i2t, _ = sim_t2q.max(-1)
        return sim_i2t
    
    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        # image_embeds_frozen = self.adaptive_pooling(image_embeds_frozen)  # Disabled
        
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)
        query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen
    
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        grid_size = tuple(cfg.get("grid_size", [8, 8]))
        
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
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            grid_size=grid_size,
        )
        
        model.load_checkpoint_from_config(cfg)
        
        return model
