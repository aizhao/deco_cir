"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from lavis.models.blip2_models.spatial_branch import QwenSpatialAdapter
from lavis.models.blip2_models.glofnd_loss import GloFNDLoss


@registry.register_model("blip2_cir_align_prompt")
class Blip2QformerCirAlignPrompt(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
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
        # 2D-RoPE Spatial Adapter configuration
        use_spatial_adapter=False,
        spatial_adapter_hidden_dim=768,
        spatial_adapter_num_heads=12,
        spatial_adapter_depth=2,
        # GloFND configuration
        use_glofnd=False,
        glofnd_data_size=50000,
        glofnd_alpha=1e-3,
        glofnd_lr_lda=0.05,
        glofnd_start_update=15,
        glofnd_lda_start=15,
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
        # new tokens
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
        
        # 2D-RoPE Spatial Adapter (Qwen-style) - Parallel Branch
        self.use_spatial_adapter = use_spatial_adapter
        if use_spatial_adapter:
            self.spatial_adapter = QwenSpatialAdapter(
                input_dim=self.visual_encoder.num_features,
                hidden_dim=spatial_adapter_hidden_dim,
                num_heads=spatial_adapter_num_heads,
                depth=spatial_adapter_depth,
            )
            # Project spatial-pooled ViT/adapter features into retrieval embedding space
            self.spatial_proj = nn.Linear(self.visual_encoder.num_features, embed_dim)
            # Project text/query embedding into spatial retrieval space (same dim as embed_dim)
            self.spatial_text_proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )
            # Separate temperature for spatial branch (initialized to match self.temp scale)
            self.temp_spatial = nn.Parameter(0.07 * torch.ones([]))
            # Attention pooling query for spatial tokens (in ViT hidden dim)
            self.spatial_pool_query = nn.Parameter(torch.zeros(1, self.visual_encoder.num_features))
            nn.init.normal_(self.spatial_pool_query, std=0.02)
            logging.info(
                f"Initialized 2D-RoPE Spatial Adapter (Parallel Branch): "
                f"hidden_dim={spatial_adapter_hidden_dim}, "
                f"num_heads={spatial_adapter_num_heads}, "
                f"depth={spatial_adapter_depth}"
            )
        else:
            self.spatial_adapter = None
            self.spatial_proj = None
            self.spatial_text_proj = None
            self.temp_spatial = None
            self.spatial_pool_query = None
        
        # GloFND (Global False Negative Detection) - Optional module
        self.use_glofnd = use_glofnd
        if use_glofnd:
            # Create separate GloFND losses for ITC and RTC
            # Note: temperature will be dynamically updated from self.temp during forward
            self.glofnd_loss_itc = GloFNDLoss(
                data_size=glofnd_data_size,
                temperature=0.07,  # Initial value, will be updated dynamically
                alpha=glofnd_alpha,
                lr_lda=glofnd_lr_lda,
                start_update=glofnd_start_update,
                lda_start=glofnd_lda_start,
            )
            self.glofnd_loss_rtc = GloFNDLoss(
                data_size=glofnd_data_size,
                temperature=0.07,  # Initial value, will be updated dynamically
                alpha=glofnd_alpha,
                lr_lda=glofnd_lr_lda,
                start_update=glofnd_start_update,
                lda_start=glofnd_lda_start,
            )
            logging.info(
                f"Initialized GloFND Loss: data_size={glofnd_data_size}, "
                f"alpha={glofnd_alpha}, lr_lda={glofnd_lr_lda}, "
                f"start_update={glofnd_start_update}, lda_start={glofnd_lda_start}"
            )
        else:
            self.glofnd_loss_itc = None
            self.glofnd_loss_rtc = None
    
    def _apply_spatial_branch(self, vit_output):
        """
        Apply 2D-RoPE Spatial Adapter as a parallel branch.
        Returns:
            enhanced_embeds: Spatially enhanced embeddings, shape [B, N, C]
        """
        # Separate CLS token and patch tokens
        cls_token = vit_output[:, :1, :]      # [B, 1, C]
        patch_tokens = vit_output[:, 1:, :]   # [B, N-1, C]
        
        # Parallel branch: compute spatial delta
        # 使用 torch.is_grad_enabled() 更可靠地检测是否需要计算梯度
        # 只有在训练模式且梯度启用时才计算梯度
        if torch.is_grad_enabled() and self.training:
            spatial_delta = self.spatial_adapter(patch_tokens)  # [B, N-1, C]
        else:
            with torch.no_grad():
                spatial_delta = self.spatial_adapter(patch_tokens)  # [B, N-1, C]
        
        # Feature fusion: add spatial delta to original semantic features
        enhanced_patches = patch_tokens + spatial_delta  # [B, N-1, C]
        
        # Reconstruct full sequence with CLS token
        enhanced_embeds = torch.cat([cls_token, enhanced_patches], dim=1)  # [B, N, C]
        
        return enhanced_embeds
    
    def _extract_visual_features(self, image, use_grad_for_vit=False):
        """
        Extract visual features with optional spatial enhancement.
        
        Args:
            image: Input image tensor, shape [B, 3, H, W]
            use_grad_for_vit: Whether to compute gradients for ViT
                             (False for frozen ViT, True if needed)
        
        Returns:
            image_embeds: Visual embeddings after LayerNorm and optional spatial enhancement
        """
        if use_grad_for_vit:
            # Compute gradients for ViT (not typical for SPRC)
            vit_output = self.visual_encoder(image)
            image_embeds = self.ln_vision(vit_output)
        else:
            # Frozen ViT: no gradients (包括 ln_vision)
            with torch.no_grad():
                vit_output = self.visual_encoder(image)
                image_embeds = self.ln_vision(vit_output)
        
        return image_embeds

    def _extract_spatial_vec(self, vit_output, normalize=True):
        """
        Compute a pooled spatial feature vector from ViT tokens using the RoPE spatial adapter.

        NOTE: This DOES NOT affect Q-Former inputs. It's a parallel branch feature for rerank/loss.
        """
        if not self.use_spatial_adapter:
            return None
        # vit_output: [B, 1+P, C]
        enhanced = self._apply_spatial_branch(vit_output)  # [B, 1+P, C]
        patch_tokens = enhanced[:, 1:, :]                  # [B, P, C]
        # Attention pooling (learnable query) instead of mean pooling
        # scores: [B, P]
        scores = torch.matmul(patch_tokens, self.spatial_pool_query.t()).squeeze(-1)
        scores = scores / (patch_tokens.shape[-1] ** 0.5)
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, P, 1]
        pooled = (patch_tokens * attn).sum(dim=1)           # [B, C]
        vec = self.spatial_proj(pooled)                    # [B, embed_dim]
        if normalize:
            vec = F.normalize(vec, dim=-1)
        return vec


    def forward(self, samples):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]

        ###============== reference text fusion ===================###
        # reference image feature for Q-Former (DO NOT apply spatial branch here)
        image_embeds = self._extract_visual_features(image, use_grad_for_vit=False)
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        # fusion reference image and text tokens into a set of multi-modal tokens
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

        ###============== Fusion-target Contrastive ===================###
        # target image feature for Q-Former (DO NOT apply spatial branch here)
        taregt_embeds = self._extract_visual_features(target, use_grad_for_vit=False)
        
        target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=taregt_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )
        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state), dim=-1
        )

        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        sim_i2t, _ = sim_t2q.max(-1)
        temp_main = torch.clamp(self.temp, min=1e-3)
        sim_i2t = sim_i2t / temp_main
        bs = image.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(
            image.device
        )
        
        # Get sample indices for GloFND (if provided in samples, otherwise use batch indices)
        indices = samples.get("indices", None)
        if indices is None:
            indices = torch.arange(bs, device=image.device, dtype=torch.long)
        else:
            indices = indices.to(image.device)
        
        # Compute ITC loss (with or without GloFND)
        if self.use_glofnd and self.glofnd_loss_itc is not None:
            # Update temperature from learnable parameter
            if isinstance(self.temp, nn.Parameter):
                self.glofnd_loss_itc.temperature = torch.clamp(self.temp.detach(), min=1e-3)
            # Aggregate target features for GloFND (mean pooling over query tokens)
            target_feats_agg = target_feats.mean(dim=1)  # [B, D]
            loss_itc, log_dict_itc = self.glofnd_loss_itc(
                anchor_features=fusion_feats,
                target_features=target_feats_agg,
                indices=indices,
            )
        else:
            # Standard InfoNCE loss
            loss_itc = F.cross_entropy(sim_i2t, targets)
            log_dict_itc = {}

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
        sim_r2t = sim_r2t / temp_main
        
        # Compute RTC loss (with or without GloFND)
        if self.use_glofnd and self.glofnd_loss_rtc is not None:
            # Update temperature from learnable parameter
            if isinstance(self.temp, nn.Parameter):
                self.glofnd_loss_rtc.temperature = torch.clamp(self.temp.detach(), min=1e-3)
            # Aggregate target features for GloFND (mean pooling over query tokens)
            target_feats_agg = target_feats.mean(dim=1)  # [B, D]
            loss_rtc, log_dict_rtc = self.glofnd_loss_rtc(
                anchor_features=text_only_feat,
                target_features=target_feats_agg,
                indices=indices,
            )
        else:
            # Standard InfoNCE loss
            loss_rtc = F.cross_entropy(sim_r2t, targets)
            log_dict_rtc = {}

        loss_align = F.mse_loss(fusion_output.last_hidden_state[:, : query_tokens.size(1), :].mean(1), 
                                prompt_tokens.clone().detach().mean(1))

        out = {
            'loss_itc': loss_itc, 
            'loss_rtc': loss_rtc,
            'loss_align': loss_align
        }
        
        # Add GloFND logging statistics
        if self.use_glofnd:
            if log_dict_itc:
                out.update({
                    'glofnd_itc_lda_mean': log_dict_itc.get('lda_mean', 0.0),
                    'glofnd_itc_lda_std': log_dict_itc.get('lda_std', 0.0),
                    'glofnd_itc_filtered_ratio': log_dict_itc.get('filtered_ratio', 0.0),
                    'glofnd_itc_num_negatives': log_dict_itc.get('num_negatives_per_sample', 0.0),
                })
            if log_dict_rtc:
                out.update({
                    'glofnd_rtc_lda_mean': log_dict_rtc.get('lda_mean', 0.0),
                    'glofnd_rtc_lda_std': log_dict_rtc.get('lda_std', 0.0),
                    'glofnd_rtc_filtered_ratio': log_dict_rtc.get('filtered_ratio', 0.0),
                    'glofnd_rtc_num_negatives': log_dict_rtc.get('num_negatives_per_sample', 0.0),
                })

        # ================= Parallel Spatial Branch (optional extra losses) =================
        # Use RoPE spatial adapter ONLY to produce an auxiliary similarity (does not feed Q-Former).
        if self.use_spatial_adapter and self.spatial_proj is not None:
            # Spatial vectors from ref/target images (B, embed_dim)
            spatial_ref_vec = self._extract_spatial_vec(image_embeds, normalize=True)
            spatial_target_vec = self._extract_spatial_vec(taregt_embeds, normalize=True)
            if spatial_ref_vec is not None and spatial_target_vec is not None:
                # Build spatial queries conditioned on text/query embedding
                # ITC: use fusion_feats (ref+text) as conditioning
                spatial_q_itc = F.normalize(
                    spatial_ref_vec + self.spatial_text_proj(fusion_feats),
                    dim=-1,
                )
                # RTC: use text_only_feat (relative/prompt) as conditioning (anchored on ref spatial)
                spatial_q_rtc = F.normalize(
                    spatial_ref_vec + self.spatial_text_proj(text_only_feat),
                    dim=-1,
                )

                # Proper batch contrastive logits: (B, B)
                logits_spatial_itc = (spatial_q_itc @ spatial_target_vec.t()) / self.temp_spatial
                loss_spatial_itc = F.cross_entropy(logits_spatial_itc, targets)

                logits_spatial_rtc = (spatial_q_rtc @ spatial_target_vec.t()) / self.temp_spatial
                loss_spatial_rtc = F.cross_entropy(logits_spatial_rtc, targets)

                out["loss_spatial_itc"] = loss_spatial_itc
                out["loss_spatial_rtc"] = loss_spatial_rtc

        return out

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        # Extract visual features for Q-Former (DO NOT apply spatial branch here)
        image_embeds = self._extract_visual_features(image, use_grad_for_vit=False)

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    @torch.no_grad()
    def forward_image(self, image):
        # Extract visual features for Q-Former (DO NOT apply spatial branch here)
        image_embeds = self._extract_visual_features(image, use_grad_for_vit=False)
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit
    

    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text):
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
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

        # text-image similarity: aggregate across all query tokens
        sim_i2t, _ = sim_t2q.max(-1)
        # sim_i2t, _ = torch.topk(sim_t2q, k=5, dim=-1)
        # sim_i2t = sim_i2t.mean(-1)
        return sim_i2t


    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            vit_output = self.visual_encoder(image)
            image_embeds_frozen = self.ln_vision(vit_output)
        image_embeds_frozen = image_embeds_frozen.float()
        
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                vit_output = self.visual_encoder(image)
                image_embeds_frozen = self.ln_vision(vit_output)
            image_embeds_frozen = image_embeds_frozen.float()
            
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                vit_output = self.visual_encoder(image)
                image_embeds_frozen = self.ln_vision(vit_output)
            image_embeds_frozen = image_embeds_frozen.float()
            
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

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
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)
        
        # 2D-RoPE Spatial Adapter configuration
        use_spatial_adapter = cfg.get("use_spatial_adapter", False)
        spatial_adapter_hidden_dim = cfg.get("spatial_adapter_hidden_dim", 768)
        spatial_adapter_num_heads = cfg.get("spatial_adapter_num_heads", 12)
        spatial_adapter_depth = cfg.get("spatial_adapter_depth", 2)
        
        # GloFND configuration
        use_glofnd = cfg.get("use_glofnd", False)
        glofnd_data_size = cfg.get("glofnd_data_size", 50000)
        glofnd_alpha = cfg.get("glofnd_alpha", 1e-3)
        glofnd_lr_lda = cfg.get("glofnd_lr_lda", 0.05)
        glofnd_start_update = cfg.get("glofnd_start_update", 15)
        glofnd_lda_start = cfg.get("glofnd_lda_start", 15)

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
            # 2D-RoPE Spatial Adapter
            use_spatial_adapter=use_spatial_adapter,
            spatial_adapter_hidden_dim=spatial_adapter_hidden_dim,
            spatial_adapter_num_heads=spatial_adapter_num_heads,
            spatial_adapter_depth=spatial_adapter_depth,
            # GloFND
            use_glofnd=use_glofnd,
            glofnd_data_size=glofnd_data_size,
            glofnd_alpha=glofnd_alpha,
            glofnd_lr_lda=glofnd_lr_lda,
            glofnd_start_update=glofnd_start_update,
            glofnd_lda_start=glofnd_lda_start,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
    
    def get_spatial_adapter_gate_value(self):
        """
        Get the current gate value of the spatial adapter for monitoring.
        
        Returns:
            float: The gate value, or None if spatial adapter is not enabled
        """
        if self.use_spatial_adapter:
            return self.spatial_adapter.get_gate_value()
        return None
    
    def set_glofnd_epoch(self, epoch: int):
        """
        Set the current epoch for GloFND loss modules.
        This controls when lambda thresholds start updating and filtering.
        
        Args:
            epoch: Current training epoch
        """
        if self.use_glofnd:
            if self.glofnd_loss_itc is not None:
                self.glofnd_loss_itc.set_epoch(epoch)
            if self.glofnd_loss_rtc is not None:
                self.glofnd_loss_rtc.set_epoch(epoch)
