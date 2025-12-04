"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import math

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
        use_instruction_injection=False,
        instruction_alpha=0.1,
        use_spatial_adapter=False,
        spatial_kernel_size=3,
        spatial_alpha=0.1,
        use_spatial_gating=False,
        use_multi_scale_spatial=False,
        # Text-Guided Fine-Grained Branch (文本条件化细粒度分支)
        use_text_guided_finegrain=False,
        finegrain_num_tokens=8,
        finegrain_alpha=0.3,
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

        # lightweight instruction injection config
        self.use_instruction_injection = use_instruction_injection
        self.instruction_alpha = instruction_alpha

        # optional MLP: text CLS -> bias on query tokens
        if self.use_instruction_injection:
            self.instruction_mlp = nn.Sequential(
                nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size),
                nn.Tanh(),
            )
        else:
            self.instruction_mlp = None

        # very light spatial adapter on ViT patch tokens (depthwise conv, no pooling)
        self.use_spatial_adapter = use_spatial_adapter
        self.spatial_kernel_size = spatial_kernel_size
        self.use_spatial_gating = use_spatial_gating
        self.use_multi_scale_spatial = use_multi_scale_spatial
        if self.use_spatial_adapter:
            hidden = self.visual_encoder.num_features
            padding = spatial_kernel_size // 2
            
            if self.use_multi_scale_spatial:
                # multi-scale spatial adapter: 3x3, 5x5, 7x7
                self.spatial_conv_3 = nn.Conv2d(
                    hidden, hidden, kernel_size=3, padding=1, groups=hidden
                )
                self.spatial_conv_5 = nn.Conv2d(
                    hidden, hidden, kernel_size=5, padding=2, groups=hidden
                )
                self.spatial_conv_7 = nn.Conv2d(
                    hidden, hidden, kernel_size=7, padding=3, groups=hidden
                )
                # fusion weights for multi-scale
                self.spatial_fusion = nn.Parameter(torch.ones(3) / 3)
                for conv in [self.spatial_conv_3, self.spatial_conv_5, self.spatial_conv_7]:
                    nn.init.zeros_(conv.weight)
                    if conv.bias is not None:
                        nn.init.zeros_(conv.bias)
                self.spatial_conv = None
            else:
                # single-scale spatial adapter
                self.spatial_conv = nn.Conv2d(
                    hidden,
                    hidden,
                    kernel_size=spatial_kernel_size,
                    padding=padding,
                    groups=hidden,
                )
                nn.init.zeros_(self.spatial_conv.weight)
                if self.spatial_conv.bias is not None:
                    nn.init.zeros_(self.spatial_conv.bias)
                self.spatial_conv_3 = self.spatial_conv_5 = self.spatial_conv_7 = None
                self.spatial_fusion = None
            
            # optional gating mechanism: learnable gate to control spatial adapter strength
            if self.use_spatial_gating:
                self.spatial_gate = nn.Sequential(
                    nn.Linear(hidden, hidden // 4),
                    nn.ReLU(),
                    nn.Linear(hidden // 4, 1),
                    nn.Sigmoid()
                )
            else:
                self.spatial_gate = None
            
            self.spatial_ln = nn.LayerNorm(hidden)
            # residual strength for spatial adapter
            self.spatial_alpha = spatial_alpha
        else:
            self.spatial_conv = None
            self.spatial_ln = None
            self.spatial_gate = None
            self.spatial_conv_3 = self.spatial_conv_5 = self.spatial_conv_7 = None
            self.spatial_fusion = None

        # new tokens
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)

        # Text-Guided Fine-Grained Branch (文本条件化细粒度分支)
        # 用文本特征作为 query，从 ViT patch tokens 中提取相关的细粒度信息
        # 绕过 Q-Former 的信息瓶颈，但保持文本条件性
        self.use_text_guided_finegrain = use_text_guided_finegrain
        self.finegrain_num_tokens = finegrain_num_tokens
        self.finegrain_alpha = finegrain_alpha
        
        if self.use_text_guided_finegrain:
            vit_hidden = self.visual_encoder.num_features  # ViT 特征维度 (e.g., 1408)
            qformer_hidden = self.Qformer.config.hidden_size  # Q-Former 特征维度 (e.g., 768)
            
            # 将文本特征投影为多个 query tokens
            self.finegrain_text_proj = nn.Sequential(
                nn.Linear(qformer_hidden, qformer_hidden),
                nn.GELU(),
                nn.Linear(qformer_hidden, finegrain_num_tokens * qformer_hidden),
            )
            
            # 将 ViT 特征投影到 Q-Former 特征空间 (K, V)
            self.finegrain_vit_to_kv = nn.Linear(vit_hidden, qformer_hidden * 2)
            
            # Cross-Attention: text query -> image patches
            self.finegrain_cross_attn = nn.MultiheadAttention(
                embed_dim=qformer_hidden,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.finegrain_ln = nn.LayerNorm(qformer_hidden)
            
            # 将细粒度特征投影到最终 embedding 空间
            self.finegrain_out_proj = nn.Linear(qformer_hidden, embed_dim)
            
            # 初始化
            nn.init.normal_(self.finegrain_text_proj[0].weight, std=0.02)
            nn.init.normal_(self.finegrain_text_proj[2].weight, std=0.02)
            nn.init.normal_(self.finegrain_vit_to_kv.weight, std=0.02)
            nn.init.normal_(self.finegrain_out_proj.weight, std=0.02)
        else:
            self.finegrain_text_proj = None
            self.finegrain_vit_to_kv = None
            self.finegrain_cross_attn = None
            self.finegrain_ln = None
            self.finegrain_out_proj = None

    def _apply_spatial_adapter(self, image_embeds):
        """
        Apply spatial adapter to image embeddings (ViT patch tokens).
        Returns processed embeddings with same shape (B, N, D).
        """
        if not self.use_spatial_adapter:
            return image_embeds
        
        B, N, D = image_embeds.size()
        n_patches = N - 1
        H = W = int(math.sqrt(n_patches))
        
        if H * W != n_patches or n_patches == 0:
            # cannot form square grid, just apply LayerNorm
            return self.spatial_ln(image_embeds)
        
        cls_tok = image_embeds[:, :1, :]
        patch_tok = image_embeds[:, 1:, :].view(B, H, W, D).permute(0, 3, 1, 2)
        
        # multi-scale or single-scale spatial adapter
        if self.use_multi_scale_spatial:
            patch_res_3 = self.spatial_conv_3(patch_tok)
            patch_res_5 = self.spatial_conv_5(patch_tok)
            patch_res_7 = self.spatial_conv_7(patch_tok)
            # weighted fusion
            weights = F.softmax(self.spatial_fusion, dim=0)
            patch_res = (
                weights[0] * patch_res_3 +
                weights[1] * patch_res_5 +
                weights[2] * patch_res_7
            )
        else:
            patch_res = self.spatial_conv(patch_tok)
        
        # optional gating: learnable gate per patch
        if self.use_spatial_gating:
            # compute gate from original patch features
            patch_flat = patch_tok.permute(0, 2, 3, 1).contiguous().view(B * H * W, D)
            gate_vals = self.spatial_gate(patch_flat).view(B, 1, H, W)
            patch_res = patch_res * gate_vals
        
        # residual spatial adapter: patch + alpha * conv(patch)
        patch_tok = patch_tok + self.spatial_alpha * patch_res
        patch_tok = (
            patch_tok.permute(0, 2, 3, 1)
            .contiguous()
            .view(B, n_patches, D)
        )
        image_embeds = torch.cat([cls_tok, patch_tok], dim=1)
        return self.spatial_ln(image_embeds)

    def _extract_text_guided_finegrain_features(self, vit_features, text_features):
        """
        文本条件化细粒度特征提取
        
        Args:
            vit_features: ViT 输出的 patch tokens (B, N, vit_hidden)，包含 CLS token
            text_features: 文本 CLS 特征 (B, qformer_hidden)
        
        Returns:
            finegrain_feats: 文本引导的细粒度特征 (B, embed_dim)
        """
        B = vit_features.size(0)
        
        # 只使用 patch tokens，不使用 CLS token
        patch_tokens = vit_features[:, 1:, :]  # (B, N-1, vit_hidden)
        
        # 将文本特征转换为多个 query tokens
        # text_features: (B, qformer_hidden) -> (B, num_tokens, qformer_hidden)
        text_queries = self.finegrain_text_proj(text_features)  # (B, num_tokens * hidden)
        text_queries = text_queries.view(B, self.finegrain_num_tokens, -1)  # (B, num_tokens, hidden)
        
        # 将 ViT patch tokens 投影为 K, V
        kv = self.finegrain_vit_to_kv(patch_tokens)  # (B, N-1, hidden * 2)
        k, v = kv.chunk(2, dim=-1)  # 各 (B, N-1, hidden)
        
        # Cross-Attention: text queries attend to image patches
        # 文本特征作为 Q，图像 patch 特征作为 K, V
        attn_output, _ = self.finegrain_cross_attn(
            query=text_queries,  # (B, num_tokens, hidden)
            key=k,               # (B, N-1, hidden)
            value=v,             # (B, N-1, hidden)
        )  # (B, num_tokens, hidden)
        
        # LayerNorm + 残差连接
        attn_output = self.finegrain_ln(attn_output + text_queries)
        
        # 聚合多个 tokens 为单个特征向量
        finegrain_feats = attn_output.mean(dim=1)  # (B, hidden)
        
        # 投影到最终 embedding 空间并归一化
        finegrain_feats = F.normalize(self.finegrain_out_proj(finegrain_feats), dim=-1)  # (B, embed_dim)
        
        return finegrain_feats

    def forward(self, samples):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]

        ###============== reference text fusion ===================###
        # reference image feature
        image_embeds_raw = self.ln_vision(self.visual_encoder(image))  # 保存原始 ViT 特征用于细粒度分支
        image_embeds = image_embeds_raw
        # optional very-light spatial adapter (keeps sequence length)
        image_embeds = self._apply_spatial_adapter(image_embeds)
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

        # optional: lightweight instruction injection from text CLS to query tokens
        if self.use_instruction_injection:
            text_cls_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            # (B, H)
            text_cls = text_cls_output.last_hidden_state[:, 0, :]
            bias = self.instruction_mlp(text_cls).unsqueeze(1)  # (B, 1, H)
            query_tokens = query_tokens + self.instruction_alpha * bias

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

        # Q-Former 融合特征
        qformer_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )
        
        # 文本条件化细粒度分支（可选）
        if self.use_text_guided_finegrain:
            # 获取文本 CLS 特征作为条件
            text_cls_for_finegrain = text_output.last_hidden_state[:, 32, :]  # 使用融合后的文本位置特征
            
            # 提取文本引导的细粒度特征
            finegrain_feats = self._extract_text_guided_finegrain_features(
                vit_features=image_embeds_raw,  # 使用原始 ViT 特征（未经 spatial adapter）
                text_features=text_cls_for_finegrain,
            )
            
            # 融合 Q-Former 特征和细粒度特征
            # fusion_feats = (1 - α) * qformer_feats + α * finegrain_feats
            fusion_feats = (1 - self.finegrain_alpha) * qformer_feats + self.finegrain_alpha * finegrain_feats
            fusion_feats = F.normalize(fusion_feats, dim=-1)
        else:
            fusion_feats = qformer_feats

        ###============== Fusion-target Contrastive ===================###
        # target image feature
        taregt_embeds = self.ln_vision(self.visual_encoder(target))
        taregt_embeds = self._apply_spatial_adapter(taregt_embeds)
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
        sim_i2t = sim_i2t / self.temp
        bs = image.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(
            image.device
        )
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

        loss_align = F.mse_loss(fusion_output.last_hidden_state[:, : query_tokens.size(1), :].mean(1), 
                                prompt_tokens.clone().detach().mean(1))

        return {
            'loss_itc': loss_itc, 
            'loss_rtc': loss_rtc,
            'loss_align': loss_align
        }

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
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = self._apply_spatial_adapter(image_embeds)

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

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = self._apply_spatial_adapter(image_embeds)
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
    def inference(self, reference_embeds, target_feats, text, reference_embeds_raw=None):
        """
        推理函数：计算融合特征与目标特征的相似度
        
        Args:
            reference_embeds: 经过 spatial adapter 处理的参考图像特征
            target_feats: 目标图像特征
            text: 文本描述
            reference_embeds_raw: 原始 ViT 特征（用于细粒度分支，可选）
        """
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

        # Q-Former 融合特征
        qformer_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )
        
        # 文本条件化细粒度分支（可选）
        if self.use_text_guided_finegrain:
            # 如果未提供原始 ViT 特征，则使用 reference_embeds（通常已经是原始 ViT 特征）
            vit_features_for_finegrain = reference_embeds_raw if reference_embeds_raw is not None else reference_embeds
            
            # 获取文本 CLS 特征作为条件
            text_cls_for_finegrain = text_output.last_hidden_state[:, 32, :]
            
            # 提取文本引导的细粒度特征
            finegrain_feats = self._extract_text_guided_finegrain_features(
                vit_features=vit_features_for_finegrain,
                text_features=text_cls_for_finegrain,
            )
            
            # 融合 Q-Former 特征和细粒度特征
            fusion_feats = (1 - self.finegrain_alpha) * qformer_feats + self.finegrain_alpha * finegrain_feats
            fusion_feats = F.normalize(fusion_feats, dim=-1)
        else:
            fusion_feats = qformer_feats

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
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_embeds = self._apply_spatial_adapter(image_embeds_frozen)
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
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
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_embeds = self._apply_spatial_adapter(image_embeds_frozen)

            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
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
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_embeds = self._apply_spatial_adapter(image_embeds_frozen)

            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1
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
                encoder_hidden_states=image_embeds,
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
        use_instruction_injection = cfg.get("use_instruction_injection", False)
        instruction_alpha = cfg.get("instruction_alpha", 0.1)
        use_spatial_adapter = cfg.get("use_spatial_adapter", False)
        spatial_kernel_size = cfg.get("spatial_kernel_size", 3)
        spatial_alpha = cfg.get("spatial_alpha", 0.1)
        use_spatial_gating = cfg.get("use_spatial_gating", False)
        use_multi_scale_spatial = cfg.get("use_multi_scale_spatial", False)
        
        # Text-Guided Fine-Grained Branch
        use_text_guided_finegrain = cfg.get("use_text_guided_finegrain", False)
        finegrain_num_tokens = cfg.get("finegrain_num_tokens", 8)
        finegrain_alpha = cfg.get("finegrain_alpha", 0.3)

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
            use_instruction_injection=use_instruction_injection,
            instruction_alpha=instruction_alpha,
            use_spatial_adapter=use_spatial_adapter,
            spatial_kernel_size=spatial_kernel_size,
            spatial_alpha=spatial_alpha,
            use_spatial_gating=use_spatial_gating,
            use_multi_scale_spatial=use_multi_scale_spatial,
            use_text_guided_finegrain=use_text_guided_finegrain,
            finegrain_num_tokens=finegrain_num_tokens,
            finegrain_alpha=finegrain_alpha,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
