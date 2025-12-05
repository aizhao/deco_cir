"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import math
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import torchvision.transforms.functional as TF

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from lavis.models.blip2_models.flair_query_enhancer import build_flair_query_enhancer


class HardNegativeGenerator:
    """
    轻量级难负样本生成器
    通过简单的图像变换生成难负样本，帮助模型学习更细粒度的区分能力
    
    变换类型:
    - color_jitter: 颜色抖动（色相、饱和度、亮度）
    - spatial_shift: 空间位移
    - horizontal_flip: 水平翻转
    - rotation: 小角度旋转
    - grayscale: 灰度化（移除颜色信息）
    """
    
    def __init__(
        self,
        color_jitter_prob=0.5,
        spatial_shift_prob=0.3,
        flip_prob=0.3,
        rotation_prob=0.3,
        grayscale_prob=0.2,
        color_jitter_strength=0.4,
        spatial_shift_range=0.1,
        rotation_range=15,
    ):
        self.color_jitter_prob = color_jitter_prob
        self.spatial_shift_prob = spatial_shift_prob
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
        self.grayscale_prob = grayscale_prob
        
        self.color_jitter_strength = color_jitter_strength
        self.spatial_shift_range = spatial_shift_range
        self.rotation_range = rotation_range
    
    def color_jitter(self, image):
        """颜色抖动：随机改变色相、饱和度、亮度"""
        s = self.color_jitter_strength
        # 随机调整亮度
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, 1 + random.uniform(-s, s))
        # 随机调整对比度
        if random.random() < 0.5:
            image = TF.adjust_contrast(image, 1 + random.uniform(-s, s))
        # 随机调整饱和度
        if random.random() < 0.5:
            image = TF.adjust_saturation(image, 1 + random.uniform(-s, s))
        # 随机调整色相
        if random.random() < 0.5:
            image = TF.adjust_hue(image, random.uniform(-s/2, s/2))
        return image
    
    def spatial_shift(self, image):
        """空间位移：随机平移图像"""
        _, h, w = image.shape
        max_shift_h = int(h * self.spatial_shift_range)
        max_shift_w = int(w * self.spatial_shift_range)
        
        shift_h = random.randint(-max_shift_h, max_shift_h)
        shift_w = random.randint(-max_shift_w, max_shift_w)
        
        # 使用 affine 变换实现平移
        image = TF.affine(
            image, 
            angle=0, 
            translate=[shift_w, shift_h], 
            scale=1.0, 
            shear=0
        )
        return image
    
    def horizontal_flip(self, image):
        """水平翻转"""
        return TF.hflip(image)
    
    def rotation(self, image):
        """小角度旋转"""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        return TF.rotate(image, angle)
    
    def grayscale(self, image):
        """灰度化（移除颜色信息）"""
        gray = TF.rgb_to_grayscale(image, num_output_channels=3)
        return gray
    
    def generate(self, images):
        """
        为一批图像生成难负样本
        
        Args:
            images: (B, C, H, W) 输入图像张量
            
        Returns:
            hard_negatives: (B, C, H, W) 难负样本张量
        """
        batch_size = images.size(0)
        hard_negatives = []
        
        for i in range(batch_size):
            img = images[i]
            
            # 随机选择变换组合
            if random.random() < self.color_jitter_prob:
                img = self.color_jitter(img)
            
            if random.random() < self.spatial_shift_prob:
                img = self.spatial_shift(img)
            
            if random.random() < self.flip_prob:
                img = self.horizontal_flip(img)
            
            if random.random() < self.rotation_prob:
                img = self.rotation(img)
            
            if random.random() < self.grayscale_prob:
                img = self.grayscale(img)
            
            hard_negatives.append(img)
        
        return torch.stack(hard_negatives, dim=0)
    
    def generate_specific(self, images, transform_type='color'):
        """
        生成特定类型的难负样本
        
        Args:
            images: (B, C, H, W) 输入图像张量
            transform_type: 变换类型 ('color', 'spatial', 'flip', 'grayscale')
            
        Returns:
            hard_negatives: (B, C, H, W) 难负样本张量
        """
        batch_size = images.size(0)
        hard_negatives = []
        
        transform_fn = {
            'color': self.color_jitter,
            'spatial': self.spatial_shift,
            'flip': self.horizontal_flip,
            'rotation': self.rotation,
            'grayscale': self.grayscale,
        }.get(transform_type, self.color_jitter)
        
        for i in range(batch_size):
            img = transform_fn(images[i])
            hard_negatives.append(img)
        
        return torch.stack(hard_negatives, dim=0)


class GatedCrossAttention(nn.Module):
    """
    借鉴 CAMS 的门控交叉注意力机制
    让 Query 决定哪些 Attention 输出需要保留
    Gate = sigmoid(Wz·Q × Uz·AttnOutput)
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # 门控机制权重
        self.gate_wz = nn.Linear(dim, dim)
        self.gate_uz = nn.Linear(dim, dim)
        
        # 初始化
        nn.init.normal_(self.gate_wz.weight, std=0.02)
        nn.init.normal_(self.gate_uz.weight, std=0.02)
        nn.init.zeros_(self.gate_wz.bias)
        nn.init.zeros_(self.gate_uz.bias)
        
    def forward(self, query, key, value):
        """
        Args:
            query: (B, N_q, D)
            key: (B, N_k, D)
            value: (B, N_k, D)
        Returns:
            gated_output: (B, N_q, D)
            attn_weights: (B, N_q, N_k)
        """
        # 标准 Cross-Attention
        attn_output, attn_weights = self.attn(query, key, value)
        
        # 门控机制：Query 决定过滤哪些信息
        gate = torch.sigmoid(self.gate_wz(query) * self.gate_uz(attn_output))
        
        # 应用门控
        gated_output = gate * attn_output
        
        return gated_output, attn_weights


class MultiSpaceTransformer(nn.Module):
    """
    借鉴 CAMS 的多空间解耦
    使用单独的 Transformer 层将共享特征解耦到不同语义空间
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
    def forward(self, x):
        return self.transformer(x)


class DifferenceEncoder(nn.Module):
    """
    Image-Difference Guided Text Enhancement - Teacher Branch
    从参考图像和目标图像的差异中提取完整的修改语义
    只在训练时使用（因为测试时没有目标图像）
    """
    def __init__(self, vit_dim=1408, hidden_dim=768, output_dim=256):
        super().__init__()
        
        # 全局差异编码：CLS token 差异
        self.global_diff_encoder = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, ref_feats, tar_feats):
        """
        Args:
            ref_feats: 参考图像 ViT 特征 (B, 257, vit_dim)
            tar_feats: 目标图像 ViT 特征 (B, 257, vit_dim)
        Returns:
            diff_feats: 图像差异表示 (B, output_dim)
        """
        # 使用 CLS token 计算全局差异
        ref_cls = ref_feats[:, 0, :]  # (B, vit_dim)
        tar_cls = tar_feats[:, 0, :]  # (B, vit_dim)
        
        # 编码差异
        diff = tar_cls - ref_cls
        diff_feats = self.global_diff_encoder(diff)  # (B, output_dim)
        
        return diff_feats


class TextEnrichmentModule(nn.Module):
    """
    Image-Difference Guided Text Enhancement - Student Branch
    从文本特征学习预测图像差异的语义
    训练时：向 Teacher (DifferenceEncoder) 学习
    测试时：独立使用，已学会"模拟"图像差异信息
    """
    def __init__(self, text_dim=768, output_dim=256):
        super().__init__()
        
        # 文本特征增强网络
        self.text_enrichment = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Linear(text_dim, output_dim),
        )
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, text_feats):
        """
        Args:
            text_feats: 文本特征 (B, text_dim)
        Returns:
            enriched_feats: 增强后的文本表示 (B, output_dim)
        """
        return self.text_enrichment(text_feats)


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
        # Gated Cross-Attention (借鉴 CAMS 的门控机制)
        use_gated_attention=False,
        # Multi-Space Disentanglement (借鉴 CAMS 的多空间解耦)
        use_multi_space=False,
        multi_space_loss_weight=0.1,
        # Image-Difference Guided Text Enhancement (图像差异引导的文本增强)
        use_diff_text_enhancement=False,
        diff_enhancement_alpha=0.3,
        diff_contrastive_weight=0.5,
        diff_contrastive_temp=0.07,
        # Hard Negative Mining (难负样本挖掘)
        use_hard_negative=False,
        hard_negative_weight=0.3,
        hard_negative_types=['color', 'spatial'],
        # FLAIR Query Enhancement (FLAIR 增强 Q-Former Query)
        use_flair_query_enhancement=False,
        flair_model_name='merged30m',
        flair_enhancement_type='attention',  # 'bias', 'attention', 'gate'
        flair_enhancement_alpha=0.3,
        freeze_flair=True,
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
        
        # 借鉴 CAMS 的创新
        self.use_gated_attention = use_gated_attention
        self.use_multi_space = use_multi_space
        self.multi_space_loss_weight = multi_space_loss_weight
        
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
            # 可选使用 Gated Cross-Attention（借鉴 CAMS）
            if self.use_gated_attention:
                self.finegrain_cross_attn = GatedCrossAttention(
                    dim=qformer_hidden,
                    num_heads=8,
                    dropout=0.1,
                )
            else:
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
            
            # Multi-Space Disentanglement（借鉴 CAMS 的多空间解耦）
            # 将细粒度特征解耦为：保留空间、修改空间、融合空间
            if self.use_multi_space:
                # 三个独立的 Transformer 层用于解耦
                self.preserve_transformer = MultiSpaceTransformer(qformer_hidden, num_heads=8)
                self.modify_transformer = MultiSpaceTransformer(qformer_hidden, num_heads=8)
                self.compose_transformer = MultiSpaceTransformer(qformer_hidden, num_heads=8)
                
                # 三个空间的独立投影
                self.preserve_proj = nn.Linear(qformer_hidden, embed_dim)
                self.modify_proj = nn.Linear(qformer_hidden, embed_dim)
                self.compose_proj = nn.Linear(qformer_hidden, embed_dim)
                
                # 正交性损失的投影（确保三个空间正交）
                self.orthogonal_proj = nn.Linear(embed_dim * 3, embed_dim)
                
                # 初始化
                nn.init.normal_(self.preserve_proj.weight, std=0.02)
                nn.init.normal_(self.modify_proj.weight, std=0.02)
                nn.init.normal_(self.compose_proj.weight, std=0.02)
                nn.init.normal_(self.orthogonal_proj.weight, std=0.02)
            else:
                self.preserve_transformer = None
                self.modify_transformer = None
                self.compose_transformer = None
                self.preserve_proj = None
                self.modify_proj = None
                self.compose_proj = None
                self.orthogonal_proj = None
        else:
            self.finegrain_text_proj = None
            self.finegrain_vit_to_kv = None
            self.finegrain_cross_attn = None
            self.finegrain_ln = None
            self.finegrain_out_proj = None
            # Multi-Space 相关
            self.preserve_transformer = None
            self.modify_transformer = None
            self.compose_transformer = None
            self.preserve_proj = None
            self.modify_proj = None
            self.compose_proj = None
            self.orthogonal_proj = None

        # Image-Difference Guided Text Enhancement (图像差异引导的文本增强)
        # 训练时利用图像差异信息增强文本表示，测试时模型已学会从文本"模拟"差异信息
        self.use_diff_text_enhancement = use_diff_text_enhancement
        self.diff_enhancement_alpha = diff_enhancement_alpha
        self.diff_contrastive_weight = diff_contrastive_weight
        self.diff_contrastive_temp = diff_contrastive_temp
        
        if self.use_diff_text_enhancement:
            vit_hidden = self.visual_encoder.num_features  # ViT 特征维度 (e.g., 1408)
            qformer_hidden = self.Qformer.config.hidden_size  # Q-Former 特征维度 (e.g., 768)
            
            # Teacher: 图像差异编码器（只在训练时使用）
            self.difference_encoder = DifferenceEncoder(
                vit_dim=vit_hidden,
                hidden_dim=qformer_hidden,
                output_dim=embed_dim,
            )
            
            # Student: 文本增强模块（训练和测试都使用）
            self.text_enrichment_module = TextEnrichmentModule(
                text_dim=qformer_hidden,
                output_dim=embed_dim,
            )
        else:
            self.difference_encoder = None
            self.text_enrichment_module = None

        # Hard Negative Mining (难负样本挖掘)
        # 通过轻量级图像变换生成难负样本，帮助模型学习更细粒度的区分
        self.use_hard_negative = use_hard_negative
        self.hard_negative_weight = hard_negative_weight
        self.hard_negative_types = hard_negative_types
        # self.hard_negative_margin = 0.05  # 减小 margin，因为难负样本确实应该相似
        # self.hard_negative_temp = 0.1     # 温度参数，用于软化损失
        
        if self.use_hard_negative:
            self.hard_negative_generator = HardNegativeGenerator(
                color_jitter_prob=0.8 if 'color' in hard_negative_types else 0.0,
                spatial_shift_prob=0.8 if 'spatial' in hard_negative_types else 0.0,
                flip_prob=0.5 if 'flip' in hard_negative_types else 0.0,
                rotation_prob=0.5 if 'rotation' in hard_negative_types else 0.0,
                grayscale_prob=0.3 if 'grayscale' in hard_negative_types else 0.0,
            )
        else:
            self.hard_negative_generator = None

        # FLAIR Query Enhancement (FLAIR 增强 Q-Former Query)
        # 使用 FLAIR 的细粒度视觉特征增强 Q-Former 的 query tokens
        self.use_flair_query_enhancement = use_flair_query_enhancement
        self.flair_model_name = flair_model_name
        self.flair_enhancement_type = flair_enhancement_type
        self.flair_enhancement_alpha = flair_enhancement_alpha
        self.freeze_flair = freeze_flair
        
        if self.use_flair_query_enhancement:
            logging.info(f"Initializing FLAIR Query Enhancement:")
            logging.info(f"  - FLAIR model: {flair_model_name}")
            logging.info(f"  - Enhancement type: {flair_enhancement_type}")
            logging.info(f"  - Enhancement alpha: {flair_enhancement_alpha}")
            logging.info(f"  - Freeze FLAIR: {freeze_flair}")
            
            self.flair_query_enhancer = build_flair_query_enhancer(
                flair_model_name=flair_model_name,
                qformer_hidden_size=self.Qformer.config.hidden_size,
                num_query_tokens=num_query_token,
                enhancement_type=flair_enhancement_type,
                enhancement_alpha=flair_enhancement_alpha,
                freeze_flair=freeze_flair,
            )
        else:
            self.flair_query_enhancer = None

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

    def _extract_text_guided_finegrain_features(self, vit_features, text_features, return_multi_space=False):
        """
        文本条件化细粒度特征提取
        
        Args:
            vit_features: ViT 输出的 patch tokens (B, N, vit_hidden)，包含 CLS token
            text_features: 文本 CLS 特征 (B, qformer_hidden)
            return_multi_space: 是否返回多空间特征（用于训练时计算正交损失）
        
        Returns:
            finegrain_feats: 文本引导的细粒度特征 (B, embed_dim)
            multi_space_feats: (可选) 包含 preserve, modify, compose 特征的字典
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
        # 可选使用 Gated Cross-Attention（借鉴 CAMS）
        if self.use_gated_attention:
            attn_output, _ = self.finegrain_cross_attn(
                query=text_queries,  # (B, num_tokens, hidden)
                key=k,               # (B, N-1, hidden)
                value=v,             # (B, N-1, hidden)
            )  # (B, num_tokens, hidden)
        else:
            attn_output, _ = self.finegrain_cross_attn(
                query=text_queries,  # (B, num_tokens, hidden)
                key=k,               # (B, N-1, hidden)
                value=v,             # (B, N-1, hidden)
            )  # (B, num_tokens, hidden)
        
        # LayerNorm + 残差连接
        attn_output = self.finegrain_ln(attn_output + text_queries)  # (B, num_tokens, hidden)
        
        # Multi-Space Disentanglement（借鉴 CAMS 的多空间解耦）
        if self.use_multi_space and self.preserve_transformer is not None:
            # 将共享特征解耦到三个独立空间
            preserve_feats = self.preserve_transformer(attn_output)  # (B, num_tokens, hidden)
            modify_feats = self.modify_transformer(attn_output)      # (B, num_tokens, hidden)
            compose_feats = self.compose_transformer(attn_output)    # (B, num_tokens, hidden)
            
            # 聚合并投影到 embedding 空间
            preserve_feats = F.normalize(self.preserve_proj(preserve_feats.mean(dim=1)), dim=-1)  # (B, embed_dim)
            modify_feats = F.normalize(self.modify_proj(modify_feats.mean(dim=1)), dim=-1)        # (B, embed_dim)
            compose_feats = F.normalize(self.compose_proj(compose_feats.mean(dim=1)), dim=-1)     # (B, embed_dim)
            
            # 融合三个空间的特征
            # compose_feats 代表"组合空间"，是最终用于检索的特征
            # 同时使用正交投影融合所有信息
            concat_feats = torch.cat([preserve_feats, modify_feats, compose_feats], dim=-1)
            finegrain_feats = F.normalize(self.orthogonal_proj(concat_feats), dim=-1)
            
            if return_multi_space:
                multi_space_feats = {
                    'preserve': preserve_feats,
                    'modify': modify_feats,
                    'compose': compose_feats,
                }
                return finegrain_feats, multi_space_feats
            return finegrain_feats
        
        # 聚合多个 tokens 为单个特征向量
        finegrain_feats = attn_output.mean(dim=1)  # (B, hidden)
        
        # 投影到最终 embedding 空间并归一化
        finegrain_feats = F.normalize(self.finegrain_out_proj(finegrain_feats), dim=-1)  # (B, embed_dim)
        
        if return_multi_space:
            return finegrain_feats, None
        return finegrain_feats
    
    def _compute_orthogonal_loss(self, multi_space_feats):
        """
        计算正交性损失，确保三个空间语义独立
        借鉴 CAMS 的 Multi-Space Disentanglement
        
        Args:
            multi_space_feats: 包含 preserve, modify, compose 特征的字典
        
        Returns:
            ortho_loss: 正交性损失标量
        """
        if multi_space_feats is None:
            return 0.0
        
        preserve = multi_space_feats['preserve']  # (B, embed_dim)
        modify = multi_space_feats['modify']      # (B, embed_dim)
        compose = multi_space_feats['compose']    # (B, embed_dim)
        
        # 计算两两之间的余弦相似度，目标是让它们接近 0（正交）
        sim_pm = (preserve * modify).sum(dim=-1).mean()   # preserve-modify
        sim_pc = (preserve * compose).sum(dim=-1).mean()  # preserve-compose
        sim_mc = (modify * compose).sum(dim=-1).mean()    # modify-compose
        
        # 正交性损失 = 相似度的平方和
        ortho_loss = sim_pm ** 2 + sim_pc ** 2 + sim_mc ** 2
        
        return ortho_loss
    
    def _compute_diff_contrastive_loss(self, text_enriched_feats, diff_feats):
        """
        计算图像差异引导的对比损失
        让文本增强特征与对应的图像差异特征在 batch 内对齐
        
        Args:
            text_enriched_feats: 文本增强模块输出 (B, embed_dim)
            diff_feats: 图像差异编码器输出 (B, embed_dim)
        
        Returns:
            contrastive_loss: 对比损失标量
        """
        # 归一化（双方都参与训练，共同学习对齐）
        text_enriched_feats = F.normalize(text_enriched_feats, dim=-1)
        diff_feats = F.normalize(diff_feats, dim=-1)  # 不 detach，让双方共同优化
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(text_enriched_feats, diff_feats.T) / self.diff_contrastive_temp  # (B, B)
        
        # 对角线是正样本
        labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
        
        # 双向对比损失
        loss_t2d = F.cross_entropy(sim_matrix, labels)      # text_enriched → diff
        loss_d2t = F.cross_entropy(sim_matrix.T, labels)    # diff → text_enriched
        
        return (loss_t2d + loss_d2t) / 2
    
    def _compute_hard_negative_loss(self, fusion_feats, target_feats, hard_neg_feats):
        """
        计算难负样本对比损失
        让模型区分真正的目标图像和难负样本（通过变换生成的相似但错误的图像）
        
        Args:
            fusion_feats: 融合特征 (B, embed_dim)，已归一化
            target_feats: 正样本（目标图像）特征 (B, num_query, embed_dim)，已归一化
            hard_neg_feats: 难负样本特征 (B, num_query, embed_dim)，已归一化
        
        Returns:
            hard_neg_loss: 难负样本对比损失
        """
        # 确保特征归一化
        # fusion_feats = F.normalize(fusion_feats, dim=-1)
        # target_feats = F.normalize(target_feats, dim=-1)
        # hard_neg_feats = F.normalize(hard_neg_feats, dim=-1)
        
        # 计算融合特征与正样本的相似度（余弦相似度）
        sim_pos = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()  # (B, num_query)
        sim_pos, _ = sim_pos.max(-1)  # (B,) 取最大相似度
        
        # 计算融合特征与难负样本的相似度
        sim_neg = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), hard_neg_feats.permute(0, 2, 1)
        ).squeeze()  # (B, num_query)
        sim_neg, _ = sim_neg.max(-1)  # (B,) 取最大相似度
        
        # 使用更温和的损失函数
        # 方案1：平滑的 margin loss（使用 sigmoid 软化）
        margin = 0.2
        # diff = sim_neg - sim_pos + margin  # 如果 diff > 0，说明难负样本太相似了
        
        # 使用平滑的损失：smooth_relu = log(1 + exp(x))，比 relu 更平滑
        loss = F.relu(sim_neg - sim_pos + margin).mean()
        
        # 方案2（备选）：如果方案1还是太大，可以用这个更温和的版本
        # loss = (diff ** 2).mean()  # 平方损失，更温和
        
        return loss

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

        # FLAIR Query Enhancement: 使用 FLAIR 的细粒度特征增强 query tokens
        # 注意：FLAIR 需要特定的图像预处理，这里直接使用原始图像
        # 在实际部署时，可能需要为 FLAIR 单独预处理图像
        if self.use_flair_query_enhancement:
            # 使用 FLAIR 提取文本条件化的细粒度特征，增强 query tokens
            query_tokens = self.flair_query_enhancer(
                query_tokens=query_tokens,
                flair_image=image,  # 使用原始图像（FLAIR 内部会处理）
                flair_text=text,    # 修改文本
            )

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
        multi_space_feats = None  # 用于正交损失计算
        if self.use_text_guided_finegrain:
            # 获取纯文本 CLS 特征作为条件（不含图像信息，避免信息泄露）
            pure_text_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            text_cls_for_finegrain = pure_text_output.last_hidden_state[:, 0, :]  # 纯文本 CLS
            
            # 提取文本引导的细粒度特征
            # 如果启用了多空间解耦，同时返回多空间特征用于正交损失
            if self.use_multi_space:
                finegrain_feats, multi_space_feats = self._extract_text_guided_finegrain_features(
                    vit_features=image_embeds_raw,  # 使用原始 ViT 特征（未经 spatial adapter）
                    text_features=text_cls_for_finegrain,
                    return_multi_space=True,
                )
            else:
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

        ###============== Image-Difference Guided Text Enhancement ===================###
        # 图像差异引导的文本增强（可选）
        diff_feats = None
        text_enriched_feats = None
        if self.use_diff_text_enhancement:
            # 获取目标图像的原始 ViT 特征（用于计算图像差异）
            target_embeds_raw = self.ln_vision(self.visual_encoder(target))
            
            # Teacher: 计算图像差异特征
            diff_feats = self.difference_encoder(image_embeds_raw, target_embeds_raw)
            
            # Student: 从纯文本特征预测差异
            # 使用纯文本 CLS 特征
            pure_text_output_for_diff = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            text_cls_for_diff = pure_text_output_for_diff.last_hidden_state[:, 0, :]
            text_enriched_feats = self.text_enrichment_module(text_cls_for_diff)
            
            # 将增强后的文本特征融合到最终特征中
            text_enriched_feats_norm = F.normalize(text_enriched_feats, dim=-1)
            fusion_feats = (1 - self.diff_enhancement_alpha) * fusion_feats + self.diff_enhancement_alpha * text_enriched_feats_norm
            fusion_feats = F.normalize(fusion_feats, dim=-1)

        ###============== Fusion-target Contrastive ===================###
        # target image feature
        if self.use_diff_text_enhancement:
            # 复用已计算的目标图像特征
            taregt_embeds = self._apply_spatial_adapter(target_embeds_raw)
        else:
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

        # 计算正交性损失（如果启用了多空间解耦）
        loss_dict = {
            'loss_itc': loss_itc, 
            'loss_rtc': loss_rtc,
            'loss_align': loss_align
        }
        
        if self.use_multi_space and multi_space_feats is not None:
            loss_ortho = self._compute_orthogonal_loss(multi_space_feats)
            loss_dict['loss_ortho'] = self.multi_space_loss_weight * loss_ortho
        
        # 计算图像差异引导的对比损失（如果启用）
        if self.use_diff_text_enhancement and diff_feats is not None and text_enriched_feats is not None:
            loss_diff_contrastive = self._compute_diff_contrastive_loss(text_enriched_feats, diff_feats)
            loss_dict['loss_diff'] = self.diff_contrastive_weight * loss_diff_contrastive
        
        # 难负样本挖掘（如果启用）
        if self.use_hard_negative and self.training:
            # 从目标图像生成难负样本
            with torch.no_grad():
                hard_neg_images = self.hard_negative_generator.generate(target)
            
            # 提取难负样本特征
            hard_neg_embeds = self.ln_vision(self.visual_encoder(hard_neg_images))
            hard_neg_embeds = self._apply_spatial_adapter(hard_neg_embeds)
            hard_neg_atts = torch.ones(hard_neg_embeds.size()[:-1], dtype=torch.long).to(image.device)
            
            hard_neg_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=hard_neg_embeds,
                encoder_attention_mask=hard_neg_atts,
                use_cache=True,
                return_dict=True,
            )
            hard_neg_feats = F.normalize(
                self.vision_proj(hard_neg_output.last_hidden_state), dim=-1
            )
            
            # 计算难负样本损失
            loss_hard_neg = self._compute_hard_negative_loss(fusion_feats, target_feats, hard_neg_feats)
            loss_dict['loss_hard_neg'] = self.hard_negative_weight * loss_hard_neg
        
        return loss_dict

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
    def inference(self, reference_embeds, target_feats, text, reference_embeds_raw=None, reference_images=None):
        """
        推理函数：计算融合特征与目标特征的相似度
        
        Args:
            reference_embeds: 经过 spatial adapter 处理的参考图像特征
            target_feats: 目标图像特征
            text: 文本描述
            reference_embeds_raw: 原始 ViT 特征（用于细粒度分支，可选）
            reference_images: 原始参考图像（用于 FLAIR 增强，可选）
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

        # FLAIR Query Enhancement（推理时）
        if self.use_flair_query_enhancement and reference_images is not None:
            query_tokens = self.flair_query_enhancer(
                query_tokens=query_tokens,
                flair_image=reference_images,
                flair_text=text,
            )

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
            
            # 获取纯文本 CLS 特征作为条件（不含图像信息，避免信息泄露）
            pure_text_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            text_cls_for_finegrain = pure_text_output.last_hidden_state[:, 0, :]  # 纯文本 CLS
            
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

        # Image-Difference Guided Text Enhancement（推理时）
        # 测试时没有目标图像，但 Student (text_enrichment_module) 已学会从文本"模拟"差异信息
        if self.use_diff_text_enhancement:
            # 获取纯文本 CLS 特征
            pure_text_output_for_diff = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            text_cls_for_diff = pure_text_output_for_diff.last_hidden_state[:, 0, :]
            
            # 文本增强：Student 已学会预测差异信息
            text_enriched_feats = self.text_enrichment_module(text_cls_for_diff)
            text_enriched_feats = F.normalize(text_enriched_feats, dim=-1)
            
            # 融合到最终特征
            fusion_feats = (1 - self.diff_enhancement_alpha) * fusion_feats + self.diff_enhancement_alpha * text_enriched_feats
            fusion_feats = F.normalize(fusion_feats, dim=-1)

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
        
        # CAMS-inspired innovations (借鉴 CAMS 的创新)
        use_gated_attention = cfg.get("use_gated_attention", False)
        use_multi_space = cfg.get("use_multi_space", False)
        multi_space_loss_weight = cfg.get("multi_space_loss_weight", 0.1)
        
        # Image-Difference Guided Text Enhancement (图像差异引导的文本增强)
        use_diff_text_enhancement = cfg.get("use_diff_text_enhancement", False)
        diff_enhancement_alpha = cfg.get("diff_enhancement_alpha", 0.3)
        diff_contrastive_weight = cfg.get("diff_contrastive_weight", 0.5)
        diff_contrastive_temp = cfg.get("diff_contrastive_temp", 0.07)
        
        # Hard Negative Mining (难负样本挖掘)
        use_hard_negative = cfg.get("use_hard_negative", False)
        hard_negative_weight = cfg.get("hard_negative_weight", 0.3)
        hard_negative_types = cfg.get("hard_negative_types", ['color', 'spatial'])
        
        # FLAIR Query Enhancement (FLAIR 增强 Q-Former Query)
        use_flair_query_enhancement = cfg.get("use_flair_query_enhancement", False)
        flair_model_name = cfg.get("flair_model_name", "merged30m")
        flair_enhancement_type = cfg.get("flair_enhancement_type", "attention")
        flair_enhancement_alpha = cfg.get("flair_enhancement_alpha", 0.3)
        freeze_flair = cfg.get("freeze_flair", True)

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
            use_gated_attention=use_gated_attention,
            use_multi_space=use_multi_space,
            multi_space_loss_weight=multi_space_loss_weight,
            use_diff_text_enhancement=use_diff_text_enhancement,
            diff_enhancement_alpha=diff_enhancement_alpha,
            diff_contrastive_weight=diff_contrastive_weight,
            diff_contrastive_temp=diff_contrastive_temp,
            use_hard_negative=use_hard_negative,
            hard_negative_weight=hard_negative_weight,
            hard_negative_types=hard_negative_types,
            # FLAIR Query Enhancement
            use_flair_query_enhancement=use_flair_query_enhancement,
            flair_model_name=flair_model_name,
            flair_enhancement_type=flair_enhancement_type,
            flair_enhancement_alpha=flair_enhancement_alpha,
            freeze_flair=freeze_flair,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
