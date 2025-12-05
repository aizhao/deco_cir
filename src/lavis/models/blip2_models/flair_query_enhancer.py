"""
FLAIR Query Enhancer: 使用 FLAIR 的细粒度特征增强 Q-Former 的 Query Tokens

设计思路：
1. FLAIR 擅长文本条件化的细粒度图文对齐
2. Q-Former 使用可学习的 query tokens 从视觉特征中提取语义信息
3. 本模块让 FLAIR 的细粒度特征指导 query tokens 的学习

核心创新：
- 不是简单地替换或拼接，而是使用 FLAIR 特征作为"先验"来增强 query
- 通过 cross-attention 让 query tokens 关注 FLAIR 提取的文本相关区域
- 残差连接保证训练稳定性，初期主要依赖 Q-Former 原有的 query

消融实验选项：
- flair_model_name: FLAIR 预训练模型选择
- enhancement_type: 增强方式 ('bias', 'attention', 'gate')
- enhancement_alpha: 增强强度（残差权重）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
import os

# FLAIR 路径配置
FLAIR_PATH = os.path.join(os.path.dirname(__file__), "../../../../../flair/src")
FLAIR_PATH = os.path.abspath(FLAIR_PATH)


class FlairQueryEnhancer(nn.Module):
    """
    使用 FLAIR 增强 Q-Former 的 Query Tokens
    
    增强策略：
    1. bias: 将 FLAIR 特征作为偏置加到 query tokens 上
    2. attention: 让 query tokens 通过 cross-attention 关注 FLAIR 局部特征
    3. gate: 使用门控机制动态融合 FLAIR 信息
    """
    
    FLAIR_MODELS = {
        'cc3m': 'flair-cc3m-recap.pt',
        'cc12m': 'flair-cc12m-recap.pt',
        'yfcc15m': 'flair-yfcc15m-recap.pt',
        'merged30m': 'flair-merged30m.pt',
    }
    
    def __init__(
        self,
        flair_model_name: str = 'merged30m',
        qformer_hidden_size: int = 768,
        num_query_tokens: int = 32,
        enhancement_type: str = 'attention',  # 'bias', 'attention', 'gate'
        enhancement_alpha: float = 0.3,
        freeze_flair: bool = True,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.enhancement_type = enhancement_type
        self.num_query_tokens = num_query_tokens
        self.qformer_hidden_size = qformer_hidden_size
        self.freeze_flair = freeze_flair
        
        # 添加 FLAIR 到 Python 路径
        if FLAIR_PATH not in sys.path:
            sys.path.insert(0, FLAIR_PATH)
        
        import flair as flair_lib
        
        # 下载并加载 FLAIR 预训练模型
        logging.info(f"[FlairQueryEnhancer] Loading FLAIR model: {flair_model_name}")
        pretrained_path = flair_lib.download_weights_from_hf(
            model_repo='xiaorui638/flair',
            filename=self.FLAIR_MODELS[flair_model_name]
        )
        
        self.flair, _, self.preprocess = flair_lib.create_model_and_transforms(
            'ViT-B-16-FLAIR',
            pretrained=pretrained_path
        )
        self.tokenizer = flair_lib.get_tokenizer('ViT-B-16-FLAIR')
        
        self.flair_embed_dim = 512  # FLAIR 的嵌入维度
        
        # 冻结 FLAIR backbone
        if freeze_flair:
            for param in self.flair.parameters():
                param.requires_grad = False
            self.flair.eval()
            logging.info("[FlairQueryEnhancer] FLAIR backbone frozen")
        
        # FLAIR -> Q-Former 维度适配
        self.flair_to_qformer = nn.Linear(self.flair_embed_dim, qformer_hidden_size)
        
        # 根据增强类型初始化不同的模块
        if enhancement_type == 'bias':
            # 简单的偏置增强：FLAIR 全局特征 -> 偏置向量
            self.bias_proj = nn.Sequential(
                nn.Linear(qformer_hidden_size, qformer_hidden_size),
                nn.Tanh(),
            )
            # 残差权重（可学习）
            self.alpha = nn.Parameter(torch.tensor(enhancement_alpha))
            logging.info("[FlairQueryEnhancer] Using BIAS enhancement")
            
        elif enhancement_type == 'attention':
            # Cross-Attention 增强：query tokens attend to FLAIR 局部特征
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=qformer_hidden_size,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(qformer_hidden_size)
            # 可学习的残差权重，初始化较小
            self.alpha = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1) ≈ 0.27
            logging.info("[FlairQueryEnhancer] Using ATTENTION enhancement")
            
        elif enhancement_type == 'gate':
            # 门控增强：每个 query token 有独立的门控
            self.gate_proj = nn.Sequential(
                nn.Linear(qformer_hidden_size * 2, qformer_hidden_size),
                nn.ReLU(),
                nn.Linear(qformer_hidden_size, 1),
                nn.Sigmoid(),
            )
            self.value_proj = nn.Linear(qformer_hidden_size, qformer_hidden_size)
            logging.info("[FlairQueryEnhancer] Using GATE enhancement")
        
        else:
            raise ValueError(f"Unknown enhancement_type: {enhancement_type}")
        
        self._init_weights()
        
        # 统计参数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"[FlairQueryEnhancer] Total params: {total_params / 1e6:.2f}M")
        logging.info(f"[FlairQueryEnhancer] Trainable params: {trainable_params / 1e6:.2f}M")
    
    def _init_weights(self):
        """初始化新增层的权重"""
        for name, m in self.named_modules():
            if 'flair' in name:
                continue  # 跳过 FLAIR 的参数
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode_image_with_flair(self, image):
        """
        使用 FLAIR 编码图像
        
        Args:
            image: (B, C, H, W) 输入图像（需要是 FLAIR 预处理后的格式）
        
        Returns:
            global_feat: (B, 512) 全局特征
            local_feats: (B, 196, 512) 局部特征
        """
        if self.freeze_flair:
            with torch.no_grad():
                global_feat, local_feats = self.flair.encode_image(image)
                global_feat = self.flair.image_post(global_feat)
                local_feats = self.flair.image_post(local_feats)
        else:
            global_feat, local_feats = self.flair.encode_image(image)
            global_feat = self.flair.image_post(global_feat)
            local_feats = self.flair.image_post(local_feats)
        return global_feat, local_feats
    
    def encode_text_with_flair(self, text, device):
        """
        使用 FLAIR 编码文本
        
        Args:
            text: List[str] 文本列表
            device: 设备
        
        Returns:
            text_feat: (B, 512) 文本全局特征
        """
        text_tokens = self.tokenizer(text).to(device)
        if self.freeze_flair:
            with torch.no_grad():
                global_text, _ = self.flair.encode_text(text_tokens)
                global_text = self.flair.text_post(global_text)
        else:
            global_text, _ = self.flair.encode_text(text_tokens)
            global_text = self.flair.text_post(global_text)
        return global_text
    
    def forward(
        self,
        query_tokens,
        flair_image=None,
        flair_text=None,
        flair_global_feat=None,
        flair_local_feats=None,
        flair_text_feat=None,
    ):
        """
        使用 FLAIR 特征增强 Q-Former 的 query tokens
        
        Args:
            query_tokens: (B, num_query, hidden) Q-Former 的 query tokens
            flair_image: (B, C, H, W) FLAIR 预处理后的图像（可选）
            flair_text: List[str] 文本（可选）
            flair_global_feat: (B, 512) 预先提取的 FLAIR 全局特征（可选）
            flair_local_feats: (B, 196, 512) 预先提取的 FLAIR 局部特征（可选）
            flair_text_feat: (B, 512) 预先提取的 FLAIR 文本特征（可选）
        
        Returns:
            enhanced_query: (B, num_query, hidden) 增强后的 query tokens
        """
        device = query_tokens.device
        B = query_tokens.size(0)
        
        # 获取 FLAIR 特征（如果没有预先提取）
        if flair_global_feat is None or flair_local_feats is None:
            if flair_image is None:
                raise ValueError("Either flair_image or (flair_global_feat, flair_local_feats) must be provided")
            flair_global_feat, flair_local_feats = self.encode_image_with_flair(flair_image)
        
        if flair_text_feat is None and flair_text is not None:
            flair_text_feat = self.encode_text_with_flair(flair_text, device)
        
        # 将 FLAIR 特征投影到 Q-Former 空间
        flair_global_proj = self.flair_to_qformer(flair_global_feat)  # (B, hidden)
        flair_local_proj = self.flair_to_qformer(flair_local_feats)   # (B, 196, hidden)
        
        if flair_text_feat is not None:
            flair_text_proj = self.flair_to_qformer(flair_text_feat)  # (B, hidden)
        else:
            flair_text_proj = None
        
        # 根据增强类型应用不同的策略
        if self.enhancement_type == 'bias':
            return self._enhance_with_bias(query_tokens, flair_global_proj, flair_text_proj)
        elif self.enhancement_type == 'attention':
            return self._enhance_with_attention(query_tokens, flair_local_proj, flair_text_proj)
        elif self.enhancement_type == 'gate':
            return self._enhance_with_gate(query_tokens, flair_global_proj, flair_local_proj, flair_text_proj)
    
    def _enhance_with_bias(self, query_tokens, flair_global, flair_text=None):
        """
        偏置增强：FLAIR 全局特征 -> 偏置向量 -> 加到 query tokens
        
        核心思想：让 FLAIR 的全局语义信息作为"先验"指导 query tokens
        """
        # 融合图像和文本信息（如果有文本）
        if flair_text is not None:
            flair_feat = flair_global + flair_text  # 简单相加
        else:
            flair_feat = flair_global
        
        # 生成偏置
        bias = self.bias_proj(flair_feat).unsqueeze(1)  # (B, 1, hidden)
        
        # 残差增强
        enhanced_query = query_tokens + self.alpha * bias
        
        return enhanced_query
    
    def _enhance_with_attention(self, query_tokens, flair_local, flair_text=None):
        """
        Cross-Attention 增强：query tokens 通过注意力关注 FLAIR 局部特征
        
        核心思想：让每个 query token 学习关注 FLAIR 识别出的重要区域
        这样 query tokens 就能捕获文本相关的细粒度信息
        """
        # 如果有文本特征，将其加入到局部特征中作为额外的 key/value
        if flair_text is not None:
            flair_text_expanded = flair_text.unsqueeze(1)  # (B, 1, hidden)
            kv = torch.cat([flair_local, flair_text_expanded], dim=1)  # (B, 197, hidden)
        else:
            kv = flair_local  # (B, 196, hidden)
        
        # Cross-Attention: query tokens attend to FLAIR features
        attn_output, _ = self.cross_attn(
            query=query_tokens,  # (B, num_query, hidden)
            key=kv,
            value=kv,
        )  # (B, num_query, hidden)
        
        # 残差连接 + LayerNorm
        alpha = torch.sigmoid(self.alpha)  # 0~1 之间的权重
        enhanced_query = self.norm(query_tokens + alpha * attn_output)
        
        return enhanced_query
    
    def _enhance_with_gate(self, query_tokens, flair_global, flair_local, flair_text=None):
        """
        门控增强：每个 query token 有独立的门控决定融合多少 FLAIR 信息
        
        核心思想：不同的 query token 可能需要不同程度的 FLAIR 增强
        例如：与颜色相关的 query 可能需要更多 FLAIR 的颜色信息
        """
        B, N, H = query_tokens.shape
        
        # 计算每个 query token 的 FLAIR 增强值
        # 使用全局特征的 pooled 版本（可以换成 attention pooling）
        flair_pooled = flair_local.mean(dim=1)  # (B, hidden)
        
        if flair_text is not None:
            flair_pooled = flair_pooled + flair_text  # 融合文本信息
        
        flair_expanded = flair_pooled.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden)
        
        # 拼接 query 和 FLAIR 特征，计算门控
        concat_feat = torch.cat([query_tokens, flair_expanded], dim=-1)  # (B, N, hidden*2)
        gate = self.gate_proj(concat_feat)  # (B, N, 1)
        
        # 计算增强值
        enhancement = self.value_proj(flair_expanded)  # (B, N, hidden)
        
        # 门控融合
        enhanced_query = query_tokens + gate * enhancement
        
        return enhanced_query


def build_flair_query_enhancer(
    flair_model_name: str = 'merged30m',
    qformer_hidden_size: int = 768,
    num_query_tokens: int = 32,
    enhancement_type: str = 'attention',
    enhancement_alpha: float = 0.3,
    freeze_flair: bool = True,
    num_attention_heads: int = 8,
    dropout: float = 0.1,
):
    """工厂函数：创建 FlairQueryEnhancer"""
    return FlairQueryEnhancer(
        flair_model_name=flair_model_name,
        qformer_hidden_size=qformer_hidden_size,
        num_query_tokens=num_query_tokens,
        enhancement_type=enhancement_type,
        enhancement_alpha=enhancement_alpha,
        freeze_flair=freeze_flair,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
    )

