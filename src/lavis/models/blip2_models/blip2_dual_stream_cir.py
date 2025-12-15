"""
双流架构 CIR 模型
结合语义理解和视觉保持，处理抽象操作指令

特点:
1. 流 1 (语义理解流): 使用 Q-Former 多模态融合 (图像+Prompt文本)
2. 流 2 (视觉保持流): 使用 Q-Former 提取纯视觉特征
3. 语义引导融合模块: 将语义特征引导视觉特征的提取
4. 所有新增模块可配置开关，方便消融实验

消融实验配置:
- use_semantic_stream: 是否使用语义理解流
- use_fusion_module: 是否使用语义引导融合
- use_dual_level_fusion: 是否使用双层特征融合
- semantic_model_type: 语义模型类型 ('qformer_native', 'lightweight', 'qwen2_vl')
- fusion_type: 融合类型 ('cross_attention', 'gated', 'concat', 'add')
"""
import math
import logging
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures


# ==========================================
# Qwen2.5-VL 语义理解模块
# ==========================================
class Qwen2VLSemanticModule(nn.Module):
    """
    使用 Qwen2.5-VL 进行语义理解
    
    Qwen2.5-VL 是一个强大的多模态视觉语言模型，
    能够理解图像内容并根据指令生成语义表示。
    """
    
    # 支持的 Qwen2-VL 模型
    QWEN2_VL_MODELS = {
        "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    }
    
    def __init__(
        self,
        model_name: str = "qwen2.5-vl-3b",
        output_dim: int = 768,
        freeze_vlm: bool = True,
        use_image_input: bool = True,
        max_length: int = 128,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.use_image_input = use_image_input
        self.max_length = max_length
        
        # 延迟加载以避免未安装时报错
        self.vlm = None
        self.processor = None
        self.vlm_loaded = False
        self.freeze_vlm = freeze_vlm
        
        # 获取模型路径
        if model_name in self.QWEN2_VL_MODELS:
            self.model_path = self.QWEN2_VL_MODELS[model_name]
        else:
            self.model_path = model_name  # 允许直接传入路径
        
        logging.info(f"Qwen2VLSemanticModule initialized with model: {self.model_path}")
        logging.info(f"VLM will be loaded on first forward pass (lazy loading)")
    
    def _load_vlm(self, device):
        """延迟加载 VLM 模型"""
        if self.vlm_loaded:
            return
        
        try:
            import transformers
            transformers_version = transformers.__version__
            logging.info(f"Transformers version: {transformers_version}")
            
            # 根据模型类型选择加载方式
            is_qwen25 = "qwen2.5" in self.model_path.lower()
            
            if is_qwen25:
                # Qwen2.5-VL 需要 transformers >= 4.45.0
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
                    model_class = Qwen2_5_VLForConditionalGeneration
                    processor_class = Qwen2_5_VLProcessor
                    logging.info("Using Qwen2.5-VL specific classes")
                except ImportError:
                    try:
                        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
                        model_class = Qwen2VLForConditionalGeneration
                        processor_class = Qwen2VLProcessor
                        logging.info("Using Qwen2-VL classes for Qwen2.5-VL")
                    except ImportError:
                        from transformers import AutoModelForVision2Seq, AutoTokenizer
                        model_class = AutoModelForVision2Seq
                        processor_class = None  # 会使用 AutoTokenizer
                        logging.info("Using AutoModelForVision2Seq (fallback)")
            else:
                # Qwen2-VL
                try:
                    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
                    model_class = Qwen2VLForConditionalGeneration
                    processor_class = Qwen2VLProcessor
                    logging.info("Using Qwen2-VL specific classes")
                except ImportError:
                    from transformers import AutoModelForVision2Seq, AutoTokenizer
                    model_class = AutoModelForVision2Seq
                    processor_class = None
                    logging.info("Using AutoModelForVision2Seq (fallback)")
            
            logging.info(f"Loading Qwen VLM model: {self.model_path}")
            
            # 加载处理器
            if processor_class is not None:
                try:
                    self.processor = processor_class.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                    )
                except Exception as e:
                    logging.warning(f"Failed to load processor with specific class: {e}")
                    # 回退方案：分别加载 tokenizer 和 image_processor
                    from transformers import AutoTokenizer, AutoImageProcessor
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                    )
                    try:
                        self.image_processor = AutoImageProcessor.from_pretrained(
                            self.model_path,
                            trust_remote_code=True,
                        )
                    except:
                        self.image_processor = None
                        logging.warning("Image processor not available, text-only mode")
                    self.processor = None
            else:
                # 回退方案
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
                self.processor = None
            
            # 加载模型
            self.vlm = model_class.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            
            # 冻结 VLM 参数
            if self.freeze_vlm:
                for param in self.vlm.parameters():
                    param.requires_grad = False
                self.vlm.eval()
                logging.info("Qwen VLM model frozen")
            
            # 创建投影层 (VLM hidden size -> output_dim)
            vlm_hidden_size = self.vlm.config.hidden_size
            self.semantic_projector = nn.Sequential(
                nn.Linear(vlm_hidden_size, self.output_dim * 2),
                nn.GELU(),
                nn.Linear(self.output_dim * 2, self.output_dim),
                nn.LayerNorm(self.output_dim),
            ).to(device)
            
            self.vlm_loaded = True
            logging.info(f"Qwen VLM loaded successfully. Hidden size: {vlm_hidden_size}")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import model class: {e}. "
                "Please install: pip install transformers>=4.45.0 qwen-vl-utils"
            )
        except Exception as e:
            logging.error(f"Failed to load Qwen VLM: {e}")
            raise
    
    def forward(
        self,
        images: Optional[torch.Tensor],
        instructions: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        使用 Qwen2-VL 提取语义特征
        
        Args:
            images: 参考图像 (B, C, H, W) 或 None
            instructions: 修改指令文本列表
            device: 设备
        Returns:
            semantic_features: (B, output_dim)
        """
        # 延迟加载
        self._load_vlm(device)
        
        batch_size = len(instructions)
        
        # CIR 任务专用 prompt - 更简洁，专注于修改意图
        prompt_template = (
            "Image modification instruction: {instruction}\n"
            "What visual changes should be made?"
        )
        
        # 处理每个样本
        semantic_features_list = []
        
        with torch.no_grad():
            for i, instruction in enumerate(instructions):
                prompt = prompt_template.format(instruction=instruction)
                
                # 根据是否有 processor 选择不同处理方式
                if self.processor is not None:
                    # 使用完整的 processor
                    try:
                        # 准备输入 - 使用消息格式
                        if self.use_image_input and images is not None:
                            # 将 tensor 转换为 PIL Image
                            img = images[i]
                            if img.dim() == 3:
                                img = img.cpu()
                                # 反归一化 (假设 ImageNet 标准化)
                                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                                img = img * std + mean
                                img = img.clamp(0, 1)
                                # 转换为 PIL
                                from PIL import Image
                                img_pil = Image.fromarray(
                                    (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
                                )
                                
                                messages = [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "image", "image": img_pil},
                                            {"type": "text", "text": prompt},
                                        ],
                                    }
                                ]
                        else:
                            img_pil = None
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                    ],
                                }
                            ]
                        
                        # 处理输入
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        
                        if img_pil is not None:
                            inputs = self.processor(
                                text=[text],
                                images=[img_pil],
                                padding=True,
                                return_tensors="pt",
                            ).to(device)
                        else:
                            inputs = self.processor(
                                text=[text],
                                padding=True,
                                return_tensors="pt",
                            ).to(device)
                            
                    except Exception as e:
                        logging.warning(f"Processor failed, using text-only: {e}")
                        # 回退到纯文本
                        inputs = self.processor(
                            text=[prompt],
                            padding=True,
                            return_tensors="pt",
                        ).to(device)
                else:
                    # 使用 tokenizer (回退方案)
                    inputs = self.tokenizer(
                        prompt,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(device)
                
                # 获取 hidden states
                outputs = self.vlm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                # 取最后一层的 hidden states
                last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_size)
                
                # 池化: 取最后一个 token 或平均池化
                pooled = last_hidden[:, -1, :]  # (1, hidden_size)
                
                semantic_features_list.append(pooled)
        
        # 拼接所有样本
        semantic_features = torch.cat(semantic_features_list, dim=0)  # (B, hidden_size)
        
        # 投影到目标维度
        semantic_features = self.semantic_projector(semantic_features.float())
        
        return semantic_features


# ==========================================
# BLIP2 生成式语义理解模块 (使用 T5 生成展开文本)
# ==========================================
class Blip2GenerativeSemanticModule(nn.Module):
    """
    使用 BLIP2 + FlanT5 生成能力来展开抽象指令
    
    工作流程:
    1. 构造提示词，让 T5 理解抽象指令
    2. 使用 T5 生成展开的视觉描述
    3. 编码展开后的文本作为语义特征
    
    优点:
    - 能将 "make it casual" 展开为 "change to relaxed style, casual fabric..."
    - 利用 T5 的语言理解能力
    
    缺点:
    - 需要额外的 T5 模型 (~250M for base, ~780M for large)
    - 推理时有生成延迟
    """
    
    # 支持的 T5 模型
    T5_MODELS = {
        "flan-t5-small": "google/flan-t5-small",   # 80M
        "flan-t5-base": "google/flan-t5-base",     # 250M (推荐)
        "flan-t5-large": "google/flan-t5-large",   # 780M
        "flan-t5-xl": "google/flan-t5-xl",         # 3B
    }
    
    def __init__(
        self,
        t5_model_name: str = "flan-t5-base",
        output_dim: int = 768,
        max_gen_length: int = 64,
        freeze_t5: bool = True,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.max_gen_length = max_gen_length
        self.freeze_t5 = freeze_t5
        
        # 获取模型路径
        if t5_model_name in self.T5_MODELS:
            self.model_path = self.T5_MODELS[t5_model_name]
        else:
            self.model_path = t5_model_name
        
        # 延迟加载
        self.t5_model = None
        self.t5_tokenizer = None
        self.t5_loaded = False
        
        # 输出投影层 (T5 hidden -> output_dim)
        self.output_proj = None  # 延迟创建
        
        # 提示词模板
        self.prompt_template = (
            "Instruction: '{instruction}'\n"
            "Describe what visual changes this instruction implies for an image. "
            "Focus on specific visual attributes like color, style, shape, texture, and composition. "
            "Answer:"
        )
        
        logging.info(f"Blip2GenerativeSemanticModule initialized: "
                     f"t5_model={t5_model_name}, freeze={freeze_t5}")
    
    def _load_t5(self, device):
        """延迟加载 T5 模型"""
        if self.t5_loaded:
            return
        
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            logging.info(f"Loading T5 model: {self.model_path}")
            
            self.t5_tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
            ).to(device)
            
            if self.freeze_t5:
                for param in self.t5_model.parameters():
                    param.requires_grad = False
                self.t5_model.eval()
                logging.info("T5 model frozen")
            
            # 创建输出投影层
            t5_hidden = self.t5_model.config.d_model
            self.output_proj = nn.Sequential(
                nn.Linear(t5_hidden, self.output_dim),
                nn.LayerNorm(self.output_dim),
            ).to(device)
            
            self.t5_loaded = True
            logging.info(f"T5 loaded successfully. Hidden size: {t5_hidden}")
            
        except Exception as e:
            logging.error(f"Failed to load T5: {e}")
            raise
    
    def generate_expansion(
        self,
        instructions: List[str],
        device: torch.device,
    ) -> List[str]:
        """
        使用 T5 生成指令展开文本
        
        Args:
            instructions: 原始指令列表
        Returns:
            expanded_texts: 展开后的文本列表
        """
        self._load_t5(device)
        
        # 构造提示词
        prompts = [
            self.prompt_template.format(instruction=inst)
            for inst in instructions
        ]
        
        # 编码
        inputs = self.t5_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        # 生成
        with torch.no_grad():
            outputs = self.t5_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=self.max_gen_length,
                num_beams=1,  # greedy for speed
                do_sample=False,
            )
        
        # 解码
        expanded_texts = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        
        return expanded_texts
    
    def forward(
        self,
        instructions: List[str],
        text_encoder,  # Q-Former 的 BERT
        tokenizer,     # BERT tokenizer
        device: torch.device,
        max_txt_len: int = 64,
    ) -> torch.Tensor:
        """
        生成展开文本并编码为语义特征
        
        Args:
            instructions: 原始指令列表
            text_encoder: Q-Former 的 BERT 编码器
            tokenizer: BERT tokenizer
            device: 设备
        Returns:
            semantic_features: (B, output_dim)
        """
        # Step 1: 生成展开文本
        expanded_texts = self.generate_expansion(instructions, device)
        
        # 打印展开结果 (调试用)
        if len(expanded_texts) > 0:
            logging.debug(f"Instruction: {instructions[0]}")
            logging.debug(f"Expanded: {expanded_texts[0]}")
        
        # Step 2: 用 BERT 编码展开后的文本
        text_tokens = tokenizer(
            expanded_texts,
            padding="max_length",
            truncation=True,
            max_length=max_txt_len,
            return_tensors="pt",
        ).to(device)
        
        text_output = text_encoder(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        
        # Step 3: 池化
        # 使用 [CLS] token 或平均池化
        text_features = text_output.last_hidden_state
        mask = text_tokens.attention_mask.unsqueeze(-1).float()
        pooled = (text_features * mask).sum(1) / mask.sum(1).clamp(min=1)
        
        # Step 4: 投影到目标维度
        semantic_features = self.output_proj(pooled.float())
        
        return semantic_features


# ==========================================
# BLIP2 语义理解模块 (轻量级替代方案)
# ==========================================
class Blip2SemanticModule(nn.Module):
    """
    基于 BLIP2 Q-Former 结构的语义理解模块
    
    比 lightweight 模式更强大，但比 Qwen2-VL 更轻量
    使用可学习的 semantic query tokens 从文本中提取语义特征
    可选地结合图像信息进行多模态语义理解
    
    特点:
    1. 使用 BERT 编码文本指令
    2. 使用可学习的 query tokens 提取语义特征 (类似 FLAIR 思想)
    3. 可选地使用图像特征辅助语义理解
    """
    
    def __init__(
        self,
        text_encoder_config,  # BertConfig 或类似配置
        num_semantic_tokens: int = 8,
        output_dim: int = 768,
        use_image_guidance: bool = True,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_semantic_tokens = num_semantic_tokens
        self.output_dim = output_dim
        self.use_image_guidance = use_image_guidance
        self.hidden_dim = text_encoder_config.hidden_size
        
        # 可学习的语义 query tokens
        self.semantic_query = nn.Parameter(
            torch.zeros(1, num_semantic_tokens, self.hidden_dim)
        )
        self.semantic_query.data.normal_(mean=0.0, std=0.02)
        
        # 文本条件化 attention (类似 FLAIR)
        # query: semantic_query, key/value: text_features
        self.text_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln_text = nn.LayerNorm(self.hidden_dim)
        self.ln_query = nn.LayerNorm(self.hidden_dim)
        
        # 图像引导 attention (可选)
        if use_image_guidance:
            self.image_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.ln_image = nn.LayerNorm(self.hidden_dim)
            self.ln_query_img = nn.LayerNorm(self.hidden_dim)
            
            # 图像特征投影 (如果维度不匹配)
            self.image_proj = None  # 将在 forward 时动态创建
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Dropout(dropout),
        )
        self.ln_ffn = nn.LayerNorm(self.hidden_dim)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        logging.info(f"Blip2SemanticModule initialized: "
                     f"num_tokens={num_semantic_tokens}, "
                     f"output_dim={output_dim}, "
                     f"use_image_guidance={use_image_guidance}")
    
    def forward(
        self,
        text_features: torch.Tensor,  # (B, L, hidden_dim)
        text_mask: torch.Tensor,      # (B, L)
        image_embeds: Optional[torch.Tensor] = None,  # (B, N, vit_dim)
    ) -> torch.Tensor:
        """
        从文本指令中提取语义特征
        
        Args:
            text_features: BERT 编码的文本特征
            text_mask: 文本 attention mask
            image_embeds: 可选的图像特征 (用于多模态语义理解)
        Returns:
            semantic_features: (B, num_semantic_tokens, output_dim) 或池化后 (B, output_dim)
        """
        B = text_features.shape[0]
        device = text_features.device
        
        # 扩展 semantic query
        query = self.semantic_query.expand(B, -1, -1)  # (B, num_tokens, hidden_dim)
        
        # Step 1: 文本条件化 attention
        # semantic_query attend to text_features
        query_norm = self.ln_query(query)
        text_norm = self.ln_text(text_features)
        
        # attention mask 处理
        key_padding_mask = ~text_mask.bool() if text_mask is not None else None
        
        attn_out, _ = self.text_attn(
            query=query_norm,
            key=text_norm,
            value=text_norm,
            key_padding_mask=key_padding_mask
        )
        query = query + attn_out  # 残差连接
        
        # Step 2: 图像引导 attention (可选)
        if self.use_image_guidance and image_embeds is not None:
            # 动态创建投影层 (如果需要)
            if self.image_proj is None and image_embeds.shape[-1] != self.hidden_dim:
                self.image_proj = nn.Linear(
                    image_embeds.shape[-1], self.hidden_dim
                ).to(device)
            
            # 投影图像特征
            if self.image_proj is not None:
                image_features = self.image_proj(image_embeds)
            else:
                image_features = image_embeds
            
            query_norm = self.ln_query_img(query)
            image_norm = self.ln_image(image_features)
            
            attn_out, _ = self.image_attn(
                query=query_norm,
                key=image_norm,
                value=image_norm
            )
            query = query + attn_out  # 残差连接
        
        # Step 3: FFN
        query = query + self.ffn(self.ln_ffn(query))
        
        # Step 4: 输出投影
        semantic_features = self.output_proj(query)  # (B, num_tokens, output_dim)
        
        # 池化为单一向量 (可选，根据下游需求)
        # 这里返回所有 tokens，让下游决定如何使用
        semantic_features_pooled = semantic_features.mean(dim=1)  # (B, output_dim)
        
        return semantic_features_pooled


# ==========================================
# 辅助模块: 语义引导融合
# ==========================================
class SemanticGuidedFusion(nn.Module):
    """
    语义引导的特征融合模块
    使用语义特征作为 query，引导视觉特征的聚合
    
    支持多种融合方式 (用于消融实验):
    - cross_attention: 标准 cross-attention
    - gated: 门控融合
    - concat: 拼接后投影
    - add: 直接相加
    """
    
    def __init__(
        self,
        visual_dim: int = 768,
        semantic_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_type: str = 'cross_attention',
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        
        # 对齐维度
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)
        
        if fusion_type == 'cross_attention':
            # Cross-Attention 融合
            self.cross_attn_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(num_layers)
            ])
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers * 2)
            ])
            self.ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ])
            
        elif fusion_type == 'gated':
            # 门控融合 (受 CAMS 启发)
            self.gate_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.output_proj = nn.Linear(hidden_dim, hidden_dim)
            
        elif fusion_type == 'concat':
            # 拼接融合
            self.concat_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            
        elif fusion_type == 'add':
            # 简单相加 (baseline)
            self.output_norm = nn.LayerNorm(hidden_dim)
            
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        # 输出投影
        self.output_proj_final = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        visual_features: torch.Tensor,    # (B, N, visual_dim)
        semantic_features: torch.Tensor,  # (B, semantic_dim) or (B, M, semantic_dim)
    ) -> torch.Tensor:
        """
        Args:
            visual_features: 视觉特征 (B, N, visual_dim)
            semantic_features: 语义特征 (B, semantic_dim) 或 (B, M, semantic_dim)
        Returns:
            fused_features: 融合后的特征 (B, N, hidden_dim)
        """
        B, N, _ = visual_features.shape
        
        # 投影到统一空间
        visual_feat = self.visual_proj(visual_features)  # (B, N, hidden_dim)
        
        # 处理语义特征维度
        if semantic_features.dim() == 2:
            semantic_feat = self.semantic_proj(semantic_features)  # (B, hidden_dim)
            semantic_feat = semantic_feat.unsqueeze(1)  # (B, 1, hidden_dim)
        else:
            semantic_feat = self.semantic_proj(semantic_features)  # (B, M, hidden_dim)
        
        if self.fusion_type == 'cross_attention':
            # 语义特征作为 query，视觉特征作为 key/value
            # 这样语义特征可以选择性地从视觉特征中提取信息
            fused = visual_feat
            for i, (attn, ffn) in enumerate(zip(self.cross_attn_layers, self.ffns)):
                # Cross-attention: 用语义引导视觉特征
                # 这里我们让视觉特征 attend to 语义特征
                attn_out, _ = attn(
                    query=fused,
                    key=semantic_feat.expand(-1, fused.shape[1], -1) if semantic_feat.shape[1] == 1 else semantic_feat,
                    value=semantic_feat.expand(-1, fused.shape[1], -1) if semantic_feat.shape[1] == 1 else semantic_feat,
                )
                fused = self.layer_norms[i * 2](fused + attn_out)
                fused = self.layer_norms[i * 2 + 1](fused + ffn(fused))
            
        elif self.fusion_type == 'gated':
            # 门控融合
            semantic_expanded = semantic_feat.expand(-1, N, -1)  # (B, N, hidden_dim)
            gate = self.gate_proj(torch.cat([visual_feat, semantic_expanded], dim=-1))
            fused = gate * visual_feat + (1 - gate) * semantic_expanded
            fused = self.output_proj(fused)
            
        elif self.fusion_type == 'concat':
            # 拼接融合
            semantic_expanded = semantic_feat.expand(-1, N, -1)
            fused = self.concat_proj(torch.cat([visual_feat, semantic_expanded], dim=-1))
            
        elif self.fusion_type == 'add':
            # 简单相加
            semantic_expanded = semantic_feat.expand(-1, N, -1)
            fused = self.output_norm(visual_feat + semantic_expanded)
        
        fused = self.output_proj_final(fused)
        return fused


# ==========================================
# 辅助模块: 指令展开模块
# ==========================================
class InstructionExpansionModule(nn.Module):
    """
    指令展开模块
    将抽象指令展开为具体的视觉语义描述
    
    实现方式: 使用可学习的 prompt + cross-attention
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        num_expansion_tokens: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 可学习的展开 tokens
        self.expansion_tokens = nn.Parameter(
            torch.zeros(1, num_expansion_tokens, text_dim)
        )
        self.expansion_tokens.data.normal_(mean=0.0, std=0.02)
        
        # 用于理解指令的 cross-attention
        self.instruction_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
        )
    
    def forward(
        self,
        instruction_features: torch.Tensor,  # (B, L, text_dim)
        instruction_mask: Optional[torch.Tensor] = None,  # (B, L)
    ) -> torch.Tensor:
        """
        展开指令特征
        
        Args:
            instruction_features: 原始指令特征
            instruction_mask: attention mask
        Returns:
            expanded_features: 展开后的语义特征 (B, num_expansion_tokens, text_dim)
        """
        B = instruction_features.shape[0]
        
        # 扩展可学习 tokens
        expansion_tokens = self.expansion_tokens.expand(B, -1, -1)
        
        # Cross-attention: expansion_tokens attend to instruction
        key_padding_mask = ~instruction_mask.bool() if instruction_mask is not None else None
        
        expanded, _ = self.instruction_attn(
            query=expansion_tokens,
            key=instruction_features,
            value=instruction_features,
            key_padding_mask=key_padding_mask,
        )
        
        # 投影
        expanded = self.output_proj(expanded)
        
        return expanded


# ==========================================
# 辅助模块: 查询生成模块
# ==========================================
class QueryGenerationModule(nn.Module):
    """
    查询生成模块
    将融合后的特征转换为最终的检索查询向量
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 256,
        num_query_tokens: int = 32,
        aggregation: str = 'attention',  # 'attention', 'mean', 'first'
    ):
        super().__init__()
        self.aggregation = aggregation
        self.num_query_tokens = num_query_tokens
        
        if aggregation == 'attention':
            # 可学习的聚合权重
            self.query_weight = nn.Parameter(torch.zeros(1, num_query_tokens, 1))
            self.query_weight.data.normal_(mean=0.0, std=0.02)
            
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_features: (B, N, input_dim)
        Returns:
            query_features: (B, output_dim) or (B, N, output_dim)
        """
        if self.aggregation == 'attention':
            # 加权聚合
            weights = F.softmax(self.query_weight, dim=1)
            pooled = (fused_features[:, :self.num_query_tokens, :] * weights).sum(dim=1)
            return F.normalize(self.output_proj(pooled), dim=-1)
            
        elif self.aggregation == 'mean':
            pooled = fused_features[:, :self.num_query_tokens, :].mean(dim=1)
            return F.normalize(self.output_proj(pooled), dim=-1)
            
        elif self.aggregation == 'first':
            pooled = fused_features[:, 0, :]
            return F.normalize(self.output_proj(pooled), dim=-1)
            
        else:
            # 返回所有 tokens 的特征
            return F.normalize(self.output_proj(fused_features), dim=-1)


# ==========================================
# 双层特征融合模块 (借鉴 Qwen-Image VAE 思想)
# ==========================================
class DualLevelFeatureFusion(nn.Module):
    """
    双层特征融合模块 (不修改预训练模型)
    
    借鉴 Qwen-Image 的 VAE 设计思想:
    - 高层语义特征: 理解修改意图，与语义指令交互
    - 低层视觉特征: 保持视觉一致性，增强局部关联
    
    关键设计:
    1. 从 Q-Former 输出通过两个不同的投影分支学习分离
    2. 高层分支: 与语义特征进行 Cross-Attention (理解"改什么")
    3. 低层分支: 通过 Self-Attention 增强视觉一致性 (理解"保留什么")
    4. 可选的 Change/Preserve 区域检测
    5. 自适应权重融合
    
    所有模块都是新增的，不修改预训练的 ViT 和 Q-Former
    """
    
    def __init__(
        self,
        visual_dim: int = 768,           # Q-Former 输出维度
        semantic_dim: int = 768,         # 语义特征维度
        hidden_dim: int = 768,           # 内部处理维度
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        # 消融实验开关
        use_spatial_align: bool = True,       # 是否使用空间感知对齐
        use_change_preserve: bool = True,     # 是否使用 Change/Preserve 分离
        adaptive_fusion: bool = True,         # 是否使用自适应权重融合
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.semantic_dim = semantic_dim
        self.hidden_dim = hidden_dim
        self.use_spatial_align = use_spatial_align
        self.use_change_preserve = use_change_preserve
        self.adaptive_fusion = adaptive_fusion
        
        # ==========================================
        # 双层投影: 从同一输入学习不同层次的表示
        # ==========================================
        
        # 高层语义投影 (关注全局语义变化)
        self.high_level_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 低层视觉投影 (关注局部视觉细节)
        self.low_level_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 语义特征投影
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)
        
        # ==========================================
        # 高层分支: 语义交互 (Cross-Attention with semantic)
        # ==========================================
        self.high_cross_attn_layers = nn.ModuleList()
        self.high_ln_layers = nn.ModuleList()
        self.high_ffn_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.high_cross_attn_layers.append(
                SpatialAwareCrossAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_2d_pos_enc=use_spatial_align,  # 使用 2D 位置编码实现空间感知
                    learnable_pos=True,
                ) if use_spatial_align else nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
            self.high_ln_layers.extend([
                nn.LayerNorm(hidden_dim),
                nn.LayerNorm(hidden_dim),
            ])
            self.high_ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ))
        
        # ==========================================
        # 低层分支: 视觉增强 (Self-Attention)
        # ==========================================
        self.low_self_attn_layers = nn.ModuleList()
        self.low_ln_layers = nn.ModuleList()
        self.low_ffn_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.low_self_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
            self.low_ln_layers.extend([
                nn.LayerNorm(hidden_dim),
                nn.LayerNorm(hidden_dim),
            ])
            self.low_ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ))
        
        # ==========================================
        # Change/Preserve 区域检测器 (可选)
        # ==========================================
        if use_change_preserve:
            self.change_preserve_detector = ChangePreserveDetector(
                hidden_dim=hidden_dim,
                semantic_dim=hidden_dim,
            )
        
        # ==========================================
        # 自适应融合权重
        # ==========================================
        if adaptive_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            )
        else:
            # 固定权重 (高层 0.6, 低层 0.4)
            self.register_buffer('fixed_weights', torch.tensor([0.6, 0.4]))
        
        # ==========================================
        # 输出投影
        # ==========================================
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, visual_dim),
            nn.LayerNorm(visual_dim),
        )
        
        logging.info(f"DualLevelFeatureFusion initialized:")
        logging.info(f"  - visual_dim: {visual_dim}, semantic_dim: {semantic_dim}")
        logging.info(f"  - use_spatial_align: {use_spatial_align}")
        logging.info(f"  - use_change_preserve: {use_change_preserve}")
        logging.info(f"  - adaptive_fusion: {adaptive_fusion}")
    
    def forward(
        self,
        visual_features: torch.Tensor,      # (B, N, visual_dim) - Q-Former 输出
        semantic_features: torch.Tensor,    # (B, semantic_dim) 或 (B, M, semantic_dim)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        双层特征融合
        
        通过 2D 位置编码实现空间感知，不需要显式的空间词汇表
        模型自动学习空间对应关系
        
        Args:
            visual_features: Q-Former 输出的视觉特征 (B, N, visual_dim)
            semantic_features: 语义理解模块输出的特征
        
        Returns:
            fused_features: 融合后的特征 (B, N, visual_dim)
            aux_outputs: 辅助输出 (用于可视化和损失计算)
        """
        B, N, _ = visual_features.shape
        device = visual_features.device
        
        # ==========================================
        # Step 1: 双层投影
        # ==========================================
        high_level_feat = self.high_level_proj(visual_features)  # (B, N, hidden_dim)
        low_level_feat = self.low_level_proj(visual_features)    # (B, N, hidden_dim)
        
        # ==========================================
        # Step 2: 处理语义特征
        # ==========================================
        if semantic_features.dim() == 2:
            semantic_feat = self.semantic_proj(semantic_features).unsqueeze(1)  # (B, 1, hidden_dim)
        else:
            semantic_feat = self.semantic_proj(semantic_features)  # (B, M, hidden_dim)
        
        # ==========================================
        # Step 3: 高层分支 - 语义交互 (带 2D 空间位置编码)
        # ==========================================
        for i, (attn, ffn) in enumerate(zip(self.high_cross_attn_layers, self.high_ffn_layers)):
            ln1 = self.high_ln_layers[i * 2]
            ln2 = self.high_ln_layers[i * 2 + 1]
            
            # Cross-Attention: 高层视觉 attend to 语义
            # 使用 2D 位置编码实现空间感知，无需显式词汇表
            if self.use_spatial_align and isinstance(attn, SpatialAwareCrossAttention):
                attn_out, attn_weights = attn(
                    query=high_level_feat,
                    key=semantic_feat,
                    value=semantic_feat,
                    num_patches=N,  # 用于计算网格大小
                )
            else:
                # 标准 MultiheadAttention
                # 扩展语义特征以匹配 key/value 维度
                semantic_expanded = semantic_feat.expand(-1, N, -1) if semantic_feat.shape[1] == 1 else semantic_feat
                attn_out, attn_weights = attn(
                    query=high_level_feat,
                    key=semantic_expanded,
                    value=semantic_expanded,
                )
            
            high_level_feat = ln1(high_level_feat + attn_out)
            high_level_feat = ln2(high_level_feat + ffn(high_level_feat))
        
        # ==========================================
        # Step 4: 低层分支 - 视觉增强
        # ==========================================
        for i, (attn, ffn) in enumerate(zip(self.low_self_attn_layers, self.low_ffn_layers)):
            ln1 = self.low_ln_layers[i * 2]
            ln2 = self.low_ln_layers[i * 2 + 1]
            
            # Self-Attention: 增强局部视觉关联
            attn_out, _ = attn(
                query=low_level_feat,
                key=low_level_feat,
                value=low_level_feat,
            )
            
            low_level_feat = ln1(low_level_feat + attn_out)
            low_level_feat = ln2(low_level_feat + ffn(low_level_feat))
        
        # ==========================================
        # Step 5: Change/Preserve 调制 (可选)
        # ==========================================
        change_mask, preserve_mask = None, None
        if self.use_change_preserve:
            change_mask, preserve_mask = self.change_preserve_detector(
                high_level_feat, semantic_feat
            )
            # 调制特征
            high_level_feat = high_level_feat * change_mask
            low_level_feat = low_level_feat * preserve_mask
        
        # ==========================================
        # Step 6: 自适应融合
        # ==========================================
        if self.adaptive_fusion:
            # 从语义特征预测融合权重
            semantic_pooled = semantic_feat.mean(dim=1)  # (B, hidden_dim)
            high_pooled = high_level_feat.mean(dim=1)    # (B, hidden_dim)
            
            gate_input = torch.cat([semantic_pooled, high_pooled], dim=-1)
            weights = self.fusion_gate(gate_input)  # (B, 2)
            
            weight_high = weights[:, 0:1].unsqueeze(-1)  # (B, 1, 1)
            weight_low = weights[:, 1:2].unsqueeze(-1)   # (B, 1, 1)
        else:
            weight_high = self.fixed_weights[0]
            weight_low = self.fixed_weights[1]
            weights = self.fixed_weights
        
        # 加权融合
        fused_features = weight_high * high_level_feat + weight_low * low_level_feat
        
        # ==========================================
        # Step 7: 输出投影 (恢复到 visual_dim)
        # ==========================================
        fused_features = self.output_proj(fused_features)
        
        # 辅助输出
        aux_outputs = {
            'high_level_feat': high_level_feat,
            'low_level_feat': low_level_feat,
            'fusion_weights': weights,
            'change_mask': change_mask,
            'preserve_mask': preserve_mask,
        }
        
        return fused_features, aux_outputs


class Spatial2DPositionEncoding(nn.Module):
    """
    2D 空间位置编码
    
    借鉴 Qwen2-VL 的 M-RoPE 思想:
    使用 2D 位置编码让模型自动学习空间对应关系
    不需要显式的词汇表
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        max_grid_size: int = 32,
        learnable: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_grid_size = max_grid_size
        self.learnable = learnable
        
        if learnable:
            # 可学习的 2D 位置编码
            self.row_embed = nn.Embedding(max_grid_size, embed_dim // 2)
            self.col_embed = nn.Embedding(max_grid_size, embed_dim // 2)
        else:
            # 固定的 sin/cos 位置编码
            self.register_buffer('pos_encoding', self._create_sincos_encoding())
    
    def _create_sincos_encoding(self) -> torch.Tensor:
        """创建 2D sin/cos 位置编码"""
        d_model = self.embed_dim
        max_len = self.max_grid_size
        
        pe = torch.zeros(max_len, max_len, d_model)
        
        # 行方向编码 (使用前半维度)
        for i in range(max_len):
            for j in range(d_model // 4):
                pe[i, :, 2*j] = math.sin(i / (10000 ** (2*j / (d_model // 2))))
                pe[i, :, 2*j+1] = math.cos(i / (10000 ** (2*j / (d_model // 2))))
        
        # 列方向编码 (使用后半维度)
        for j in range(max_len):
            for k in range(d_model // 4):
                pe[:, j, d_model//2 + 2*k] = math.sin(j / (10000 ** (2*k / (d_model // 2))))
                pe[:, j, d_model//2 + 2*k+1] = math.cos(j / (10000 ** (2*k / (d_model // 2))))
        
        return pe
    
    def forward(self, num_patches: int, device: torch.device) -> torch.Tensor:
        """
        生成 2D 位置编码
        
        Args:
            num_patches: patch 数量 (假设是 H*W)
            device: 设备
        
        Returns:
            pos_encoding: (num_patches, embed_dim)
        """
        # 计算网格大小
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            grid_size = int(math.ceil(math.sqrt(num_patches)))
        
        if self.learnable:
            # 生成行列索引
            rows = torch.arange(grid_size, device=device)
            cols = torch.arange(grid_size, device=device)
            
            # 获取编码
            row_enc = self.row_embed(rows)  # (grid_size, embed_dim/2)
            col_enc = self.col_embed(cols)  # (grid_size, embed_dim/2)
            
            # 组合为 2D 编码
            # 每个位置 (i, j) 的编码 = [row_enc[i], col_enc[j]]
            pos_enc = torch.zeros(grid_size, grid_size, self.embed_dim, device=device)
            for i in range(grid_size):
                for j in range(grid_size):
                    pos_enc[i, j, :self.embed_dim//2] = row_enc[i]
                    pos_enc[i, j, self.embed_dim//2:] = col_enc[j]
            
            # 展平
            pos_enc = pos_enc.view(-1, self.embed_dim)[:num_patches]
        else:
            pos_enc = self.pos_encoding[:grid_size, :grid_size].reshape(-1, self.embed_dim)[:num_patches]
            pos_enc = pos_enc.to(device)
        
        return pos_enc


class SpatialAwareCrossAttention(nn.Module):
    """
    空间感知的 Cross-Attention (基于 2D 位置编码)
    
    借鉴 Qwen2-VL 的 M-RoPE 思想:
    通过 2D 位置编码让模型自动学习空间对应关系
    不需要显式的空间词汇表
    
    核心思想:
    - 为图像 patches 添加 2D 位置编码 (row, col)
    - 模型通过训练自动学习 "左边" → 左侧 patches 的对应
    - 端到端学习，无需手动规则
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_2d_pos_enc: bool = True,
        learnable_pos: bool = True,
        max_grid_size: int = 32,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_2d_pos_enc = use_2d_pos_enc
        
        # QKV 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        
        # 2D 空间位置编码
        if use_2d_pos_enc:
            self.pos_encoding = Spatial2DPositionEncoding(
                embed_dim=embed_dim,
                max_grid_size=max_grid_size,
                learnable=learnable_pos,
            )
    
    def forward(
        self,
        query: torch.Tensor,      # (B, N, embed_dim) - 视觉特征
        key: torch.Tensor,        # (B, M, embed_dim) - 语义特征
        value: torch.Tensor,      # (B, M, embed_dim)
        num_patches: Optional[int] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        空间感知的 Cross-Attention
        
        通过 2D 位置编码，模型自动学习空间关系:
        - 文本 "左边的狗" 会自动关注左侧 patches
        - 无需显式词汇表
        
        Args:
            query: 视觉特征 (B, N, embed_dim)
            key, value: 语义特征 (B, M, embed_dim)
            num_patches: patch 数量 (用于确定网格大小)
        """
        B, N, _ = query.shape
        M = key.shape[1]
        
        if num_patches is None:
            num_patches = N
        
        # LayerNorm
        query = self.ln_q(query)
        key = self.ln_kv(key)
        value = self.ln_kv(value)
        
        # 添加 2D 位置编码到 query (视觉特征)
        if self.use_2d_pos_enc:
            pos_enc = self.pos_encoding(num_patches, query.device)  # (N, embed_dim)
            query = query + pos_enc.unsqueeze(0)  # (B, N, embed_dim)
        
        # QKV 投影
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, M)
        
        # 应用 mask (如果有)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, v)  # (B, heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        # 返回平均的注意力权重 (用于可视化)
        attn_weights_mean = attn_weights.mean(dim=1)  # (B, N, M)
        
        return attn_output, attn_weights_mean


class ChangePreserveDetector(nn.Module):
    """
    Change/Preserve 区域检测器
    
    预测图像中哪些区域应该改变，哪些区域应该保持
    借鉴 Qwen-Image 文本渲染中"精确定位"的思想
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        semantic_dim: int = 768,
        use_mutual_exclusion: bool = True,
    ):
        super().__init__()
        self.use_mutual_exclusion = use_mutual_exclusion
        
        # 改变区域检测
        self.change_predictor = nn.Sequential(
            nn.Linear(hidden_dim + semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 保持区域检测
        self.preserve_predictor = nn.Sequential(
            nn.Linear(hidden_dim + semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,    # (B, N, hidden_dim)
        semantic_features: torch.Tensor,  # (B, 1, semantic_dim) or (B, M, semantic_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测每个 patch 的 change/preserve 分数
        
        Returns:
            change_mask: (B, N, 1) 改变区域的权重
            preserve_mask: (B, N, 1) 保持区域的权重
        """
        B, N, _ = visual_features.shape
        
        # 扩展语义特征
        if semantic_features.shape[1] == 1:
            semantic_expanded = semantic_features.expand(-1, N, -1)
        else:
            # 池化后扩展
            semantic_expanded = semantic_features.mean(dim=1, keepdim=True).expand(-1, N, -1)
        
        # 拼接特征
        combined = torch.cat([visual_features, semantic_expanded], dim=-1)
        
        # 预测分数
        change_scores = self.change_predictor(combined)      # (B, N, 1)
        preserve_scores = self.preserve_predictor(combined)  # (B, N, 1)
        
        if self.use_mutual_exclusion:
            # 互斥约束: 归一化使得 change + preserve ≈ 1
            total = change_scores + preserve_scores + 1e-6
            change_mask = change_scores / total
            preserve_mask = preserve_scores / total
        else:
            change_mask = change_scores
            preserve_mask = preserve_scores
        
        return change_mask, preserve_mask


# ==========================================
# 主模型: 双流架构 CIR
# ==========================================
@registry.register_model("blip2_dual_stream_cir")
class Blip2DualStreamCIR(Blip2Base):
    """
    双流架构的组合图像检索模型
    
    消融实验配置:
    - use_semantic_stream: 是否使用语义理解流 (默认 True)
    - use_fusion_module: 是否使用语义引导融合 (默认 True)
    - use_instruction_expansion: 是否使用指令展开 (默认 True)
    - fusion_type: 融合类型 ('cross_attention', 'gated', 'concat', 'add')
    
    当所有新增模块关闭时，退化为标准的 Q-Former CIR 模型
    """
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_dual_stream_cir.yaml",
    }
    
    def __init__(
        self,
        # 基础参数
        vit_model: str = "eva_clip_g",
        img_size: int = 224,
        drop_path_rate: float = 0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = "fp16",
        freeze_vit: bool = True,
        num_query_token: int = 32,
        cross_attention_freq: int = 2,
        embed_dim: int = 256,
        max_txt_len: int = 32,
        # ====== 消融实验开关 ======
        use_semantic_stream: bool = True,      # 是否使用语义理解流
        use_fusion_module: bool = True,        # 是否使用语义引导融合
        use_instruction_expansion: bool = True, # 是否使用指令展开 (轻量级模式有效)
        fusion_type: str = 'cross_attention',  # 融合类型
        # ====== 语义流参数 ======
        semantic_model_type: str = 'blip2_gen',  # 'lightweight', 'blip2', 'blip2_gen', 'qwen2_vl'
        qwen_model_name: str = 'qwen2.5-vl-3b',    # Qwen 模型名称 (仅 qwen2_vl 模式)
        t5_model_name: str = 'flan-t5-base',       # T5 模型名称 (仅 blip2_gen 模式)
        use_image_in_semantic: bool = True,  # BLIP2 语义模块是否使用图像
        semantic_hidden_dim: int = 768,
        num_expansion_tokens: int = 8,
        freeze_vlm: bool = True,               # 是否冻结 VLM
        use_vlm_image_input: bool = True,      # VLM 是否使用图像输入
        # ====== 融合模块参数 ======
        num_fusion_layers: int = 2,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.1,
        # ====== 查询生成参数 ======
        query_aggregation: str = 'attention',
        # ====== 双层特征融合参数 (借鉴 Qwen-Image) ======
        use_dual_level_fusion: bool = False,     # 是否使用双层特征融合
        use_spatial_align: bool = True,          # 是否使用空间感知对齐
        use_change_preserve: bool = True,        # 是否使用 Change/Preserve 分离
        adaptive_fusion: bool = True,            # 是否使用自适应权重融合
        dual_level_layers: int = 2,              # 双层融合的层数
    ):
        super().__init__()
        
        # 保存配置
        self.max_txt_len = max_txt_len
        self.embed_dim = embed_dim
        self.num_query_token = num_query_token
        
        # 消融实验开关
        self.use_semantic_stream = use_semantic_stream
        self.use_fusion_module = use_fusion_module
        self.use_instruction_expansion = use_instruction_expansion
        self.fusion_type = fusion_type
        self.semantic_model_type = semantic_model_type
        self.use_vlm_image_input = use_vlm_image_input
        self.use_image_in_semantic = use_image_in_semantic
        
        # 双层融合开关
        self.use_dual_level_fusion = use_dual_level_fusion
        self.use_spatial_align = use_spatial_align
        self.use_change_preserve = use_change_preserve
        self.adaptive_fusion = adaptive_fusion
        
        logging.info(f"=== 双流 CIR 模型配置 ===")
        logging.info(f"use_semantic_stream: {use_semantic_stream}")
        logging.info(f"semantic_model_type: {semantic_model_type}")
        logging.info(f"use_fusion_module: {use_fusion_module}")
        logging.info(f"use_instruction_expansion: {use_instruction_expansion}")
        logging.info(f"fusion_type: {fusion_type}")
        logging.info(f"use_dual_level_fusion: {use_dual_level_fusion}")
        if use_dual_level_fusion:
            logging.info(f"  - use_spatial_align: {use_spatial_align}")
            logging.info(f"  - use_change_preserve: {use_change_preserve}")
            logging.info(f"  - adaptive_fusion: {adaptive_fusion}")
        
        # ==========================================
        # 基础组件: tokenizer
        # ==========================================
        self.tokenizer = self.init_tokenizer()
        
        # ==========================================
        # 共享视觉编码器
        # ==========================================
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("Freeze vision encoder")
        
        # ==========================================
        # 流 2: 视觉保持流 (Q-Former) - 始终存在
        # ==========================================
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        
        # 复制预训练权重
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        
        # 投影层
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        
        # ITM head (可选，用于 re-ranking)
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        
        # 温度参数
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        
        # ==========================================
        # 流 1: 语义理解流 (可选)
        # ==========================================
        if self.use_semantic_stream:
            if self.semantic_model_type == 'qformer_native':
                # 【推荐】使用 Q-Former 原生多模态能力 (图像 + Prompt 文本)
                # 不需要额外的模块，直接使用 Q-Former
                # 只需要一个投影层
                self.semantic_proj = nn.Linear(
                    self.Qformer.config.hidden_size, 
                    semantic_hidden_dim
                )
                logging.info(f"Initialized qformer_native mode: Q-Former multimodal fusion (image + prompted text)")
                logging.info(f"  - Using CIR prompt template for text preprocessing")
                logging.info(f"  - Output dim: {semantic_hidden_dim}")
                
            elif self.semantic_model_type == 'qwen2_vl':
                # 使用 Qwen2.5-VL 进行语义理解 (最强大但最重，需要 transformers>=4.45)
                self.qwen_semantic = Qwen2VLSemanticModule(
                    model_name=qwen_model_name,
                    output_dim=semantic_hidden_dim,
                    freeze_vlm=freeze_vlm,
                    use_image_input=use_vlm_image_input,
                    max_length=max_txt_len * 4,  # VLM 需要更长的上下文
                )
                logging.info(f"Initialized Qwen2VLSemanticModule with model: {qwen_model_name}")
                
            elif self.semantic_model_type == 'blip2_gen':
                # BLIP2 生成式语义模块 (使用 T5 生成展开文本)
                # 注意：此模式只使用文本，不使用图像
                self.blip2_gen_semantic = Blip2GenerativeSemanticModule(
                    t5_model_name=t5_model_name,
                    output_dim=semantic_hidden_dim,
                    max_gen_length=64,
                    freeze_t5=freeze_vlm,
                )
                logging.info(f"Initialized Blip2GenerativeSemanticModule with T5: {t5_model_name}")
                logging.info(f"  WARNING: blip2_gen mode only uses text, not image!")
                
            elif self.semantic_model_type == 'blip2':
                # BLIP2 语义模块 (使用额外的 attention 层)
                self.blip2_semantic = Blip2SemanticModule(
                    text_encoder_config=self.Qformer.config,
                    num_semantic_tokens=num_expansion_tokens,
                    output_dim=semantic_hidden_dim,
                    use_image_guidance=use_image_in_semantic,
                    num_heads=fusion_heads,
                    dropout=fusion_dropout,
                )
                logging.info(f"Initialized Blip2SemanticModule with {num_expansion_tokens} tokens, "
                            f"use_image_guidance={use_image_in_semantic}")
                
            else:
                # 轻量级方案: lightweight 模式 (最轻但能力有限，只使用文本)
                if self.use_instruction_expansion:
                    self.instruction_expansion = InstructionExpansionModule(
                        text_dim=self.Qformer.config.hidden_size,
                        num_expansion_tokens=num_expansion_tokens,
                        num_heads=fusion_heads,
                        dropout=fusion_dropout,
                    )
                    logging.info(f"Initialized InstructionExpansionModule with {num_expansion_tokens} tokens")
                
                # 语义特征投影
                self.semantic_proj = nn.Linear(
                    self.Qformer.config.hidden_size, 
                    semantic_hidden_dim
                )
                logging.info(f"Initialized lightweight mode: text-only semantic understanding")
        
        # ==========================================
        # 语义引导融合模块 (可选)
        # 注意: 当启用双层融合时，fusion_module 不会被使用
        # ==========================================
        if self.use_fusion_module and self.use_semantic_stream:
            # 当启用双层融合时，自动禁用 fusion_module 以节省内存
            if self.use_dual_level_fusion:
                logging.warning("use_dual_level_fusion=True 时 fusion_module 将不会被使用，跳过初始化以节省内存")
                self.use_fusion_module = False  # 更新标志
            else:
                self.fusion_module = SemanticGuidedFusion(
                    visual_dim=self.Qformer.config.hidden_size,
                    semantic_dim=semantic_hidden_dim,
                    hidden_dim=self.Qformer.config.hidden_size,
                    num_layers=num_fusion_layers,
                    num_heads=fusion_heads,
                    dropout=fusion_dropout,
                    fusion_type=fusion_type,
                )
                logging.info(f"Initialized SemanticGuidedFusion with type: {fusion_type}")
        
        # ==========================================
        # 双层特征融合模块 (借鉴 Qwen-Image，可选)
        # ==========================================
        if self.use_dual_level_fusion and self.use_semantic_stream:
            self.dual_level_fusion = DualLevelFeatureFusion(
                visual_dim=self.Qformer.config.hidden_size,
                semantic_dim=semantic_hidden_dim,
                hidden_dim=self.Qformer.config.hidden_size,
                num_heads=fusion_heads,
                num_layers=dual_level_layers,
                dropout=fusion_dropout,
                use_spatial_align=use_spatial_align,
                use_change_preserve=use_change_preserve,
                adaptive_fusion=adaptive_fusion,
            )
            logging.info(f"Initialized DualLevelFeatureFusion (Qwen-Image style)")
        
        # ==========================================
        # 查询生成模块
        # ==========================================
        self.query_generator = QueryGenerationModule(
            input_dim=self.Qformer.config.hidden_size,
            output_dim=embed_dim,
            num_query_tokens=num_query_token,
            aggregation=query_aggregation,
        )
        logging.info(f"Initialized QueryGenerationModule with aggregation: {query_aggregation}")
    
    def encode_text_features(self, text: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码文本特征
        
        Returns:
            text_features: (B, L, hidden_dim)
            text_mask: (B, L)
        """
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        
        return text_output.last_hidden_state, text_tokens.attention_mask
    
    # CIR 专用 Prompt 模板
    CIR_PROMPT_TEMPLATE = "Modify the image according to: {instruction}"
    CIR_PROMPT_TEMPLATE_DETAILED = (
        "Given the reference image, apply this modification: {instruction}. "
        "Focus on what should change and what should remain."
    )
    
    def _prepare_prompted_text(self, text_list: List[str], use_detailed: bool = False) -> List[str]:
        """为文本添加 CIR 专用 Prompt"""
        template = self.CIR_PROMPT_TEMPLATE_DETAILED if use_detailed else self.CIR_PROMPT_TEMPLATE
        return [template.format(instruction=text) for text in text_list]
    
    def forward_semantic_stream(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        text_list: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        流 1: 语义理解流
        处理图像+文本指令，生成语义特征
        
        推荐使用 'qformer_native' 模式：
        - 同时输入图像和带 Prompt 的文本
        - 直接利用 Q-Former 预训练的多模态能力
        
        Args:
            text_features: (B, L, hidden_dim) - 轻量级模式使用
            text_mask: (B, L) - 轻量级模式使用
            images: (B, C, H, W) - Qwen2-VL 模式使用
            image_embeds: (B, N, vit_dim) - qformer_native/blip2 模式使用 (ViT 特征)
            text_list: 原始文本列表 - qformer_native/qwen2_vl 模式使用
        Returns:
            semantic_features: (B, semantic_dim)
        """
        device = text_features.device if text_features is not None else image_embeds.device
        
        if self.semantic_model_type == 'qformer_native':
            # 【推荐】使用 Q-Former 原生多模态能力 (图像 + Prompt 文本)
            assert text_list is not None, "text_list is required for qformer_native mode"
            assert image_embeds is not None, "image_embeds is required for qformer_native mode"
            
            batch_size = image_embeds.shape[0]
            
            # 添加 CIR 专用 Prompt
            prompted_texts = self._prepare_prompted_text(text_list, use_detailed=True)
            
            # 文本编码
            text_tokens = self.tokenizer(
                prompted_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            
            # 准备 attention masks
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            
            # Q-Former 多模态融合 (图像 + 文本)
            # 这是 BLIP2 的核心：query tokens 通过 cross-attention 同时融合图像和文本
            semantic_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,  # 图像特征
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            # 提取语义特征 (取 query tokens 部分，已融合图像+文本信息)
            query_output = semantic_output.last_hidden_state[:, :query_tokens.size(1), :]
            
            # 池化并投影
            semantic_features = query_output.mean(dim=1)  # (B, hidden_dim)
            semantic_features = self.semantic_proj(semantic_features)  # (B, semantic_dim)
            
        elif self.semantic_model_type == 'qwen2_vl':
            # 使用 Qwen2.5-VL 进行语义理解 (最强大，但需要新版 transformers)
            assert text_list is not None, "text_list is required for Qwen2-VL mode"
            semantic_features = self.qwen_semantic(
                images=images if self.use_vlm_image_input else None,
                instructions=text_list,
                device=device,
            )
        elif self.semantic_model_type == 'blip2_gen':
            # 使用 BLIP2 生成式语义模块 (T5 展开文本)
            # 注意：此模式只使用文本，不使用图像
            assert text_list is not None, "text_list is required for blip2_gen mode"
            semantic_features = self.blip2_gen_semantic(
                instructions=text_list,
                text_encoder=self.Qformer.bert,
                tokenizer=self.tokenizer,
                device=device,
                max_txt_len=self.max_txt_len,
            )
        elif self.semantic_model_type == 'blip2':
            # 使用 BLIP2 语义模块 (额外的 attention 层)
            semantic_features = self.blip2_semantic(
                text_features=text_features,
                text_mask=text_mask,
                image_embeds=image_embeds if self.use_image_in_semantic else None,
            )
        else:
            # 轻量级模式 (最轻，只使用文本)
            if self.use_instruction_expansion:
                # 使用指令展开模块
                expanded = self.instruction_expansion(text_features, text_mask)
                semantic_features = self.semantic_proj(expanded)
            else:
                # 简单池化
                mask = text_mask.unsqueeze(-1).float()
                pooled = (text_features * mask).sum(1) / mask.sum(1).clamp(min=1)
                semantic_features = self.semantic_proj(pooled)
        
        return semantic_features
    
    def forward_visual_stream(
        self,
        image_embeds: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        流 2: 视觉保持流
        使用 Q-Former 提取视觉特征
        
        Args:
            image_embeds: ViT 输出的图像特征 (B, N, vit_dim)
            text_features: 文本特征 (可选，用于 multimodal 融合)
            text_mask: 文本 mask
        Returns:
            visual_features: (B, num_query_token, hidden_dim)
        """
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        if text_features is not None and text_mask is not None:
            # Multimodal fusion in Q-Former
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            attention_mask = torch.cat([query_atts, text_mask], dim=1)
            
            # 获取 text input ids (需要从外部传入或重新编码)
            # 这里我们使用 text_features 作为 query_embeds 的一部分
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                attention_mask=query_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            # 纯视觉特征提取
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        return query_output.last_hidden_state
    
    def forward(self, samples: Dict) -> Dict:
        """
        训练前向传播
        
        Args:
            samples: {
                'image': 参考图像 (B, C, H, W),
                'target': 目标图像 (B, C, H, W),
                'text_input': 修改文本 List[str],
            }
        Returns:
            losses: Dict of losses
        """
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]
        
        device = image.device
        batch_size = image.shape[0]
        
        # ==========================================
        # Step 1: 编码参考图像
        # ==========================================
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        
        # ==========================================
        # Step 2: 主干流程 (保持原始 SPRC 设计)
        # ==========================================
        # 【优化】移除冗余的 encode_text_features，直接使用 tokenizer
        
        # 文本 tokenization
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        
        # 使用原始的 query_tokens (预训练好的)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        
        # 主干 Q-Former: 图像+文本融合 (原始 SPRC 方式)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,  # ← 恢复使用原始 query_tokens
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        # 使用第一个 text token 位置作为融合特征
        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, self.num_query_token, :]), 
            dim=-1
        )
        
        # ==========================================
        # Step 4: 双层融合作为辅助增强 (可选)
        # 【优化】复用主干 Q-Former 输出，避免重复计算
        # ==========================================
        dual_level_aux = None
        enhanced_feats = None
        semantic_features = None
        
        if self.use_semantic_stream:
            # 获取视觉特征 (轻量级线性层)
            visual_features = self.forward_visual_stream(image_embeds)
            
            # 【关键优化】复用主干 Q-Former 输出作为语义特征
            # 主干 Q-Former 已经融合了 图像+文本 信息，直接复用
            # fusion_output.last_hidden_state: (B, num_query + text_len, hidden_dim)
            # query 部分 [:, :num_query_token, :] 已包含图像-文本融合信息
            qformer_semantic = fusion_output.last_hidden_state[:, :self.num_query_token, :]
            semantic_features = self.semantic_proj(qformer_semantic.mean(dim=1))  # (B, semantic_dim)
            
            # 双层融合增强
            if self.use_dual_level_fusion:
                enhanced_feats, dual_level_aux = self.dual_level_fusion(
                    visual_features=visual_features,
                    semantic_features=semantic_features,
                )
            elif self.use_fusion_module:
                enhanced_feats = self.fusion_module(visual_features, semantic_features)
            else:
                # 简单融合
                if semantic_features.dim() == 3:
                    semantic_pooled = semantic_features.mean(dim=1)
                else:
                    semantic_pooled = semantic_features
                enhanced_feats = visual_features + semantic_pooled.unsqueeze(1)
        
        # 生成辅助查询特征 (用于辅助损失)
        if enhanced_feats is not None:
            query_features = self.query_generator(enhanced_feats)
        else:
            # 如果没有增强，使用主干输出
            query_features = self.query_generator(fusion_output.last_hidden_state[:, :self.num_query_token, :])
        
        # ==========================================
        # Step 5: 编码目标图像
        # ==========================================
        with self.maybe_autocast():
            target_embeds = self.ln_vision(self.visual_encoder(target))
        target_embeds = target_embeds.float()
        
        target_atts = torch.ones(target_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens_target = self.query_tokens.expand(batch_size, -1, -1)
        
        target_output = self.Qformer.bert(
            query_embeds=query_tokens_target,
            encoder_hidden_states=target_embeds,
            encoder_attention_mask=target_atts,
            return_dict=True,
        )
        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state), dim=-1
        )
        
        # ==========================================
        # Step 6: 计算损失
        # ==========================================
        
        # Loss 1: Fusion-Target Contrastive (主要损失)
        # fusion_feats: (B, embed_dim), target_feats: (B, num_query_token, embed_dim)
        # 计算每个查询与所有目标的相似度
        # sim_f2t[i, j, k] = fusion_feats[i] 与 target_feats[j, k, :] 的相似度
        sim_f2t = torch.einsum('bd,bnd->bn', fusion_feats, target_feats)  # (B, num_query_token)
        sim_f2t_max, _ = sim_f2t.max(dim=-1)  # (B,) - 每个样本取最大相似度
        
        # 构建完整的相似度矩阵 (B, B) 用于对比学习
        # 需要计算 fusion_feats[i] 与所有 target_feats[j] 的相似度
        sim_matrix = torch.einsum('id,jnd->ijn', fusion_feats, target_feats)  # (B, B, num_query_token)
        sim_matrix_max, _ = sim_matrix.max(dim=-1)  # (B, B)
        sim_matrix_max = sim_matrix_max / self.temp
        
        targets = torch.arange(batch_size, dtype=torch.long, device=device)
        if batch_size > 1:
            loss_ftc = F.cross_entropy(sim_matrix_max, targets)
        else:
            loss_ftc = -torch.log(torch.sigmoid(sim_f2t_max / self.temp)).mean()
        
        # Loss 2: Query-Target Contrastive (辅助损失)
        target_feats_mean = target_feats.mean(dim=1)  # (B, embed_dim)
        sim_q2t = torch.matmul(query_features, target_feats_mean.T) / self.temp  # (B, B)
        
        if batch_size > 1:
            loss_qtc = F.cross_entropy(sim_q2t, targets)
        else:
            loss_qtc = -torch.log(torch.sigmoid(sim_q2t.diag())).mean()
        
        # Loss 3: 语义一致性损失 (可选) - 鼓励增强特征保持语义信息
        loss_semantic = torch.tensor(0.0, device=device)
        if self.use_semantic_stream and enhanced_feats is not None and semantic_features is not None:
            # 鼓励增强后的特征与语义特征对齐
            enhanced_pooled = enhanced_feats.mean(dim=1)
            if semantic_features.dim() == 3:
                semantic_pooled = semantic_features.mean(dim=1)
            else:
                semantic_pooled = semantic_features
            
            # 使用余弦相似度作为一致性度量
            cos_sim = F.cosine_similarity(enhanced_pooled, semantic_pooled, dim=-1)
            loss_semantic = (1 - cos_sim).mean()
        
        # Loss 4: 双层融合辅助损失 (可选)
        loss_dual_level = torch.tensor(0.0, device=device)
        if self.use_dual_level_fusion and dual_level_aux is not None:
            # 4.1 高层-低层特征正交性损失 (鼓励两层特征互补而非冗余)
            if dual_level_aux.get('high_level_feat') is not None and dual_level_aux.get('low_level_feat') is not None:
                high_pooled = dual_level_aux['high_level_feat'].mean(dim=1)
                low_pooled = dual_level_aux['low_level_feat'].mean(dim=1)
                # 鼓励正交 (余弦相似度趋近于 0)
                ortho_sim = F.cosine_similarity(high_pooled, low_pooled, dim=-1).abs()
                loss_ortho = ortho_sim.mean() * 0.1  # 权重 0.1
                loss_dual_level = loss_dual_level + loss_ortho
            
            # 4.2 Change/Preserve 区分度损失 (鼓励明确的区域划分)
            if dual_level_aux.get('change_mask') is not None and dual_level_aux.get('preserve_mask') is not None:
                change_mask = dual_level_aux['change_mask']
                preserve_mask = dual_level_aux['preserve_mask']
                # 熵损失: 鼓励每个 patch 要么明确是 change，要么明确是 preserve
                entropy_change = -(change_mask * torch.log(change_mask + 1e-6) + 
                                   (1 - change_mask) * torch.log(1 - change_mask + 1e-6))
                loss_entropy = entropy_change.mean() * 0.05  # 权重 0.05
                loss_dual_level = loss_dual_level + loss_entropy
        
        # 返回格式与训练脚本兼容
        # loss_itc: 主对比损失 (直接使用，权重=1)
        # loss_rtc: 辅助对比损失 (使用 kwargs['loss_rtc'] 权重，默认 0.4)
        # loss_align: 语义一致性损失 (使用 kwargs['loss_align'] 权重，默认 0.4)
        losses = {
            'loss_itc': loss_ftc,      # 主对比损失 (Fusion-Target)
            'loss_rtc': loss_qtc,      # 辅助对比损失 (Query-Target)
            'loss_align': loss_semantic,  # 语义一致性损失
        }
        
        # 添加双层融合损失 (如果启用)
        if self.use_dual_level_fusion:
            losses['loss_dual'] = loss_dual_level
        
        return losses
    
    @torch.no_grad()
    def extract_target_features(
        self, 
        image: torch.Tensor,
        mode: str = 'mean'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取目标图像特征 (用于构建检索库)
        
        Args:
            image: (B, C, H, W)
            mode: 'mean' 或 'all'
        Returns:
            image_features: (B, embed_dim) 或 (B, num_query_token, embed_dim)
            image_embeds: (B, N, vit_dim) - 原始 ViT 特征
        """
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        image_features = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )
        
        if mode == 'mean':
            return image_features.mean(dim=1), image_embeds
        else:
            return image_features, image_embeds
    
    @torch.no_grad()
    def inference(
        self,
        reference_embeds: torch.Tensor,
        target_feats: torch.Tensor,
        text: List[str],
        fusion_weight: float = 0,  # 主干特征权重
    ) -> torch.Tensor:
        """
        推理: 计算查询与目标的相似度
        
        Args:
            reference_embeds: 参考图像 ViT 特征 (B, N, vit_dim)
            target_feats: 目标图像特征 (num_targets, num_query_token, embed_dim)
            text: 修改文本
            fusion_weight: 主干特征与增强特征的融合权重 (0-1)
                          1.0 = 仅使用主干特征
                          0.0 = 仅使用增强特征
        Returns:
            similarities: (B, num_targets)
        """
        device = reference_embeds.device
        batch_size = reference_embeds.shape[0]
        
        # ==========================================
        # Step 1: 主干 Q-Former 特征 (原始 SPRC)
        # ==========================================
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(device)
        
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, self.num_query_token, :]),
            dim=-1
        )  # (B, embed_dim)
        
        # ==========================================
        # Step 2: 双层融合增强特征 (可选)
        # ==========================================
        final_feats = fusion_feats  # 默认使用主干特征
        
        if self.use_dual_level_fusion and fusion_weight < 1.0:
            # 编码文本特征 (用于语义流)
            text_features, text_mask = self.encode_text_features(text, device)
            
            # 视觉流
            visual_features = self.forward_visual_stream(reference_embeds)
            
            # 语义流
            semantic_features = self.forward_semantic_stream(
                text_features, text_mask,
                images=None,  # 推理时不使用原始图像
                image_embeds=reference_embeds,
                text_list=text,
            )
            
            # 双层融合
            enhanced_feats, _ = self.dual_level_fusion(
                visual_features=visual_features,
                semantic_features=semantic_features,
            )
            
            # 生成增强查询特征
            enhanced_query = self.query_generator(enhanced_feats)  # (B, embed_dim)
            enhanced_query = F.normalize(enhanced_query, dim=-1)
            
            # 特征融合: 主干特征 + 增强特征
            final_feats = fusion_weight * fusion_feats + (1 - fusion_weight) * enhanced_query
            final_feats = F.normalize(final_feats, dim=-1)
        
        # ==========================================
        # Step 3: 计算相似度
        # ==========================================
        if target_feats.dim() == 2:
            # 2D case: (num_targets, embed_dim)
            similarities = torch.matmul(final_feats, target_feats.T)  # (B, num_targets)
        else:
            # 3D case: (num_targets, num_query_token, embed_dim)
            # final_feats: (B, embed_dim)
            # target_feats: (num_targets, num_query_token, embed_dim)
            # 计算 sim[b, t, k] = final_feats[b] · target_feats[t, k]
            sim_t2q = torch.einsum('bd,tnd->btn', final_feats, target_feats)  # (B, num_targets, num_query_token)
            
            # 取每个目标的最大相似度
            similarities, _ = sim_t2q.max(dim=-1)  # (B, num_targets)
        
        return similarities
    
    @classmethod
    def from_config(cls, cfg):
        """从配置文件创建模型"""
        # 基础参数
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size", 224)
        num_query_token = cfg.get("num_query_token", 32)
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        
        max_txt_len = cfg.get("max_txt_len", 32)
        embed_dim = cfg.get("embed_dim", 256)
        
        # 消融实验开关
        use_semantic_stream = cfg.get("use_semantic_stream", True)
        use_fusion_module = cfg.get("use_fusion_module", True)
        use_instruction_expansion = cfg.get("use_instruction_expansion", True)
        fusion_type = cfg.get("fusion_type", "cross_attention")
        
        # 语义流参数
        semantic_model_type = cfg.get("semantic_model_type", "qformer_native")  # 默认使用 qformer_native (推荐)
        qwen_model_name = cfg.get("qwen_model_name", "qwen2.5-vl-3b")
        t5_model_name = cfg.get("t5_model_name", "flan-t5-base")  # T5 模型 (blip2_gen 模式)
        semantic_hidden_dim = cfg.get("semantic_hidden_dim", 768)
        num_expansion_tokens = cfg.get("num_expansion_tokens", 8)
        freeze_vlm = cfg.get("freeze_vlm", True)
        use_vlm_image_input = cfg.get("use_vlm_image_input", True)
        use_image_in_semantic = cfg.get("use_image_in_semantic", True)
        
        # 融合模块参数
        num_fusion_layers = cfg.get("num_fusion_layers", 2)
        fusion_heads = cfg.get("fusion_heads", 8)
        fusion_dropout = cfg.get("fusion_dropout", 0.1)
        
        # 查询生成参数
        query_aggregation = cfg.get("query_aggregation", "attention")
        
        # 双层特征融合参数 (借鉴 Qwen-Image)
        use_dual_level_fusion = cfg.get("use_dual_level_fusion", False)
        use_spatial_align = cfg.get("use_spatial_align", True)
        use_change_preserve = cfg.get("use_change_preserve", True)
        adaptive_fusion = cfg.get("adaptive_fusion", True)
        dual_level_layers = cfg.get("dual_level_layers", 2)
        
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
            # 消融实验开关
            use_semantic_stream=use_semantic_stream,
            use_fusion_module=use_fusion_module,
            use_instruction_expansion=use_instruction_expansion,
            fusion_type=fusion_type,
            # 语义流参数
            semantic_model_type=semantic_model_type,
            qwen_model_name=qwen_model_name,
            t5_model_name=t5_model_name,
            semantic_hidden_dim=semantic_hidden_dim,
            num_expansion_tokens=num_expansion_tokens,
            freeze_vlm=freeze_vlm,
            use_vlm_image_input=use_vlm_image_input,
            use_image_in_semantic=use_image_in_semantic,
            # 融合模块参数
            num_fusion_layers=num_fusion_layers,
            fusion_heads=fusion_heads,
            fusion_dropout=fusion_dropout,
            # 查询生成参数
            query_aggregation=query_aggregation,
            # 双层特征融合参数 (借鉴 Qwen-Image)
            use_dual_level_fusion=use_dual_level_fusion,
            use_spatial_align=use_spatial_align,
            use_change_preserve=use_change_preserve,
            adaptive_fusion=adaptive_fusion,
            dual_level_layers=dual_level_layers,
        )
        
        model.load_checkpoint_from_config(cfg)
        
        return model

