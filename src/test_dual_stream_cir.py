"""
双流架构 CIR 模型测试脚本

用于验证模型能否正常初始化和前向传播，
以及测试各种消融配置。

使用方法:
    # 测试轻量级模式
    python src/test_dual_stream_cir.py
    
    # 测试 Qwen2.5-VL 模式 (需要安装 transformers>=4.37.0)
    python src/test_dual_stream_cir.py --test-qwen
"""
import os
import sys
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# 导入模型
from lavis.models.blip2_models.blip2_dual_stream_cir import (
    Blip2DualStreamCIR,
    SemanticGuidedFusion,
    InstructionExpansionModule,
    QueryGenerationModule,
    Qwen2VLSemanticModule,
)


def test_semantic_guided_fusion():
    """测试语义引导融合模块"""
    print("\n" + "=" * 60)
    print("测试 SemanticGuidedFusion 模块")
    print("=" * 60)
    
    batch_size = 2
    num_tokens = 32
    visual_dim = 768
    semantic_dim = 768
    
    visual_features = torch.randn(batch_size, num_tokens, visual_dim)
    semantic_features = torch.randn(batch_size, semantic_dim)
    
    fusion_types = ['cross_attention', 'gated', 'concat', 'add']
    
    for fusion_type in fusion_types:
        print(f"\n--- 融合类型: {fusion_type} ---")
        
        fusion = SemanticGuidedFusion(
            visual_dim=visual_dim,
            semantic_dim=semantic_dim,
            hidden_dim=768,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            fusion_type=fusion_type,
        )
        
        # 统计参数量
        num_params = sum(p.numel() for p in fusion.parameters())
        print(f"参数量: {num_params:,}")
        
        # 前向传播
        fused = fusion(visual_features, semantic_features)
        print(f"输入 visual: {visual_features.shape}")
        print(f"输入 semantic: {semantic_features.shape}")
        print(f"输出: {fused.shape}")
        
        assert fused.shape == (batch_size, num_tokens, 768), f"输出形状错误: {fused.shape}"
        print("✓ 测试通过")


def test_instruction_expansion():
    """测试指令展开模块"""
    print("\n" + "=" * 60)
    print("测试 InstructionExpansionModule 模块")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 32
    text_dim = 768
    num_expansion_tokens = 8
    
    expansion = InstructionExpansionModule(
        text_dim=text_dim,
        num_expansion_tokens=num_expansion_tokens,
        num_heads=8,
        dropout=0.1,
    )
    
    # 统计参数量
    num_params = sum(p.numel() for p in expansion.parameters())
    print(f"参数量: {num_params:,}")
    
    instruction_features = torch.randn(batch_size, seq_len, text_dim)
    instruction_mask = torch.ones(batch_size, seq_len)
    instruction_mask[:, seq_len//2:] = 0  # 模拟 padding
    
    expanded = expansion(instruction_features, instruction_mask)
    print(f"输入: {instruction_features.shape}")
    print(f"输出: {expanded.shape}")
    
    assert expanded.shape == (batch_size, num_expansion_tokens, text_dim), f"输出形状错误"
    print("✓ 测试通过")


def test_query_generation():
    """测试查询生成模块"""
    print("\n" + "=" * 60)
    print("测试 QueryGenerationModule 模块")
    print("=" * 60)
    
    batch_size = 2
    num_tokens = 32
    input_dim = 768
    output_dim = 256
    
    aggregations = ['attention', 'mean', 'first']
    
    for agg in aggregations:
        print(f"\n--- 聚合方式: {agg} ---")
        
        generator = QueryGenerationModule(
            input_dim=input_dim,
            output_dim=output_dim,
            num_query_tokens=num_tokens,
            aggregation=agg,
        )
        
        fused_features = torch.randn(batch_size, num_tokens, input_dim)
        query = generator(fused_features)
        print(f"输入: {fused_features.shape}")
        print(f"输出: {query.shape}")
        
        assert query.shape == (batch_size, output_dim), f"输出形状错误"
        print("✓ 测试通过")


def test_full_model_initialization():
    """测试完整模型初始化"""
    print("\n" + "=" * 60)
    print("测试 Blip2DualStreamCIR 完整模型初始化")
    print("=" * 60)
    
    # 测试不同的消融配置
    ablation_configs = [
        {
            "name": "完整模型",
            "use_semantic_stream": True,
            "use_fusion_module": True,
            "use_instruction_expansion": True,
            "fusion_type": "cross_attention",
        },
        {
            "name": "Baseline (无新增模块)",
            "use_semantic_stream": False,
            "use_fusion_module": False,
            "use_instruction_expansion": False,
            "fusion_type": "add",
        },
        {
            "name": "仅语义流",
            "use_semantic_stream": True,
            "use_fusion_module": False,
            "use_instruction_expansion": True,
            "fusion_type": "add",
        },
        {
            "name": "门控融合",
            "use_semantic_stream": True,
            "use_fusion_module": True,
            "use_instruction_expansion": True,
            "fusion_type": "gated",
        },
    ]
    
    for config in ablation_configs:
        print(f"\n--- {config['name']} ---")
        
        try:
            # 使用小型 ViT 进行测试 (避免 OOM)
            model = Blip2DualStreamCIR(
                vit_model="clip_L",  # 使用较小的 ViT
                img_size=224,
                freeze_vit=True,
                num_query_token=32,
                embed_dim=256,
                max_txt_len=32,
                use_semantic_stream=config["use_semantic_stream"],
                use_fusion_module=config["use_fusion_module"],
                use_instruction_expansion=config["use_instruction_expansion"],
                fusion_type=config["fusion_type"],
            )
            
            # 统计参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"总参数量: {total_params:,}")
            print(f"可训练参数: {trainable_params:,}")
            print(f"冻结参数: {total_params - trainable_params:,}")
            print("✓ 初始化成功")
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"✗ 初始化失败: {e}")


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试前向传播")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 创建模型
        model = Blip2DualStreamCIR(
            vit_model="clip_L",
            img_size=224,
            freeze_vit=True,
            num_query_token=32,
            embed_dim=256,
            max_txt_len=32,
            use_semantic_stream=True,
            use_fusion_module=True,
            use_instruction_expansion=True,
            fusion_type="cross_attention",
        )
        model = model.to(device)
        model.eval()
        
        # 创建模拟输入
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224).to(device)
        target = torch.randn(batch_size, 3, 224, 224).to(device)
        text = ["make it red", "flip horizontally"]
        
        samples = {
            "image": image,
            "target": target,
            "text_input": text,
        }
        
        # 训练模式前向传播
        model.train()
        with torch.cuda.amp.autocast() if device == "cuda" else torch.no_grad():
            outputs = model(samples)
        
        print("\n训练输出:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value}")
        
        # 推理模式测试
        model.eval()
        with torch.no_grad():
            # 提取目标特征
            target_feats, target_embeds = model.extract_target_features(target)
            print(f"\n目标特征形状: {target_feats.shape}")
            
            # 提取参考图像 ViT 特征
            with model.maybe_autocast():
                ref_embeds = model.ln_vision(model.visual_encoder(image))
            ref_embeds = ref_embeds.float()
            
            # 推理
            similarities = model.inference(ref_embeds, target_feats.unsqueeze(0).expand(batch_size, -1, -1), text)
            print(f"相似度形状: {similarities.shape}")
        
        print("\n✓ 前向传播测试通过")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        import traceback
        print(f"✗ 前向传播失败: {e}")
        traceback.print_exc()


def test_ablation_comparison():
    """比较不同消融配置的参数量"""
    print("\n" + "=" * 60)
    print("消融配置参数量对比")
    print("=" * 60)
    
    configs = [
        ("Baseline", False, False, False, "add"),
        ("+ 语义流", True, False, True, "add"),
        ("+ 简单融合", True, True, True, "add"),
        ("+ 拼接融合", True, True, True, "concat"),
        ("+ 门控融合", True, True, True, "gated"),
        ("+ Cross-Attention", True, True, True, "cross_attention"),
    ]
    
    results = []
    
    for name, sem, fus, exp, ftype in configs:
        try:
            model = Blip2DualStreamCIR(
                vit_model="clip_L",
                img_size=224,
                freeze_vit=True,
                num_query_token=32,
                embed_dim=256,
                use_semantic_stream=sem,
                use_fusion_module=fus,
                use_instruction_expansion=exp,
                fusion_type=ftype,
            )
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            results.append((name, trainable))
            
            del model
            
        except Exception as e:
            results.append((name, f"Error: {e}"))
    
    # 打印对比表格
    print("\n配置名称                      可训练参数")
    print("-" * 50)
    baseline_params = results[0][1] if isinstance(results[0][1], int) else 0
    for name, params in results:
        if isinstance(params, int):
            diff = params - baseline_params
            diff_str = f"(+{diff:,})" if diff > 0 else ""
            print(f"{name:<30} {params:>12,} {diff_str}")
        else:
            print(f"{name:<30} {params}")


def test_qwen_semantic_module():
    """测试 Qwen2.5-VL 语义理解模块"""
    print("\n" + "=" * 60)
    print("测试 Qwen2VLSemanticModule 模块")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 创建模块 (延迟加载)
        qwen_module = Qwen2VLSemanticModule(
            model_name="qwen2.5-vl-3b",
            output_dim=768,
            freeze_vlm=True,
            use_image_input=False,  # 先测试纯文本模式
        )
        
        # 测试文本输入
        instructions = ["mirror the image", "make it red"]
        
        # 创建模拟的 text_features (用于获取 device)
        dummy_text_features = torch.randn(2, 32, 768).to(device)
        
        print("开始加载 Qwen2-VL 模型...")
        semantic_features = qwen_module(
            images=None,
            instructions=instructions,
            device=device,
        )
        
        print(f"输入指令: {instructions}")
        print(f"输出语义特征形状: {semantic_features.shape}")
        print("✓ Qwen2VLSemanticModule 测试通过")
        
    except ImportError as e:
        print(f"⚠ 跳过测试: {e}")
        print("请确保安装了 transformers>=4.37.0")
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        traceback.print_exc()


def test_qwen_full_model():
    """测试使用 Qwen2.5-VL 的完整模型"""
    print("\n" + "=" * 60)
    print("测试 Qwen2.5-VL 完整模型")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print("初始化模型 (使用 Qwen2.5-VL)...")
        model = Blip2DualStreamCIR(
            vit_model="clip_L",
            img_size=224,
            freeze_vit=True,
            num_query_token=32,
            embed_dim=256,
            max_txt_len=32,
            # 使用 Qwen2.5-VL
            use_semantic_stream=True,
            semantic_model_type='qwen2_vl',
            qwen_model_name='qwen2.5-vl-3b',
            freeze_vlm=True,
            use_vlm_image_input=False,  # 先测试纯文本模式
            # 融合配置
            use_fusion_module=True,
            fusion_type='cross_attention',
        )
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 测试前向传播
        model = model.to(device)
        model.eval()
        
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224).to(device)
        target = torch.randn(batch_size, 3, 224, 224).to(device)
        text = ["mirror the image horizontally", "change color to red"]
        
        samples = {
            "image": image,
            "target": target,
            "text_input": text,
        }
        
        print("\n开始前向传播...")
        with torch.cuda.amp.autocast() if device == "cuda" else torch.no_grad():
            outputs = model(samples)
        
        print("训练输出:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
        
        print("\n✓ Qwen2.5-VL 完整模型测试通过")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except ImportError as e:
        print(f"⚠ 跳过测试: {e}")
        print("请确保安装了 transformers>=4.37.0")
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="双流架构 CIR 模型测试")
    parser.add_argument("--test-qwen", action="store_true", help="测试 Qwen2.5-VL 模式")
    args = parser.parse_args()
    
    print("=" * 60)
    print("双流架构 CIR 模型测试")
    print("=" * 60)
    
    # 测试各个模块
    test_semantic_guided_fusion()
    test_instruction_expansion()
    test_query_generation()
    
    # 测试完整模型 (轻量级)
    test_full_model_initialization()
    
    # 测试前向传播 (需要 GPU)
    if torch.cuda.is_available():
        test_forward_pass()
    else:
        print("\n⚠ 跳过前向传播测试 (无 GPU)")
    
    # 参数量对比
    test_ablation_comparison()
    
    # 测试 Qwen2.5-VL (可选)
    if args.test_qwen:
        print("\n" + "=" * 60)
        print("Qwen2.5-VL 测试")
        print("=" * 60)
        
        if torch.cuda.is_available():
            test_qwen_semantic_module()
            test_qwen_full_model()
        else:
            print("⚠ Qwen2.5-VL 测试需要 GPU")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

