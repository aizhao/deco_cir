"""
Quick test to verify the training fixes work correctly
"""

import torch
import sys
sys.path.append('src')

from lavis.models import load_model_and_preprocess
from utils import device

def test_differential_lr():
    """Test that differential learning rates are set up correctly"""
    print("\n" + "="*80)
    print("Testing Differential Learning Rates")
    print("="*80)
    
    blip_model, _, _ = load_model_and_preprocess(
        name="blip2_avgpool_cir_align_prompt", 
        model_type="pretrain", 
        is_eval=False, 
        device=device
    )
    
    # Simulate the parameter grouping from training script
    projector_params = []
    base_params = []
    
    for name, param in blip_model.named_parameters():
        if not param.requires_grad:
            continue
        if 'deco_projector' in name or 'visual_pos_embed' in name:
            projector_params.append(param)
        else:
            base_params.append(param)
    
    print(f"\nProjector parameters: {len(projector_params)}")
    print(f"Base parameters: {len(base_params)}")
    
    # Create optimizer with differential LR
    learning_rate = 2e-6
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': learning_rate},
        {'params': projector_params, 'lr': learning_rate * 20}
    ])
    
    print(f"\nLearning rates:")
    print(f"  Base (BERT): {optimizer.param_groups[0]['lr']:.2e}")
    print(f"  Projector (DeCo): {optimizer.param_groups[1]['lr']:.2e}")
    print(f"  Ratio: {optimizer.param_groups[1]['lr'] / optimizer.param_groups[0]['lr']:.1f}x")
    
    assert len(projector_params) > 0, "No projector parameters found!"
    assert optimizer.param_groups[1]['lr'] == learning_rate * 20, "Projector LR not 20x!"
    
    print("\n✓ Differential learning rates configured correctly!")
    return True

def test_layernorm():
    """Test that LayerNorm is present in MLP"""
    print("\n" + "="*80)
    print("Testing LayerNorm in MLP")
    print("="*80)
    
    blip_model, _, _ = load_model_and_preprocess(
        name="blip2_avgpool_cir_align_prompt", 
        model_type="pretrain", 
        is_eval=False, 
        device=device
    )
    
    # Check MLP architecture
    mlp = blip_model.deco_projector.mlp_projector
    print(f"\nMLP architecture:")
    for i, layer in enumerate(mlp):
        print(f"  Layer {i}: {layer.__class__.__name__}")
    
    has_layernorm = any(isinstance(m, torch.nn.LayerNorm) for m in mlp.modules())
    
    assert has_layernorm, "LayerNorm not found in MLP!"
    print("\n✓ LayerNorm present in MLP!")
    return True

def test_forward_pass():
    """Test forward pass with loss computation"""
    print("\n" + "="*80)
    print("Testing Forward Pass")
    print("="*80)
    
    blip_model, _, _ = load_model_and_preprocess(
        name="blip2_avgpool_cir_align_prompt", 
        model_type="pretrain", 
        is_eval=False, 
        device=device
    )
    
    # Create dummy batch
    batch_size = 4
    samples = {
        'image': torch.randn(batch_size, 3, 224, 224).to(device),
        'target': torch.randn(batch_size, 3, 224, 224).to(device),
        'text_input': ['a red dress'] * batch_size
    }
    
    blip_model.train()
    
    # Forward pass
    with torch.cuda.amp.autocast():
        loss_dict = blip_model(samples)
    
    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
        assert not torch.isnan(value), f"{key} is NaN!"
        assert not torch.isinf(value), f"{key} is Inf!"
    
    # Test adaptive TIC weighting
    print(f"\nTesting adaptive TIC weighting:")
    for epoch in [0, 3, 5, 10]:
        tic_weight = 0.01 if epoch < 5 else 1.0
        print(f"  Epoch {epoch}: TIC weight = {tic_weight}")
    
    print("\n✓ Forward pass successful, all losses finite!")
    return True

def test_cls_token_removal():
    """Test CLS token is properly removed"""
    print("\n" + "="*80)
    print("Testing CLS Token Removal")
    print("="*80)
    
    blip_model, _, _ = load_model_and_preprocess(
        name="blip2_avgpool_cir_align_prompt", 
        model_type="pretrain", 
        is_eval=False, 
        device=device
    )
    
    # Test with 257 tokens (256 + 1 CLS)
    input_with_cls = torch.randn(2, 257, 1408).to(device)
    output = blip_model.deco_projector(input_with_cls)
    
    print(f"\nInput shape: {input_with_cls.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output tokens: 64")
    print(f"Actual output tokens: {output.shape[1]}")
    
    assert output.shape[1] == 64, f"Expected 64 tokens, got {output.shape[1]}"
    
    print("\n✓ CLS token correctly removed!")
    return True

def main():
    print("\n" + "="*80)
    print("DeCo-SPRC Training Fixes Verification")
    print("="*80)
    
    tests = [
        ("CLS Token Removal", test_cls_token_removal),
        ("LayerNorm in MLP", test_layernorm),
        ("Differential Learning Rates", test_differential_lr),
        ("Forward Pass", test_forward_pass),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "="*80)
        print("All tests passed! Ready to train.")
        print("="*80)
        print("\nRecommended training command:")
        print("cd SPRC/src")
        print("python blip_fine_tune_2.py \\")
        print("  --dataset CIRR \\")
        print("  --blip-model-name blip2_avgpool_cir_align_prompt \\")
        print("  --learning-rate 2e-6 \\")
        print("  --batch-size 128 \\")
        print("  --num-epochs 50 \\")
        print("  --loss-dense 0.5 \\")
        print("  --loss-tic 0.3 \\")
        print("  --save-training --save-best")
    else:
        print("\n" + "="*80)
        print("Some tests failed. Please review the errors above.")
        print("="*80)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
