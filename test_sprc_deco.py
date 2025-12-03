"""
Test script for SPRC_DeCo model
Verifies model loading and forward pass
"""
import torch
from lavis.models import load_model_and_preprocess

def test_sprc_deco():
    print("=" * 60)
    print("Testing SPRC_DeCo Model")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load model
    print("\n1. Loading SPRC_DeCo model...")
    try:
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="sprc_deco",
            model_type="pretrain",
            is_eval=False,
            device=device
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Check model components
    print("\n2. Checking model components...")
    print(f"   - Spatial Projector: {hasattr(model, 'spatial_projector')}")
    print(f"   - Position Embeddings: {hasattr(model, 'pos_embed')}")
    print(f"   - Grid size: {model.grid_size}")
    print(f"   - Num spatial tokens: {model.num_spatial_tokens}")
    
    # Create dummy data
    print("\n3. Creating dummy data...")
    batch_size = 2
    dummy_samples = {
        "image": torch.randn(batch_size, 3, 224, 224).to(device),
        "target": torch.randn(batch_size, 3, 224, 224).to(device),
        "text_input": ["a red shirt", "blue jeans"]
    }
    print(f"   - Batch size: {batch_size}")
    print(f"   - Image shape: {dummy_samples['image'].shape}")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    model.train()
    try:
        with torch.cuda.amp.autocast():
            loss_dict = model(dummy_samples)
        
        print("✓ Forward pass successful")
        print(f"\n   Loss components:")
        for key, value in loss_dict.items():
            print(f"   - {key}: {value.item():.4f}")
        
        # Compute total loss
        total_loss = sum(loss_dict.values())
        print(f"\n   Total loss: {total_loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    try:
        total_loss.backward()
        print("✓ Backward pass successful")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return
    
    # Test feature extraction
    print("\n6. Testing feature extraction...")
    model.eval()
    try:
        with torch.no_grad():
            target_feats, vit_feats = model.extract_target_features(
                dummy_samples["target"]
            )
        print("✓ Feature extraction successful")
        print(f"   - Target features shape: {target_feats.shape}")
        print(f"   - Expected: [{batch_size}, 64, 256]")
        
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return
    
    # Test inference
    print("\n7. Testing inference...")
    try:
        with torch.no_grad():
            # Extract reference features
            ref_vit_feats = model.ln_vision(
                model.visual_encoder(dummy_samples["image"])
            ).float()
            
            # Run inference
            sim_scores = model.inference(
                ref_vit_feats,
                target_feats,
                dummy_samples["text_input"]
            )
        
        print("✓ Inference successful")
        print(f"   - Similarity scores shape: {sim_scores.shape}")
        print(f"   - Expected: [{batch_size}, {batch_size}]")
        print(f"   - Scores:\n{sim_scores}")
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nYou can now train the model using:")
    print("python blip_fine_tune_2.py --dataset CIRR --blip-model-name sprc_deco --backbone pretrain ...")

if __name__ == "__main__":
    test_sprc_deco()
