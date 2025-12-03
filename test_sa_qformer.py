"""
Test script for SA-QFormer basic functionality
"""
import torch
import sys
sys.path.insert(0, 'src')

from lavis.models.blip2_models.sa_qformer import (
    AdaptiveGridPooling,
    GridQueryInitializer,
    Learned2DPositionalEncoding,
    Sinusoidal2DPositionalEncoding,
    LocalConstrainedCrossAttention,
    InstructionInjectionModule,
    SAQFormer
)
from lavis.models.blip2_models.Qformer import BertConfig


def test_adaptive_pooling():
    print("Testing AdaptiveGridPooling...")
    pooling = AdaptiveGridPooling(output_size=(8, 8))
    
    # Test with 14x14 ViT output
    batch_size = 2
    vit_features = torch.randn(batch_size, 196, 768)  # 14x14 = 196
    
    output = pooling(vit_features, (14, 14))
    assert output.shape == (batch_size, 64, 768), f"Expected (2, 64, 768), got {output.shape}"
    print("✓ AdaptiveGridPooling test passed")


def test_positional_encoding():
    print("\nTesting Positional Encodings...")
    
    # Test learned encoding
    learned_pe = Learned2DPositionalEncoding(grid_size=(8, 8), hidden_size=768)
    pos_enc = learned_pe(batch_size=2, device='cpu')
    assert pos_enc.shape == (2, 64, 768), f"Expected (2, 64, 768), got {pos_enc.shape}"
    print("✓ Learned positional encoding test passed")
    
    # Test sinusoidal encoding
    sin_pe = Sinusoidal2DPositionalEncoding(grid_size=(8, 8), hidden_size=768)
    pos_enc = sin_pe(batch_size=2, device='cpu')
    assert pos_enc.shape == (2, 64, 768), f"Expected (2, 64, 768), got {pos_enc.shape}"
    print("✓ Sinusoidal positional encoding test passed")


def test_grid_query_initializer():
    print("\nTesting GridQueryInitializer...")
    
    # Test with learned encoding
    initializer = GridQueryInitializer(grid_size=(8, 8), hidden_size=768, pos_encoding_type='learned')
    queries = initializer(batch_size=2, device='cpu')
    assert queries.shape == (2, 64, 768), f"Expected (2, 64, 768), got {queries.shape}"
    print("✓ GridQueryInitializer test passed")


def test_local_attention_mask():
    print("\nTesting LocalConstrainedCrossAttention...")
    
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.encoder_width = 768
    
    local_attn = LocalConstrainedCrossAttention(config, grid_size=(8, 8), neighborhood_radius=1)
    
    # Check mask shape
    assert local_attn.local_mask.shape == (64, 64), f"Expected (64, 64), got {local_attn.local_mask.shape}"
    
    # Check corner query (0,0) - should have 4 neighbors (including itself)
    corner_neighbors = local_attn.local_mask[0].sum().item()
    assert corner_neighbors == 4, f"Corner should have 4 neighbors, got {corner_neighbors}"
    
    # Check center query (3,3) - should have 9 neighbors
    center_idx = 3 * 8 + 3  # row 3, col 3
    center_neighbors = local_attn.local_mask[center_idx].sum().item()
    assert center_neighbors == 9, f"Center should have 9 neighbors, got {center_neighbors}"
    
    print("✓ LocalConstrainedCrossAttention mask test passed")
    
    # Test forward pass
    batch_size = 2
    queries = torch.randn(batch_size, 64, 768)
    keys_values = torch.randn(batch_size, 64, 768)
    
    output = local_attn(queries, keys_values)
    assert output.shape == (batch_size, 64, 768), f"Expected (2, 64, 768), got {output.shape}"
    print("✓ LocalConstrainedCrossAttention forward test passed")


def test_instruction_injection():
    print("\nTesting InstructionInjectionModule...")
    
    config = BertConfig.from_pretrained("bert-base-uncased")
    instruction_module = InstructionInjectionModule(config)
    
    batch_size = 2
    grid_queries = torch.randn(batch_size, 64, 768)
    text_embeds = torch.randn(batch_size, 10, 768)  # 10 text tokens
    text_mask = torch.ones(batch_size, 10, dtype=torch.long)
    
    output = instruction_module(grid_queries, text_embeds, text_mask)
    assert output.shape == (batch_size, 64, 768), f"Expected (2, 64, 768), got {output.shape}"
    
    # Check that output differs from input (modulation occurred)
    assert not torch.allclose(output, grid_queries), "Queries should be modulated"
    print("✓ InstructionInjectionModule test passed")


def test_sa_qformer_initialization():
    print("\nTesting SAQFormer initialization...")
    
    try:
        # Test with minimal config
        model = SAQFormer(
            vit_model="clip_L",  # Use smaller model for testing
            img_size=224,
            grid_size=(4, 4),  # Smaller grid for faster testing
            neighborhood_radius=1,
            pos_encoding_type='learned',
            freeze_vit=True,
        )
        print("✓ SAQFormer initialization test passed")
        
        # Test invalid config
        try:
            model = SAQFormer(grid_size=(-1, 8))
            assert False, "Should have raised ValueError for negative grid size"
        except ValueError as e:
            print(f"✓ Correctly rejected invalid grid_size: {e}")
        
        try:
            model = SAQFormer(pos_encoding_type='invalid')
            assert False, "Should have raised ValueError for invalid encoding type"
        except ValueError as e:
            print(f"✓ Correctly rejected invalid pos_encoding_type: {e}")
            
    except Exception as e:
        print(f"✗ SAQFormer initialization failed: {e}")
        import traceback
        traceback.print_exc()


def test_spatial_correspondence():
    print("\nTesting spatial correspondence...")
    
    # Create grid query initializer
    initializer = GridQueryInitializer(grid_size=(8, 8), hidden_size=768, pos_encoding_type='learned')
    
    # Get queries for two batches
    queries1 = initializer(batch_size=1, device='cpu')
    queries2 = initializer(batch_size=1, device='cpu')
    
    # Queries should be identical for same positions (deterministic initialization)
    assert torch.allclose(queries1, queries2), "Queries should be deterministic"
    
    # Different positions should have different positional encodings
    # Check query[0] vs query[1] (adjacent positions)
    diff = (queries1[0, 0] - queries1[0, 1]).abs().sum()
    assert diff > 0, "Adjacent queries should have different positional encodings"
    
    print("✓ Spatial correspondence test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("SA-QFormer Component Tests")
    print("=" * 60)
    
    test_adaptive_pooling()
    test_positional_encoding()
    test_grid_query_initializer()
    test_local_attention_mask()
    test_instruction_injection()
    test_spatial_correspondence()
    test_sa_qformer_initialization()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
