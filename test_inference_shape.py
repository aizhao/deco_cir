import torch

# Test the inference shape logic
B = 32  # batch size
N = 2297  # number of targets
embed_dim = 256
num_tokens = 64

# Simulate fusion features and target features
fusion_feats = torch.randn(B, embed_dim)
target_feats = torch.randn(N, num_tokens, embed_dim)

print(f"fusion_feats shape: {fusion_feats.shape}")
print(f"target_feats shape: {target_feats.shape}")

# Method 1: Using element-wise multiplication (our new implementation)
fusion_feats_expanded = fusion_feats.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, embed_dim]
target_feats_expanded = target_feats.unsqueeze(0)  # [1, N, 64, embed_dim]
sim_matrix = (fusion_feats_expanded * target_feats_expanded).sum(-1)  # [B, N, 64]
sim_i2t, _ = sim_matrix.max(-1)  # [B, N]

print(f"\nMethod 1 (element-wise):")
print(f"sim_matrix shape: {sim_matrix.shape}")
print(f"sim_i2t shape: {sim_i2t.shape}")

# Verify it's [B, N]
assert sim_i2t.shape == (B, N), f"Expected shape ({B}, {N}), got {sim_i2t.shape}"
print("✓ Shape is correct!")
