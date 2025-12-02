# Before vs After: Critical Training Fixes

## Overview
This document shows the exact changes made to fix the performance drop after replacing Q-former with DeCo Projector.

---

## Fix 1: MLP Architecture with LayerNorm

### ❌ BEFORE (avgpool_projector.py)
```python
def build_net(self):
    # MLP projection layers
    modules = [nn.Linear(self.mm_hidden_size, self.llm_hidden_size)]
    for _ in range(1, self.layer_num):
        modules.append(nn.GELU())
        modules.append(nn.Linear(self.llm_hidden_size, self.llm_hidden_size))
    self.mlp_projector = nn.Sequential(*modules)
```

**Problem**: No LayerNorm → unstable feature distributions

### ✅ AFTER (avgpool_projector.py)
```python
def build_net(self):
    # Enhanced MLP projection with LayerNorm for stable training
    modules = [
        nn.Linear(self.mm_hidden_size, self.llm_hidden_size),
        nn.LayerNorm(self.llm_hidden_size),  # Critical!
        nn.GELU()
    ]
    for _ in range(1, self.layer_num):
        modules.append(nn.Linear(self.llm_hidden_size, self.llm_hidden_size))
        modules.append(nn.LayerNorm(self.llm_hidden_size))  # Critical!
        modules.append(nn.GELU())
    self.mlp_projector = nn.Sequential(*modules)
```

**Fix**: LayerNorm after each Linear layer → stable visual-text alignment

---

## Fix 2: Differential Learning Rates (FashionIQ)

### ❌ BEFORE (blip_fine_tune_2.py - FashionIQ function)
```python
# Define the optimizer, the loss and the grad scaler
optimizer = optim.AdamW(
    [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 
      'lr': learning_rate,
      'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])

scheduler = OneCycleLR(optimizer, max_lr=learning_rate, 
                      pct_start=1.5/num_epochs, div_factor=100., 
                      steps_per_epoch=len(relative_train_loader), 
                      epochs=num_epochs)
```

**Problem**: Same LR for pretrained BERT and random DeCo → can't converge

### ✅ AFTER (blip_fine_tune_2.py - FashionIQ function)
```python
# Define the optimizer with differential learning rates
# DeCo Projector (randomly initialized) needs 10-20x higher LR than pretrained BERT
projector_params = []
base_params = []

for name, param in blip_model.named_parameters():
    if not param.requires_grad:
        continue
    # DeCo Projector and visual position embeddings need higher LR
    if 'deco_projector' in name or 'visual_pos_embed' in name:
        projector_params.append(param)
    else:
        base_params.append(param)

print(f"Projector params: {len(projector_params)}, Base params: {len(base_params)}")

optimizer = optim.AdamW([
    {'params': base_params, 'lr': learning_rate, 
     'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05},
    {'params': projector_params, 'lr': learning_rate * 20,  # 20x higher!
     'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05}
])

scheduler = OneCycleLR(optimizer, max_lr=[learning_rate, learning_rate * 20],  # Two LRs!
                      pct_start=1.5/num_epochs, div_factor=100., 
                      steps_per_epoch=len(relative_train_loader), 
                      epochs=num_epochs)
```

**Fix**: 
- BERT: 2e-6 (small adjustments to pretrained weights)
- DeCo: 4e-5 (20x higher, learns visual mapping fast)

---

## Fix 3: Differential Learning Rates (CIRR)

### ❌ BEFORE (blip_fine_tune_2.py - CIRR function)
```python
# Define the optimizer, the loss and the grad scaler
optimizer = optim.AdamW(
    [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 
      'lr': learning_rate,
      'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])

scheduler = OneCycleLR(optimizer, max_lr=learning_rate, 
                      pct_start=1/50, 
                      steps_per_epoch=len(relative_train_loader), 
                      epochs=80)
```

**Problem**: Same as FashionIQ

### ✅ AFTER (blip_fine_tune_2.py - CIRR function)
```python
# Define the optimizer with differential learning rates
projector_params = []
base_params = []

for name, param in blip_model.named_parameters():
    if not param.requires_grad:
        continue
    if 'deco_projector' in name or 'visual_pos_embed' in name:
        projector_params.append(param)
    else:
        base_params.append(param)

print(f"Projector params: {len(projector_params)}, Base params: {len(base_params)}")

optimizer = optim.AdamW([
    {'params': base_params, 'lr': learning_rate, 
     'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05},
    {'params': projector_params, 'lr': learning_rate * 20, 
     'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05}
])

scheduler = OneCycleLR(optimizer, max_lr=[learning_rate, learning_rate * 20], 
                      pct_start=1/50, 
                      steps_per_epoch=len(relative_train_loader), 
                      epochs=80)
```

**Fix**: Same differential LR strategy

---

## Fix 4: Warmup Strategy (FashionIQ)

### ❌ BEFORE (blip_fine_tune_2.py - FashionIQ training loop)
```python
# Start with the training loop
print('Training loop started')
for epoch in range(num_epochs):
    train_running_results = {'images_in_epoch': 0}
    train_bar = tqdm(relative_train_loader, ncols=150)
    for idx, (reference_images, target_images, captions) in enumerate(train_bar):
        # ... training code
```

**Problem**: BERT receives random noise from DeCo immediately

### ✅ AFTER (blip_fine_tune_2.py - FashionIQ training loop)
```python
# Start with the training loop
print('Training loop started')
warmup_epochs = 1  # First epoch: freeze BERT, only train DeCo Projector

for epoch in range(num_epochs):
    # Warmup strategy: freeze BERT in first epoch to let DeCo Projector adapt
    if epoch < warmup_epochs:
        print(f"[Warmup Epoch {epoch}] Freezing BERT, only training DeCo Projector")
        for name, param in blip_model.named_parameters():
            if 'Qformer' in name and 'deco_projector' not in name:
                param.requires_grad = False
    elif epoch == warmup_epochs:
        print(f"[Epoch {epoch}] Unfreezing BERT, full model training")
        for name, param in blip_model.named_parameters():
            if 'Qformer' in name:
                param.requires_grad = True
    
    train_running_results = {'images_in_epoch': 0}
    train_bar = tqdm(relative_train_loader, ncols=150)
    for idx, (reference_images, target_images, captions) in enumerate(train_bar):
        # ... training code
```

**Fix**: Epoch 0 freezes BERT, lets DeCo learn basic visual compression first

---

## Fix 5: Warmup Strategy (CIRR)

### ❌ BEFORE (blip_fine_tune_2.py - CIRR training loop)
```python
for epoch in range(num_epochs):
    train_running_results = {'images_in_epoch': 0}
    train_bar = tqdm(relative_train_loader, ncols=150)
    for idx, (reference_images, target_images, captions) in enumerate(train_bar):
        # ... training code
```

**Problem**: Same as FashionIQ

### ✅ AFTER (blip_fine_tune_2.py - CIRR training loop)
```python
warmup_epochs = 1  # First epoch: freeze BERT, only train DeCo Projector

for epoch in range(num_epochs):
    # Warmup strategy: freeze BERT in first epoch to let DeCo Projector adapt
    if epoch < warmup_epochs:
        print(f"[Warmup Epoch {epoch}] Freezing BERT, only training DeCo Projector")
        for name, param in blip_model.named_parameters():
            if 'Qformer' in name and 'deco_projector' not in name:
                param.requires_grad = False
    elif epoch == warmup_epochs:
        print(f"[Epoch {epoch}] Unfreezing BERT, full model training")
        for name, param in blip_model.named_parameters():
            if 'Qformer' in name:
                param.requires_grad = True
    
    train_running_results = {'images_in_epoch': 0}
    train_bar = tqdm(relative_train_loader, ncols=150)
    for idx, (reference_images, target_images, captions) in enumerate(train_bar):
        # ... training code
```

**Fix**: Same warmup strategy

---

## Fix 6: Adaptive TIC Loss Weighting (FashionIQ)

### ❌ BEFORE (blip_fine_tune_2.py - FashionIQ loss computation)
```python
# Extract the features, compute the logits and the loss
with torch.cuda.amp.autocast():
    loss_dict = blip_model({"image":reference_images, 
                           "target":target_images, 
                           "text_input":captions})
    loss = 0.
    for key in loss_dict.keys():
        loss += loss_dict[key]
```

**Problem**: TIC loss pushes features apart before DeCo learns anything

### ✅ AFTER (blip_fine_tune_2.py - FashionIQ loss computation)
```python
# Extract the features, compute the logits and the loss
with torch.cuda.amp.autocast():
    loss_dict = blip_model({"image":reference_images, 
                           "target":target_images, 
                           "text_input":captions})
    loss = 0.
    
    # Adaptive loss weighting: reduce TIC loss in early epochs
    tic_weight = 0.01 if epoch < 5 else 1.0
    
    for key in loss_dict.keys():
        if key == 'loss_tic':
            loss += tic_weight * loss_dict[key]  # Reduced early!
        else:
            loss += loss_dict[key]
```

**Fix**: TIC weight = 0.01 for epochs 0-4, then 1.0 from epoch 5

---

## Fix 7: Adaptive TIC Loss Weighting (CIRR)

### ❌ BEFORE (blip_fine_tune_2.py - CIRR loss computation)
```python
# Extract the features, compute the logits and the loss
with torch.cuda.amp.autocast():
    loss_dict = blip_model({"image":reference_images, 
                           "target":target_images, 
                           "text_input":captions})
    loss = 0.
    for key in loss_dict.keys():
        if key != 'loss_itc':
            loss += kwargs[key] * loss_dict[key]
        else:
            loss += loss_dict[key]
```

**Problem**: Same as FashionIQ

### ✅ AFTER (blip_fine_tune_2.py - CIRR loss computation)
```python
# Extract the features, compute the logits and the loss
with torch.cuda.amp.autocast():
    loss_dict = blip_model({"image":reference_images, 
                           "target":target_images, 
                           "text_input":captions})
    loss = 0.
    
    # Adaptive loss weighting: reduce TIC loss in early epochs
    tic_weight = 0.01 if epoch < 5 else kwargs.get('loss_tic', 1.0)
    
    for key in loss_dict.keys():
        if key == 'loss_tic':
            loss += tic_weight * loss_dict[key]  # Reduced early!
        elif key != 'loss_itc':
            loss += kwargs[key] * loss_dict[key]
        else:
            loss += loss_dict[key]
```

**Fix**: Same adaptive weighting with kwargs support

---

## Summary of Changes

| Fix | File | Lines Changed | Impact |
|-----|------|---------------|--------|
| LayerNorm in MLP | avgpool_projector.py | ~15 | High - Stabilizes features |
| Differential LR (FashionIQ) | blip_fine_tune_2.py | ~20 | **Critical** - Enables learning |
| Differential LR (CIRR) | blip_fine_tune_2.py | ~20 | **Critical** - Enables learning |
| Warmup (FashionIQ) | blip_fine_tune_2.py | ~12 | High - Prevents BERT damage |
| Warmup (CIRR) | blip_fine_tune_2.py | ~12 | High - Prevents BERT damage |
| Adaptive TIC (FashionIQ) | blip_fine_tune_2.py | ~8 | Medium - Improves convergence |
| Adaptive TIC (CIRR) | blip_fine_tune_2.py | ~8 | Medium - Improves convergence |

**Total**: ~95 lines changed across 2 files

---

## Expected Performance Recovery

### Training Dynamics

**Epoch 0 (Warmup)**:
- Only DeCo trains (BERT frozen)
- loss_itc should decrease from ~7.0 to ~5.0
- TIC weight = 0.01 (minimal impact)

**Epoch 1-5 (Early Training)**:
- Full model training
- loss_itc should reach ~3.0-4.0
- R@5 should reach 5-8%
- TIC weight = 0.01 (still minimal)

**Epoch 6-20 (Mid Training)**:
- loss_itc should reach ~2.0-2.5
- R@5 should reach 10-12%
- TIC weight = 1.0 (full strength)
- loss_tic starts decreasing

**Epoch 20+ (Late Training)**:
- loss_itc should stabilize ~1.5-2.0
- R@5 should reach 12-15%
- All losses stable

---

## Verification

Run these commands to verify fixes:

```bash
# Check syntax
cd SPRC/src
python -m py_compile ../test_fixes.py
python -m py_compile blip_fine_tune_2.py
python -m py_compile lavis/models/projectors/avgpool_projector.py

# Run verification tests
cd SPRC
python test_fixes.py

# Start training
cd src
python blip_fine_tune_2.py \
  --dataset CIRR \
  --blip-model-name blip2_avgpool_cir_align_prompt \
  --learning-rate 2e-6 \
  --batch-size 128 \
  --num-epochs 50 \
  --loss-dense 0.5 \
  --loss-tic 0.3 \
  --save-training --save-best
```
