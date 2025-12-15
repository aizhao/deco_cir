"""
GloFND (Global False Negative Detection) Loss Module for SPRC
Adapted from GloFND paper: https://github.com/vibalcam/GloFND

This module implements the GloFND mechanism to detect and filter false negative samples
in contrastive learning, preventing the model from pushing away semantically similar samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import logging


class LambdaThreshold(nn.Module):
    """
    LambdaThreshold module for learning per-sample thresholds to detect false negatives.
    
    This module maintains a learnable threshold (lda) for each sample in the dataset.
    Samples with similarity above the threshold are considered false negatives and filtered out.
    """
    
    def __init__(
        self,
        data_size: int,
        alpha: float = 1e-3,
        lr_lda: float = 0.05,
        lda_beta1: float = 0.9,
        lda_beta2: float = 0.98,
        eps: float = 1e-8,
        start_update: int = 15,
        lda_start: int = 15,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            data_size: Total number of samples in the dataset
            alpha: Target false negative rate (controls sensitivity)
            lr_lda: Learning rate for lambda threshold updates
            lda_beta1: Adam beta1 for lambda threshold optimizer
            lda_beta2: Adam beta2 for lambda threshold optimizer
            eps: Small epsilon for numerical stability
            start_update: Epoch to start updating lambda thresholds
            lda_start: Epoch to start filtering false negatives
            device: Device to use (for compatibility, but buffers are stored on CPU)
        """
        super().__init__()
        self.data_size = data_size
        self.alpha = alpha
        self.lr_lda = lr_lda
        self.lda_beta1 = lda_beta1
        self.lda_beta2 = lda_beta2
        self.eps = eps
        self.start_update = start_update
        self.lda_start = lda_start
        
        # Store buffers on CPU to save GPU memory
        # lda: learnable threshold for each sample [data_size, 1]
        # self.register_buffer("lda", torch.ones(data_size, device="cpu").reshape(-1, 1))
        self.register_buffer("lda", torch.zeros(data_size, device="cpu").reshape(-1, 1))
        # m_grad: first moment estimate for Adam [data_size, 1]
        self.register_buffer("m_grad", torch.zeros(data_size, device="cpu").reshape(-1, 1))
        # v_grad: second moment estimate for Adam [data_size, 1]
        self.register_buffer("v_grad", torch.zeros(data_size, device="cpu").reshape(-1, 1))
        
        logging.info(
            f"LambdaThreshold initialized: data_size={data_size}, alpha={alpha}, "
            f"lr_lda={lr_lda}, start_update={start_update}, lda_start={lda_start}"
        )

    @torch.no_grad()
    def update(
        self,
        sim: torch.Tensor,  # [B, B] similarity matrix
        idx: torch.Tensor,  # [B] sample indices
        neg_mask: torch.Tensor,  # [B, B] negative mask (1=negative, 0=positive/self)
        epoch: int,
    ) -> Dict[str, float]:
        """
        Update lambda thresholds based on current similarities.
        
        Args:
            sim: Similarity matrix [B, B]
            idx: Sample indices [B]
            neg_mask: Negative mask [B, B] (1 for negatives, 0 for positives/self)
            epoch: Current training epoch
            
        Returns:
            Dictionary with logging statistics
        """
        if epoch < self.start_update:
            return {}
        
        batch_size = sim.shape[0]
        buffer_device = self.lda.device
        idx_buf = idx.to(buffer_device)
        lda_orig = self.lda.index_select(0, idx_buf).to(sim.device)  # [B, 1]

        # Optional warm-start: initialize with batch quantile at first update step
        if epoch == self.start_update:
            lda_list = []
            for i in range(batch_size):
                neg_sim = sim[i][neg_mask[i].bool()]
                if neg_sim.numel() > 0:
                    q = torch.quantile(neg_sim.float(), 1 - self.alpha, dim=0, keepdim=True)
                else:
                    q = lda_orig[i]
                lda_list.append(q)
            lda_orig = torch.stack(lda_list, dim=0).to(sim.device)  # [B,1]

        lda = lda_orig.expand(-1, sim.size(1))   # [B, B]
        
        # Compute gradient: alpha - (proportion of negatives with sim > lda)
        # This encourages lda to be set such that alpha fraction of negatives are filtered
        g_mask = (sim > lda).float() * neg_mask.float()  # [B, B]
        num_negatives = neg_mask.sum(dim=-1, keepdim=True).float()  # [B, 1]
        g_mask_sum = g_mask.sum(dim=-1, keepdim=True)  # [B, 1]
        
        # Gradient: positive when too many negatives are above threshold (need to increase lda)
        # negative when too few negatives are above threshold (need to decrease lda)
        lda_grad = self.alpha - g_mask_sum / (num_negatives + self.eps)  # [B, 1]
        
        # Adam update for lambda thresholds
        m_grad = self.m_grad.index_select(0, idx_buf).to(sim.device)
        v_grad = self.v_grad.index_select(0, idx_buf).to(sim.device)
        
        m_grad = self.lda_beta1 * m_grad + (1 - self.lda_beta1) * lda_grad
        v_grad = self.lda_beta2 * v_grad + (1 - self.lda_beta2) * (lda_grad ** 2)
        
        # Bias correction
        m_hat = m_grad / (1 - self.lda_beta1 ** (epoch + 1))
        v_hat = v_grad / (1 - self.lda_beta2 ** (epoch + 1))
        
        # Update lda (clamp to [-1, 1] range)
        # Allow lambda to follow logits scale (logits already clamped elsewhere)
        lda_new = (lda_orig - self.lr_lda * m_hat / (v_hat.sqrt() + self.eps)).clamp(min=-20, max=20)
        
        # Write back to CPU buffers
        lda_new_buf = lda_new.to(buffer_device)
        m_grad_buf = m_grad.to(buffer_device)
        v_grad_buf = v_grad.to(buffer_device)
        self.lda.index_copy_(0, idx_buf, lda_new_buf)
        self.m_grad.index_copy_(0, idx_buf, m_grad_buf)
        self.v_grad.index_copy_(0, idx_buf, v_grad_buf)
        
        return {
            'lda_mean': lda.mean().item(),
            'lda_std': lda.std().item(),
            'lda_min': lda.min().item(),
            'lda_max': lda.max().item(),
            'filtered_ratio': (g_mask_sum / (num_negatives + self.eps)).mean().item(),
        }

    @torch.no_grad()
    def get_mask(
        self,
        sim: torch.Tensor,  # [B, B]
        idx: torch.Tensor,  # [B]
        neg_mask: torch.Tensor,  # [B, B]
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get mask to filter false negatives based on learned thresholds.
        
        Args:
            sim: Similarity matrix [B, B]
            idx: Sample indices [B]
            neg_mask: Negative mask [B, B]
            epoch: Current training epoch
            
        Returns:
            mask: [B, B] Filtered mask (1=keep, 0=filter out)
            mask_sum: [B, 1] Number of negatives kept per sample
        """
        if epoch < self.lda_start:
            # Before lda_start, don't filter any negatives (use standard InfoNCE)
            mask_sum = neg_mask.sum(dim=-1, keepdim=True).float()
            return neg_mask.float(), mask_sum
        
        idx_cpu = idx.to("cpu")
        lda_orig = self.lda[idx_cpu].to(sim.device)  # [B, 1]
        lda = lda_orig.expand(-1, sim.size(1))   # [B, B]
        
        # Filter negatives with similarity > lda (these are false negatives)
        # mask = 1 for negatives with sim <= lda (true negatives)
        # mask = 0 for negatives with sim > lda (false negatives) or positives/self
        mask = (sim <= lda).float() * neg_mask.float()  # [B, B]
        mask_sum = mask.sum(dim=-1, keepdim=True)  # [B, 1]
        
        return mask, mask_sum


class GloFNDLoss(nn.Module):
    """
    GloFND-enhanced contrastive loss that replaces standard InfoNCE.
    
    This loss function:
    1. Computes similarity matrix between anchors and targets
    2. Updates lambda thresholds to detect false negatives
    3. Filters false negatives from the loss computation
    4. Computes standard cross-entropy loss on filtered similarities
    """
    
    def __init__(
        self,
        data_size: int,
        temperature: float = 0.07,
        alpha: float = 1e-3,
        lr_lda: float = 0.05,
        start_update: int = 15,
        lda_start: int = 15,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            data_size: Total number of samples in the dataset
            temperature: Temperature scaling for similarity
            alpha: Target false negative rate
            lr_lda: Learning rate for lambda threshold updates
            start_update: Epoch to start updating lambda thresholds
            lda_start: Epoch to start filtering false negatives
            device: Device (for compatibility)
        """
        super().__init__()
        # Temperature can be updated dynamically; store as tensor for easy clamp
        self.register_buffer("temperature", torch.tensor(temperature))
        self.lambda_threshold = LambdaThreshold(
            data_size=data_size,
            alpha=alpha,
            lr_lda=lr_lda,
            start_update=start_update,
            lda_start=lda_start,
            device=device,
        )
        self.epoch = 0
        
    def set_epoch(self, epoch: int):
        """Set current epoch for controlling lambda updates and filtering."""
        self.epoch = epoch
        
    def forward(
        self,
        anchor_features: torch.Tensor,  # [B, D] Anchor features (fusion_feats or text_only_feat)
        target_features: torch.Tensor,  # [B, D] Target image features
        indices: torch.Tensor,  # [B] Sample indices for updating lambda thresholds
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GloFND-enhanced contrastive loss.
        
        Args:
            anchor_features: [B, D] Anchor features (e.g., fusion_feats or text_only_feat)
            target_features: [B, D] Target image features (aggregated from target tokens)
            indices: [B] Sample indices for lambda threshold updates
            
        Returns:
            loss: Scalar loss value
            log_dict: Dictionary with logging statistics
        """
        batch_size = anchor_features.shape[0]
        device = anchor_features.device
        
        # Compute similarity matrix [B, B]
        # Use current temperature (can be updated from model's learnable temp)
        temp = self.temperature
        if isinstance(temp, torch.Tensor):
            temp = temp.item()
        temp = max(temp, 1e-3)  # clamp to avoid exploding logits
        logits = (anchor_features @ target_features.t()) / temp  # [B, B]
        
        # Build negative mask (exclude diagonal, which is positive pairs)
        neg_mask = 1 - torch.eye(batch_size, device=device)  # [B, B]
        
        # Update lambda thresholds (detached to avoid affecting gradients)
        log_dict = self.lambda_threshold.update(
            sim=logits.detach(),
            idx=indices,
            neg_mask=neg_mask,
            epoch=self.epoch,
        )
        
        # Get mask to filter false negatives
        mask, mask_sum = self.lambda_threshold.get_mask(
            sim=logits.detach(),
            idx=indices,
            neg_mask=neg_mask,
            epoch=self.epoch,
        )
        
        # Apply mask: keep positives (diagonal) always, filter selected negatives
        pos_mask = torch.eye(batch_size, device=device)
        mask_with_pos = (mask + pos_mask).clamp(max=1)  # ensure positives are kept
        masked_logits = torch.where(mask_with_pos.bool(), logits, torch.full_like(logits, -1e4))
        masked_logits = torch.clamp(masked_logits, -50, 50)
        
        # Standard cross-entropy loss (targets are diagonal indices)
        targets = torch.arange(batch_size, device=device)
        loss = F.cross_entropy(masked_logits, targets)
        
        log_dict.update({
            'loss': loss.item(),
            'num_negatives_per_sample': mask_sum.mean().item(),
            'num_negatives_total': mask_sum.sum().item(),
        })
        
        return loss, log_dict

