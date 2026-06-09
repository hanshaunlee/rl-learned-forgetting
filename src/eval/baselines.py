"""Discrete baseline selectors. Each exposes select(batch, k) -> (mask, valid),
both (B, MENU_SIZE) bool, fed straight into RewardFunction. The continuous
compressors (PCA/AE/VIB) live in continuous.py — different interface."""
from __future__ import annotations

import numpy as np
import torch

from src.data.feature_dataset import MAX_SLOTS, MENU_SIZE


# Menu layout: [siglip, dinov2_cls, slot_0..slot_11, lowlevel]
SIGLIP_POS = 0
DINOV2_CLS_POS = 1
SLOTS_START = 2
SLOTS_END = SLOTS_START + MAX_SLOTS
LOWLEVEL_POS = SLOTS_END


def base_valid_mask(batch: dict) -> torch.Tensor:
    """(B, MENU_SIZE) bool — globals always valid, slots gated by slot_valid."""
    B = batch["siglip"].shape[0]
    device = batch["siglip"].device
    valid = torch.ones(B, MENU_SIZE, dtype=torch.bool, device=device)
    valid[:, SLOTS_START:SLOTS_END] = batch["slot_valid"]
    return valid


def topk_mask(scores: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic top-K under a validity mask; picks min(k, n_valid) per row."""
    B, M = scores.shape
    masked = scores.masked_fill(~valid, float("-inf"))
    mask = torch.zeros(B, M, dtype=torch.bool, device=scores.device)
    n_valid = valid.sum(dim=1)
    # sort once, keep the leading entries up to each row's budget (handles n_valid < k)
    order = masked.argsort(dim=1, descending=True)
    take = torch.minimum(n_valid, torch.full_like(n_valid, k))
    pos = torch.arange(M, device=scores.device).unsqueeze(0)
    keep_pos = pos < take.unsqueeze(1)
    mask.scatter_(1, order, keep_pos)
    return mask


class BaselineSelector:
    name: str = "base"

    def select(self, batch: dict, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class RandomSelector(BaselineSelector):
    """Uniform random K-subset over valid candidates. The floor."""
    name = "random"

    def __init__(self, seed: int = 231):
        self.rng = np.random.default_rng(seed)

    def select(self, batch, k):
        valid = base_valid_mask(batch)
        # random scores + top-K == sampling without replacement, but vectorised
        # and the n_valid < k edge case falls out of topk_mask
        B, M = valid.shape
        scores = torch.from_numpy(
            self.rng.random((B, M)).astype(np.float32)
        ).to(valid.device)
        return topk_mask(scores, valid, k), valid


class GlobalsOnlySelector(BaselineSelector):
    """Commit to the global gists in priority order: siglip, dinov2_cls, then lowlevel.

    Caps out at 3 tokens regardless of k. Any learned selector should beat this
    once object slots actually carry information.
    """
    name = "globals_only"
    PRIORITY = [SIGLIP_POS, DINOV2_CLS_POS, LOWLEVEL_POS]

    def select(self, batch, k):
        valid = base_valid_mask(batch)
        B, M = valid.shape
        mask = torch.zeros(B, M, dtype=torch.bool, device=valid.device)
        for pos in self.PRIORITY[:k]:
            mask[:, pos] = True
        return mask & valid, valid


class SaliencySelector(BaselineSelector):
    """Top-K slots by area (extractor stores slots largest-first), globals fill the rest.

    Stand-in for "top-K by DINOv2 attention norm" — per-patch norms weren't cached,
    and area correlates with them well enough for a baseline.
    """
    name = "saliency"

    def __init__(self):
        # slots score above all globals, so slots go first and globals only
        # backfill when an image runs out of valid slots
        scores = torch.zeros(MENU_SIZE)
        for i in range(MAX_SLOTS):
            scores[SLOTS_START + i] = MAX_SLOTS - i
        scores[SIGLIP_POS] = 0.3
        scores[DINOV2_CLS_POS] = 0.2
        scores[LOWLEVEL_POS] = 0.1
        self.priority_scores = scores

    def select(self, batch, k):
        valid = base_valid_mask(batch)
        B = valid.shape[0]
        scores = self.priority_scores.to(valid.device).unsqueeze(0).expand(B, -1).contiguous()
        return topk_mask(scores, valid, k), valid
