"""Recognition probe: embed a candidate menu (full or sparse) into a 256-dim vector.

The SAME encoder embeds both the agent's memory trace (sparse subset) and the
reference image (full menu). Trained contrastively so a menu's embedding is close
to its own image's reference and far from other images'. Forces retention of
distinguishing detail, not just semantic gist.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.feature_dataset import (
    MAX_SLOTS, MENU_SIZE,
    FAM_SIGLIP, FAM_DINOV2_CLS, FAM_OBJECT_SLOT, FAM_LOWLEVEL,
    DIM_SIGLIP, DIM_DINOV2, DIM_LOWLEVEL,
)

N_FAMILIES = 4


class RecognitionProbe(nn.Module):
    def __init__(self, d_model: int = 256, embed_dim: int = 32, n_heads: int = 4,
                 k: int = 4):
        super().__init__()
        self.d_model = d_model
        self.k = k

        # Per-family adapters (same as attribute probe)
        self.adapt_siglip = nn.Linear(DIM_SIGLIP, d_model)
        self.adapt_dinov2 = nn.Linear(DIM_DINOV2, d_model)
        self.adapt_lowlevel = nn.Linear(DIM_LOWLEVEL, d_model)
        self.family_embed = nn.Embedding(N_FAMILIES, d_model)

        # Attention pooling: a single learned query attends over candidates
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Projection head to final embedding space
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, embed_dim),
        )

        fam_ids = ([FAM_SIGLIP] + [FAM_DINOV2_CLS]
                   + [FAM_OBJECT_SLOT] * MAX_SLOTS + [FAM_LOWLEVEL])
        self.register_buffer("family_ids", torch.tensor(fam_ids, dtype=torch.long))

    # def build_candidates(self, batch: dict, sample_k: bool):
    #     """Project all families to d_model, assemble (B, MENU_SIZE, D) + valid mask.

    #     When sample_k is True, pick exactly self.k valid candidates per image
    #     (or all of them if fewer than k are valid). Globals are NOT protected:
    #     siglip and dinov2_cls are eligible for sampling like any other candidate.
    #     """
    #     B = batch["siglip"].shape[0]
    #     device = batch["siglip"].device

    #     siglip = self.adapt_siglip(batch["siglip"]).unsqueeze(1)
    #     dinov2_cls = self.adapt_dinov2(batch["dinov2_cls"]).unsqueeze(1)
    #     slots = self.adapt_dinov2(batch["slots"])
    #     lowlevel = self.adapt_lowlevel(batch["lowlevel"]).unsqueeze(1)

    #     candidates = torch.cat([siglip, dinov2_cls, slots, lowlevel], dim=1)
    #     fam = self.family_embed(self.family_ids).unsqueeze(0)
    #     candidates = candidates + fam

    #     valid = torch.ones(B, MENU_SIZE, dtype=torch.bool, device=device)
    #     valid[:, 2:2 + MAX_SLOTS] = batch["slot_valid"]

    #     if sample_k:
    #         scores = torch.rand(B, MENU_SIZE, device=device)
    #         scores = scores.masked_fill(~valid, float("-inf"))
    #         _, top_idx = scores.topk(min(self.k, MENU_SIZE), dim=1)
    #         sampled = torch.zeros_like(valid)
    #         sampled.scatter_(1, top_idx, True)
    #         valid = valid & sampled   # AND drops -inf picks when fewer than k were valid

    #     return candidates, valid

    def build_candidates(self, batch: dict, sample_k: bool = True,
                         selection_mask: torch.Tensor = None):
        """Project families to d_model, assemble (B, MENU_SIZE, D) + valid mask.

        If selection_mask is given (B, MENU_SIZE bool), use it as the active set
        (the agent's choice). Otherwise, if sample_k, randomly keep K valid candidates.
        Otherwise use the full valid menu.
        """
        B = batch["siglip"].shape[0]
        device = batch["siglip"].device

        siglip = self.adapt_siglip(batch["siglip"]).unsqueeze(1)
        dinov2_cls = self.adapt_dinov2(batch["dinov2_cls"]).unsqueeze(1)
        slots = self.adapt_dinov2(batch["slots"])
        lowlevel = self.adapt_lowlevel(batch["lowlevel"]).unsqueeze(1)

        candidates = torch.cat([siglip, dinov2_cls, slots, lowlevel], dim=1)
        fam = self.family_embed(self.family_ids).unsqueeze(0)
        candidates = candidates + fam

        # Base validity: which slots are real (padding excluded)
        base_valid = torch.ones(B, MENU_SIZE, dtype=torch.bool, device=device)
        base_valid[:, 2:2 + MAX_SLOTS] = batch["slot_valid"]

        if selection_mask is not None:
            # Use the agent's explicit selection, intersected with real candidates
            valid = base_valid & selection_mask.to(device)
        elif sample_k:
            valid = self._sample_k_mask(base_valid, device)
        else:
            valid = base_valid

        return candidates, valid
    
    def _sample_k_mask(self, base_valid: torch.Tensor, device) -> torch.Tensor:
        """Randomly keep exactly K valid candidates per row (or all if fewer than K)."""
        B, M = base_valid.shape
        valid = torch.zeros_like(base_valid)
        for i in range(B):
            idxs = base_valid[i].nonzero(as_tuple=True)[0]
            if len(idxs) <= self.k:
                valid[i, idxs] = True
            else:
                chosen = idxs[torch.randperm(len(idxs), device=device)[:self.k]]
                valid[i, chosen] = True
        return valid

    def encode(self, batch: dict, sample_k: bool = True, selection_mask=None) -> torch.Tensor:
        candidates, valid = self.build_candidates(batch, sample_k, selection_mask)
        B = candidates.shape[0]

        # Guard: rows with zero valid candidates would make attention produce NaN.
        # Force at least one attended position (we'll zero their output below).
        empty_rows = ~valid.any(dim=1)               # (B,) True where nothing is valid
        if empty_rows.any():
            valid = valid.clone()
            valid[empty_rows, 0] = True              # let them attend to slot 0 to avoid NaN

        query = self.pool_query.expand(B, -1, -1)
        key_padding_mask = ~valid
        pooled, _ = self.pool_attn(query, candidates, candidates,
                                   key_padding_mask=key_padding_mask)
        pooled = pooled.squeeze(1)

        embed = self.proj(pooled)
        embed = F.normalize(embed, dim=-1)

        # Zero out embeddings for genuinely-empty selections so they can't spuriously match
        if empty_rows.any():
            embed = embed.clone()
            embed[empty_rows] = 0.0

        return embed

    def forward(self, batch: dict, sample_k: bool = True,
                selection_mask: torch.Tensor = None) -> torch.Tensor:
        return self.encode(batch, sample_k=sample_k, selection_mask=selection_mask)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.data.feature_dataset import FeatureDataset, collate_menus

    ds = FeatureDataset(split="train")
    loader = DataLoader(ds, batch_size=8, num_workers=0, collate_fn=collate_menus)
    batch = next(iter(loader))

    probe = RecognitionProbe()
    probe.train()

    # Embed the same batch twice: once "full" (all valid), once "K-sampled" (anchor)
    emb_full = probe.encode(batch, sample_k=False)
    emb_sparse = probe.encode(batch, sample_k=True)
    print(f"Embedding shape: {tuple(emb_full.shape)}")           # (8, 256)
    print(f"L2 norms (should be ~1.0): {emb_full.norm(dim=-1)[:4].tolist()}")

    # Cosine sim between full and sparse embeddings of the SAME images
    # (should be positive — same image, even with dropout)
    cos_same = (emb_full * emb_sparse).sum(dim=-1)
    print(f"Cosine(full, sparse) for same images: {cos_same[:4].tolist()}")

    print(f"Param count: {sum(p.numel() for p in probe.parameters()):,}")