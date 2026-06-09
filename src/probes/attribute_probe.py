"""Attribute QA probe: given a candidate menu + a question, predict yes/no.

Pretrained on full menus (with random slot dropout for robustness), then frozen
during agent training. The probe is part of the agent's environment — it converts
the agent's retained features into a reward signal.
"""
import torch
import torch.nn as nn

from src.data.feature_dataset import (
    MAX_SLOTS, MENU_SIZE,
    FAM_SIGLIP, FAM_DINOV2_CLS, FAM_OBJECT_SLOT, FAM_LOWLEVEL,
    DIM_SIGLIP, DIM_DINOV2, DIM_LOWLEVEL,
)

N_FAMILIES = 4
N_QUESTION_TYPES = 3   # object_presence, spatial, count

# A question is identified by (qtype, qid_index). For object presence, qid_index
# is the COCO category index (0-79). For spatial, 0-3. For count, 0-2.
# We give every distinct question template its own embedding.
N_OBJECT_CATEGORIES = 80
N_SPATIAL = 4
N_COUNT = 3
N_QUESTION_TEMPLATES = N_OBJECT_CATEGORIES + N_SPATIAL + N_COUNT  # 87


class AttributeProbe(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                k: int = 4):
        super().__init__()
        self.d_model = d_model
        self.k = k

        # Per-family adapters: raw dim -> common d_model
        self.adapt_siglip = nn.Linear(DIM_SIGLIP, d_model)
        self.adapt_dinov2 = nn.Linear(DIM_DINOV2, d_model)   # shared for cls + slots
        self.adapt_lowlevel = nn.Linear(DIM_LOWLEVEL, d_model)

        # Learned embeddings
        self.family_embed = nn.Embedding(N_FAMILIES, d_model)
        self.question_embed = nn.Embedding(N_QUESTION_TEMPLATES, d_model)

        # Set Transformer over [question_token, candidate_1, ..., candidate_15]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Binary head on the question token's output
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # Family ID tensor for the fixed layout (1 siglip, 1 dinov2_cls, 12 slots, 1 lowlevel)
        fam_ids = ([FAM_SIGLIP] + [FAM_DINOV2_CLS]
                   + [FAM_OBJECT_SLOT] * MAX_SLOTS + [FAM_LOWLEVEL])
        self.register_buffer("family_ids", torch.tensor(fam_ids, dtype=torch.long))

    # def build_candidates(self, batch: dict, apply_dropout: bool):
    #     """Project all families into d_model and assemble (B, MENU_SIZE, d_model) + valid mask."""
    #     B = batch["siglip"].shape[0]
    #     device = batch["siglip"].device

    #     siglip = self.adapt_siglip(batch["siglip"]).unsqueeze(1)      # (B, 1, D)
    #     dinov2_cls = self.adapt_dinov2(batch["dinov2_cls"]).unsqueeze(1)  # (B, 1, D)
    #     slots = self.adapt_dinov2(batch["slots"])                    # (B, MAX_SLOTS, D)
    #     lowlevel = self.adapt_lowlevel(batch["lowlevel"]).unsqueeze(1)    # (B, 1, D)

    #     candidates = torch.cat([siglip, dinov2_cls, slots, lowlevel], dim=1)  # (B, MENU_SIZE, D)

    #     # Add family embeddings
    #     fam = self.family_embed(self.family_ids).unsqueeze(0)        # (1, MENU_SIZE, D)
    #     candidates = candidates + fam

    #     # Validity: siglip, dinov2_cls, lowlevel always valid; slots per slot_valid
    #     valid = torch.ones(B, MENU_SIZE, dtype=torch.bool, device=device)
    #     valid[:, 2:2 + MAX_SLOTS] = batch["slot_valid"]

    #     # Option B: random slot dropout during training
    #     if apply_dropout and self.training:
    #         drop = torch.rand(B, MENU_SIZE, device=device) < self.slot_dropout
    #         # Never drop the always-on global tokens (keep at least the gist)
    #         drop[:, 0] = False   # siglip
    #         drop[:, 1] = False   # dinov2_cls
    #         valid = valid & ~drop

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

    def forward(self, batch: dict, question_template_idx: torch.Tensor,
                sample_k: bool = True, selection_mask: torch.Tensor = None):
        """
        batch: collated menu dict
        question_template_idx: (B,) long — which question template (0..86)
        sample_k: if True and no selection_mask, internally sample K candidates
        selection_mask: (B, MENU_SIZE) bool — explicit agent selection (overrides sample_k)
        Returns: (B,) logits for yes/no.
        """
        candidates, valid = self.build_candidates(batch, sample_k=sample_k,
                                                   selection_mask=selection_mask)
        B = candidates.shape[0]

        q_tok = self.question_embed(question_template_idx).unsqueeze(1)
        tokens = torch.cat([q_tok, candidates], dim=1)

        q_valid = torch.ones(B, 1, dtype=torch.bool, device=valid.device)
        full_valid = torch.cat([q_valid, valid], dim=1)
        pad_mask = ~full_valid

        encoded = self.transformer(tokens, src_key_padding_mask=pad_mask)
        q_out = encoded[:, 0]
        logit = self.head(q_out).squeeze(-1)
        return logit


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.data.feature_dataset import FeatureDataset, collate_menus

    ds = FeatureDataset(split="train")
    loader = DataLoader(ds, batch_size=8, num_workers=0, collate_fn=collate_menus)
    batch = next(iter(loader))

    probe = AttributeProbe()
    probe.train()

    # Dummy question template indices
    q_idx = torch.randint(0, N_QUESTION_TEMPLATES, (8,))
    logits = probe(batch, q_idx)
    print(f"Output logits shape: {tuple(logits.shape)}")  # expect (8,)
    print(f"Sample logits: {logits[:4].tolist()}")
    print(f"Param count: {sum(p.numel() for p in probe.parameters()):,}")