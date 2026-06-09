"""Selection policy: score every candidate in the menu, sample K to keep."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.feature_dataset import (
    MAX_SLOTS, MENU_SIZE,
    FAM_SIGLIP, FAM_DINOV2_CLS, FAM_OBJECT_SLOT, FAM_LOWLEVEL,
    DIM_SIGLIP, DIM_DINOV2, DIM_LOWLEVEL,
)

N_FAMILIES = 4


class Selector(nn.Module):
    # Same backbone as the probes (per-family adapters + family embeddings + a
    # set transformer over candidates); the head just emits one score per slot.
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model

        self.adapt_siglip = nn.Linear(DIM_SIGLIP, d_model)
        self.adapt_dinov2 = nn.Linear(DIM_DINOV2, d_model)
        self.adapt_lowlevel = nn.Linear(DIM_LOWLEVEL, d_model)
        self.family_embed = nn.Embedding(N_FAMILIES, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        fam_ids = ([FAM_SIGLIP] + [FAM_DINOV2_CLS]
                   + [FAM_OBJECT_SLOT] * MAX_SLOTS + [FAM_LOWLEVEL])
        self.register_buffer("family_ids", torch.tensor(fam_ids, dtype=torch.long))

    def forward(self, batch: dict):
        B = batch["siglip"].shape[0]
        device = batch["siglip"].device

        siglip = self.adapt_siglip(batch["siglip"]).unsqueeze(1)
        dinov2_cls = self.adapt_dinov2(batch["dinov2_cls"]).unsqueeze(1)
        slots = self.adapt_dinov2(batch["slots"])
        lowlevel = self.adapt_lowlevel(batch["lowlevel"]).unsqueeze(1)

        candidates = torch.cat([siglip, dinov2_cls, slots, lowlevel], dim=1)  # (B, M, D)
        candidates = candidates + self.family_embed(self.family_ids).unsqueeze(0)

        valid = torch.ones(B, MENU_SIZE, dtype=torch.bool, device=device)
        valid[:, 2:2 + MAX_SLOTS] = batch["slot_valid"]

        encoded = self.transformer(candidates, src_key_padding_mask=~valid)
        scores = self.score_head(encoded).squeeze(-1)

        # -inf at padded/invalid slots so they're unreachable by softmax/argmax
        scores = scores.masked_fill(~valid, float("-inf"))
        return scores, valid


def sample_selection(scores: torch.Tensor, valid: torch.Tensor, k: int,
                     greedy: bool = False):
    """K picks without replacement (Plackett-Luce). Returns the mask and summed log-prob.

    greedy=True takes the top-K deterministically — that's what eval uses.
    """
    B, M = scores.shape
    device = scores.device
    n_valid = valid.sum(dim=1)

    mask = torch.zeros(B, M, dtype=torch.bool, device=device)
    log_prob = torch.zeros(B, device=device)
    cur_scores = scores.clone()

    for step in range(k):
        active = (mask.sum(dim=1) < n_valid) & (n_valid > 0)
        if not active.any():
            break

        # Exhausted rows (all picks taken) have all -inf scores -> softmax = NaN
        # -> multinomial CUDA-asserts. Substitute a safe uniform dist there; the
        # sample is discarded via the `active` mask below.
        safe_scores = cur_scores.masked_fill(~active.unsqueeze(1), 0.0)
        probs = F.softmax(safe_scores, dim=1)
        if greedy:
            choice = cur_scores.argmax(dim=1)
        else:
            choice = torch.multinomial(probs, num_samples=1).squeeze(1)

        step_logp = torch.log(probs.gather(1, choice.unsqueeze(1)).squeeze(1) + 1e-12)
        log_prob = log_prob + step_logp * active.float()

        rows = torch.arange(B, device=device)
        mask[rows[active], choice[active]] = True
        cur_scores[rows[active], choice[active]] = float("-inf")

    return mask, log_prob


if __name__ == "__main__":
    # scratch: eyeball that selections are exactly-K and stay inside the valid set
    from torch.utils.data import DataLoader
    from src.data.feature_dataset import FeatureDataset, collate_menus

    ds = FeatureDataset(split="train")
    loader = DataLoader(ds, batch_size=8, num_workers=0, collate_fn=collate_menus)
    batch = next(iter(loader))

    sel = Selector()
    scores, valid = sel(batch)
    mask, log_prob = sample_selection(scores, valid, k=4)
    print("selected:", mask.sum(dim=1).tolist())
    print("valid:   ", valid.sum(dim=1).tolist())
    print("log_prob:", log_prob.tolist())
    print("in valid set:", bool((mask & ~valid).sum() == 0))
    print("params:", sum(p.numel() for p in sel.parameters()))
