"""Continuous compression baselines: PCA, AE, VIB.

Each compresses a fixed per-image feature pool to a K-dim z; a small attribute
probe and recognition probe then read z for the same two tasks the discrete
selector is rewarded on. The probes consume z directly rather than being forced
back into the token menu — the discrete probes were trained on real menu tokens,
so feeding them a reshaped z would just be out-of-distribution. Slots are
mean-pooled into one object summary; keeping per-slot identity would smuggle a
discrete selection back in.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.feature_dataset import DIM_SIGLIP, DIM_DINOV2, DIM_LOWLEVEL
from src.probes.attribute_probe import N_QUESTION_TEMPLATES


POOL_DIM = DIM_SIGLIP + DIM_DINOV2 + DIM_DINOV2 + DIM_LOWLEVEL  # 768+768+768+108=2412
RECOG_EMBED_DIM = 32


def make_pool(batch: dict) -> torch.Tensor:
    """Concatenate [siglip, dinov2_cls, slot_mean, lowlevel] into a (B, POOL_DIM) pool."""
    # invalid slots are zero in the dataset, so a valid-count divide gives the
    # right mean; an image with no slots gets a zero slot_mean
    slot_sum = batch["slots"].sum(dim=1)
    n_valid = batch["slot_valid"].sum(dim=1, keepdim=True).float().clamp(min=1.0)
    slot_mean = slot_sum / n_valid
    return torch.cat([batch["siglip"], batch["dinov2_cls"], slot_mean,
                      batch["lowlevel"]], dim=-1)


# --- continuous probes (shared by all three compressors) ---
class ContinuousAttributeProbe(nn.Module):
    """K-dim z + question template -> binary yes/no logit."""
    def __init__(self, z_dim: int, d_model: int = 256):
        super().__init__()
        self.adapt_z = nn.Linear(z_dim, d_model)
        self.question_embed = nn.Embedding(N_QUESTION_TEMPLATES, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, z: torch.Tensor, question_template_idx: torch.Tensor) -> torch.Tensor:
        h = self.adapt_z(z)
        q = self.question_embed(question_template_idx)
        return self.head(torch.cat([h, q], dim=-1)).squeeze(-1)


class ContinuousRecognitionProbe(nn.Module):
    """K-dim z -> 32-dim L2-normalised embedding.

    Each method gets its own gallery from its own probe, so the embedding space is
    method-specific — but retrieval correctness is space-invariant, which keeps the
    cross-method comparison honest.
    """
    def __init__(self, z_dim: int, d_model: int = 256, embed_dim: int = RECOG_EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(z_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(z), dim=-1)


# --- compressors ---
class Compressor(nn.Module):
    """pool -> z. encode() returns (B, k)."""
    name: str = "base"
    k: int

    def encode(self, pool: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def transform(self, batch: dict) -> torch.Tensor:
        return self.encode(make_pool(batch))


class PCACompressor(Compressor):
    """Linear PCA via truncated SVD on the train pool. The no-learning baseline."""
    name = "pca"

    def __init__(self, k: int, pool_dim: int = POOL_DIM):
        super().__init__()
        self.k = k
        self.register_buffer("mean", torch.zeros(pool_dim))
        self.register_buffer("components", torch.zeros(k, pool_dim))
        self._fitted = False

    @torch.no_grad()
    def fit(self, loader, device: str = "cuda"):
        pools = []
        for batch in loader:
            b = {kk: v.to(device) if torch.is_tensor(v) else v for kk, v in batch.items()}
            pools.append(make_pool(b).cpu())
        X = torch.cat(pools, dim=0)
        mean = X.mean(dim=0)
        Xc = X - mean
        # economy SVD; top-K rows of Vt are the principal axes
        _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
        self.mean.copy_(mean)
        self.components.copy_(Vt[: self.k])
        self._fitted = True

    def encode(self, pool: torch.Tensor) -> torch.Tensor:
        return (pool - self.mean) @ self.components.T


class AECompressor(Compressor):
    """Bottleneck-K autoencoder, MSE reconstruction. Nonlinear but task-agnostic —
    the natural contrast for VIB (same architecture, no task signal)."""
    name = "ae"

    def __init__(self, k: int, pool_dim: int = POOL_DIM, hidden: int = 512):
        super().__init__()
        self.k = k
        self.encoder = nn.Sequential(
            nn.Linear(pool_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, k),
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, pool_dim),
        )

    def encode(self, pool: torch.Tensor) -> torch.Tensor:
        return self.encoder(pool)

    def forward(self, pool: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(pool)
        recon = self.decoder(z)
        return z, recon


class VIBCompressor(Compressor):
    """Variational Information Bottleneck (Alemi et al., 2017).

    Encoder gives q(z|pool) = N(mu, diag(sigma^2)); trained jointly with the probes
    on task loss + beta*KL(q || N(0,I)). The KL squeezes info through the bottleneck,
    the task loss keeps what survives relevant. Eval uses the posterior mean.
    """
    name = "vib"

    def __init__(self, k: int, pool_dim: int = POOL_DIM, hidden: int = 512):
        super().__init__()
        self.k = k
        self.encoder = nn.Sequential(
            nn.Linear(pool_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * k),         # concatenated [mu, logvar]
        )

    def encode_dist(self, pool: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(pool)
        mu, logvar = h[..., : self.k], h[..., self.k :]
        # Clamp logvar to avoid blow-up early in training.
        logvar = logvar.clamp(-8.0, 8.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def encode(self, pool: torch.Tensor) -> torch.Tensor:
        """Deterministic encoding (posterior mean) — for eval and PCA-style use."""
        mu, _ = self.encode_dist(pool)
        return mu

    def forward(self, pool: torch.Tensor):
        """Stochastic encoding for training. Returns (z, mu, logvar)."""
        mu, logvar = self.encode_dist(pool)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return z, mu, logvar


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)) summed over latent dims, mean over batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()


def build_compressor(method: str, k: int) -> Compressor:
    if method == "pca":
        return PCACompressor(k)
    if method == "ae":
        return AECompressor(k)
    if method == "vib":
        return VIBCompressor(k)
    raise ValueError(f"unknown method {method!r}; expected one of pca/ae/vib")


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.data.feature_dataset import FeatureDataset, collate_menus

    K = 2
    ds = FeatureDataset(split="val")
    loader = DataLoader(ds, batch_size=8, num_workers=0, collate_fn=collate_menus)
    pool = make_pool(next(iter(loader)))
    print("pool:", tuple(pool.shape))
    for method in ["pca", "ae", "vib"]:
        c = build_compressor(method, K)
        if method != "pca":
            print(method, "z:", tuple(c.encode(pool).shape))
