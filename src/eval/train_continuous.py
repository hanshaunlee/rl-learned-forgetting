"""Train one continuous-compression baseline + its probes at a fixed K.

    pca: SVD-fit the train pool once, then train probes on frozen z.
    ae:  train encoder/decoder on reconstruction, then train probes on frozen z.
    vib: train encoder + both probes jointly with task loss + KL bottleneck.

    python -m src.eval.train_continuous --method vib --k 2
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.paths import CHECKPOINTS_DIR
from src.data.feature_dataset import FeatureDataset, collate_menus
from src.probes.question_bank import (
    QuestionBank, _build_category_index, question_to_template_idx,
)
from src.eval.continuous import (
    make_pool,
    build_compressor,
    PCACompressor, AECompressor, VIBCompressor,
    ContinuousAttributeProbe, ContinuousRecognitionProbe,
    kl_to_standard_normal,
)

DEVICE = "cuda"
SEED = 231
BATCH_SIZE = 128
P_CAPTION_RELEVANT = 0.7
INFONCE_TEMP = 0.07

# epochs kept near the discrete probes so the capacity comparison stays fair
AE_EPOCHS = 12
PROBE_EPOCHS = 12
VIB_EPOCHS = 15
VIB_BETA = 1e-3       # KL weight; small because K is already small
LR = 3e-4


def sample_questions_for_batch(image_ids, qb, cat_id_to_idx, rng):
    """One question per image, same as the discrete attribute trainer."""
    template_idx, answers = [], []
    for img_id in image_ids.tolist():
        q = qb.sample_question(img_id, p_caption_relevant=P_CAPTION_RELEVANT, rng=rng)
        template_idx.append(question_to_template_idx(q, cat_id_to_idx))
        answers.append(q["answer"])
    return (torch.tensor(template_idx, dtype=torch.long),
            torch.tensor(answers, dtype=torch.float))


def info_nce(anchor, reference, temperature=INFONCE_TEMP):
    """Symmetric InfoNCE on (B, D) L2-normalised embeddings."""
    logits_a = (anchor @ reference.T) / temperature
    logits_r = (reference @ anchor.T) / temperature
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return 0.5 * (F.cross_entropy(logits_a, labels) + F.cross_entropy(logits_r, labels))


@torch.no_grad()
def eval_probes(compressor, attr_probe, recog_probe, val_loader,
                qb, cat_id_to_idx, rng) -> dict:
    """Training-time proxies for the reward: binary attr accuracy and a recognition margin.

    Margin is mean(self_sim − best_other_sim). We don't use in-batch top-1 because z is
    deterministic here, so self-cosine is always 1.0 and top-1 would be trivially perfect.
    """
    compressor.eval(); attr_probe.eval(); recog_probe.eval()
    correct_attr, total_attr = 0, 0
    margin_sum, margin_n = 0.0, 0
    for batch in val_loader:
        q_idx, answers = sample_questions_for_batch(batch["image_id"], qb, cat_id_to_idx, rng)
        b = {k: v.to(DEVICE) for k, v in batch.items()}
        q_idx, answers = q_idx.to(DEVICE), answers.to(DEVICE)

        z = compressor.encode(make_pool(b))
        logits = attr_probe(z, q_idx)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct_attr += (preds == answers).sum().item()
        total_attr += answers.numel()

        emb = recog_probe(z)
        sims = emb @ emb.T                                # (B, B)
        self_sim = sims.diagonal()
        sims_no_self = sims.clone()
        sims_no_self.fill_diagonal_(-1e9)
        best_other = sims_no_self.max(dim=1).values
        margin_sum += (self_sim - best_other).sum().item()
        margin_n += emb.shape[0]
    compressor.train(); attr_probe.train(); recog_probe.train()
    return {
        "attr_acc": correct_attr / max(total_attr, 1),
        "recog_margin": margin_sum / max(margin_n, 1),
    }


def save_bundle(method: str, k: int, compressor, attr_probe, recog_probe, path):
    torch.save({
        "method": method,
        "k": k,
        "compressor": compressor.state_dict(),
        "attr_probe": attr_probe.state_dict(),
        "recog_probe": recog_probe.state_dict(),
    }, path)


def fit_pca(compressor: PCACompressor, train_loader):
    """One-shot fit over the entire train pool."""
    print("Fitting PCA over train pool (single-shot SVD)...")
    compressor.fit(train_loader, device=DEVICE)
    print(f"  fitted: components {tuple(compressor.components.shape)}  "
          f"mean ||={compressor.mean.norm().item():.3f}")


def train_autoencoder(compressor: AECompressor, train_loader, epochs: int):
    """Reconstruction-only training of the AE compressor (task-agnostic)."""
    opt = torch.optim.Adam(compressor.parameters(), lr=LR)
    compressor.train()
    for epoch in range(epochs):
        epoch_loss, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"AE recon {epoch+1}/{epochs}"):
            b = {k: v.to(DEVICE) for k, v in batch.items()}
            pool = make_pool(b)
            _, recon = compressor(pool)
            loss = F.mse_loss(recon, pool)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item(); n += 1
        print(f"  AE epoch {epoch+1}: mse={epoch_loss/n:.4f}")


def train_probes_on_frozen_z(compressor, attr_probe, recog_probe,
                             train_loader, val_loader, qb, cat_id_to_idx, rng,
                             epochs: int) -> dict:
    """Train both probes with the compressor frozen. Used by PCA and AE."""
    compressor.eval()
    for p in compressor.parameters():
        p.requires_grad_(False)

    params = list(attr_probe.parameters()) + list(recog_probe.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    bce = nn.BCEWithLogitsLoss()

    best = {"attr_acc": 0.0, "recog_margin": float("-inf")}
    for epoch in range(epochs):
        attr_probe.train(); recog_probe.train()
        ep_attr, ep_recog, n = 0.0, 0.0, 0
        for batch in tqdm(train_loader, desc=f"probes {epoch+1}/{epochs}"):
            q_idx, answers = sample_questions_for_batch(
                batch["image_id"], qb, cat_id_to_idx, rng)
            b = {k: v.to(DEVICE) for k, v in batch.items()}
            q_idx, answers = q_idx.to(DEVICE), answers.to(DEVICE)

            with torch.no_grad():
                z = compressor.encode(make_pool(b))

            attr_loss = bce(attr_probe(z, q_idx), answers)
            emb = recog_probe(z)
            recog_loss = info_nce(emb, emb)
            loss = attr_loss + recog_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_attr += attr_loss.item(); ep_recog += recog_loss.item(); n += 1

        metrics = eval_probes(compressor, attr_probe, recog_probe,
                              val_loader, qb, cat_id_to_idx, rng)
        print(f"  epoch {epoch+1}: attr_loss={ep_attr/n:.4f}  "
              f"recog_loss={ep_recog/n:.4f}  "
              f"val_attr={metrics['attr_acc']:.4f}  "
              f"val_recog_margin={metrics['recog_margin']:.4f}")
        # "best epoch" for the printout only — real metric is the reward in
        # run_continuous.py. Pick on attr_acc, break ties on margin (different scale).
        if (metrics['attr_acc'], metrics['recog_margin']) > \
           (best['attr_acc'], best['recog_margin']):
            best = metrics
    return best


def train_vib_joint(compressor: VIBCompressor, attr_probe, recog_probe,
                    train_loader, val_loader, qb, cat_id_to_idx, rng,
                    epochs: int, beta: float) -> dict:
    """End-to-end: encoder + probes share gradients; KL bottleneck applies."""
    params = (list(compressor.parameters())
              + list(attr_probe.parameters())
              + list(recog_probe.parameters()))
    opt = torch.optim.Adam(params, lr=LR)
    bce = nn.BCEWithLogitsLoss()

    best = {"attr_acc": 0.0, "recog_margin": float("-inf")}
    for epoch in range(epochs):
        compressor.train(); attr_probe.train(); recog_probe.train()
        ep_attr, ep_recog, ep_kl, n = 0.0, 0.0, 0.0, 0
        for batch in tqdm(train_loader, desc=f"VIB joint {epoch+1}/{epochs}"):
            q_idx, answers = sample_questions_for_batch(
                batch["image_id"], qb, cat_id_to_idx, rng)
            b = {k: v.to(DEVICE) for k, v in batch.items()}
            q_idx, answers = q_idx.to(DEVICE), answers.to(DEVICE)

            pool = make_pool(b)
            # two posterior samples: the second gives InfoNCE a non-degenerate
            # anchor/reference pair, which is what makes z stable under the noise
            z_attr, mu, logvar = compressor(pool)
            z_recog = compressor.reparameterize(mu, logvar)

            attr_loss = bce(attr_probe(z_attr, q_idx), answers)
            anchor = recog_probe(z_attr)
            reference = recog_probe(z_recog)
            recog_loss = info_nce(anchor, reference)
            kl = kl_to_standard_normal(mu, logvar)
            loss = attr_loss + recog_loss + beta * kl

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_attr += attr_loss.item(); ep_recog += recog_loss.item()
            ep_kl += kl.item(); n += 1

        metrics = eval_probes(compressor, attr_probe, recog_probe,
                              val_loader, qb, cat_id_to_idx, rng)
        print(f"  epoch {epoch+1}: attr_loss={ep_attr/n:.4f}  "
              f"recog_loss={ep_recog/n:.4f}  kl={ep_kl/n:.3f}  "
              f"val_attr={metrics['attr_acc']:.4f}  "
              f"val_recog_margin={metrics['recog_margin']:.4f}")
        # same best-epoch bookkeeping as above
        if (metrics['attr_acc'], metrics['recog_margin']) > \
           (best['attr_acc'], best['recog_margin']):
            best = metrics
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["pca", "ae", "vib"], required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--vib-beta", type=float, default=VIB_BETA)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    qb = QuestionBank()
    cat_id_to_idx = _build_category_index(qb.coco)

    train_ds = FeatureDataset(split="train")
    val_ds = FeatureDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_menus)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_menus)

    compressor = build_compressor(args.method, args.k).to(DEVICE)
    attr_probe = ContinuousAttributeProbe(z_dim=args.k).to(DEVICE)
    recog_probe = ContinuousRecognitionProbe(z_dim=args.k).to(DEVICE)

    print(f"Continuous baseline: method={args.method}  K={args.k}  "
          f"compressor params={sum(p.numel() for p in compressor.parameters()):,}  "
          f"probe params={sum(p.numel() for p in attr_probe.parameters()) + sum(p.numel() for p in recog_probe.parameters()):,}")

    if args.method == "pca":
        # PCA loader: shuffle off, single pass.
        pca_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate_menus)
        fit_pca(compressor, pca_loader)
        best = train_probes_on_frozen_z(compressor, attr_probe, recog_probe,
                                        train_loader, val_loader, qb, cat_id_to_idx,
                                        rng, PROBE_EPOCHS)

    elif args.method == "ae":
        train_autoencoder(compressor, train_loader, AE_EPOCHS)
        best = train_probes_on_frozen_z(compressor, attr_probe, recog_probe,
                                        train_loader, val_loader, qb, cat_id_to_idx,
                                        rng, PROBE_EPOCHS)

    elif args.method == "vib":
        best = train_vib_joint(compressor, attr_probe, recog_probe,
                               train_loader, val_loader, qb, cat_id_to_idx,
                               rng, VIB_EPOCHS, args.vib_beta)

    ckpt = CHECKPOINTS_DIR / f"continuous_{args.method}_k{args.k}.pt"
    save_bundle(args.method, args.k, compressor, attr_probe, recog_probe, ckpt)
    print(f"\nBest val proxies — attr={best['attr_acc']:.4f}  "
          f"recog_margin={best['recog_margin']:.4f}")
    print(f"Saved bundle to {ckpt}")


if __name__ == "__main__":
    main()
