"""Contrastive pretraining for the recognition probe (InfoNCE, in-batch negatives).

Anchor = sparse (dropped-out) menu embedding; reference = full menu embedding.
Pulls each anchor toward its own reference, pushes from other images' references.
Evaluates retrieval against both random and hard (semantic/perceptual NN) distractors.
"""
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.paths import CHECKPOINTS_DIR, FEATURES_DIR
from src.data.feature_dataset import FeatureDataset, collate_menus
from src.probes.recognition_probe import RecognitionProbe

DEVICE = "cuda"
BATCH_SIZE = 128
N_EPOCHS = 15
LR = 3e-4
TEMPERATURE = 0.07
SEED = 231
LINEUP_SIZE = 50   # 1 true + (LINEUP_SIZE-1) distractors; chance = 1/LINEUP_SIZE


def info_nce_loss(anchor, reference, temperature):
    """Standard InfoNCE with in-batch negatives. anchor, reference: (B, D) normalized."""
    logits = (anchor @ reference.T) / temperature   # (B, B)
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, labels)


def compute_val_neighbors(val_ids):
    """Compute within-val nearest neighbors over raw SigLIP / DINOv2-CLS embeddings.

    Returns (semantic_nn, perceptual_nn), each shape (N_val, N_val) with val-local
    indices sorted by descending cosine similarity (col 0 = closest non-self).
    """
    val_ids = list(val_ids)
    with h5py.File(FEATURES_DIR / "siglip_val2017.h5", "r") as f:
        siglip = np.stack([f[str(i)][:] for i in val_ids]).astype(np.float32)
    with h5py.File(FEATURES_DIR / "dinov2_val2017.h5", "r") as f:
        dinov2 = np.stack([f[str(i)]["cls"][:] for i in val_ids]).astype(np.float32)

    def cosine_nn(emb):
        normed = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sim = normed @ normed.T
        np.fill_diagonal(sim, -np.inf)
        return np.argsort(-sim, axis=1)

    return cosine_nn(siglip), cosine_nn(dinov2)


@torch.no_grad()
def evaluate_retrieval(probe, val_ds, semantic_nn, perceptual_nn, rng):
    """Top-1 retrieval accuracy against random / semantic / perceptual distractors."""
    probe.eval()

    val_ids = val_ds.image_ids
    full_dataset = FeatureDataset(split="val")
    loader = DataLoader(full_dataset, batch_size=256, num_workers=0, collate_fn=collate_menus)

    ref_embeds = {}
    anchor_embeds = {}
    for batch in loader:
        ids = batch["image_id"].tolist()
        b = {k: v.to(DEVICE) for k, v in batch.items()}
        refs = probe.encode(b, sample_k=False).cpu()   # full menu
        ancs = probe.encode(b, sample_k=True).cpu()    # K-sampled menu
        for j, iid in enumerate(ids):
            ref_embeds[iid] = refs[j]
            anchor_embeds[iid] = ancs[j]

    val_pos = {iid: i for i, iid in enumerate(val_ids)}

    def retrieval_acc(distractor_fn):
        correct = 0
        for iid in val_ids:
            anchor = anchor_embeds[iid]
            distractor_ids = distractor_fn(iid)
            lineup_ids = [iid] + distractor_ids
            lineup = torch.stack([ref_embeds[d] for d in lineup_ids])
            sims = lineup @ anchor
            pred = sims.argmax().item()
            if pred == 0:
                correct += 1
        return correct / len(val_ids)

    def random_distractors(iid):
        choices = [x for x in val_ids if x != iid]
        return rng.choice(choices, size=LINEUP_SIZE - 1, replace=False).tolist()

    def nn_distractors(iid, nn_table):
        pos = val_pos[iid]
        neighbor_local = nn_table[pos][:LINEUP_SIZE - 1]
        return [val_ids[n] for n in neighbor_local]

    acc_random = retrieval_acc(random_distractors)
    acc_semantic = retrieval_acc(lambda iid: nn_distractors(iid, semantic_nn))
    acc_perceptual = retrieval_acc(lambda iid: nn_distractors(iid, perceptual_nn))
    return acc_random, acc_semantic, acc_perceptual


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = FeatureDataset(split="train")
    val_ds = FeatureDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_menus)

    print("Computing within-val nearest neighbors for hard distractors...")
    semantic_nn, perceptual_nn = compute_val_neighbors(val_ds.image_ids)

    probe = RecognitionProbe().to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=LR)

    print(f"Training recognition probe: {len(train_ds)} train, {len(val_ds)} val  "
          f"(K={probe.k} sampling)")
    print(f"Chance retrieval accuracy ({LINEUP_SIZE}-way): {1/LINEUP_SIZE:.3f}\n")

    best_hard = 0.0
    for epoch in range(N_EPOCHS):
        probe.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}"):
            b = {k: v.to(DEVICE) for k, v in batch.items()}
            anchor = probe.encode(b, sample_k=True)      # K-sampled
            reference = probe.encode(b, sample_k=False)  # full menu
            loss = info_nce_loss(anchor, reference, TEMPERATURE)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        acc_r, acc_s, acc_p = evaluate_retrieval(
            probe, val_ds, semantic_nn, perceptual_nn, rng)
        print(f"Epoch {epoch+1}: loss={epoch_loss/len(train_loader):.4f}  "
              f"retrieval acc -> random={acc_r:.3f}  semantic={acc_s:.3f}  perceptual={acc_p:.3f}")

        # Select on the harder metric (mean of semantic + perceptual)
        hard = (acc_s + acc_p) / 2
        if hard > best_hard:
            best_hard = hard
            torch.save(probe.state_dict(), CHECKPOINTS_DIR / "recognition_probe.pt")

    print(f"\nBest hard-distractor accuracy: {best_hard:.3f}")
    print(f"Saved to {CHECKPOINTS_DIR / 'recognition_probe.pt'}")


if __name__ == "__main__":
    main()