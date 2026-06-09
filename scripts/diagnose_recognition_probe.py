"""Diagnostics for the recognition probe's suspiciously-high hard-distractor accuracy.

Investigates four hypotheses without modifying any model/training code:

  H1  Lineup size: are nn_distractors lineups often < N_WAY because the neighbor
      table is filtered to the val split?
  H2  Global-token fingerprinting: do the protected siglip/dinov2_cls globals make
      every image trivially self-identifiable, even against hard neighbors?
  H3  Effective dropout strength: with globals protected and few object slots per
      image, how many candidates actually get dropped on average?
  H4  Anchor vs reference leak: after the recent guard change, is dropout actually
      active at eval (so anchor != reference for the same image)?
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.paths import CHECKPOINTS_DIR, FEATURES_DIR
from src.data.feature_dataset import FeatureDataset, collate_menus, MAX_SLOTS, MENU_SIZE
from src.probes.recognition_probe import RecognitionProbe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_WAY = 4
SEED = 231


def banner(title):
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}")


# -------------------------------------------------------------------------
# H1: lineup-size distribution
# -------------------------------------------------------------------------
def h1_lineup_sizes():
    banner("H1  LINEUP-SIZE DISTRIBUTION (val-split filtered)")

    data = np.load(FEATURES_DIR / "neighbors.npz")
    image_ids_all = data["image_ids"]
    semantic_nn = data["semantic_nn"]
    perceptual_nn = data["perceptual_nn"]
    id_to_idx = {int(iid): i for i, iid in enumerate(image_ids_all)}

    val_ds = FeatureDataset(split="val")
    val_ids = val_ds.image_ids
    val_id_set = set(val_ids)
    print(f"val split: {len(val_ids)} images "
          f"(of {len(image_ids_all)} total in neighbor table)")
    print(f"neighbor TOP_K stored: {semantic_nn.shape[1]}")
    print(f"N_WAY = {N_WAY}  (need >= {N_WAY-1} distractors)")

    def lineup_stats(nn_table, label):
        # Mirror evaluate_retrieval.nn_distractors exactly: keep neighbors in val,
        # then slice to first (N_WAY-1).
        n_in_val_among_topk = []        # how many of the top-K stored are in val
        lineup_distractor_counts = []   # actual distractors used per query
        for iid in val_ids:
            idx = id_to_idx[iid]
            neighbor_ids = [int(image_ids_all[n]) for n in nn_table[idx]
                            if int(image_ids_all[n]) in val_id_set]
            n_in_val_among_topk.append(len(neighbor_ids))
            lineup_distractor_counts.append(min(len(neighbor_ids), N_WAY - 1))

        n_in_val_among_topk = np.array(n_in_val_among_topk)
        ld = np.array(lineup_distractor_counts)
        ls = ld + 1   # lineup includes the true image

        print(f"\n--- {label} ---")
        print(f"Neighbors-in-val (out of top-{nn_table.shape[1]} stored)  "
              f"min={n_in_val_among_topk.min()}  mean={n_in_val_among_topk.mean():.1f}  "
              f"median={int(np.median(n_in_val_among_topk))}  max={n_in_val_among_topk.max()}")

        print(f"Actual lineup size (1 true + distractors):")
        hist = Counter(ls.tolist())
        for size in sorted(hist):
            pct = 100 * hist[size] / len(ls)
            bar = "#" * int(pct / 2)
            print(f"  size={size}: {hist[size]:4d}  ({pct:5.1f}%)  {bar}")

        n_short = int((ls < N_WAY).sum())
        n_singleton = int((ls == 1).sum())
        chance = 1.0 / ls.mean()
        print(f"Lineups smaller than N_WAY={N_WAY}:  "
              f"{n_short}/{len(ls)} ({100*n_short/len(ls):.1f}%)")
        print(f"Singleton lineups (zero distractors!): {n_singleton}/{len(ls)} "
              f"({100*n_singleton/len(ls):.1f}%)")
        print(f"Effective avg chance accuracy: 1 / mean_lineup_size "
              f"= 1/{ls.mean():.2f} = {chance:.3f}  "
              f"(vs nominal 1/{N_WAY} = {1/N_WAY:.3f})")

    lineup_stats(semantic_nn, "SEMANTIC (SigLIP NN)")
    lineup_stats(perceptual_nn, "PERCEPTUAL (DINOv2 NN)")


# -------------------------------------------------------------------------
# H2 & H3 & H4: need the model
# -------------------------------------------------------------------------
def load_probe():
    probe = RecognitionProbe().to(DEVICE)
    ckpt = CHECKPOINTS_DIR / "recognition_probe.pt"
    if ckpt.exists():
        probe.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("WARNING: no checkpoint found, using random init.")
    probe.eval()
    return probe


def embed_split(probe, split="val", apply_dropout=False, seed=None):
    """Embed every image once. Returns dict iid -> (D,) tensor on CPU."""
    if seed is not None:
        torch.manual_seed(seed)
    ds = FeatureDataset(split=split)
    loader = DataLoader(ds, batch_size=256, num_workers=0, collate_fn=collate_menus)
    out = {}
    with torch.no_grad():
        for batch in loader:
            ids = batch["image_id"].tolist()
            b = {k: v.to(DEVICE) for k, v in batch.items()}
            emb = probe.encode(b, apply_dropout=apply_dropout).cpu()
            for j, iid in enumerate(ids):
                out[iid] = emb[j]
    return out


def h2_global_fingerprinting(probe):
    banner("H2  GLOBAL-TOKEN FINGERPRINTING")

    data = np.load(FEATURES_DIR / "neighbors.npz")
    image_ids_all = data["image_ids"]
    semantic_nn = data["semantic_nn"]
    perceptual_nn = data["perceptual_nn"]
    id_to_idx = {int(iid): i for i, iid in enumerate(image_ids_all)}

    val_ds = FeatureDataset(split="val")
    val_ids = val_ds.image_ids
    val_id_set = set(val_ids)

    # Reference: no dropout
    refs = embed_split(probe, split="val", apply_dropout=False)
    # Anchor with dropout active (the "sparse" view)
    ancs = embed_split(probe, split="val", apply_dropout=True, seed=SEED)
    # Anchor with NO dropout (pure-fingerprint ablation: anchor == reference modulo attention)
    ancs_full = embed_split(probe, split="val", apply_dropout=False)

    def cos(a, b):
        return float((a * b).sum())

    self_sims = []
    self_sims_no_dropout = []   # ancs_full vs refs (should be 1.0 -- same encoder, same input)
    sem_hardest_sims = []
    perc_hardest_sims = []
    sem_hardest_minus_self = []
    perc_hardest_minus_self = []
    sem_pred_correct = []
    perc_pred_correct = []

    for iid in val_ids:
        a = ancs[iid]
        r_self = refs[iid]
        s_self = cos(a, r_self)
        self_sims.append(s_self)
        self_sims_no_dropout.append(cos(ancs_full[iid], r_self))

        idx = id_to_idx[iid]

        for nn_table, sims_list, minus_list, correct_list in [
            (semantic_nn, sem_hardest_sims, sem_hardest_minus_self, sem_pred_correct),
            (perceptual_nn, perc_hardest_sims, perc_hardest_minus_self, perc_pred_correct),
        ]:
            neighbor_ids = [int(image_ids_all[n]) for n in nn_table[idx]
                            if int(image_ids_all[n]) in val_id_set]
            distractor_ids = neighbor_ids[:N_WAY - 1]
            if not distractor_ids:
                continue
            distractor_sims = [cos(a, refs[d]) for d in distractor_ids]
            hardest = max(distractor_sims)
            sims_list.append(hardest)
            minus_list.append(s_self - hardest)
            # The lineup includes the true image; correct iff cos(a, r_self) > any distractor sim.
            correct_list.append(s_self > hardest)

    def summary(name, arr):
        a = np.array(arr)
        print(f"  {name:42s}  n={len(a):4d}  "
              f"mean={a.mean(): .4f}  median={np.median(a): .4f}  "
              f"std={a.std():.4f}  min={a.min(): .4f}  max={a.max(): .4f}")

    print("\nCosine similarities (anchor with dropout vs reference, val split):")
    summary("(a) self  cos(anchor, own_ref)",          self_sims)
    summary("(*) no-dropout self (sanity, ~1.0?)",     self_sims_no_dropout)
    summary("(b) hardest semantic NN distractor",      sem_hardest_sims)
    summary("(c) hardest perceptual NN distractor",    perc_hardest_sims)
    print()
    summary("self - hardest_semantic_NN  (margin)",    sem_hardest_minus_self)
    summary("self - hardest_perceptual_NN (margin)",   perc_hardest_minus_self)
    print()
    print(f"  Implied top-1 vs hardest semantic NN:    {np.mean(sem_pred_correct):.4f}")
    print(f"  Implied top-1 vs hardest perceptual NN:  {np.mean(perc_pred_correct):.4f}")

    # --- Ablation: re-embed with the global rows literally masked out ---
    # We don't touch the model; we tweak the batch so siglip and dinov2_cls are zeroed.
    # This isolates: "if you remove the unique globals, is hard retrieval still trivial?"
    banner("H2-ablation  ZERO-OUT GLOBALS (siglip + dinov2_cls)")
    print("Re-embedding val with siglip and dinov2_cls fed as ZEROS (everything else unchanged).")
    print("If hard accuracy plummets, the globals were doing the work.")

    val_ds = FeatureDataset(split="val")
    loader = DataLoader(val_ds, batch_size=256, num_workers=0, collate_fn=collate_menus)

    refs_no_glob = {}
    ancs_no_glob = {}
    torch.manual_seed(SEED)
    with torch.no_grad():
        for batch in loader:
            ids = batch["image_id"].tolist()
            b = {k: v.to(DEVICE) for k, v in batch.items()}
            b["siglip"] = torch.zeros_like(b["siglip"])
            b["dinov2_cls"] = torch.zeros_like(b["dinov2_cls"])
            r = probe.encode(b, apply_dropout=False).cpu()
            a = probe.encode(b, apply_dropout=True).cpu()
            for j, iid in enumerate(ids):
                refs_no_glob[iid] = r[j]
                ancs_no_glob[iid] = a[j]

    sem_correct = perc_correct = total = 0
    for iid in val_ids:
        a = ancs_no_glob[iid]
        r_self = refs_no_glob[iid]
        s_self = cos(a, r_self)
        idx = id_to_idx[iid]
        for nn_table, name in [(semantic_nn, "sem"), (perceptual_nn, "perc")]:
            neighbor_ids = [int(image_ids_all[n]) for n in nn_table[idx]
                            if int(image_ids_all[n]) in val_id_set]
            distractor_ids = neighbor_ids[:N_WAY - 1]
            if not distractor_ids:
                continue
            sims = [cos(a, refs_no_glob[d]) for d in distractor_ids]
            ok = s_self > max(sims)
            if name == "sem":
                sem_correct += int(ok)
            else:
                perc_correct += int(ok)
        total += 1

    print(f"  No-globals semantic   top-1 vs hardest NN: {sem_correct/total:.4f}")
    print(f"  No-globals perceptual top-1 vs hardest NN: {perc_correct/total:.4f}")


# -------------------------------------------------------------------------
# H3: effective dropout strength
# -------------------------------------------------------------------------
def h3_dropout_strength(probe):
    banner("H3  EFFECTIVE DROPOUT STRENGTH")

    ds = FeatureDataset(split="val")
    loader = DataLoader(ds, batch_size=256, num_workers=0, collate_fn=collate_menus)

    valid_full = []     # candidates valid before dropout (excludes padded slots)
    valid_after = []    # candidates valid after dropout
    n_slots_real = []   # how many object slots are real (not padding)
    n_dropped = []      # how many were dropped
    n_dropped_droppable = []  # of the droppable subset

    torch.manual_seed(SEED)
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(DEVICE) for k, v in batch.items()}
            _, valid_pre = probe.build_candidates(b, apply_dropout=False)
            _, valid_post = probe.build_candidates(b, apply_dropout=True)
            n_real = b["slot_valid"].sum(dim=1).cpu()              # (B,)
            vf = valid_pre.sum(dim=1).cpu()
            va = valid_post.sum(dim=1).cpu()
            valid_full.append(vf)
            valid_after.append(va)
            n_slots_real.append(n_real)
            n_dropped.append(vf - va)
            # Droppable rows = (everything except protected globals) that started valid.
            # Globals (idx 0,1) are protected; their `valid` is always True before dropout.
            # So droppable starting count = vf - 2
            n_dropped_droppable.append((vf - va).float() / (vf - 2).clamp(min=1).float())

    vf = torch.cat(valid_full)
    va = torch.cat(valid_after)
    nr = torch.cat(n_slots_real)
    nd = torch.cat(n_dropped)
    rate = torch.cat(n_dropped_droppable)

    print(f"\nMenu layout: MENU_SIZE={MENU_SIZE}  (1 SigLIP + 1 DINOv2_CLS "
          f"+ {MAX_SLOTS} slots + 1 lowlevel = {MENU_SIZE})")
    print(f"Configured slot_dropout = {probe.slot_dropout}\n")

    print(f"Real object slots per image (excluding padding):")
    print(f"  mean={nr.float().mean():.2f}  median={int(nr.median())}  "
          f"min={int(nr.min())}  max={int(nr.max())}")
    hist = Counter(nr.tolist())
    for k in sorted(hist):
        pct = 100*hist[k]/len(nr)
        bar = "#" * int(pct/2)
        print(f"    {k:2d} slots: {hist[k]:4d}  ({pct:5.1f}%)  {bar}")

    print(f"\nValid candidates (pre-dropout, i.e. 2 globals + real slots + 1 lowlevel):")
    print(f"  mean={vf.float().mean():.2f}  min={int(vf.min())}  max={int(vf.max())}")

    print(f"\nValid candidates (post-dropout):")
    print(f"  mean={va.float().mean():.2f}  min={int(va.min())}  max={int(va.max())}")

    print(f"\nCandidates actually dropped per image:")
    print(f"  mean={nd.float().mean():.2f}  min={int(nd.min())}  max={int(nd.max())}")
    hist = Counter(nd.tolist())
    for k in sorted(hist):
        pct = 100*hist[k]/len(nd)
        bar = "#" * int(pct/2)
        print(f"    drop {k:2d}: {hist[k]:4d}  ({pct:5.1f}%)  {bar}")

    print(f"\nFraction of *droppable* candidates dropped (effective dropout rate):")
    print(f"  mean={rate.mean():.3f}  (configured = {probe.slot_dropout})")

    n_only_globals = int((va == 2).sum())
    n_drop_zero = int((nd == 0).sum())
    print(f"\nImages where ONLY globals survive (everything droppable was dropped): "
          f"{n_only_globals}/{len(va)} ({100*n_only_globals/len(va):.1f}%)")
    print(f"Images with zero drops: {n_drop_zero}/{len(nd)} "
          f"({100*n_drop_zero/len(nd):.1f}%)")


# -------------------------------------------------------------------------
# H4: anchor != reference at eval (dropout active in eval)
# -------------------------------------------------------------------------
def h4_anchor_vs_reference(probe):
    banner("H4  ANCHOR vs REFERENCE AT EVAL (is dropout actually active?)")

    ds = FeatureDataset(split="val")
    loader = DataLoader(ds, batch_size=64, num_workers=0, collate_fn=collate_menus)

    probe.eval()
    is_training = probe.training
    print(f"probe.training = {is_training}  (should be False at eval)\n")

    # Compare candidate sets directly to prove dropout is firing
    batch = next(iter(loader))
    b = {k: v.to(DEVICE) for k, v in batch.items()}
    torch.manual_seed(SEED)
    with torch.no_grad():
        _, valid_no_drop = probe.build_candidates(b, apply_dropout=False)
        _, valid_drop_a = probe.build_candidates(b, apply_dropout=True)
        _, valid_drop_b = probe.build_candidates(b, apply_dropout=True)

    print("First-batch valid-mask sums (apply_dropout=False, =True call#1, =True call#2):")
    for i in range(min(6, valid_no_drop.shape[0])):
        print(f"  image_id={int(batch['image_id'][i])}: "
              f"no_drop={int(valid_no_drop[i].sum())}  "
              f"drop_a={int(valid_drop_a[i].sum())}  "
              f"drop_b={int(valid_drop_b[i].sum())}")
    print()

    any_drop_a = (valid_drop_a != valid_no_drop).any().item()
    any_drop_b = (valid_drop_b != valid_no_drop).any().item()
    differ_aa = (valid_drop_a != valid_drop_b).any().item()
    print(f"apply_dropout=True changes valid mask vs no-drop?  call#1: {any_drop_a}  "
          f"call#2: {any_drop_b}")
    print(f"Two consecutive dropped passes differ from each other? {differ_aa}  "
          f"(should be True -- random)")

    # Globals must always survive even under dropout
    glob0_kept_drop = bool(valid_drop_a[:, 0].all().item() and valid_drop_b[:, 0].all().item())
    glob1_kept_drop = bool(valid_drop_a[:, 1].all().item() and valid_drop_b[:, 1].all().item())
    print(f"siglip (idx 0) always kept under dropout?  {glob0_kept_drop}")
    print(f"dinov2_cls (idx 1) always kept under dropout? {glob1_kept_drop}")

    # Direct embedding comparison: same input, dropout on vs off
    torch.manual_seed(SEED)
    with torch.no_grad():
        ref = probe.encode(b, apply_dropout=False).cpu()
        anc = probe.encode(b, apply_dropout=True).cpu()
        anc2 = probe.encode(b, apply_dropout=True).cpu()
    cos_self = (ref * anc).sum(dim=-1)
    cos_anc_anc2 = (anc * anc2).sum(dim=-1)
    print()
    print(f"cos(reference, anchor) for same image  -- mean={cos_self.mean():.4f}  "
          f"min={cos_self.min():.4f}  max={cos_self.max():.4f}")
    print(f"cos(anchor#1, anchor#2) same image, two dropout draws -- "
          f"mean={cos_anc_anc2.mean():.4f}  min={cos_anc_anc2.min():.4f}")
    print()
    if cos_self.mean().item() > 0.9999:
        print(">> ref ~= anchor (cos > 0.9999). Dropout is NOT changing the embedding.")
    elif cos_self.mean().item() > 0.95:
        print(">> Anchor very close to reference. Dropout is firing but barely changes the embedding.")
    else:
        print(">> Anchor visibly differs from reference. Dropout is having a real effect.")


def main():
    h1_lineup_sizes()
    probe = load_probe()
    h2_global_fingerprinting(probe)
    h3_dropout_strength(probe)
    h4_anchor_vs_reference(probe)


if __name__ == "__main__":
    main()
